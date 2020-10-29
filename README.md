![](/fig/head_img.png)

## Prompt
Having been provided with two historical data sets for 311 calls and weather of New York City, the goal of this task is two-fold.
- Explore the available data sets to extract insights/trends and formulate hypotheses.
- Build a model to predict the inbound 311 calls for the next day.

**Evaluation Criteria**
1. Ability to work with messy and massive amount of data
2. Ability to implement a sensible model
3. Ability to justify your data processing and modeling choices
4. Ability to visualize and explain the findings and insights

____

## Data Processing
**311 Calls**

The entire raw dataset was too large to process on my local machine. I tried a few different strategies to process it.
- Filter/aggregate before extraction
    - I spent a little time exploring the Socrata API to filter or aggregate the data into a smaller chunk but chose a path requiring less effort
- Big Data tools
    - Tried to load raw data into BigQuery but hit a snag with timestamp formats, decided to go a different way
- Path of least resistance (Google Colab env)
    - Download full dataset to Drive
    - Launch High-RAM session
    - Load, transform, and export using H2O (built in parellelization, memory-optimized)
- Identified only features of interest and loaded subset of columns
- Filtered for `date >= '2016-01-01'`
- Aggregate at a daily level

**Weather**

No issues working with this dataset _except_ NYC's weather data points are missing after Oct 2017.

___

## Exploratory Analysis

### Calls Data

##### Visualize Different Time Series
![](/fig/daily_all_time.png)

![](/fig/daily_gte2016.png)

![](/fig/daily_5mo.png)

![](/fig/min_hrly.png)

##### Side Information

I closely examined each of the features in my subset of data. I've highlighted a few of the interesting distributions below.

`location_type` suggests the bulk of calls are residential complaints.

| location_type              | cnt     | perc   |
|----------------------------|---------|--------|
| RESIDENTIAL BUILDING       | 2482443 | 0.2635 |
| Street/Sidewalk            | 2368539 | 0.2514 |
| Residential Building/House | 1291237 |  0.137 |
| Street                     | 1030477 | 0.1094 |
| Sidewalk                   |  974511 | 0.1034 |
| Store/Commercial           |  220154 | 0.0234 |
| Park                       |   99650 | 0.0106 |
| Club/Bar/Restaurant        |   92230 | 0.0098 |


`complaint_type` does the best at describing the nature of or reason for a call. There are over four hundred different categories but the top 15 categories make up for ~60% of calls. Also notice there are several "Noise" descriptions which can likely be bucketed together.

| complaint_type                      | cnt     | perc   |
|-------------------------------------|---------|--------|
| Noise - Residential                 | 1243053 | 0.1021 |
| HEAT/HOT WATER                      |  971516 | 0.0798 |
| Illegal Parking                     |  780959 | 0.0641 |
| Blocked Driveway                    |  618701 | 0.0508 |
| Noise - Street/Sidewalk             |  493411 | 0.0405 |
| Street Condition                    |  420689 | 0.0346 |
| Street Light Condition              |  371703 | 0.0305 |
| UNSANITARY CONDITION                |  346304 | 0.0284 |
| Request Large Bulky Item Collection |  326335 | 0.0268 |
| Water System                        |  320690 | 0.0263 |
| Noise                               |  271307 | 0.0223 |
| General Construction/Plumbing       |  236913 | 0.0195 |
| PAINT/PLASTER                       |  231519 |  0.019 |
| Noise - Commercial                  |  218585 |  0.018 |
| PLUMBING                            |  216504 | 0.0178 |
| other                               | 5107295 | 0.4191 |

![](/fig/type_time.png)


##### Seasonality & Decomposition

With the trusty "eyeball test" we can see there is a weekly pattern in the time series plotted above. The ridgline plot below shows the distribution of total calls for a given day of week without the time dimension, a different view of the weekly cycle.

![](/fig/dow_dist.png)

Using `statsmodels` we can quickly perform decomposition on our time series data to smooth and determine viability for forecasting.

Also a quick ADF stat test confirms our time series data is indeed stationary with the result `p-value = 0.001`

![](/fig/decomp.png)


## Weather

##### Summarize
![](/fig/all_weather_dist.png)

![](/fig/all_weather_time.png)

##### Correlation

Contrary to what I expected there was not a significant correlation between average daily temperature and total number of calls. However when we separate "Noise" and "Other" calls into separate categories there are some relationships worth noting:

- `noise_calls` are positively but somewhat weakly correlated
- `other_calls` are negatively and more strongly correlated


|            | temp_mean | humid_mean | num_other | num_noise |
|------------|-----------|------------|-----------|-----------|
| **temp_mean**  |         1 |   0.088265 | -0.141484 |  0.323677 |
| **humid_mean** |  0.088265 |          1 | -0.057679 | -0.074224 |
| **num_other**  | -0.141484 |  -0.057679 |         1 | -0.760228 |
| **num_noise**  |  0.323677 |  -0.074224 | -0.760228 |         1 |


![](/fig/calls_temp.png)

___

## Forecast Daily Calls

#### Model Universe

I decided to limit my universe to the time range with complete weather data `'2016-01-01':'2017-10-28'`. My goal was to evaluate the lift in accuracy with weather data vs without. TBD...

#### Train Test Split

The universe contains 667 days, I chose to hold out the last 100 days of the series for my test set. This should provide a decent understanding of how well the model can forecast future daily call volumes.

#### Baseline (ARIMA)

**Accuracy**: To measure the accuracy of a given forecasting model we'll consider `mean_absolute_error` as our success metric. It's easy to understand and is in the same unit scale as the observed data.

I chose the classic statistical model ARIMA to establish a baseline `mae` score.

To work with the ARIMA model there are a few transformation steps required. First we remove seasonality and then remove overall trend using differencing. I tested 1st order and 2nd order differencing with 1st order yielding better results.

To determind the correct parameters `statsmodels` has some handy built in plotting functions. This makes it easy to iterate through different values for `p` & `q` and examine the results.

**Autoregressive order `p`**

![](/fig/arima_p.png)

**Moving average order `q`**

![](/fig/arima_q.png)

**Results**

After iterating through potential parameters I landed on the optimal combination of `(p=1, d=1, q=0)` yielding the following:

![](/fig/arima_rslt.png)

![](/fig/arima_pred.png)

> ##### Baseline MAE:     722


#### Optimize (LSTM)

Time series problems are well suited for recurrent neural networks given thier ability to pick up on recent and long-term signals. I chose to start with a bare-bones LSTM with only the time series sequence data to learn from.

The primary parameter to choose is `n_time_steps` or the number of previous days (sequence) to consider at each interval. I tested many different values but my first and obvious choice was the best. Given the weekly cycle we've observed in the data thus far it makes sense feed the network with the previous 7 days of data.

I also tried to include a number of different feature combinations including the weather data.
- Include weather data points a various time sequence lengths
- I tried two models, one model for `noise_calls` and another for `other_calls`. The separate forecasters were indeed accurate one thier own but when combining for a final prediction the success metric was worse overall
- The one model where weather made a positive impact was `other_calls`, after including `avg_temperature` & `avg_humidity` the MAE decreased a moderate amount.

At the end of the day the winning model was the simple LSTM trained solely on historical daily calls.

![](/fig/lstm_pred.png)

>  ##### LSTM MAE:     402
> ##### Reduced baseline error by ~45%

____

## Next Steps

**Temperature - Correlation vs Causation?**

Unfortunately I can't definitively answer this question. I'm certain that the weather signal can be incorporated in some fashion to improve the forecast. Also simply visually looking at the time series there appears to be an annual cycle which aligns nicely with temperature. Much more to flush out here...

**COVID Curveball**

How well would the model perform predicting inbound call for 2020? I chose not to focus on this time range given the shift in distribution beginning in 2019. In retrospect that could have been the starting index for a separate model universe. Or could one develop a single model to adjust accordingly?

**More Data**

Retrieve complete weather data for the full time range to allow for a larger training set.

**More Features**

One of the strengths of neural networks is the ability to process high-dimensional noisy data and distill the latent or hidden relationships in the data. In that sense there is much work left to be done on feature engineering and injecting side information into the LSTM framework. Especially for the text data available - clean, tokenize, and vectorize into input features.

**Model Tuning**

I did not get a chance to spend any time of network architecture or fine-tuning the LSTM parameters. This is likely to yield better results.

**Methodology**

I beleive a Probabibilistic time series model would be well suited for this problem. My goal was to utilize this third algorithm type to compare against the LSTM. Perhaps another day...

**Clean up and refactor code!!**
