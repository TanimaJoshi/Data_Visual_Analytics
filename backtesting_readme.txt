There are a couple different ways we can go about solving this problem:
- classification to determine which jobs will actually be presetn 
- times series modeling for each of the available jobs


The second method is the route that I have chosen. This is because we will assume that the jobs will remain
but that they will grow or shrink according to trends in the time-series data. This means that we will forgo
using classification models such as SVM or random forests. We will instead use time-series models such as 
prophet, xgboost, linear regression, arima and exponential smoothing on a subset of the available jobs. 


Backtesting Pseudo code: 
Load the data from Final_Combined_Data_clean_no_dupes
Find the top 5 occupations by total employment
Apply the three remaining forecasting models:

Linear Regression
Random Forest
Prophet


Create separate visualizations for each model
Compare the performance of all models
Identify which model works best for each occupation

All other functionality remains the same, including:

The detailed visualizations for each model
MAPE calculations for accuracy comparison
Summary tables showing results by model and occupation
Identification of the best overall model


Forecasting Pseudo Code: 

Loads the same occupation data from your database
Identifies the top 5 jobs by total employment
For each job:

Trains a Prophet model on all historical data
Predicts employment for the next 5 years
Prints the forecasted values for each future year


Creates a scatter plot showing:

Historical data as solid lines with circles
Future predictions as dashed lines with stars
A vertical line marking where predictions begin
