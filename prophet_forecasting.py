import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Connect to database and load data
conn = sqlite3.connect("occupation_salaries (2).db")
query = "SELECT YEAR, OCC_TITLE, TOT_EMP FROM Final_Combined_Data_clean_no_dupes"
df = pd.read_sql_query(query, conn)
conn.close()

# Convert TOT_EMP to numeric and YEAR to integer
df['TOT_EMP'] = pd.to_numeric(df['TOT_EMP'], errors='coerce').fillna(0)
df['YEAR'] = df['YEAR'].astype(int)

# Filter out generic categories 
exclude_terms = ['total', 'all occupation', 'combined']
filtered_df = df[~df['OCC_TITLE'].str.lower().str.contains('|'.join(exclude_terms))]

# Get top 5 jobs by total employment
top_jobs = filtered_df.groupby('OCC_TITLE')['TOT_EMP'].sum().nlargest(5)
print("\nTop 5 jobs by total employment:")
for job, emp in top_jobs.items():
    print(f"{job}: {emp:,.0f}")

# Define colors for plotting
colors = ['blue', 'red', 'green', 'purple', 'orange']

# Create plot
plt.figure(figsize=(15, 10))

# Get last year in data and calculate future years
last_year = df['YEAR'].max()
future_years = list(range(last_year + 1, last_year + 6))  # 5 years into future
all_years = sorted(df['YEAR'].unique().tolist() + future_years)

# Process each job
for i, (job, color) in enumerate(zip(top_jobs.index, colors)):
    print(f"\nForecasting for: {job}")
    
    # Get data for this job by year
    job_data = filtered_df[filtered_df['OCC_TITLE'] == job].copy()
    yearly_data = job_data.groupby('YEAR')['TOT_EMP'].sum().reset_index()
    
    # Make sure data is sorted by year
    yearly_data = yearly_data.sort_values('YEAR')
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_data = yearly_data.rename(columns={'YEAR': 'ds', 'TOT_EMP': 'y'})
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')
    
    # Train Prophet model using all available data
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_data)
    
    # Create future dataframe for prediction
    future = pd.DataFrame({'ds': pd.date_range(
        start=f'{min(yearly_data["YEAR"])}-01-01',
        end=f'{max(future_years)}-12-31',
        freq='YS'  # Year start frequency
    )})
    
    # Make predictions
    forecast = model.predict(future)
    
    # Filter forecast to include actual years and future years
    forecast_years = forecast['ds'].dt.year.values
    mask = np.isin(forecast_years, all_years)
    filtered_forecast = forecast[mask]
    
    # Extract values for future years
    future_predictions = filtered_forecast[filtered_forecast['ds'].dt.year > last_year]['yhat'].values
    
    # Print predictions
    print("Year\tPredicted Employment")
    for year, pred in zip(future_years, future_predictions):
        print(f"{year}\t{pred:,.0f}")
    
    # Plot actual data points
    plt.scatter(yearly_data['YEAR'], yearly_data['TOT_EMP'], 
                color=color, marker='o', s=80, label=f"{job} (Actual)")
    
    # Connect actual data points with lines
    plt.plot(yearly_data['YEAR'], yearly_data['TOT_EMP'], 
            color=color, linestyle='-', alpha=0.7)
    
    # Plot future predictions with different marker
    plt.scatter(future_years, future_predictions, 
                color=color, marker='*', s=150)
    
    # Connect predictions with dashed line
    plt.plot(future_years, future_predictions, 
            color=color, linestyle='--', alpha=0.7)
    
    # Add last actual to future prediction line to connect them
    connecting_years = [yearly_data['YEAR'].iloc[-1]] + future_years
    connecting_values = [yearly_data['TOT_EMP'].iloc[-1]] + list(future_predictions)
    plt.plot(connecting_years, connecting_values, 
            color=color, linestyle='--', alpha=0.7)

# Enhance the plot
plt.title('Top 5 Jobs Employment Forecast - 5 Years into Future (Prophet)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Employment', fontsize=14)
plt.grid(True, alpha=0.3)

# Fix the x-axis to show all years
plt.xticks(all_years, rotation=45)

# Add vertical line at last actual year
plt.axvline(x=last_year, color='black', linestyle=':', linewidth=2, 
           label=f"Last Actual Data ({last_year})")

# Format y-axis with comma separator
plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

# Add legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save and show the plot
plt.tight_layout()
plt.savefig('five_year_prophet_forecast.png', bbox_inches='tight')
print("\nForecast plot saved as 'five_year_prophet_forecast.png'")