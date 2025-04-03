import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")  # Suppress convergence warnings

# Connect to database and load data using the updated table name
conn = sqlite3.connect("occupation_salaries (2).db")
query = "SELECT YEAR, OCC_TITLE, TOT_EMP FROM Final_Combined_Data_clean_no_dupes"
df = pd.read_sql_query(query, conn)
conn.close()

# Convert TOT_EMP to numeric
df['TOT_EMP'] = pd.to_numeric(df['TOT_EMP'], errors='coerce').fillna(0)

# Ensure YEAR is properly formatted
try:
    df['YEAR'] = df['YEAR'].astype(int)
except ValueError:
    # Keep as strings if there are non-numeric years
    pass

# Print data summary
print(f"Dataset has {len(df)} rows spanning years {min(df['YEAR'])} to {max(df['YEAR'])}")

# Filter out generic categories 
exclude_terms = ['total', 'all occupation', 'combined']
filtered_df = df[~df['OCC_TITLE'].str.lower().str.contains('|'.join(exclude_terms))]

# Get top 5 jobs by total employment across all years
top_jobs = filtered_df.groupby('OCC_TITLE')['TOT_EMP'].sum().nlargest(5)
print("\nTop 5 jobs by total employment:")
for job, emp in top_jobs.items():
    print(f"{job}: {emp:,.0f}")

# Define colors and markers for consistent plotting
colors = ['blue', 'red', 'green', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'v']

# Store results for each model
all_results = {
    'Linear Regression': [],
    'Random Forest': [],
    'Prophet': []
}

# Function to create plot for a specific model
def create_model_plot(model_name, forecast_results, all_years):
    plt.figure(figsize=(15, 8))
    
    for result in forecast_results:
        job = result['Job']
        color = result['Color']
        yearly_data = result['YearlyData']
        test_year = result['TestYear']
        test_pred = result['Predicted']
        actual_val = result['Actual']
        mape = result['MAPE']
        
        # Plot actual data points
        plt.scatter(yearly_data['YEAR'], yearly_data['TOT_EMP'], 
                    color=color, marker='o', s=80, label=f"{job} (Actual)")
        
        # Connect points with lines
        plt.plot(yearly_data['YEAR'], yearly_data['TOT_EMP'], 
                color=color, linestyle='-', alpha=0.5)
        
        # Plot predicted point
        plt.scatter(test_year, test_pred, 
                    color=color, marker='*', s=200, 
                    label=f"{job} ({model_name} Pred)")
        
        # Draw a line between actual and predicted for the test year
        plt.plot([test_year, test_year], [test_pred, actual_val], 
                color=color, linestyle=':', linewidth=2)
        
        # Add text annotation
        plt.annotate(f"MAPE: {mape:.1f}%", 
                    xy=(test_year, (test_pred + actual_val) / 2), 
                    xytext=(10, 0), textcoords='offset points',
                    color=color, fontweight='bold')
    
    # Enhance the plot with labels and formatting
    plt.title(f'Top 5 Jobs by Total Employment - {model_name} Prediction', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Employment', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Fix the x-axis to show all years properly
    plt.xticks(all_years, rotation=45)
    plt.xlim(min(all_years)-0.5, max(all_years)+0.5)
    
    # Format y-axis to use comma separator
    plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Adjust legend to avoid overlapping
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    filename = f'top_5_jobs_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {model_name} as '{filename}'")

# Process each job for all models
for i, (job, color, marker) in enumerate(zip(top_jobs.index, colors, markers)):
    print(f"\nAnalyzing: {job}")
    
    # Get data for this job by year
    job_data = filtered_df[filtered_df['OCC_TITLE'] == job].copy()
    yearly_data = job_data.groupby('YEAR')['TOT_EMP'].sum().reset_index()
    
    # Make sure data is sorted by year
    yearly_data = yearly_data.sort_values('YEAR')
    print(yearly_data)
    
    # Only process if we have at least 3 data points
    if len(yearly_data) >= 3:
        # Split into train/test
        train = yearly_data.iloc[:-1]  # all but last year
        test = yearly_data.iloc[-1:]   # last year
        test_year = test['YEAR'].values[0]
        actual_val = test['TOT_EMP'].values[0]
        
        # Common data for all models
        job_result = {
            'Job': job,
            'Color': color,
            'YearlyData': yearly_data,
            'TestYear': test_year,
            'Actual': actual_val
        }
        
        # ----- 1. Linear Regression -----
        try:
            # Prepare data for Linear Regression
            X_train = train[['YEAR']].values.reshape(-1, 1)
            y_train = train['TOT_EMP'].values
            
            # Train model
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            
            # Predict for test year
            X_test = np.array([[test_year]])
            test_pred = lr_model.predict(X_test)[0]
            
            # Calculate error
            error = abs(actual_val - test_pred)
            mape = 100 * error / actual_val if actual_val != 0 else float('inf')
            
            # Store results
            lr_job_result = job_result.copy()
            lr_job_result.update({
                'Predicted': test_pred,
                'MAPE': mape,
                'R-squared': lr_model.score(X_train, y_train),
                'Slope': lr_model.coef_[0]
            })
            all_results['Linear Regression'].append(lr_job_result)
            
            print(f"  Linear Regression - Actual: {actual_val:,.0f}, Predicted: {test_pred:,.0f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  Error in Linear Regression for {job}: {str(e)}")
        
        # ----- 2. Random Forest -----
        try:
            # Prepare data for Random Forest
            X_train = train[['YEAR']].values.reshape(-1, 1)
            y_train = train['TOT_EMP'].values
            
            # Train model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predict for test year
            X_test = np.array([[test_year]])
            test_pred = rf_model.predict(X_test)[0]
            
            # Calculate error
            error = abs(actual_val - test_pred)
            mape = 100 * error / actual_val if actual_val != 0 else float('inf')
            
            # Store results
            rf_job_result = job_result.copy()
            rf_job_result.update({
                'Predicted': test_pred,
                'MAPE': mape
            })
            all_results['Random Forest'].append(rf_job_result)
            
            print(f"  Random Forest - Actual: {actual_val:,.0f}, Predicted: {test_pred:,.0f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  Error in Random Forest for {job}: {str(e)}")
        
        # ----- 3. Prophet -----
        try:
            # Prophet requires 'ds' and 'y' columns
            prophet_data = train.rename(columns={'YEAR': 'ds', 'TOT_EMP': 'y'})
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')
            
            # Train model
            model = Prophet(yearly_seasonality=True)
            model.fit(prophet_data)
            
            # Create future dataframe for prediction
            future = pd.DataFrame({'ds': [pd.to_datetime(str(test_year), format='%Y')]})
            
            # Make forecast
            forecast = model.predict(future)
            test_pred = forecast['yhat'].values[0]
            
            # Calculate error
            error = abs(actual_val - test_pred)
            mape = 100 * error / actual_val if actual_val != 0 else float('inf')
            
            # Store results
            prophet_job_result = job_result.copy()
            prophet_job_result.update({
                'Predicted': test_pred,
                'MAPE': mape
            })
            all_results['Prophet'].append(prophet_job_result)
            
            print(f"  Prophet - Actual: {actual_val:,.0f}, Predicted: {test_pred:,.0f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  Error in Prophet for {job}: {str(e)}")
    else:
        print(f"  Not enough data points for {job} to make predictions")

# Get all years for plotting
all_years = sorted(df['YEAR'].unique())

# Create plots for each model
for model_name, results in all_results.items():
    if results:
        create_model_plot(model_name, results, all_years)
        
        # Calculate average MAPE
        avg_mape = np.mean([r['MAPE'] for r in results])
        print(f"\nAverage MAPE with {model_name}: {avg_mape:.2f}%")
        
        # Create results dataframe
        results_df = pd.DataFrame([{
            'Job': r['Job'],
            'Actual': r['Actual'],
            'Predicted': r['Predicted'],
            'MAPE': r['MAPE']
        } for r in results])
        
        print(f"\n===== {model_name.upper()} RESULTS =====")
        pd.set_option('display.float_format', '{:,.2f}'.format)
        print(results_df)

# Create a summary comparison
print("\n===== MODEL COMPARISON =====")
model_comparison = []

for job in top_jobs.index:
    job_comparison = {'Job': job}
    
    for model_name, results in all_results.items():
        job_results = [r for r in results if r['Job'] == job]
        if job_results:
            job_comparison[f'{model_name} MAPE'] = job_results[0]['MAPE']
    
    model_comparison.append(job_comparison)

comparison_df = pd.DataFrame(model_comparison)
print(comparison_df)

# Find best model for each job
print("\n===== BEST MODEL BY JOB =====")
for job in top_jobs.index:
    job_row = comparison_df[comparison_df['Job'] == job]
    if not job_row.empty:
        mape_columns = [col for col in job_row.columns if 'MAPE' in col]
        if mape_columns:
            best_model = min(mape_columns, key=lambda col: job_row[col].values[0])
            best_mape = job_row[best_model].values[0]
            best_model_name = best_model.replace(' MAPE', '')
            print(f"{job}: {best_model_name} (MAPE: {best_mape:.2f}%)")

# Overall best model
model_avg_mapes = {}
for model_name, results in all_results.items():
    if results:
        model_avg_mapes[model_name] = np.mean([r['MAPE'] for r in results])

if model_avg_mapes:
    best_model = min(model_avg_mapes, key=model_avg_mapes.get)
    print(f"\nBest overall model: {best_model} (Avg MAPE: {model_avg_mapes[best_model]:.2f}%)")