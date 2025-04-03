import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('occupation_salaries (2).db')

# First, let's check the current row count
row_count_query = "SELECT COUNT(*) FROM Final_Combined_Data_clean"
initial_count = pd.read_sql_query(row_count_query, conn).iloc[0, 0]
print(f"Initial row count: {initial_count}")

# Read the table into a DataFrame
df = pd.read_sql_query("SELECT * FROM Final_Combined_Data_clean", conn)

# Remove duplicates based on the specified columns
df_cleaned = df.drop_duplicates(subset=['OCC_TITLE', 'AREA_TITLE', 'PRIM_STATE', 'TOT_EMP'])

# Calculate how many duplicates were removed
duplicates_removed = len(df) - len(df_cleaned)
print(f"Duplicates removed: {duplicates_removed}")

# Create a new clean table (or replace the existing one)
# Option 1: Create a new table
df_cleaned.to_sql("Final_Combined_Data_clean_no_dupes", conn, index=False, if_exists='replace')

# Option 2: Replace the existing table (uncomment if you want to overwrite)
# First backup the original table
# df.to_sql("Final_Combined_Data_clean_backup", conn, index=False, if_exists='replace')
# Then replace the original table
# df_cleaned.to_sql("Final_Combined_Data_clean", conn, index=False, if_exists='replace')

# Verify the new table row count
if 'Final_Combined_Data_clean_no_dupes' in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].values:
    new_count_query = "SELECT COUNT(*) FROM Final_Combined_Data_clean_no_dupes"
    new_count = pd.read_sql_query(new_count_query, conn).iloc[0, 0]
    print(f"New table row count: {new_count}")

# Close the connection
conn.close()

print("Duplicate removal complete!")