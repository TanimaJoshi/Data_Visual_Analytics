import sqlite3
import pandas as pd
import random

# Connect to the database
conn = sqlite3.connect('occupation_salaries (2).db')

# Get a random occupation title for a specific year (e.g., 2021)
query_random_occupation = """
SELECT DISTINCT OCC_TITLE 
FROM "Final_Combined_Data_clean_no_dupes" 
WHERE YEAR = 2021
ORDER BY RANDOM() 
LIMIT 1
"""

random_occupation = pd.read_sql_query(query_random_occupation, conn).iloc[0, 0]

# Get all area titles for this occupation title in the chosen year
query_areas = """
SELECT OCC_TITLE, AREA_TITLE, PRIM_STATE, TOT_EMP 
FROM "Final_Combined_Data_clean_no_dupes" 
WHERE OCC_TITLE = ? AND YEAR = 2021
"""

df = pd.read_sql_query(query_areas, conn, params=(random_occupation,))

# Display the results
print(f"Occupation: {random_occupation}")
print(f"Number of different areas: {len(df)}")
print(df)

conn.close()