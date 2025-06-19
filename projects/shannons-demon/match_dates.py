import pandas as pd

# Expected start and end dates
expected_start_date = pd.to_datetime('2023-01-01')
expected_end_date = pd.to_datetime('2023-01-05')

# Create a fake DataFrame
data = {
    'id': [1, 2, 3, 4, 5],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
}
df = pd.DataFrame(data)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Check the first and last date in the date column
actual_start_date = df['date'].min()
actual_end_date = df['date'].max()

# Compare actual and expected dates
if actual_start_date == expected_start_date and actual_end_date == expected_end_date:
    print("Dates match.")
else:
    print("Dates do not match.")
