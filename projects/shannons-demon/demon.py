"""
Shannon's Demon Demonstration
"""

#
# Suppress Warnings
#

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

#
# Imports
#

import glob
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

#
# Initialize simulation
#

capital = 10000
cash_value = capital / 2.0
position_value = capital - cash_value
rebalance_pct = 0.5

#
# Use either simulated data or real data
#

simulate_data = False
if simulate_data:
    symbol = 'DEMON'
    # Create date range
    n_days = 100
    date_end = pd.to_datetime('today')
    date_end_str = date_end.strftime('%Y-%m-%d')
    date_start = date_end - pd.DateOffset(days=n_days)
    date_start_str = date_start.strftime('%Y-%m-%d')
    dates = pd.date_range(start=date_start_str, end=date_end_str, freq='D')
    # Create close prices that randomly halve and double.
    close_values = [1.0]
    for _ in range(1, len(dates)):
        last_price = close_values[-1]
        new_price = last_price * random.choice([0.5, 2.0])
        close_values.append(new_price)
    df = pd.DataFrame({'date': dates, 'close': close_values})
else:
    symbol = 'BTC-USD'
    path = './data'
    extension = 'csv'
    os.chdir(path)
    files = glob.glob('*.{}'.format(extension))
    filename = '.'.join([symbol, 'csv'])
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower()

#
# Function get_new_targets
#

def get_new_targets(rebalance_pct, current_price):
    low_target = current_price * (1.0 - rebalance_pct)
    high_target = current_price / (1.0 - rebalance_pct)
    return low_target, high_target

#
# Calculate the initial targets
#

cp = df.iloc[0]['close']
nshares_hold = capital / cp
nshares = position_value / cp
low_target, high_target = get_new_targets(rebalance_pct, cp)

#
# Iterate through the price data, rebalancing as necessary
#

all_records = []
for index, row in df[1:].iterrows():
    cdate = row['date']
    cp = row['close']
    position_value = nshares * cp
    total_balance = cash_value + position_value
    half_balance = total_balance / 2.0
    if cp < low_target:
        print(f"\nDate: {cdate} Low Target Hit : {low_target:.2f}")
        rebalance_amount = half_balance - position_value
        shares_to_add = rebalance_amount / cp
        nshares += shares_to_add
        cash_value -= rebalance_amount
        position_value = nshares * cp
        low_target, high_target = get_new_targets(rebalance_pct, cp)
        print(f"Date: {cdate} New Low Target: {low_target:.2f}, New High Target; {high_target:.2f}")
    elif cp > high_target:
        print(f"\nDate: {cdate} High Target Hit: {high_target:.2f}")
        rebalance_amount = position_value - half_balance
        shares_to_subtract = rebalance_amount / cp
        nshares -= shares_to_subtract
        cash_value += rebalance_amount
        position_value = nshares * cp
        low_target, high_target = get_new_targets(rebalance_pct, cp)
        print(f"Date: {cdate} New Low Target: {low_target:.2f}, New High Target: {high_target:.2f}")
    all_records.append((symbol, cdate, cp, nshares, position_value, cash_value, total_balance))
print(f"\nFinal Portfolio Value: {total_balance:.2f}")

#
# Prepare data for comparing results
#

df1 = pd.DataFrame(all_records, columns=['symbol', 'date', 'close', 'shares', 'position_value', 'cash_value', 'total_balance'])
df1['total_balance'] = df1['close'] * nshares_hold
df1['daily_return'] = df1['close'].pct_change()
df1['cumulative_return'] = 100 * (np.exp(np.log1p(df1['daily_return']).cumsum()) - 1.0)
df1['type'] = 'Buy and Hold'

df2 = pd.DataFrame(all_records, columns=['symbol', 'date', 'close', 'shares', 'position_value', 'cash_value', 'total_balance'])
df2['daily_return'] = df2['total_balance'].pct_change()
df2['cumulative_return'] = 100 * (np.exp(np.log1p(df2['daily_return']).cumsum()) - 1.0)
df2['type'] = 'Rebalance'

df_both = pd.concat([df1, df2])
df_both['date'] = pd.to_datetime(df_both['date'])
df_both.dropna(inplace=True)
df_both.reset_index(inplace=True)

#
# Plot the Results
#

sns.set(rc={'figure.figsize': (12, 8)})
palette = {'Buy and Hold': 'red', 'Rebalance': 'green'}

title_sd = f"Symbol: {symbol} (Starting Portfolio: ${capital:,.0f})"
g = sns.lineplot(x='date', y='total_balance', hue='type', data=df_both, palette=palette)
g.set(title=title_sd)
g.set_ylim(-100, df_both['total_balance'].max())
g.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Format y-axis as currency
g.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
output_path = './' + symbol.lower() + '.png'
plt.savefig(output_path)
plt.show()
