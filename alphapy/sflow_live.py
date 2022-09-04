import pandas as pd
from alphapy.globals import USEP

# Read in the Live Results file.
df_live = pd.read_csv('/Users/markconway/Projects/alphapy-root/alphapy-sports/MLB/Projects/won_on_points/output/live_results.csv')
print(df_live.columns)

# Input Parameters

capital = 10000
prob_col = 'prob_test_xgb'
kelly_frac = 0.5
ml_min = 150

# Calculate winning percentages and probability deltas.

def get_odds(x):
    if x > 0:
        odds = x / 100.0 + 1.0
    elif x < 0:
        odds = 100.0 / abs(x) + 1.0
    else:
        odds = 0.0
    return odds

def get_prob_win(x):
    if x > 0:
        prob = 1.0 - (x / (x + 100.0))
    elif x < 0:
        prob = abs(x) / (abs(x) + 100.0)
    else:
        prob = 0.5
    return prob

def get_kelly_pct(row, side):
    colname = USEP.join([side, 'odds'])
    odds = row[colname]
    colname = USEP.join([side, 'prob', 'win'])
    prob_win = row[colname]
    prob_model = row[prob_col]
    kelly_pct = ((odds - 1.0) * prob_win - (1.0 - prob_model)) / (odds - 1.0) * kelly_frac
    return kelly_pct

if not df_live.empty:
    # calculate odds and win probability
    for side in ['away', 'home']:
        colname = USEP.join([side, 'money', 'line'])
        mlclose = df_live[colname]
        colname = USEP.join([side, 'odds'])
        df_live[colname] = mlclose.apply(get_odds)
        colname = USEP.join([side, 'prob', 'win'])
        df_live[colname] = mlclose.apply(get_prob_win)
        colname = USEP.join([side, 'kelly', 'pct'])
        df_live[colname] = df_live.apply(get_kelly_pct, args=(side,), axis=1)     
    #df_live.to_csv('test.csv')
    # Betting System Analysis
    if ml_min:
        df_bet = df_live.loc[(df_live['away_money_line'].abs() >= ml_min) | (df_live['home_money_line'].abs() >= ml_min)].copy()
    working_cap = capital
    for index, row in df_bet.iterrows():
        if row['away_money_line'] >= ml_min:
            pass
        if row['home_money_line'] >= ml_min:
            pass
