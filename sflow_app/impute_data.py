"""
Impute the spread lines for the missing betting data.
"""

#
# Imports
#

from alphapy.utilities import datetime_stamp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 
# List of Leagues
#

league_dict = {'mlb' : '2022-07-24',
               'nhl' : '2022-10-07'}
print("\nLeagues:\n")

#
# Input Files
#

file_dict_path = {}
file_dict_input = {}
for league in league_dict.keys():
    input_dir_spec = '/'.join(['..', '..',league.upper(), 'data'])
    input_file_spec = '.'.join(['_'.join([league, 'game', 'scores', '1g']), 'csv'])
    file_dict_path[league] = '/'.join([input_dir_spec, input_file_spec])
    file_dict_input[league] = pd.read_csv(file_dict_path[league])
print(file_dict_input)

#
# Imputation Model
#

features = ['away_point_spread', 'away_money_line', 'home_point_spread', 'home_money_line']
target_away = 'away_point_spread_line'
target_home = 'home_point_spread_line'

def run_model(league):
    print(league.upper())
    df = file_dict_input[league]

    # Training data where 'away_point_spread_line' and 'home_point_spread_line' are not null
    X_train = df.loc[df[target_away].notnull() & df[target_home].notnull(), features]
    y_train_away = df.loc[df[target_away].notnull() & df[target_home].notnull(), target_away]
    y_train_home = df.loc[df[target_away].notnull() & df[target_home].notnull(), target_home]

    # Testing data where 'away_point_spread_line' and 'home_point_spread_line' are null
    X_test = df.loc[df[target_away].isnull() & df[target_home].isnull(), features]
    print(len(X_train), len(X_test))

    # Training the models
    model_away = RandomForestRegressor(random_state=42)
    model_home = RandomForestRegressor(random_state=42)

    model_away.fit(X_train, y_train_away)
    model_home.fit(X_train, y_train_home)

    predictions_away = model_away.predict(X_train)
    predictions_home = model_home.predict(X_train)

    mse_away = mean_squared_error(y_train_away, predictions_away)
    mse_home = mean_squared_error(y_train_home, predictions_home)

    rmse_away = np.sqrt(mse_away)
    rmse_home = np.sqrt(mse_home)
    print(f"RMSE Away: {rmse_away}")
    print(f"RMSE Home: {rmse_home}")

    # Making predictions and evaluating the models
    predictions_away_test = model_away.predict(X_test)
    predictions_away_test = [round(prediction) * 1.0 for prediction in predictions_away_test]
    predictions_away_test = [100.0 if 0 <= pred < 100 else -100.0 if -100 <= pred < 0 else pred for pred in predictions_away_test]

    predictions_home_test = model_home.predict(X_test)
    predictions_home_test = [round(prediction) * 1.0 for prediction in predictions_home_test]
    predictions_home_test = [100.0 if 0 <= pred < 100 else -100.0 if -100 <= pred < 0 else pred for pred in predictions_home_test]

    # Replace missing values in away_point_spread_line
    missing_away_indices = df[df['away_point_spread_line'].isnull()].index
    df.loc[missing_away_indices, 'away_point_spread_line'] = predictions_away_test

    # Replace missing values in home_point_spread_line
    missing_home_indices = df[df['home_point_spread_line'].isnull()].index
    df.loc[missing_home_indices, 'home_point_spread_line'] = predictions_home_test

    return df

#
# Run imputation model for each league
#

for league in league_dict.keys():
    print(f"Processing {league.upper()}")
    df_out = run_model(league)
    df_out.to_csv(file_dict_path[league], index=False)
