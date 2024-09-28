"""
Generate product files for betting data.
"""

#
# Imports
#

from alphapy.space import Space
from alphapy.utilities import datetime_stamp
import pandas as pd
from sflow_main import extract_features
from sflow_globals import feature_dict
from sflow_globals import game_dict
import yaml


# 
# List of Leagues
#

league_dict = {'mlb'   : '2023-12-01',
               'nba'   : '2024-07-01',
               'ncaab' : '2024-07-01',
               'ncaaf' : '2024-03-01',
               'nfl'   : '2024-03-01',
               'nhl'   : '2024-07-01'}
print("\nCurrent Betting Leagues:\n")
print(league_dict)

#
# Create input and output specifications
#

file_dict_input = {}
file_dict_output = {}
subject = 'game'
source = 'scores'
fractal = '1g'
space = Space(subject, source, fractal)
dt_stamp = datetime_stamp()[:8]

sports_spec_dict = {}
for league in league_dict.keys():
    # Read in the sports specification for the given league.
    config_dir_spec = '/'.join(['.', 'config'])
    config_file_spec = '_'.join(['config', league]) + '.yml'
    config_spec = '/'.join([config_dir_spec, config_file_spec])
    sports_spec = yaml.safe_load(config_spec)
    with open(sports_spec, 'r') as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    sports_spec_dict[league] = specs
    # Create the input specification
    input_dir_spec = specs['data_directory']
    input_file_spec = '.'.join(['_'.join([league, subject, source, fractal]), 'csv'])
    file_dict_input[league] = '/'.join([input_dir_spec, input_file_spec])
    # Create the output specification
    output_dir_spec = specs['prod_directory']
    output_file_spec = '.'.join(['_'.join([league, subject, source, fractal, dt_stamp]), 'csv'])
    file_dict_output[league] = '/'.join([output_dir_spec, output_file_spec])

#
# Create the odds dataset for each sport.
#

cols_drop = ['match_id', 'time_est', 'last_update_time']

for f in file_dict_input.keys():
    print(f"Generating {f.upper()} Output")
    # Read the CSV file
    df = pd.read_csv(file_dict_input[f], low_memory=False)
    # Extract the features
    df = extract_features(df, sports_spec_dict[f], space)
    # Filter for season cutoff
    df = df[df['date'] <= league_dict[f]]
    # Drop columns that are not part of the product dataset
    df.drop(columns=cols_drop, inplace=True)
    df.fillna(0, inplace=True)
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['home_point_spread_line'] = df['home_point_spread_line'].astype(int)
    df['away_point_spread_line'] = df['away_point_spread_line'].astype(int)
    df['over_line'] = df['over_line'].astype(int)
    df['under_line'] = df['under_line'].astype(int)
    print(f"Output: {df.shape}")
    df.to_csv(file_dict_output[f], index=False)
