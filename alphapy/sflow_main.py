################################################################################
#
# Package   : AlphaPy
# Module    : sflow_main
# Created   : July 11, 2013
#
# Copyright 2021 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# HOW TO RUN:
#
# export ALPHAPY_ROOT=/Users/markconway/Projects/alphapy-root
# sflow --tdate 2020-01-01 --pdate 2021-11-24
#


#
# Suppress Warnings
#

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


#
# Imports
#

import argparse
from datetime import date, datetime, timedelta
import itertools
import logging
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
import yaml

from alphapy.alphapy_main import get_alphapy_config
from alphapy.alphapy_main import main_pipeline
from alphapy.frame import read_frame
from alphapy.frame import write_frame
from alphapy.globals import Partition, datasets
from alphapy.globals import SSEP, USEP
from alphapy.google_utils import authenticate_google
from alphapy.google_utils import authenticate_google_drive
from alphapy.google_utils import gdrive_dict
from alphapy.google_utils import upload_to_drive
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.space import Space
from alphapy.transforms import dateparts
from alphapy.utilities import datetime_stamp
from alphapy.utilities import most_recent_file
from alphapy.utilities import valid_date


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Sports Fields
#
# The following fields are repeated for:
#     1. 'home'
#     2. 'away'
#     3. 'delta'
#
# Note that [Target]s will not be merged into the Game table;
# these targets will be predictors in the Game table that are
# generated after each game result. All of the fields below
# are predictors and are generated a priori, i.e., we calculate
# deltas from the last previously played game for each team and
# these data go into the row for the next game to be played.
#

sports_dict = {'wins' : int,
               'losses' : int,
               'ties' : int,
               'days_since_first_game' : int,
               'days_since_previous_game' : int,
               'won_on_points' : bool,
               'lost_on_points' : bool,
               'won_on_spread' : bool,
               'lost_on_spread' : bool,
               'point_win_streak' : int,
               'point_loss_streak' : int,
               'point_margin_game' : int,
               'point_margin_season' : int,
               'point_margin_season_avg' : float,
               'point_margin_streak' : int,
               'point_margin_streak_avg' : float,
               'point_margin_ngames' : int,
               'point_margin_ngames_avg' : float,
               'cover_win_streak' : int,
               'cover_loss_streak' : int,
               'cover_margin_game' : float,
               'cover_margin_season' : float, 
               'cover_margin_season_avg' : float,
               'cover_margin_streak' : float,
               'cover_margin_streak_avg' : float,
               'cover_margin_ngames' : float,
               'cover_margin_ngames_avg' : float,
               'total_points' : int,
               'overunder_margin' : float,
               'over' : bool,
               'under' : bool,
               'over_streak' : int,
               'under_streak' : int,
               'overunder_season' : float,
               'overunder_season_avg' : float,
               'overunder_streak' : float,
               'overunder_streak_avg' : float,
               'overunder_ngames' : float,
               'overunder_ngames_avg' : float}


#
# These are the leaders. Generally, we try to predict one of these
# variables as the target and lag the remaining ones.
#

game_dict = {'point_margin_game' : int,
             'won_on_points' : bool,
             'lost_on_points' : bool,
             'cover_margin_game' : float,
             'won_on_spread' : bool,
             'lost_on_spread' : bool,
             'overunder_margin' : float,
             'over' : bool,
             'under' : bool}


#
# Convert Boolean features to int before writing the data frame
#

features_bool = ['won_on_points', 'lost_on_points',
                 'won_on_spread', 'lost_on_spread',
                 'over', 'under']


#
# Function get_sport_config
#

def get_sport_config():
    r"""Read the configuration file for SportFlow.

    Parameters
    ----------
    None : None

    Returns
    -------
    specs : dict
        The parameters for controlling SportFlow.

    """

    # Read the configuration file

    full_path = SSEP.join(['.', 'config', 'sport.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: sport

    data_directory = cfg['sport']['data_directory']
    dir_exists = os.path.isdir(data_directory)
    if dir_exists:
        specs['data_directory'] = data_directory
    else:
        raise ValueError("Directory %s does not exist" % data_directory)

    specs['league'] = cfg['sport']['league']
    specs['points_max'] = cfg['sport']['points_max']
    specs['points_min'] = cfg['sport']['points_min']
    specs['random_scoring'] = cfg['sport']['random_scoring']
    specs['rolling_window'] = cfg['sport']['rolling_window']   
    specs['seasons'] = cfg['sport']['seasons']

    # Log the sports parameters

    logger.info('SPORT PARAMETERS:')
    logger.info('league           = %s', specs['league'])
    logger.info('data_directory   = %s', specs['data_directory'])
    logger.info('points_max       = %d', specs['points_max'])
    logger.info('points_min       = %d', specs['points_min'])
    logger.info('random_scoring   = %r', specs['random_scoring'])
    logger.info('rolling_window   = %d', specs['rolling_window'])
    logger.info('seasons          = %s', specs['seasons'])

    # Game Specifications
    return specs


#
# Function get_point_margin
#

def get_point_margin(row, score, opponent_score):
    r"""Get the point margin for a game.

    Parameters
    ----------
    row : pandas.Series
        The row of a game.
    score : int
        The score for one team.
    opponent_score : int
        The score for the other team.

    Returns
    -------
    point_margin : int
        The resulting point margin (0 if NaN).

    """
    point_margin = 0
    nans = math.isnan(row[score]) or math.isnan(row[opponent_score])
    if not nans:
        point_margin = row[score] - row[opponent_score]
    return point_margin


#
# Function get_wins
#

def get_wins(point_margin):
    r"""Determine a win based on the point margin.

    Parameters
    ----------
    point_margin : int
        The point margin can be positive, zero, or negative.

    Returns
    -------
    won : int
        If the point margin is greater than 0, return 1, else 0.

    """
    won = 1 if point_margin > 0 else 0
    return won


#
# Function get_losses
#

def get_losses(point_margin):
    r"""Determine a loss based on the point margin.

    Parameters
    ----------
    point_margin : int
        The point margin can be positive, zero, or negative.

    Returns
    -------
    lost : int
        If the point margin is less than 0, return 1, else 0.

    """
    lost = 1 if point_margin < 0 else 0
    return lost


#
# Function get_ties
#

def get_ties(point_margin):
    r"""Determine a tie based on the point margin.

    Parameters
    ----------
    point_margin : int
        The point margin can be positive, zero, or negative.

    Returns
    -------
    tied : int
        If the point margin is equal to 0, return 1, else 0.

    """
    tied = 1 if point_margin == 0 else 0
    return tied


#
# Function get_day_offset
#

def get_day_offset(date_vector):
    r"""Compute the day offsets between games.

    Parameters
    ----------
    date_vector : pandas.Series
        The date column.

    Returns
    -------
    day_offset : pandas.Series
        A vector of day offsets between adjacent dates.

    """

    dv = pd.to_datetime(date_vector)
    offsets = pd.to_datetime(dv) - pd.to_datetime(dv[0])
    day_offset = (offsets / pd.Timedelta('1D')).astype(int)
    return day_offset


#
# Function get_series_diff
#

def get_series_diff(series):
    r"""Perform the difference operation on a series.

    Parameters
    ----------
    series : pandas.Series
        The series for the ``diff`` operation.

    Returns
    -------
    new_series : pandas.Series
        The differenced series.

    """
    new_series = pd.Series(len(series))
    new_series = series.diff()
    new_series[0] = 0
    return new_series


#
# Function get_streak
#

def get_streak(series, start_index, window):
    r"""Calculate the current streak.

    Parameters
    ----------
    series : pandas.Series
        A Boolean series for calculating streaks.
    start_index : int
        The offset of the series to start counting.
    window : int
        The period over which to count.

    Returns
    -------
    streak : int
        The count value for the current streak.

    """
    if window <= 0:
        window = len(series)
    i = start_index
    streak = 0
    while i >= 0 and (start_index-i+1) < window and series[i]:
        streak += 1
        i -= 1
    return streak


#
# Function add_features
#

def add_features(frame, fdict, flen, prefix=''):
    r"""Add new features to a dataframe with the specified dictionary.

    Parameters
    ----------
    frame : pandas.DataFrame
        The dataframe to extend with new features defined by ``fdict``.
    fdict : dict
        A dictionary of column names (key) and data types (value).
    flen : int
        Length of ``frame``.
    prefix : str, optional
        Prepend all columns with a prefix.

    Returns
    -------
    frame : pandas.DataFrame
        The dataframe with the added features.

    """
    # generate sequences
    seqint = [0] * flen
    seqfloat = [0.0] * flen
    seqbool = [False] * flen
    # initialize new fields in frame
    for key, value in list(fdict.items()):
        newkey = key
        if prefix:
            newkey = USEP.join([prefix, newkey])
        if value == int:
            frame[newkey] = pd.Series(seqint)
        elif value == float:
            frame[newkey] = pd.Series(seqfloat)
        elif value == bool:
            frame[newkey] = pd.Series(seqbool)
        else:
            raise ValueError("Type to generate feature series not found")
    return frame


#
# Function generate_team_frame
#

def generate_team_frame(team, tf, home_team, away_team, window):
    r"""Calculate statistics for each team.

    Parameters
    ----------
    team : str
        The abbreviation for the team.
    tf : pandas.DataFrame
        The initial team frame.
    home_team : str
        Label for the home team.
    away_team : str
        Label for the away team.
    window : int
        The value for the rolling window to calculate means and sums.

    Returns
    -------
    tf : pandas.DataFrame
        The completed team frame.

    """
    # Initialize new features
    tf = add_features(tf, sports_dict, len(tf))
    # Daily Offsets
    tf['days_since_first_game'] = get_day_offset(tf['date'])
    tf['days_since_previous_game'] = get_series_diff(tf['days_since_first_game'])
    # Team Loop
    for index, row in tf.iterrows():
        if team == row[home_team]:
            tf['point_margin_game'].at[index] = get_point_margin(row, 'home_score', 'away_score')
            spread = row['home_point_spread']
        elif team == row[away_team]:
            tf['point_margin_game'].at[index] = get_point_margin(row, 'away_score', 'home_score')
            spread = -row['home_point_spread']
        else:
            raise KeyError("Team not found in Team Frame")
        if index == 0:
            tf['wins'].at[index] = get_wins(tf['point_margin_game'].at[index])
            tf['losses'].at[index] = get_losses(tf['point_margin_game'].at[index])
            tf['ties'].at[index] = get_ties(tf['point_margin_game'].at[index])
        else:
            tf['wins'].at[index] = tf['wins'].at[index-1] + get_wins(tf['point_margin_game'].at[index])
            tf['losses'].at[index] = tf['losses'].at[index-1] + get_losses(tf['point_margin_game'].at[index])
            tf['ties'].at[index] = tf['ties'].at[index-1] + get_ties(tf['point_margin_game'].at[index])
        tf['won_on_points'].at[index] = True if tf['point_margin_game'].at[index] > 0 else False
        tf['lost_on_points'].at[index] = True if tf['point_margin_game'].at[index] < 0 else False
        tf['cover_margin_game'].at[index] = tf['point_margin_game'].at[index] + spread
        tf['won_on_spread'].at[index] = True if tf['cover_margin_game'].at[index] > 0 else False
        tf['lost_on_spread'].at[index] = True if tf['cover_margin_game'].at[index] <= 0 else False
        nans = math.isnan(row['home_score']) or math.isnan(row['away_score'])
        if not nans:
            tf['total_points'].at[index] = row['home_score'] + row['away_score']
        nans = math.isnan(row['over_under'])
        if not nans:
            tf['overunder_margin'].at[index] = tf['total_points'].at[index] - row['over_under']
        tf['over'].at[index] = True if tf['overunder_margin'].at[index] > 0 else False
        tf['under'].at[index] = True if tf['overunder_margin'].at[index] < 0 else False
        tf['point_win_streak'].at[index] = get_streak(tf['won_on_points'], index, 0)
        tf['point_loss_streak'].at[index] = get_streak(tf['lost_on_points'], index, 0)
        tf['cover_win_streak'].at[index] = get_streak(tf['won_on_spread'], index, 0)
        tf['cover_loss_streak'].at[index] = get_streak(tf['lost_on_spread'], index, 0)
        tf['over_streak'].at[index] = get_streak(tf['over'], index, 0)
        tf['under_streak'].at[index] = get_streak(tf['under'], index, 0)
        # Handle the streaks
        if tf['point_win_streak'].at[index] > 0:
            streak = tf['point_win_streak'].at[index]
        elif tf['point_loss_streak'].at[index] > 0:
            streak = tf['point_loss_streak'].at[index]
        else:
            streak = 1
        tf['point_margin_streak'].at[index] = tf['point_margin_game'][index-streak+1:index+1].sum()
        tf['point_margin_streak_avg'].at[index] = tf['point_margin_game'][index-streak+1:index+1].mean()
        if tf['cover_win_streak'].at[index] > 0:
            streak = tf['cover_win_streak'].at[index]
        elif tf['cover_loss_streak'].at[index] > 0:
            streak = tf['cover_loss_streak'].at[index]
        else:
            streak = 1
        tf['cover_margin_streak'].at[index] = tf['cover_margin_game'][index-streak+1:index+1].sum()
        tf['cover_margin_streak_avg'].at[index] = tf['cover_margin_game'][index-streak+1:index+1].mean()
        if tf['over_streak'].at[index] > 0:
            streak = tf['over_streak'].at[index]
        elif tf['under_streak'].at[index] > 0:
            streak = tf['under_streak'].at[index]
        else:
            streak = 1
        tf['overunder_streak'].at[index] = tf['overunder_margin'][index-streak+1:index+1].sum()
        tf['overunder_streak_avg'].at[index] = tf['overunder_margin'][index-streak+1:index+1].mean()
    # Rolling and Expanding Variables
    tf['point_margin_season'] = tf['point_margin_game'].cumsum()
    tf['point_margin_season_avg'] = tf['point_margin_game'].expanding().mean()
    tf['point_margin_ngames'] = tf['point_margin_game'].rolling(window=window, min_periods=1).sum()
    tf['point_margin_ngames_avg'] = tf['point_margin_game'].rolling(window=window, min_periods=1).mean()
    tf['cover_margin_season'] = tf['cover_margin_game'].cumsum()
    tf['cover_margin_season_avg'] = tf['cover_margin_game'].expanding().mean()
    tf['cover_margin_ngames'] = tf['cover_margin_game'].rolling(window=window, min_periods=1).sum()
    tf['cover_margin_ngames_avg'] = tf['cover_margin_game'].rolling(window=window, min_periods=1).mean()
    tf['overunder_season'] = tf['overunder_margin'].cumsum()
    tf['overunder_season_avg'] = tf['overunder_margin'].expanding().mean()
    tf['overunder_ngames'] = tf['overunder_margin'].rolling(window=window, min_periods=1).sum()
    tf['overunder_ngames_avg'] = tf['overunder_margin'].rolling(window=window, min_periods=1).mean()
    return tf


#
# Function get_team_frame
#

def get_team_frame(game_frame, team, home, away):
    r"""Calculate statistics for each team.

    Parameters
    ----------
    game_frame : pandas.DataFrame
        The game frame for a given season.
    team : str
        The team abbreviation.
    home : str
        The label of the home team column.
    away : int
        The label of the away team column.

    Returns
    -------
    team_frame : pandas.DataFrame
        The extracted team frame.

    """
    team_frame = game_frame[(game_frame[home] == team) | (game_frame[away] == team)]
    return team_frame


#
# Function insert_model_data
#

def insert_model_data(mf, mpos, mdict, tf, tpos, prefix):
    r"""Insert a row from the team frame into the model frame.

    Parameters
    ----------
    mf : pandas.DataFrame
        The model frame for a single season.
    mpos : int
        The position in the model frame where to insert the row.
    mdict : dict
        A dictionary of column names (key) and data types (value).
    tf : pandas.DataFrame
        The team frame for a season.
    tpos : int
        The position of the row in the team frame.
    prefix : str
        The prefix to join with the ``mdict`` key.

    Returns
    -------
    mf : pandas.DataFrame
        The model dataframe.

    """
    team_row = tf.iloc[tpos]
    for key, _ in list(mdict.items()):
        newkey = key
        if prefix:
            newkey = USEP.join([prefix, newkey])
        mf.at[mpos, newkey] = team_row[key]
    return mf


#
# Function generate_delta_data
#

def generate_delta_data(frame, fdict, prefix1, prefix2):
    r"""Subtract two similar columns to get the delta value.

    Parameters
    ----------
    frame : pandas.DataFrame
        The input model frame.
    fdict : dict
        A dictionary of column names (key) and data types (value).
    prefix1 : str
        The prefix of the first team.
    prefix2 : str
        The prefix of the second team.

    Returns
    -------
    frame : pandas.DataFrame
        The completed dataframe with the delta data.

    """
    for key, _ in list(fdict.items()):
        newkey = USEP.join(['delta', key])
        key1 = USEP.join([prefix1, key])
        key2 = USEP.join([prefix2, key])
        frame[newkey] = frame[key1] - frame[key2]
    return frame


#
# Function save_timegpt_data
#

def save_timegpt_data(model_specs):
    r"""Save the training data for TimeGPT.

    Parameters
    ----------
    model_specs : dict
        The model specifications.

    Returns
    -------
    None

    """

    # Extract model fields
    run_dir = model_specs['run_dir']

    # Get the run's training data, which contains previous results

    output_dir = '/'.join([run_dir, 'output'])
    mrf = most_recent_file(output_dir, 'ranked_train*.csv')
    df_train = pd.read_csv(mrf, low_memory=False)

    # Get a list of all unique teams from both 'home_team' and 'away_team' columns
    all_teams = pd.concat([df_train['home_team'], df_train['away_team']]).unique()

    # List to hold the individual team DataFrames
    team_frames = []

    for team in all_teams:
        # Split the data into home and away games for the current team
        home_games = df_train[df_train['home_team'] == team]
        away_games = df_train[df_train['away_team'] == team]

        # Keep only the relevant columns and rename them by removing the 'home_' or 'away_' prefix
        home_games = home_games[['season', 'date'] + [col for col in home_games.columns if col.startswith('home_')]]
        away_games = away_games[['season', 'date'] + [col for col in away_games.columns if col.startswith('away_')]]
        
        home_games.columns = home_games.columns.str.replace('home_', '')
        away_games.columns = away_games.columns.str.replace('away_', '')
        
        # Add a new column to indicate the team name
        home_games['team'] = team
        away_games['team'] = team
        
        # Append the processed frames to the list
        team_frames.append(home_games)
        team_frames.append(away_games)

    # Concatenate all the individual team frames into a single DataFrame
    merged_frame = pd.concat(team_frames)

    # Create a new ordered column list with 'season' and 'date' first

    columns_order = ['season', 'date'] + [col for col in merged_frame if col not in ['season', 'date']]
    merged_frame = merged_frame.reindex(columns=columns_order)

    # Sort the final DataFrame based on date and team name

    merged_frame['date_dt'] = pd.to_datetime(merged_frame['date']) 
    merged_frame = merged_frame.sort_values(by=['date_dt', 'team']).reset_index(drop=True)
    merged_frame.drop(columns=['date_dt'], inplace=True)

    # Save the DataFrame to a CSV file

    file_spec = '/'.join([run_dir, 'output', 'team_time_series.csv'])
    merged_frame.to_csv(file_spec, index=False)

    return


#
# Function extract_datasets
#

def extract_datasets(model_specs, df, league, gdrive=None):
    """
    Extract datasets for each category.

    Parameters
    ----------
    model_specs : dict
        The model specifications.
    df : pandas.DataFrame
        The dataframe of live results.
    league : str
        The league abbreviation.
    gdrive : GoogleDrive object, optional
        The Google Drive object.

    Returns
    -------
    dict
        A dictionary containing three dataframes: results, predictions, summary.
    """

    directory = model_specs['directory']
    target = model_specs['target']

    # Define column subsets

    game_cols = ['date', 'away_team', 'home_team']
    score_cols = ['away_score', 'home_score']
    spread_cols = ['away_point_spread', 'away_point_spread_line', 'home_point_spread', 'home_point_spread_line']
    moneyline_cols = ['away_money_line', 'home_money_line']
    over_under_cols = ['over_under', 'over_line', 'under_line']
    pred_cols = [col for col in df.columns if 'pred_' in col]
    prob_cols = [col for col in df.columns if 'prob_' in col]

    # Convert columns types

    cols_to_int = ['away_point_spread_line', 'home_point_spread_line', 'over_line', 'under_line']
    for col in cols_to_int:
        df[col] = df[col].astype(int)

    for col in prob_cols:
        df[col] = df[col].round(3)
    
    # Get date calculations for results and predictions.

    df['date'] = pd.to_datetime(df['date'])
    current_date = datetime.now()
    two_weeks_ago = current_date - timedelta(weeks=2)
    two_weeks_from_now = current_date + timedelta(weeks=2)

    # Get the columns for each dataframe.

    results_col_map = {
        'won_on_spread' : 'home_point_spread',
        'won_on_points' : 'home_money_line',
        'over'          : 'over_under',
    }

    df_results = df[df['away_score'].notna() & df['home_score'].notna()]
    results_cols = game_cols + score_cols + [target] + [results_col_map[target]]
    df_results = df_results[results_cols]
    matching_cols = [col for col in df.columns if col in df_results.columns]
    df_results = df_results[matching_cols]
    cols_to_int = ['away_score', 'home_score']
    for col in cols_to_int:
        df_results[col] = df_results[col].astype(int)
    df_results = df_results[df_results['date'] >= two_weeks_ago]
    df_results = df_results.sort_values(by='date', ascending=False)

    target_col_map = {
        'won_on_spread' : spread_cols,
        'won_on_points' : moneyline_cols,
        'over'          : over_under_cols,
    }

    df_pred1 = df[pd.isna(df['away_score']) & pd.isna(df['home_score'])]
    pred1_cols = game_cols + pred_cols + target_col_map[target]
    df_pred1 = df_pred1[pred1_cols]
    matching_cols = [col for col in df.columns if col in df_pred1.columns]
    df_pred1 = df_pred1[matching_cols]
    df_pred1.drop(columns=['pred_test_best'], inplace=True)
    df_pred1 = df_pred1[df_pred1['date'] <= two_weeks_from_now]

    df_pred2 = df[pd.isna(df['away_score']) & pd.isna(df['home_score'])]
    pred2_cols = game_cols + prob_cols + target_col_map[target]
    df_pred2 = df_pred2[pred2_cols]
    matching_cols = [col for col in df.columns if col in df_pred2.columns]
    df_pred2 = df_pred2[matching_cols]
    df_pred2.drop(columns=['prob_test_best'], inplace=True)
    df_pred2 = df_pred2[df_pred2['date'] <= two_weeks_from_now]

    summary_data = []
    df_summ = df[pd.isna(df['away_score']) & pd.isna(df['home_score'])]
    for col in pred_cols:
        matches = df_summ[df_summ[col] == df_summ[target]].shape[0]
        total_predictions = df_summ[col].count()
        mismatches = total_predictions - matches
        winning_percentage = ((matches / total_predictions) * 100).round(2)
        fade_percentage = (100.0 - winning_percentage).round(2)
        model_name = col.replace('pred_', '').replace('test_', '')
        summary_data.append({'model'       : model_name,
                             'wins'        : matches,
                             'losses'      : mismatches,
                             'total games' : total_predictions,
                             'win %'       : winning_percentage,
                             'fade %'      : fade_percentage})
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(by='win %', ascending=False)
    df_summary = df_summary[df_summary['model'] != 'best']

    # Store the dataframes in a dictionary for easy access
    datasets = {
        'results': df_results,
        'predictions_pred': df_pred1,
        'predictions_prob': df_pred2,
        'summary': df_summary
    }

    for name, dataset in datasets.items():
        file_name = f"{target}_{name}.csv"
        # Save file to local directory
        logger.info(f"Saving {file_name}")
        dataset.to_csv(f"{directory}/{file_name}", index=False)
        # Upload file to Google Drive
        if gdrive:
            if name != 'predictions_prob':
                tag = USEP.join(['nb', league.lower()])
            else:
                tag = USEP.join(['sb', league.lower()])
            folder_id = gdrive_dict[tag]
            file_id = upload_to_drive(gdrive, file_name, folder_id)
            # convert_csv_to_sheet(gdrive, file_id)

    return datasets


#
# Function update_live_results
#

def update_live_results(model_specs, df_live):
    r"""Update the live results.

    Parameters
    ----------
    model_specs : dict
        The model specifications.
    df_live : pandas.DataFrame
        The dataframe of live results.

    Returns
    -------
    df_live : pandas.DataFrame
        The dataframe of live results.

    """

    logger.info("Updating Live Results")

    # Extract model fields

    run_dir = model_specs['run_dir']
    target = model_specs['target']

    # Get the run's training data, which contains previous results

    output_dir = '/'.join([run_dir, 'output'])
    mrf = most_recent_file(output_dir, 'ranked_train*.csv')
    df_results = pd.read_csv(mrf, low_memory=False)
    df_results['date_dt'] = pd.to_datetime(df_results['date'])
    current_date = datetime.now()
    previous_date = current_date - timedelta(days=30)
    df_results = df_results[df_results['date_dt'] > previous_date]
    df_results.drop(columns=['date_dt'], inplace=True)
    df_results.set_index('match_id', inplace=True)

    # Get the run's predictions

    mrf = most_recent_file(output_dir, 'ranked_test*.csv')
    df_pred = pd.read_csv(mrf)
    game_cols = ['match_id', 'season', 'date', 'away_team', 'away_score', 'away_point_spread',
           'away_point_spread_line', 'away_money_line', 'home_team', 'home_score',
           'home_point_spread', 'home_point_spread_line', 'home_money_line',
           'over_under', 'over_line', 'under_line']
    pred_cols = df_pred.columns[df_pred.columns.str.startswith('pred_')]
    prob_cols = df_pred.columns[df_pred.columns.str.startswith('prob_')]
    df_pred_cols = list(itertools.chain(game_cols, pred_cols, prob_cols, [target]))
    df_pred = df_pred[df_pred_cols]
    df_pred.set_index('match_id', inplace=True)

    # Update the live results with the new results

    df_live = pd.concat([df_live, df_pred])
    df_live = df_live[~df_live.index.duplicated(keep='last')]

    # Update any scores

    df_live.update(df_results, errors='ignore')
    df_live = df_live.sort_values(by=['date'])

    return df_live


#
# Function record_live_results
#

def record_live_results(model_specs):
    r"""Record the live results.

    Parameters
    ----------
    model_specs : dict
        The model specifications.

    Returns
    -------
    df_live : pandas.DataFrame
        The dataframe of live results.

    """

    # Extract model fields
    directory = model_specs['directory']
    
    # Read the Live Results File.

    logger.info("Reading Live Results")

    try:
        df_live = read_frame(directory, 'live_results', model_specs['extension'], model_specs['separator'])
        df_live.set_index('match_id', inplace=True)
        logger.info("Current Live Records: %d", df_live.shape[0])
    except:
        df_live = pd.DataFrame()
        logger.info("No Live Results to Analyze")

    df_live = update_live_results(model_specs, df_live)
    logger.info("Total Live Records: %d", df_live.shape[0])

    # Save updated Live Results file.

    file_spec = '/'.join([directory, 'live_results.csv'])
    df_live.to_csv(file_spec, index_label='match_id')

    return df_live


#
# Function main
#

def main(args=None):
    r"""The main program for SportFlow.

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the game configuration.
    (4) Get the model configuration.
    (5) Generate game frames for each season.
    (6) Create statistics for each team.
    (7) Merge the team frames into the final model frame.
    (8) Run the AlphaPy pipeline.

    Raises
    ------
    ValueError
        Training date must be before prediction date.

    """

    # Argument Parsing

    parser = argparse.ArgumentParser(description="SportFlow Parser")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument('--pdate', dest='predict_date',
                        help="prediction date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_argument('--tdate', dest='train_date',
                        help="training date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--predict', dest='predict_mode', action='store_true')
    parser.add_argument('--train', dest='predict_mode', action='store_false')
    parser.set_defaults(predict_mode=False)
    parser.add_argument('--rundir', dest='run_dir',
                        help="run directory is in the format: run_YYYYMMDD_hhmmss",
                        required=False)
    args = parser.parse_args()

    # Google Drive Authorization

    creds = authenticate_google()
    gdrive = authenticate_google_drive(creds) if creds else None

    # Logging

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="sport_flow.log", filemode='a', level=log_level,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("SportFlow Start")
    logger.info('*'*80)

    # Read AlphaPy root directory

    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)

    # Read AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

    # Read model configuration file

    _, model_specs = get_model_config()
    model_specs['alphapy_root'] = alphapy_root
    
    # Extract model fields
    
    live_results = model_specs['live_results']

    # Add command line arguments to model specifications

    model_specs['predict_mode'] = args.predict_mode
    model_specs['predict_date'] = args.predict_date
    model_specs['train_date'] = args.train_date

    # If not in prediction mode, then create the training infrastructure.

    if not model_specs['predict_mode']:
        # create the directory infrastructure if necessary
        output_dirs = ['config', 'runs']
        for od in output_dirs:
            output_dir = SSEP.join([model_specs['directory'], od])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # create the run directory
        dt_stamp = datetime_stamp()
        run_dir_name = USEP.join(['run', dt_stamp])
        run_dir = SSEP.join([model_specs['directory'], 'runs', run_dir_name])
        os.makedirs(run_dir)
        # create the subdirectories of the runs directory
        sub_dirs = ['config', 'input', 'model', 'output', 'plots']
        for sd in sub_dirs:
            output_dir = SSEP.join([run_dir, sd])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # copy the market file to the config directory
        file_names = ['model.yml', 'sport.yml']
        for file_name in file_names:
            src_file = SSEP.join([model_specs['directory'], 'config', file_name])
            dst_file = SSEP.join([run_dir, 'config', file_name])
            shutil.copyfile(src_file, dst_file)
    else:
        run_dir = args.run_dir if args.run_dir else None
        if not run_dir:
            # get latest directory
            search_dir = SSEP.join([model_specs['directory'], 'runs'])
            run_dir = most_recent_file(search_dir, 'run_*')
    model_specs['run_dir'] = run_dir

    # Set train and predict dates

    if args.train_date:
        train_date = args.train_date
    else:
        train_date = pd.to_datetime('1900-01-01').strftime("%Y-%m-%d")

    if args.predict_date:
        predict_date = args.predict_date
    else:
        predict_date = date.today().strftime("%Y-%m-%d")

    # Verify that the dates are in sequence.

    if train_date >= predict_date:
        raise ValueError("Training date must be before prediction date")
    else:
        logger.info("Training Date: %s", train_date)
        logger.info("Prediction Date: %s", predict_date)

    # Read game configuration file

    sport_specs = get_sport_config()

    # Section: game

    data_directory = sport_specs['data_directory']
    league = sport_specs['league']
    points_max = sport_specs['points_max']
    points_min = sport_specs['points_min']
    random_scoring = sport_specs['random_scoring']
    seasons = sport_specs['seasons']
    window = sport_specs['rolling_window']   

    # Create the game scores space
    space = Space('game', 'scores', '1g')

    #
    # Derived Variables
    #

    series = space.source
    team1_prefix = 'home'
    team2_prefix = 'away'
    home_team = USEP.join([team1_prefix, 'team'])
    away_team = USEP.join([team2_prefix, 'team'])

    #
    # Read in the game frame. This is the feature generation phase.
    #

    logger.info("Reading Game Data")

    file_base = USEP.join([league, space.subject, space.source, space.fractal])
    df = read_frame(data_directory, file_base, model_specs['extension'], model_specs['separator'])
    logger.info("Total Game Records: %d", df.shape[0])

    #
    # Get date information
    #

    df = pd.concat([df, dateparts(df, 'date')], axis=1)

    #
    # Make all team names lower case
    #

    df['home_team'] = df['home_team'].str.lower()
    df['away_team'] = df['away_team'].str.lower()

    #
    # Run the game pipeline on a seasonal loop
    #

    if not seasons:
        # run model on all seasons
        seasons = df['season'].unique().tolist()
    df['season'] = df['season'].astype(str)

    #
    # Initialize the final frame
    #

    ff = pd.DataFrame()

    #
    # Iterate through each season of the game frame
    #

    for season in seasons:

        # Generate a frame for each season

        gf = df[df['season'] == season].copy()
        gf = gf.reset_index()

        # Generate derived variables for the game frame

        total_games = gf.shape[0]
        if random_scoring:
            gf['home_score'] = np.random.randint(points_min, points_max, total_games)
            gf['away_score'] = np.random.randint(points_min, points_max, total_games)
        gf['total_points'] = gf['home_score'] + gf['away_score']

        # gf['line_delta'] = gf['line'] - gf['line_open']
        # gf['over_under_delta'] = gf['over_under'] - gf['over_under_open']

        gf = add_features(gf, game_dict, gf.shape[0])
        for index, row in gf.iterrows():
            if not np.isnan(row['home_score']):
                gf['point_margin_game'].at[index] = get_point_margin(row, 'home_score', 'away_score')
                gf['won_on_points'].at[index] = True if gf['point_margin_game'].at[index] > 0 else False
                gf['lost_on_points'].at[index] = True if gf['point_margin_game'].at[index] < 0 else False
                gf['cover_margin_game'].at[index] = gf['point_margin_game'].at[index] + row['home_point_spread']
                gf['won_on_spread'].at[index] = True if gf['cover_margin_game'].at[index] > 0 else False
                gf['lost_on_spread'].at[index] = True if gf['cover_margin_game'].at[index] <= 0 else False
                gf['overunder_margin'].at[index] = gf['total_points'].at[index] - row['over_under']
                gf['over'].at[index] = True if gf['overunder_margin'].at[index] > 0 else False
                gf['under'].at[index] = True if gf['overunder_margin'].at[index] < 0 else False
            else:
                gf['point_margin_game'].at[index] = None
                gf['won_on_points'].at[index] = None
                gf['lost_on_points'].at[index] = None
                gf['cover_margin_game'].at[index] = None
                gf['won_on_spread'].at[index] = None
                gf['lost_on_spread'].at[index] = None
                gf['overunder_margin'].at[index] = None
                gf['over'].at[index] = None
                gf['under'].at[index] = None

        # Generate each team frame

        team_frames = {}
        teams = gf.groupby([home_team])
        for team, _ in teams:
            team_name = team[0]
            team_frame = USEP.join([league, team_name.lower(), series, str(season)])
            logger.info("Generating team frame: %s", team_frame)
            tf = get_team_frame(gf, team_name, home_team, away_team)
            tf = tf.reset_index()
            tf = generate_team_frame(team_name, tf, home_team, away_team, window)
            team_frames[team_frame] = tf

        # Create the model frame, initializing the home and away frames

        mdict = {k:v for (k,v) in list(sports_dict.items()) if v != bool}
        team1_frame = pd.DataFrame()
        team1_frame = add_features(team1_frame, mdict, gf.shape[0], prefix=team1_prefix)
        team2_frame = pd.DataFrame()
        team2_frame = add_features(team2_frame, mdict, gf.shape[0], prefix=team2_prefix)
        frames = [gf, team1_frame, team2_frame]
        mf = pd.concat(frames, axis=1)

        # Loop through each team frame, inserting data into the model frame row
        #     get index+1 [if valid]
        #     determine if team is home or away to get prefix
        #     try: np.where((gf[home_team] == 'PHI') & (gf['date'] == '09/07/14'))[0][0]
        #     Assign team frame fields to respective model frame fields: set gf.at(pos, field)

        for team, _ in teams:
            team_name = team[0]
            team_frame = USEP.join([league, team_name.lower(), series, str(season)])
            logger.info("Merging team frame %s into model frame", team_frame)
            tf = team_frames[team_frame]
            for index in range(0, tf.shape[0]-1):
                gindex = index + 1
                model_row = tf.iloc[gindex]
                key_date = model_row['date']
                at_home = False
                if team_name == model_row[home_team]:
                    at_home = True
                    key_team = model_row[home_team]
                elif team_name == model_row[away_team]:
                    key_team = model_row[away_team]
                else:
                    raise KeyError("Team %s not found in Team Frame" % team)            
                try:
                    if at_home:
                        mpos = np.where((mf[home_team] == key_team) & (mf['date'] == key_date))[0][0]
                    else:
                        mpos = np.where((mf[away_team] == key_team) & (mf['date'] == key_date))[0][0]
                except:
                    raise IndexError("Team/Date Key not found in Model Frame")
                # insert team data into model row
                mf = insert_model_data(mf, mpos, mdict, tf, index, team1_prefix if at_home else team2_prefix)

        # Compute delta data 'home' - 'away'
        mf = generate_delta_data(mf, mdict, team1_prefix, team2_prefix)

        # Append this to final frame
        frames = [ff, mf]
        ff = pd.concat(frames)
        
    # Grouped Betting Results

    for col_key in ['won_on_points', 'won_on_spread', 'over']:
        ff_means = ff.groupby('date')[col_key].mean().shift(-1)
        ds_name = USEP.join([col_key, 'daily_mean_lag1'])
        ff_means.rename(ds_name, inplace=True)
        ff = ff.merge(ff_means, how='left', on='date')
        
    # Convert Boolean Features
    
    for bf in features_bool:
        ff[bf] = ff[bf].astype(float)

    # Write out dataframes

    input_dir = SSEP.join([run_dir, 'input'])
    if args.predict_mode:
        # get the prediction frame only
        new_predict_frame = ff.loc[ff.date >= predict_date]
        if len(new_predict_frame) <= 1:
            raise ValueError("Prediction frame has length 1 or less")
        # rewrite with all the features to the train and test files
        logger.info("Saving prediction frame")
        write_frame(new_predict_frame, input_dir, datasets[Partition.test],
                    model_specs['extension'], model_specs['separator'])
    else:
        # split data into training and test data
        new_train_frame = ff.loc[(ff.date >= train_date) & (ff.date < predict_date)]
        if new_train_frame.empty:
            raise ValueError(f"Training frame has no rows. Adjust tdate {train_date} back.")
        new_test_frame = ff.loc[ff.date >= predict_date]
        if new_test_frame.empty:
            raise ValueError(f"Testing frame has no rows. Adjust pdate {predict_date} back.")
        # rewrite with all the features to the train and test files
        logger.info("Saving training frame")
        write_frame(new_train_frame, input_dir, datasets[Partition.train],
                    model_specs['extension'], model_specs['separator'])
        logger.info("Saving testing frame")
        write_frame(new_test_frame, input_dir, datasets[Partition.test],
                    model_specs['extension'], model_specs['separator'])

    # Create the model from specs
    model = Model(model_specs)

    # Run the pipeline
    model = main_pipeline(alphapy_specs, model)

    # Save the TimeGPT training data
    save_timegpt_data(model_specs)
    
    # Update the live results
    
    if live_results:
        df_live = record_live_results(model_specs)
        extract_datasets(model_specs, df_live, league, gdrive)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("SportFlow End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
