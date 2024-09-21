
################################################################################
#
# Package   : AlphaPy
# Module    : sflow_globals
# Created   : September 21, 2024
#
# Copyright 2024 ScottFree Analytics LLC
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

feature_dict = {'wins' : int,
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
