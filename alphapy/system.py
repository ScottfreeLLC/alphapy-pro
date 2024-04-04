################################################################################
#
# Package   : AlphaPy
# Module    : system
# Created   : July 11, 2013
#
# Copyright 2020 ScottFree Analytics LLC
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
# Imports
#

from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import read_frame
from alphapy.frame import write_frame
from alphapy.globals import ModelType
from alphapy.globals import Orders
from alphapy.globals import BSEP, SSEP, USEP
from alphapy.metalabel import get_daily_vol
from alphapy.space import Space
from alphapy.portfolio import Trade
from alphapy.utilities import most_recent_file

import logging
import pandas as pd
from pandas import DataFrame


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class System
#

class System(object):
    """Create a new system. All systems are stored in
    ``System.systems``. Duplicate names are not allowed.

    Parameters
    ----------
    system_name : str
        The name of the pattern.
    predict_history : int
        Historical period required to calculate predictions.
    forecast_period : int
        Holding period of a position.
    profit_factor : float
        The multiple of volatility for taking a profit.
    stoploss_factor : float
        The multiple of volatility for taking a loss.
    algo : str
        Abbreviation for algorithm.
    prob_min : float
        A probability between 0.0 and 1.0.
    prob_max : float
        A probability between 0.0 and 1.0.
    fractal : str
        Pandas offset alias.

    Attributes
    ----------
    systems : dict
        Class variable for storing all known systems

    Examples
    --------

    >>> System('closer', hc, lc)

    """

    # class variable to track all systems

    systems = {}

    # __new__

    def __new__(cls,
                system_name,
                forecast_period = 1,
                predict_history = 50,
                profit_factor = 1.0,
                stoploss_factor = 1.0,
                algo = 'xgb',
                prob_min = 0.0,
                prob_max = 1.0,
                fractal = '1D'):
        # create system name
        if system_name not in System.systems:
            return super(System, cls).__new__(cls)
        else:
            logger.info("System %s already exists", system_name)

    # __init__

    def __init__(self,
                 system_name,
                 forecast_period = 1,
                 predict_history = 50,
                 profit_factor = 1.0,
                 stoploss_factor = 1.0,
                 algo = 'xgb',
                 prob_min = 0.0,
                 prob_max = 1.0,
                 fractal = '1D'):
        # initialization
        self.system_name = system_name
        self.forecast_period = forecast_period
        self.predict_history = predict_history
        self.profit_factor = profit_factor
        self.stoploss_factor = stoploss_factor
        self.algo = algo
        self.prob_min = prob_min
        self.prob_max = prob_max
        self.fractal = fractal
        # add system to systems list
        System.systems[system_name] = self

    # __str__

    def __str__(self):
        return self.system_name


#
# Class SystemRank
#

class SystemRank(object):
    """Create a new system. All systems are stored in
    ``SystemRank.systems``. Duplicate names are not allowed.

    Parameters
    ----------
    system_name : str
        The name of the pattern.
    forecast_period : int
        Holding period of a position.
    algo : str
        Abbreviation for algorithm.
    long_rank : int
        The multiple of volatility for taking a profit.
    long_score : float
        The multiple of volatility for taking a loss.
    short_rank : int
        The minimum return required to take a profit.
    short_score : float
        A probability between 0.0 and 1.0.
    fractal : str
        Pandas offset alias.

    Attributes
    ----------
    systems : dict
        Class variable for storing all known ranking systems

    """

    # class variable to track all systems

    systems = {}

    # __new__

    def __new__(cls,
                system_name,
                forecast_period = 1,
                algo = 'xgbk',
                long_rank = 1,
                long_score = 1.0,
                short_rank = 1,
                short_score = 1.0,
                fractal = '1D'):
        # create system name
        if system_name not in SystemRank.systems:
            return super(SystemRank, cls).__new__(cls)
        else:
            logger.info("System Rank %s already exists", system_name)

    # __init__

    def __init__(self,
                 system_name,
                 forecast_period = 1,
                 algo = 'xgbk',
                 long_rank = 1,
                 long_score = 1.0,
                 short_rank = 1,
                 short_score = 1.0,
                 fractal = '1D'):
        # initialization
        self.system_name = system_name
        self.forecast_period = forecast_period
        self.algo = algo
        self.long_rank = long_rank
        self.long_score = long_score
        self.short_rank = short_rank
        self.short_score = short_score
        self.fractal = fractal
        # add system to systems list
        System.systems[system_name] = self

    # __str__

    def __str__(self):
        return self.system_name


#
# Function trade_ranking
#

def trade_ranking(symbol, quantity, system, df_rank, space, intraday):
    r"""Trade the given system.

    Parameters
    ----------
    symbol : str
        The symbol to trade.
    quantity : float
        The amount of the ``symbol`` to trade, e.g., number of shares
    system : alphapy.System
        The long/short system to run.
    df_rank : pd.DataFrame
        The dataframe containing the ranked predictions.
    space : alphapy.Space
        Namespace of all variables over all fractals.
    intraday : bool
        If True, then run an intraday system.

    Returns
    -------
    tradelist : list
        List of trade entries and exits.

    Other Parameters
    ----------------
    Frame.frames : dict
        All of the data frames containing price data.

    """

    # Unpack the system parameters.

    forecast_period = system.forecast_period
    algo = system.algo
    long_rank = system.long_rank
    long_score = system.long_score
    short_rank = system.short_rank
    short_score = system.short_score
    trade_fractal = system.fractal

    # Read in the price frame for all fractals and variables.

    symbol = symbol.lower()
    tspace = Space(space.subject, space.source, 'ALL')
    df_trade = Frame.frames[frame_name(symbol, tspace)].df.copy()

    # extract the rankings frame for the given symbol

    df_sym = df_rank.query('symbol==@symbol').copy()
    df_sym.index = pd.to_datetime(df_sym.index)

    # evaluate entries by joining price with ranked probabilities

    symbol = symbol.upper()
    logger.info("Getting ranks for %s", symbol)
    partition_tag = 'test'
    pcol = USEP.join(['pred', partition_tag, algo.lower()])
    df_trade = df_trade.merge(df_sym[[pcol, 'long_rank', 'short_rank']],
                              how='left', left_index=True, right_index=True)

    # Initialize trading state variables

    inlong = False
    inshort = False
    psize = 0
    q = quantity
    hold = 0
    tradelist = []

    # Loop through prices and generate trades

    ccol = USEP.join(['close', trade_fractal])
    hcol = USEP.join(['high', trade_fractal])
    lcol = USEP.join(['low', trade_fractal])
    icol = USEP.join(['endofday', trade_fractal])

    for dt, row in df_trade.iterrows():
        # get prices for this row
        c = row[ccol]
        # evaluate entry and exit conditions
        score = row[pcol]
        lrank = row['long_rank']
        srank = row['short_rank']
        if long_score:
            lerow = score >= long_score and lrank <= long_rank
        else:
            lerow = lrank <= long_rank
        if short_score:
            serow = score <= short_score and srank <= short_rank
        else:
            serow = srank <= short_rank
        # check for intraday positions
        end_of_day = row[icol] if intraday else False
        # process the long and short events
        if lerow:
            if inshort:
                # short active, so exit short
                tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                inshort = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go long
                tradelist.append((dt, [symbol, Orders.le, q, c]))
                inlong = True
                psize = psize + q
        if serow:
            if inlong:
                # long active, so exit long
                tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                inlong = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go short
                tradelist.append((dt, [symbol, Orders.se, -q, c]))
                inshort = True
                psize = psize - q
        # Exit when holding period is reached
        if hold >= forecast_period:
            if inlong:
                tradelist.append((dt, [symbol, Orders.lh, -psize, c]))
                inlong = False
            if inshort:
                tradelist.append((dt, [symbol, Orders.sh, -psize, c]))
                inshort = False
            hold = psize = 0
        # check current positions for exit
        if inlong or inshort:
            # increment the hold counter
            hold += 1
            # close any intraday trades at the end of the day
            if intraday and end_of_day:
                if inlong:
                    # long active, so exit long
                    tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                    inlong = False
                if inshort:
                    # short active, so exit short
                    tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                    inshort = False
                hold = psize =0
    return tradelist


#
# Function trade_metalabel
#

def trade_metalabel(symbol, quantity, system, df_rank, space, intraday,
                    ts_flag=False, use_probs=True):
    r"""Trade the given system.

    Parameters
    ----------
    symbol : str
        The symbol to trade.
    quantity : float
        The amount of the ``symbol`` to trade, e.g., number of shares
    system : alphapy.System
        The long/short system to run.
    df_rank : pd.DataFrame
        The dataframe containing the ranked predictions.
    space : alphapy.Space
        Namespace of all variables over all fractals.
    intraday : bool
        If True, then run an intraday system.
    ts_flag : bool (optional)
        True if using time series probabilities.
    use_probs : bool (optional)
        Flag indicating whether to use probabilities.

    Returns
    -------
    tradelist : list
        List of trade entries and exits.

    Other Parameters
    ----------------
    Frame.frames : dict
        All of the data frames containing price data.

    """

    # Unpack the system parameters.

    forecast_period = system.forecast_period
    profit_factor = system.profit_factor
    stoploss_factor = system.stoploss_factor
    trade_fractal = system.fractal
    algo = system.algo
    prob_min = system.prob_min
    prob_max = system.prob_max

    # Read in the price frame for all fractals and variables.

    symbol = symbol.lower()
    tspace = Space(space.subject, space.source, 'ALL')
    df_trade = Frame.frames[frame_name(symbol, tspace)].df.copy()

    # Get daily volatility and calculate the profit target and stop loss.

    close_col = USEP.join(['close', trade_fractal])
    ds_close = df_trade[close_col]
    ds_vol = get_daily_vol(ds_close)

    # extract the rankings frame for the given symbol

    df_sym = df_rank.query('symbol==@symbol').copy()
    df_sym.index = pd.to_datetime(df_sym.index)

    # entry probability function

    def assign_entry(df, prob_col, prob_min, prob_max):
        if prob_min and prob_max:
            lhs = BSEP.join(['(', prob_col, '>=', str(prob_min), ')'])
            rhs = BSEP.join(['(', prob_col, '<=', str(prob_max), ')'])
            expr = BSEP.join(['entry', '=', lhs, '&', rhs])
        elif prob_min:
            expr = BSEP.join(['entry', '=', prob_col, '>=', str(prob_min)])
        elif prob_max:
            expr = BSEP.join(['entry', '=', prob_col, '<=', str(prob_max)])
        else:
            lhs = BSEP.join(['(', prob_col, '>= 0.0)'])
            rhs = BSEP.join(['(', prob_col, '<= 1.0)'])
            expr = BSEP.join(['entry', '=', lhs, '&', rhs])
        df = df.eval(expr)
        return df

    # evaluate entries by joining price with ranked probabilities

    symbol = symbol.upper()
    logger.info("Getting probabilities for %s", symbol)
    partition_tag = 'test'
    if ts_flag:
        pcol = USEP.join(['prob', partition_tag, 'ts', algo.lower()])
    else:
        pcol = USEP.join(['prob', partition_tag, algo.lower()])
    df_trade = df_trade.merge(df_sym[pcol], how='left', left_index=True, right_index=True)
    df_sym[pcol].fillna(0.5, inplace=True)
    if use_probs:
        df_trade = assign_entry(df_trade, pcol, prob_min, prob_max)
    else:
        df_trade = assign_entry(df_trade, pcol, 0.0, 1.0)

    # Initialize trading state variables

    inlong = False
    inshort = False
    psize = 0
    q = quantity
    hold = 0
    tradelist = []

    # Loop through prices and generate trades

    ccol = USEP.join(['close', trade_fractal])
    hcol = USEP.join(['high', trade_fractal])
    lcol = USEP.join(['low', trade_fractal])
    icol = USEP.join(['endofday', trade_fractal])

    for dt, row in df_trade.iterrows():
        # get prices for this row
        c = row[ccol]
        h = row[hcol]
        l = row[lcol]
        # evaluate entry and exit conditions
        lerow = row['entry'] and row['side'] == 1
        serow = row['entry'] and row['side'] == -1
        end_of_day = row[icol] if intraday else False
        # calculate profit targets and stop losses
        try:
            daily_vol = ds_vol.loc[dt]
        except KeyError:
            try:
                daily_vol = ds_vol.iloc[0]
            except IndexError:
                daily_vol = 0.03
        profit_target = profit_factor * daily_vol * c
        stop_loss = stoploss_factor * daily_vol * c
        # process the long and short events
        if lerow:
            if inshort:
                # short active, so exit short
                tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                inshort = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go long
                tradelist.append((dt, [symbol, Orders.le, q, c]))
                inlong = True
                le_price = c
                psize = psize + q
        if serow:
            if inlong:
                # long active, so exit long
                tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                inlong = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go short
                tradelist.append((dt, [symbol, Orders.se, -q, c]))
                inshort = True
                se_price = c
                psize = psize - q
        # Exit when holding period is reached
        if hold >= forecast_period:
            if inlong:
                tradelist.append((dt, [symbol, Orders.lh, -psize, c]))
                inlong = False
            if inshort:
                tradelist.append((dt, [symbol, Orders.sh, -psize, c]))
                inshort = False
            hold = psize = 0
        # check current positions for exit
        if inlong or inshort:
            # increment the hold counter
            hold += 1
            # check for profit targets or stop losses
            if inlong and hold > 1:
                if h >= le_price + profit_target:
                    # profit target
                    tradelist.append((dt, [symbol, Orders.lx, -psize, le_price + profit_target]))
                    inlong = False
                if l <= le_price - stop_loss:
                    # stop loss
                    tradelist.append((dt, [symbol, Orders.lx, -psize, le_price - stop_loss]))
                    inlong = False
                if not inlong:
                    hold = psize = 0
            if inshort and hold > 1:
                if l <= se_price - profit_target:
                    # profit target
                    tradelist.append((dt, [symbol, Orders.sx, -psize, se_price - profit_target]))
                    inshort = False
                if h >= se_price + stop_loss:
                    # stop loss
                    tradelist.append((dt, [symbol, Orders.sx, -psize, se_price + stop_loss]))
                    inshort = False
                if not inshort:
                    hold = psize = 0
            # close any intraday trades at the end of the day
            if intraday and end_of_day:
                if inlong:
                    # long active, so exit long
                    tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                    inlong = False
                if inshort:
                    # short active, so exit short
                    tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                    inshort = False
                hold = psize =0
    return tradelist


#
# Function trade_system
#

def trade_system(symbol, quantity, system, target, df_rank, space,
                 intraday, use_probs=True):
    r"""Trade the given system.

    Parameters
    ----------
    symbol : str
        The symbol to trade.
    quantity : float
        The amount of the ``symbol`` to trade, e.g., number of shares
    system : alphapy.System
        The long/short system to run.
    target : str
        The target variable.
    df_rank : pd.DataFrame
        The dataframe containing the ranked predictions.
    space : alphapy.Space
        Namespace of all variables over all fractals.
    intraday : bool
        If True, then run an intraday system.
    use_probs : bool (optional)
        Flag indicating whether to use probabilities.

    Returns
    -------
    tradelist : list
        List of trade entries and exits.

    Other Parameters
    ----------------
    Frame.frames : dict
        All of the data frames containing price data.

    """

    # Unpack the system parameters.

    forecast_period = system.forecast_period
    profit_factor = system.profit_factor
    stoploss_factor = system.stoploss_factor
    trade_fractal = system.fractal
    algo = system.algo
    prob_min = system.prob_min
    prob_max = system.prob_max

    # Read in the price frame for all fractals and variables.

    symbol = symbol.lower()
    tspace = Space(space.subject, space.source, 'ALL')
    df_trade = Frame.frames[frame_name(symbol, tspace)].df.copy()

    # Get daily volatility and calculate the profit target and stop loss.

    close_col = USEP.join(['close', trade_fractal])
    ds_close = df_trade[close_col]
    ds_vol = get_daily_vol(ds_close)

    # extract the rankings frame for the given symbol

    df_sym = df_rank.query('symbol==@symbol').copy()
    df_sym.index = pd.to_datetime(df_sym.index)

    # entry probability function

    def assign_entry(df, prob_col, prob_min, prob_max):
        if prob_min and prob_max:
            lhs = BSEP.join(['(', prob_col, '>=', str(prob_min), ')'])
            rhs = BSEP.join(['(', prob_col, '<=', str(prob_max), ')'])
            expr = BSEP.join(['entry', '=', lhs, '&', rhs])
        elif prob_min:
            expr = BSEP.join(['entry', '=', prob_col, '>=', str(prob_min)])
        elif prob_max:
            expr = BSEP.join(['entry', '=', prob_col, '<=', str(prob_max)])
        else:
            lhs = BSEP.join(['(', prob_col, '>= 0.0)'])
            rhs = BSEP.join(['(', prob_col, '<= 1.0)'])
            expr = BSEP.join(['entry', '=', lhs, '&', rhs])
        df = df.eval(expr)
        return df

    # evaluate entries by joining price with ranked probabilities

    partition_tag = 'test'
    pcol = USEP.join(['prob', partition_tag, algo.lower()])
    df_trade = df_trade.merge(df_sym[pcol], how='left', left_index=True, right_index=True)
    df_sym[pcol].fillna(0.5, inplace=True)
    if use_probs:
        df_trade = assign_entry(df_trade, pcol, prob_min, prob_max)
    else:
        df_trade = assign_entry(df_trade, pcol, 0.0, 1.0)

    # Initialize trading state variables

    symbol = symbol.upper()
    inlong = False
    inshort = False
    psize = 0
    q = quantity
    hold = 0
    tradelist = []

    # Loop through prices and generate trades

    ccol = USEP.join(['close', trade_fractal])
    hcol = USEP.join(['high', trade_fractal])
    lcol = USEP.join(['low', trade_fractal])
    icol = USEP.join(['endofday', trade_fractal])

    for dt, row in df_trade.iterrows():
        # get prices for this row
        c = row[ccol]
        h = row[hcol]
        l = row[lcol]
        # evaluate entry and exit conditions
        lerow = row['entry'] if system_type == 'long' else None
        serow = row['entry'] if system_type == 'short' else None
        end_of_day = row[icol] if intraday else False
        # calculate profit targets and stop losses
        try:
            daily_vol = ds_vol.loc[dt]
        except KeyError:
            try:
                daily_vol = ds_vol.iloc[0]
            except IndexError:
                daily_vol = 0.03
        profit_target = profit_factor * daily_vol * c
        stop_loss = stoploss_factor * daily_vol * c
        # process the long and short events
        if lerow:
            if inshort:
                # short active, so exit short
                tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                inshort = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go long
                tradelist.append((dt, [symbol, Orders.le, q, c]))
                inlong = True
                le_price = c
                psize = psize + q
        if serow:
            if inlong:
                # long active, so exit long
                tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                inlong = False
                hold = psize = 0
            if psize == 0 and not end_of_day:
                # go short
                tradelist.append((dt, [symbol, Orders.se, -q, c]))
                inshort = True
                se_price = c
                psize = psize - q
        # Exit when holding period is reached
        if hold >= forecast_period:
            if inlong:
                tradelist.append((dt, [symbol, Orders.lh, -psize, c]))
                inlong = False
            if inshort:
                tradelist.append((dt, [symbol, Orders.sh, -psize, c]))
                inshort = False
            hold = psize = 0
        # check current positions for exit
        if inlong or inshort:
            # increment the hold counter
            hold += 1
            # check for profit targets or stop losses
            if inlong and hold > 1:
                if h >= le_price + profit_target:
                    # profit target
                    tradelist.append((dt, [symbol, Orders.lx, -psize, le_price + profit_target]))
                    inlong = False
                if l <= le_price - stop_loss:
                    # stop loss
                    tradelist.append((dt, [symbol, Orders.lx, -psize, le_price - stop_loss]))
                    inlong = False
                if not inlong:
                    hold = psize = 0
            if inshort and hold > 1:
                if l <= se_price - profit_target:
                    # profit target
                    tradelist.append((dt, [symbol, Orders.sx, -psize, se_price - profit_target]))
                    inshort = False
                if h >= se_price + stop_loss:
                    # stop loss
                    tradelist.append((dt, [symbol, Orders.sx, -psize, se_price + stop_loss]))
                    inshort = False
                if not inshort:
                    hold = psize = 0
            # close any intraday trades at the end of the day
            if intraday and end_of_day:
                if inlong:
                    # long active, so exit long
                    tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                    inlong = False
                if inshort:
                    # short active, so exit short
                    tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                    inshort = False
                hold = psize =0
    return tradelist


#
# Function run_system
#

def run_system(model,
               system,
               group,
               intraday = False,
               quantity = 1):
    r"""Run a system for a given group, creating a trades frame.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    system : alphapy.System or alphapy.SystemRank
        The system to run.
    group : alphapy.Group
        The group of symbols to trade.
    intraday : bool, optional
        If true, this is an intraday system.
    quantity : float, optional
        The amount to trade for each symbol, e.g., number of shares

    Returns
    -------
    tf : pandas.DataFrame
        All of the trades for this ``group``.

    """

    system_name = system.system_name
    logger.info("Generating Trades for System %s", system_name)

    # Unpack the model data.

    model_type = model.specs['model_type']
    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    separator = model.specs['separator']
    rank_group_id = model.specs['rank_group_id']
    target = model.specs['target']

    # Extract the group information.

    gname = group.name
    gmembers = group.members
    gspace = group.space

    # Get the latest rankings frame.

    rank_dir = SSEP.join([run_dir, 'output'])
    file_path = most_recent_file(rank_dir, 'ranked_test*')
    file_name = file_path.split(SSEP)[-1].split('.')[0]
    df_rank = read_frame(rank_dir, file_name, extension, separator, index_col='date')

    # If ranking, sort the dataframe, and assign long and short ranks

    if model_type == ModelType.ranking:
        partition_tag = 'test'
        prob_col = USEP.join(['pred', partition_tag, system.algo.lower()])
        df_rank.sort_values(by=['date', prob_col], ascending = [True, False], inplace=True)
        df_rank['long_rank'] = df_rank.groupby(rank_group_id)[prob_col].transform('rank', ascending=False)
        df_rank['short_rank'] = df_rank.groupby(rank_group_id)[prob_col].transform('rank')

    # Run the system for each member of the group

    gtlist_base = []
    gtlist_prob = []
    for symbol in gmembers:
        tlist_base = []
        tlist_prob = []
        # generate the trades for this member
        if model_type == ModelType.ranking:
            tlist_base = trade_ranking(symbol, quantity, system, df_rank, gspace, intraday)
        elif model_type == ModelType.metalabel:
            tlist_base = trade_metalabel(symbol, quantity, system, df_rank, gspace, intraday)
        else:
            tlist_prob = trade_system(symbol, quantity, system, target,
                                      df_rank, gspace, intraday)
            tlist_base = trade_system(symbol, quantity, system, target,
                                      df_rank, gspace, intraday, use_probs=False)
        if tlist_base:
            for item in tlist_base:
                gtlist_base.append(item)
        else:
            logger.info("No baseline trades for symbol %s", symbol.upper())
        if model_type != ModelType.ranking:
            if tlist_prob:
                for item in tlist_prob:
                    gtlist_prob.append(item)
            else:
                logger.info("No probability trades for symbol %s", symbol.upper())

    # Determine index column

    if intraday:
        index_column = 'datetime'
    else:
        index_column = 'date'

    # Create group trades frame

    def record_trades(gtlist, tag=''):
        tf = pd.DataFrame()
        if gtlist:
            tspace = Space(system_name, "trades", gspace.fractal)
            gtlist = sorted(gtlist, key=lambda x: x[0])
            tf1 = DataFrame.from_records(gtlist, columns=[index_column, 'trades'])
            tf2 = pd.DataFrame(tf1['trades'].to_list(), columns=Trade.states)
            tf = pd.concat([tf1[index_column], tf2], axis=1)
            tf.set_index(index_column, inplace=True)
            tfname = frame_name(gname, tspace)
            system_dir = SSEP.join([run_dir, 'systems'])
            write_frame(tf, system_dir, tfname, extension, separator, tag,
                        index=True, index_label=index_column)
            del tspace
        else:
            logger.info("No trades were found")
        return tf
            
    # Get the trading frames for all system runs.

    df_trades_base = record_trades(gtlist_base, 'base')
    if model_type != ModelType.ranking:
        df_trades_prob = record_trades(gtlist_prob, 'prob')
    else:
        df_trades_prob = pd.DataFrame()

    # Return trades frame
    return df_trades_base, df_trades_prob
