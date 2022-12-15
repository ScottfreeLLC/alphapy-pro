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
from alphapy.globals import Orders
from alphapy.globals import BSEP, SSEP, USEP
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
    signal_long : str
        The entry condition for a long position.
    signal_short : str
        The entry condition for a short position.
    profit_factor : float
        The multiple of volatility for taking a profit.
    stoploss_factor : float
        The multiple of volatility for taking a loss.
    minimum_return : float
        The minimum return required to take a profit.
    forecast_period : int
        Holding period of a position.
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
                signal_long,
                signal_short,
                profit_factor = 1.0,
                stoploss_factor = 1.0,
                minimum_return = 0.05,
                forecast_period = 1,
                fractal = '1D'):
        # create system name
        if system_name not in System.systems:
            return super(System, cls).__new__(cls)
        else:
            logger.info("System %s already exists", system_name)

    # __init__

    def __init__(self,
                 system_name,
                 signal_long,
                 signal_short,
                 profit_factor = 1.0,
                 stoploss_factor = 1.0,
                 minimum_return = 0.05,
                 forecast_period = 1,
                 fractal = '1D'):
        # initialization
        self.system_name = system_name
        self.signal_long = signal_long
        self.signal_short = signal_short
        self.profit_factor = profit_factor
        self.stoploss_factor = stoploss_factor
        self.minimum_return = minimum_return
        self.forecast_period = forecast_period
        self.fractal = fractal
        # add system to systems list
        System.systems[system_name] = self

    # __str__

    def __str__(self):
        return self.system_name


#
# Function trade_system
#

def trade_system(system, df_rank, ts_flag, space, intraday, symbol, quantity):
    r"""Trade the given system.

    Parameters
    ----------
    system : alphapy.System
        The long/short system to run.
    df_rank : pd.DataFrame
        The dataframe containing the ranked predictions.
    ts_flag : bool
        True if using time series probabilities.
    space : alphapy.Space
        Namespace of all variables over all fractals.
    intraday : bool
        If True, then run an intraday system.
    symbol : str
        The symbol to trade.
    quantity : float
        The amount of the ``symbol`` to trade, e.g., number of shares

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

    system_type = system.system_type
    forecast_period = system.forecast_period
    algo = system.algo
    prob_min = system.prob_min
    prob_max = system.prob_max
    fractal = system.fractal

    # Read in the price frame for all fractals and variables.

    symbol = symbol.lower()
    tspace = Space(space.subject, space.source, 'ALL')
    tframe = Frame.frames[frame_name(symbol, tspace)].df.copy()

    # extract the rankings frame for the given symbol

    df_sym = df_rank.query('symbol==@symbol').copy()
    df_sym.index = pd.to_datetime(df_sym.index)

    # entry probability function

    def assign_entry(df, key, prob_col, prob_min, prob_max):
        if prob_min and prob_max:
            lhs = BSEP.join(['(', prob_col, '>=', str(prob_min), ')'])
            rhs = BSEP.join(['(', prob_col, '<=', str(prob_max), ')'])
            expr = BSEP.join([key, '=', lhs, '&', rhs])
        elif prob_min:
            expr = BSEP.join([key, '=', prob_col, '>=', str(prob_min)])
        elif prob_max:
            expr = BSEP.join([key, '=', prob_col, '<=', str(prob_max)])
        else:
            lhs = BSEP.join(['(', prob_col, '>= 0.0'])
            rhs = BSEP.join(['(', prob_col, '<= 1.0'])
            expr = BSEP.join([key, '=', lhs, '&', rhs])
        df = df.eval(expr)
        return df

    # evaluate entries by joining price with ranked probabilities

    logger.info("Getting probabilities for %s", symbol.upper())
    partition_tag = 'test'
    if ts_flag:
        pcol = USEP.join(['prob', partition_tag, 'ts', algo.lower()])
    else:
        pcol = USEP.join(['prob', partition_tag, algo.lower()])
    tframe = tframe.merge(df_sym[pcol], how='left', left_index=True, right_index=True)
    df_sym[pcol].fillna(0.5, inplace=True)
    if system_type == 'short':
        df_sym[pcol] = 1.0 - df_sym[pcol]
    tframe = assign_entry(tframe, system_type, pcol, prob_min, prob_max)
    
    # Initialize trading state variables

    inlong = False
    inshort = False
    psize = 0
    q = quantity
    hold = 0
    tradelist = []

    # Loop through prices and generate trades

    ccol = USEP.join(['close', fractal])
    icol = USEP.join(['endofday', fractal])

    for dt, row in tframe.iterrows():
        # get prices for this row
        c = row[ccol]
        end_of_day = row[icol] if intraday else False
        # evaluate entry and exit conditions
        lerow = row['long'] if system_type == 'long' else None
        serow = row['short'] if system_type == 'short' else None
        # process the long and short events
        if lerow:
            if inshort:
                # short active, so exit short
                tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                inshort = False
                hold = 0
                psize = 0
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
                hold = 0
                psize = 0
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
            hold = 0
            psize = 0
        # increment the hold counter
        if inlong or inshort:
            hold += 1
            if intraday and end_of_day:
                if inlong:
                    # long active, so exit long
                    tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                    inlong = False
                if inshort:
                    # short active, so exit short
                    tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                    inshort = False
                hold = 0
                psize = 0
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
    system : alphapy.System
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

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Extract the group information.

    gname = group.name
    gmembers = group.members
    gspace = group.space

    # Get the latest rankings frame.

    rank_dir = SSEP.join([directory, 'output'])
    file_path = most_recent_file(rank_dir, 'ranked_test*')
    file_name = file_path.split(SSEP)[-1].split('.')[0]
    df_rank = read_frame(rank_dir, file_name, extension, separator, index_col='date')
    ts_flag = '_ts_' in file_name

    # Run the system for each member of the group

    gtlist = []
    for symbol in gmembers:
        # generate the trades for this member
        tlist = trade_system(system, df_rank, ts_flag, gspace, intraday, symbol, quantity)
        if tlist:
            # add trades to global trade list
            for item in tlist:
                gtlist.append(item)
        else:
            logger.info("No trades for symbol %s", symbol.upper())

    # Create group trades frame

    if intraday:
        index_column = 'datetime'
    else:
        index_column = 'date'

    tf = pd.DataFrame()
    if gtlist:
        tspace = Space(system_name, "trades", gspace.fractal)
        gtlist = sorted(gtlist, key=lambda x: x[0])
        tf1 = DataFrame.from_records(gtlist, columns=[index_column, 'trades'])
        tf2 = pd.DataFrame(tf1['trades'].to_list(), columns=Trade.states)
        tf = pd.concat([tf1[index_column], tf2], axis=1)
        tf.set_index(index_column, inplace=True)
        tfname = frame_name(gname, tspace)
        system_dir = SSEP.join([directory, 'systems'])
        write_frame(tf, system_dir, tfname, extension, separator,
                    index=True, index_label=index_column)
        del tspace
    else:
        logger.info("No trades were found")

    # Return trades frame
    return tf
