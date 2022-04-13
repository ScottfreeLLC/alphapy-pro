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
    name : str
        The system name.
    algo : str
        Abbreviation of the algorithm.
    ts_flag : bool
        Flag indicating whether to use the walk-forward or train/test probability.
    buysignal : str
        Name of the conditional feature for a long entry.
    buyexit : str, optional
        Name of the conditional feature for a long exit.
    sellsignal : str, optional
        Name of the conditional feature for a short entry.
    sellexit : str, optional
        Name of the conditional feature for a short exit.
    holdperiod : int, optional
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
                name,
                algo,
                ts_flag,
                prob_min,
                buysignal,
                buystop = None,
                buyexit = None,
                sellsignal = None,
                sellstop = None,
                sellexit = None,
                holdperiod = None,
                fractal = '1D'):
        # create system name
        if name not in System.systems:
            return super(System, cls).__new__(cls)
        else:
            logger.info("System %s already exists", name)
    
    # __init__
    
    def __init__(self,
                 name,
                 algo,
                 ts_flag,
                 prob_min,
                 buysignal,
                 buystop = None,
                 buyexit = None,
                 sellsignal = None,
                 sellstop = None,
                 sellexit = None,
                 holdperiod = None,
                 fractal = '1D'):
        # initialization
        self.name = name
        self.algo = algo
        self.ts_flag = ts_flag
        self.prob_min = prob_min
        self.buysignal = buysignal
        self.buystop = buystop
        self.buyexit = buyexit
        self.sellsignal = sellsignal
        self.sellstop = sellstop
        self.sellexit = sellexit
        self.holdperiod = holdperiod
        self.fractal = fractal
        # add system to systems list
        System.systems[name] = self
        
    # __str__

    def __str__(self):
        return self.name


#
# Function trade_system
#

def trade_system(model, system, space, intraday, symbol, quantity):
    r"""Trade the given system.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    system : alphapy.System
        The long/short system to run.
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

    # Unpack the model data.

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack the system parameters.

    algo = system.algo
    ts_flag = system.ts_flag
    prob_min = system.prob_min
    buysignal = system.buysignal
    buystop = system.buystop
    buyexit = system.buyexit
    sellsignal = system.sellsignal
    sellstop = system.sellstop
    sellexit = system.sellexit
    holdperiod = system.holdperiod
    fractal = system.fractal

    # Read in the price frame for all fractals and variables.

    symbol = symbol.lower()
    tspace = Space(space.subject, space.schema, 'ALL')
    tframe = Frame.frames[frame_name(symbol, tspace)].df

    # Initialize signal dictionary

    signals = {'buysignal'  : buysignal,
               'buystop'    : buystop,
               'buyexit'    : buyexit,
               'sellsignal' : sellsignal,
               'sellstop'   : sellstop,
               'sellexit'   : sellexit}

    # Use model output probabilities as input to the system

    if prob_min:
        logger.info("Getting probabilities for %s", symbol.upper())
        # read the rankings frame for the given symbol
        rank_dir = SSEP.join([directory, 'output'])
        file_path = most_recent_file(rank_dir, 'ranked_test*')
        file_name = file_path.split(SSEP)[-1].split('.')[0]
        df_rank = read_frame(rank_dir, file_name, extension, separator, index_col='date')
        # select the probability column for the trading system
        partition_tag = 'test_'
        ts_tag = 'ts_' if ts_flag else ''
        prob_col = ''.join(['prob_', partition_tag, ts_tag, algo])
        df_rank = df_rank.query('symbol==@symbol')
        df_rank.index = pd.to_datetime(df_rank.index)
        # join price with rankings to get probabilities for this symbol
        tframe = tframe.merge(df_rank[prob_col], how='left', left_index=True, right_index=True)
        tframe[prob_col].fillna(0.5, inplace=True)
        # substitute actual probability column into signal
        for key in signals.keys():
            if signals[key] and (key == 'buysignal' or key == 'sellsignal'):
                expr = BSEP.join([key, '=', prob_col, '>=', str(prob_min)])
                tframe.eval(expr, inplace=True)
    else:
        for key in signals.keys():
            vname = signals[key]
            if vname:
                tframe[key] = tframe[vname]

    # Initialize trading state variables

    inlong = False
    leactive = False
    hprev = 0
    inshort = False
    seactive = False
    lprev = 0
    hold = 0
    psize = 0
    q = quantity
    tradelist = []

    # Loop through prices and generate trades

    ccol = USEP.join(['close', fractal])
    hcol = USEP.join(['high', fractal])
    lcol = USEP.join(['low', fractal])
    icol = USEP.join(['endofday', fractal])

    for dt, row in tframe.iterrows():
        # get prices for this row
        c = row[ccol]
        h = row[hcol]
        l = row[lcol]
        end_of_day = row[icol] if intraday else False
        # evaluate entry and exit conditions
        lerow = row['buysignal'] if buysignal else None
        lsrow = row['buystop'] if buystop else None
        lxrow = row['buyexit'] if buyexit else None
        serow = row['sellsignal'] if sellsignal else None
        ssrow = row['sellstop'] if sellstop else None
        sxrow = row['sellexit'] if sellexit else None
        # process the long and short events
        if lerow or leactive:
            orderclose = lerow and not buystop
            orderstop = leactive and (True if h > hprev else False)
            if psize < 0 and (orderclose or orderstop):
                # short active, so exit short
                tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
                inshort = False
                hold = 0
                psize = 0
            if psize == 0:
                if (orderclose or orderstop) and not end_of_day:
                    # go long (again)
                    if orderclose:
                        tradelist.append((dt, [symbol, Orders.le, q, c]))
                    elif orderstop:
                        tradelist.append((dt, [symbol, Orders.le, q, hprev]))
                    inlong = True
                    psize = psize + q
                    leactive = False
                elif lerow and buystop:
                    leactive = True
                    hprev = lsrow
        if serow or seactive:
            orderclose = serow and not sellstop
            orderstop = seactive and (True if l < lprev else False)
            if psize > 0 and (orderclose or orderstop):
                # long active, so exit long
                tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
                inlong = False
                hold = 0
                psize = 0
            if psize == 0:
                if (orderclose or orderstop) and not end_of_day:
                    # go short (again)
                    if orderclose:
                        tradelist.append((dt, [symbol, Orders.se, -q, c]))
                    elif orderstop:
                        tradelist.append((dt, [symbol, Orders.se, -q, lprev]))
                    inshort = True
                    seactive = False
                    psize = psize - q
                elif serow and sellstop:
                    seactive = True
                    lprev = ssrow
        # check exit conditions
        if inlong and hold > 0 and lxrow:
            # long active, so exit long
            tradelist.append((dt, [symbol, Orders.lx, -psize, c]))
            inlong = False
            hold = 0
            psize = 0
        if inshort and hold > 0 and sxrow:
            # short active, so exit short
            tradelist.append((dt, [symbol, Orders.sx, -psize, c]))
            inshort = False
            hold = 0
            psize = 0
        # if a holding period was given, then check for exit
        if holdperiod and hold >= holdperiod:
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

    system_name = system.name
    logger.info("Generating Trades for System %s", system_name)

    # Unpack the model data.

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Extract the group information.

    gname = group.name
    gmembers = group.members
    gspace = group.space

    # Run the system for each member of the group

    gtlist = []
    for symbol in gmembers:
        # generate the trades for this member
        tlist = trade_system(model, system, gspace, intraday, symbol, quantity)
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
