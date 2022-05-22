################################################################################
#
# Package   : AlphaPy
# Module    : variables
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
# Variables
# ---------
#
# Numeric substitution is allowed for any number in the expression.
# Offsets are allowed in event expressions but cannot be substituted.
#
# Examples
# --------
#
# Variable('rrunder', 'rr_3_20 <= 0.9')
#
# 'rrunder_2_10_0.7'
# 'rrunder_2_10_0.9'
# 'xmaup_20_50_20_200'
# 'xmaup_10_50_20_50'
#


#
# Suppress Warnings
#

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#
# Imports
#

import builtins
from collections import OrderedDict
from importlib import import_module
import logging
import math
import re
import sys

from alphapy.alias import get_alias
from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.globals import BarType
from alphapy.globals import CARET, LOFF, ROFF, USEP
from alphapy.space import Space
from alphapy.utilities import valid_name


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Variable
#

class Variable(object):
    """Create a new variable as a key-value pair. All variables are stored
    in ``Variable.variables``. Duplicate keys or values are not allowed,
    unless the ``replace`` parameter is ``True``.

    Parameters
    ----------
    name : str
        Variable key.
    expr : str
        Variable value.
    replace : bool, optional
        Replace the current key-value pair if it already exists.

    Attributes
    ----------
    variables : dict
        Class variable for storing all known variables

    Examples
    --------

    >>> Variable('rrunder', 'rr_3_20 <= 0.9')
    >>> Variable('hc', 'higher_close')

    """

    # class variable to track all variables

    variables = {}

    # function __new__

    def __new__(cls,
                name,
                expr,
                replace = False):
        # code
        efound = expr in [Variable.variables[key].expr for key in Variable.variables]
        if efound:
            key = [key for key in Variable.variables if expr in Variable.variables[key].expr]
            logger.info("Expression '%s' already exists for key %s", expr, key)
            return
        if replace or not name in Variable.variables:
            if not valid_name(name):
                logger.info("Invalid variable key: %s", name)
            else:
                return super(Variable, cls).__new__(cls)
        logger.info("Key %s already exists", name)

    # function __init__

    def __init__(self,
                 name,
                 expr):
        # code
        self.name = name
        self.expr = expr
        # add key with expression
        Variable.variables[name] = self
       
    # function __str__

    def __str__(self):
        return self.expr


#
# Function vparse
#

def vparse(vname):
    r"""Parse a variable name into its respective components.

    Parameters
    ----------
    vname : str
        The name of the variable.

    Returns
    -------
    vxlag : str
        Original variable name without the ``lag`` component.
    root : str
        The base variable name without the parameters.
    valias : str
        Expanded name with alias substitution. 
    plist : list
        The parameter list.
    lag : int
        The offset starting with the current value [0]
        and counting back, e.g., an offset [1] means the
        previous value of the variable.

    Notes
    -----

    **AlphaPy** makes feature creation easy. The syntax
    of a variable name maps to a function call:

    xma_20_50 => xma(20, 50)

    Examples
    --------

    >>> vparse('lmin_5[2]')
    # (0, 'lmin_5', 'lmin', 'lowest_low', ['5'], 2)

    """

    # split along lag
    lsplit = vname.split(LOFF)
    vxlag = lsplit[0]
    # if necessary, substitute any alias
    vxlag1 = vxlag.split(USEP)[0]
    alias = get_alias(vxlag1)
    if alias:
        valias = vxlag.replace(vxlag1, alias)
        vsplit = valias.split(USEP)
    else:
        valias = None
        vsplit = vxlag.split(USEP)
    # get root
    root = vsplit[0]
    # parameter list
    plist = vsplit[1:]
    # extract lag
    lag = 0
    if len(lsplit) > 1:
        # lag is present
        slag = lsplit[1].replace(ROFF, '')
        if len(slag) > 0:
            lpat = r'(^-?[0-9]+$)'
            lre = re.compile(lpat)
            if lre.match(slag):
                lag = int(slag)
    # log results in debug mode
    logger.debug("vname   : %s", vname)
    logger.debug("vxlag   : %s", vxlag)
    logger.debug("root    : %s", root)
    logger.debug("valias  : %s", valias)
    logger.debug("plist   : %s", plist)
    logger.debug("lag     : %s", lag)
    # return all components
    return vxlag, root, valias, plist, lag


#
# Function allvars
#

def allvars(expr, match_fractal=True, match_lag=True):
    r"""Get the list of valid names in the expression.

    Parameters
    ----------
    expr : str
        A valid expression conforming to the Variable Definition Language.
    match_fractal : bool
        Flag to match fractal special character.
    match_lag : bool
        Flag to match fractal special character.

    Returns
    -------
    vlist : list
        List of valid variable names.

    """
    vlist = []
    logger.debug("Expression: %s", expr)
    if match_fractal and match_lag:
        pat = r'(\^)?([A-Za-z]{1}\w+)(\[\d+\])?'
    elif match_fractal:
        pat = r'[\^]?[A-Za-z]{1}\w+'
    elif match_lag:
        pat = r'[A-Za-z]{1}\w+(\[\d+\])?'
    else:
        pat = r'[A-Za-z]{1}\w+'
    vgroups = re.findall(pat, expr)
    vlist = [''.join(vgroup) for vgroup in vgroups]
    logger.debug("Variable Names: %s", vlist)
    return vlist


#
# Function vtree
#

def vtree(vname):
    r"""Get all of the antecedent variables. 

    Before applying a variable to a dataframe, we have to recursively
    get all of the child variables, beginning with the starting variable's
    expression. Then, we have to extract the variables from all the
    subsequent expressions. This process continues until all antecedent
    variables are obtained.

    Parameters
    ----------
    vname : str
        A valid variable stored in ``Variable.variables``.

    Returns
    -------
    all_variables : list
        The variables that need to be applied before ``vname``.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """
    allv = []
    def vwalk(allv, vname):
        vxlag, root, _, plist, lag = vparse(vname)
        if root in Variable.variables:
            root_expr = Variable.variables[root].expr
            expr = vsub(vxlag, root_expr)
            av = allvars(expr)
            for v in av:
                vwalk(allv, v)
        else:
            for p in plist:
                if valid_name(p):
                    vwalk(allv, p)
        allv.append(vxlag)
        if lag > 0:
            allv.append(vname)
        return allv
    allv = vwalk(allv, vname)
    all_variables = list(OrderedDict.fromkeys(allv))
    return all_variables


#
# Function vsub
#

def vsub(v, expr):
    r"""Substitute the variable parameters into the expression.

    This function performs the parameter substitution when
    applying features to a dataframe. It is a mechanism for
    the user to override the default values in any given
    expression when defining a feature, instead of having
    to programmatically call a function with new values.  

    Parameters
    ----------
    v : str
        Variable name.
    expr : str
        The expression for substitution.

    Returns
    -------
    newexpr
        The expression with the new, substituted values.

    """

    # numbers pattern
    npat = r'[-+]?[0-9]*\.?[0-9]+'
    nreg = re.compile(npat)
    # find all number locations in variable name
    viter = nreg.finditer(v)
    vlocs = []
    for match in viter:
        vlocs.append(match.span())
    # find all number locations in expression
    # find all non-number locations as well
    elen = len(expr)
    eiter = nreg.finditer(expr)
    elocs = []
    enlocs = []
    index = 0
    for match in eiter:
        eloc = match.span()
        elocs.append(eloc)
        enlocs.append((index, eloc[0]))
        index = eloc[1]
    # build new expression
    newexpr = str()
    for i, enloc in enumerate(enlocs):
        if i < len(vlocs):
            newexpr += expr[enloc[0]:enloc[1]] + v[vlocs[i][0]:vlocs[i][1]]
        else:
            newexpr += expr[enloc[0]:enloc[1]] + expr[elocs[i][0]:elocs[i][1]]
    if elocs:
        estart = elocs[len(elocs)-1][1]
    else:
        estart = 0
    newexpr += expr[estart:elen]
    return newexpr

    
#
# Function vexpr
#

def vexpr(f, v):
    r"""Get the expanded expression for a variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the variables.
    v : str
        Variable to add to the dataframe.

    Returns
    -------
    expr_new : str
        Expanded expression for evaluation.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """

    vxlag, root, _, _, _ = vparse(v)
    if root in Variable.variables:
        logger.debug("Found variable root %s: ", root)
        expr = Variable.variables[root].expr
        expr_sub = vsub(vxlag, expr)
        expr_split = re.split('(\W)', expr_sub)
        expr_new = ''.join([''.join(['`', e, '`']) if e in f.columns else e for e in expr_split])
        logger.debug("Expression: %s", expr_new)
    else:
        expr_new = None
        logger.debug("Could not find expression for variable: %s", v)
    # output expanded expression for evaluation
    return expr_new

    
#
# Function vfunc
#

def vfunc(f, v, vfuncs):
    r"""Find a function for defining a variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe to contain the new variable.
    v : str
        Variable representing a function.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    func    : function
        Function to execute for defining the variable.
    newlist : list
        Function parameter list.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """

    _, root, _, plist, _ = vparse(v)
    func_name = root
    # Convert the parameter list and prepend the data frame
    newlist = []
    for param in plist:
        try:
            newlist.append(int(param))
        except:
            try:
                newlist.append(float(param))
            except:
                newlist.append(param)
    newlist.insert(0, f)
    # Find the module and function
    module = None
    if vfuncs:
        for module_name in vfuncs:
            funcs = vfuncs[module_name]
            if func_name in funcs:
                module = module_name
                break
    # If the module was found, import the external transform function,
    # else search the local namespace and AlphaPy.
    if module:
        ext_module = import_module(module)
        func = getattr(ext_module, func_name)
    else:
        modname = globals()['__name__']
        module = sys.modules[modname]
        if func_name in dir(module):
            func = getattr(module, func_name)
        else:
            # Search the AlphaPy namespace
            try:
                ap_module = import_module('alphapy.transforms')
                func = getattr(ap_module, func_name)
            except:
                func = None
    # return function and parameter list
    logger.debug("Found function %s with parameters %s", func_name, newlist)
    return func, newlist


#
# Function vexec
#

def vexec(f, v, vfuncs=None):
    r"""Add a variable to the given dataframe.

    This is the core function for adding a variable to a dataframe.
    The default variable functions are already defined locally
    in ``alphapy.transforms``; however, you may want to define your
    own variable functions. If so, then the ``vfuncs`` parameter
    will contain the list of modules and functions to be imported
    and applied by the ``vexec`` function.

    To write your own variable function, your function must have
    a pandas *DataFrame* as an input parameter and must return
    a pandas *DataFrame* with the new variable(s).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe to contain the new variable.
    v : str
        Variable to add to the dataframe.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    f : pandas.DataFrame
        Dataframe with the new variable.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """

    vxlag, root, _, _, lag = vparse(v)
    if vxlag not in f.columns:
        # find the variable to evaluate
        veval = vexpr(f, v)
        if veval:
            try:
                f[vxlag] = f.eval(veval)
            except:
                logger.info("Variable %s: %s could not be evaluated", vxlag, veval)
        else:
            # Must be a function call
            func, newlist = vfunc(f, v, vfuncs)
            if func:
                f[v] = func(*newlist)
            elif root not in dir(builtins):
                vinfo = "Variable {} could not be defined".format(root)
                logger.info(vinfo)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
    # output frame and execution status
    return f


#
# Function map_bar_type
#

def map_dollar_bars(df, amount=1000000):
    r"""Map time bars to dollar bars.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to convert to a different bar type.
    amount : int
        The notional amount based on price times volume.
    """

    #get_dollar_bars(time_bars, dollar_threshold):

    dollar_bars = []
    running_volume = 0

    # initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    for i in range(len(df)):
        # get the timestamp, open, high, low, close, and volume of the next bar
        next_close, next_high, next_low, next_open, next_timestamp, next_volume = [time_bars[i][k] for k in ['close', 'high', 'low', 'open', 'time', 'vol']]
        # get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = ((next_open) + (next_close))/2
        # get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price
        # update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)
        # if the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:
            # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
            bar_timestamp = next_timestamp + timedelta(minutes=1)            
            # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollar_bars += [{'timestamp': bar_timestamp, 'open': next_open, 'high': running_high, 'low': running_low, 'close': next_close}]
            # reset the running volume to zero
            running_volume = 0
            # reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf
        # otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars
    return df


#
# Function map_bar_type
#

def map_bar_type(df, bar_type):
    r"""Map time bars to a different bar type.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to convert to a different bar type.
    bar_type : Enum.BarType
        The bar type for conversion (Dollar Bar, Heikin-Ashi, et al).
    """

    if bar_type == BarType.time:
        pass
    elif bar_type == BarType.dollar:
        df = map_dollar_bars(df)
    elif bar_type == BarType.heikinashi:
        ha_map = {'open'  : 'openha',
                  'high'  : 'highha',
                  'low'   : 'lowha',
                  'close' : 'closeha'}
        new_names = [x+'0' for x in ha_map.keys()]
        for v in ha_map.keys():
            df = vexec(df, ha_map[v])
        df.rename(columns=dict(zip(ha_map.keys(), new_names)), inplace=True)
        df.rename(columns=dict(zip(ha_map.values(), ha_map.keys())), inplace=True)
    else:
        raise ValueError("Unknown Bar Type: %s" % bar_type)
    return df


#
# Function vapply
#

def vapply(group, market_specs, vfuncs=None):
    r"""Apply a set of variables to multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    dfs : list
        The list of pandas dataframes to analyze.

    Other Parameters
    ----------------
    Frame.frames : dict
        Global dictionary of dataframes

    """

    # Get group information

    gspace = group.space
    gsubject = gspace.subject
    gsource = gspace.source
    symbols = [item.lower() for item in group.members]

    # Extract market specification fields

    fractals = market_specs['fractals']
    features = market_specs['features']
    bar_type = market_specs['bar_type']

    # Initialize list of dataframes, function dictionary, and possibly bar type
    dffs = []

    # Apply the variables to each frame

    for symbol in symbols:
        logger.info("Applying Variables to %s", symbol.upper())
        # apply variables to each of the fractals
        dfs = []
        for fractal in fractals:
            logger.info("Fractal: %s", fractal)           
            fspace = Space(gsubject, gsource, fractal)
            fname = frame_name(symbol.lower(), fspace)
            if fname in Frame.frames:
                df = Frame.frames[fname].df
                if not df.empty:
                    # Remap to a different bar type if specified
                    df = map_bar_type(df, bar_type)
                    # create the features in the dataframe
                    all_features = features[fractal]
                    for feature in all_features:
                        allv = vtree(feature)
                        logger.debug("%s Feature: %s_%s", symbol.upper(), fractal, feature)
                        for v in allv:
                            logger.debug("%s Variable: %s_%s", symbol.upper(), fractal, v)
                            df = vexec(df, v, vfuncs)
                else:
                    logger.info("Empty Dataframe for %s [%s]" % (symbol, fractal))
            else:
                logger.info("Dataframe Not Found for %s [%s]" % (symbol, fractal))
            # rename the columns
            df = df.add_suffix(USEP + fractal)
            # add the fractal frame to the list
            dfs.append(df)
        # join all fractal frames
        logger.info("Joining Frames: %s", fractals)
        for indexf, df in enumerate(dfs):
            # upsample successive frames
            if indexf > 0:
                # shift higher fractals
                df = df.shift(1)
                # resample for base fractal
                dfr = df.resample(fractals[0]).ffill()
                # join frames
                dfj = dfj.merge(dfr, left_index=True, right_index=True)
            else:
                dfj = df
        # add the symbol
        colsym = 'symbol'
        dfj[colsym] = symbol
        first_col = dfj.pop(colsym)
        dfj.insert(0, colsym, first_col)
        # assign global frame for trading
        tspace = Space(gsubject, gsource, 'ALL')
        _ = Frame(symbol.lower(), tspace, dfj)
        # append frame to list of dataframes
        dffs.append(dfj)
    # return all of the dataframes
    return dffs
