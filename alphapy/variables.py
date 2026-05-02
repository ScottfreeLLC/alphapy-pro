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

import warnings
import pandas as pd
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
import re
import sys

# AlphaPy Imports

from alphapy.alias import get_alias
from alphapy.globals import LOFF, ROFF, USEP
from alphapy.utilities import valid_name


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Variable
#

class Variable():
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
        expr_split = re.split(r'(\W)', expr_sub)
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
    func_name = root
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
    if func:
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
                vinfo = "Variable {} is not a function".format(root)
                logger.debug(vinfo)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
    # output frame and execution status
    return f


#
# Function cached_vtree
#

from functools import lru_cache

@lru_cache(maxsize=256)
def cached_vtree(feature):
    r"""Cached version of vtree for performance.

    Parameters
    ----------
    feature : str
        The feature name to get antecedent variables for.

    Returns
    -------
    tuple
        Tuple of all antecedent variables (tuple for hashability).

    """
    return tuple(vtree(feature))


#
# Function build_feature_order
#

def build_feature_order(features):
    r"""Build topologically sorted list of all variables needed.

    This function pre-computes the complete variable execution order
    for a list of features, eliminating redundant vtree calls when
    processing multiple symbols.

    Parameters
    ----------
    features : list
        List of feature names to compute.

    Returns
    -------
    ordered : list
        Ordered list of all variables to apply.

    """
    seen = set()
    ordered = []
    for feature in features:
        for v in cached_vtree(feature):
            if v not in seen:
                seen.add(v)
                ordered.append(v)
    return ordered


#
# Function vapply_fast
#

def vapply_fast(symbols, frames, features, vfuncs=None, max_workers=8):
    r"""Apply variables to multiple dataframes with parallelization.

    This is an optimized version of vapply that:
    1. Pre-computes the variable execution order once
    2. Processes symbols in parallel using ThreadPoolExecutor
    3. Caches vtree results to avoid redundant computation

    Parameters
    ----------
    symbols : list
        List of symbol names.
    frames : dict
        Dictionary mapping symbol names to DataFrames.
    features : list
        List of features to compute.
    vfuncs : dict, optional
        Dictionary of external modules and functions.
    max_workers : int
        Maximum number of parallel workers (default 8).

    Returns
    -------
    results : dict
        Dictionary mapping symbols to DataFrames with computed features.

    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Pre-compute variable order once for all symbols
    var_order = build_feature_order(features)
    logger.info(f"Processing {len(symbols)} symbols with {len(var_order)} variables")

    def process_symbol(symbol):
        """Process a single symbol's dataframe."""
        if symbol not in frames:
            logger.warning(f"No frame found for symbol: {symbol}")
            return symbol, None

        df = frames[symbol].copy()
        for v in var_order:
            df = vexec(df, v, vfuncs)
        return symbol, df

    # Process symbols in parallel
    results = {}
    n_workers = min(len(symbols), max_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_symbol, sym): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                symbol, df = future.result()
                if df is not None:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    logger.info(f"Completed processing {len(results)} symbols")
    return results
