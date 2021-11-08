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
# Imports
#

import builtins
from collections import OrderedDict
from importlib import import_module
import logging
import pandas as pd
import parser
import re
import sys

from alphapy.alias import get_alias
from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.globals import BSEP, CARET, ESEP, LOFF, PSEP, ROFF, USEP
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
                return
            try:
                _ = parser.expr(expr)
            except:
                logger.info("Invalid expression: %s", expr)
                return
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
    # ('lmin_5', 'lmin', 'lowest_low', ['5'], 2)

    """

    # split along lag first
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
    # return all components
    return vxlag, root, valias, plist, lag


#
# Function allvars
#

def allvars(expr):
    r"""Get the list of valid names in the expression.

    Parameters
    ----------
    expr : str
        A valid expression conforming to the Variable Definition Language.

    Returns
    -------
    vlist : list
        List of valid variable names.

    """
    vlist = []
    logger.debug("Expression: %s", expr)
    for item in expr.split(BSEP):
        match1_re = r'(\(|-)?[A-Za-z]\w+(\[\d+\])?'
        match1 = re.search(match1_re, item)
        match2_re = r'\(([^)]+)\)'
        match2 = re.search(match2_re, item)
        if match1 or match2:
            if match2:
                var = match2.group(0)[1:-1]
            elif match1:
                var = match1.group(0)
            var = re.sub('[()-]', '', var)
            vlist.append(var)
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
    info : str
        String describing the outcome:
            ``'variable added'``
            ``'function called'``
            ``'variable lagged'``
            ``'fractal deferred'``

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """

    vxlag, root, valias, plist, lag = vparse(v)
    logger.debug("vexec  : %s", v)
    logger.debug("vxlag  : %s", vxlag)
    logger.debug("root   : %s", root)
    logger.debug("valias : %s", valias)
    logger.debug("plist  : %s", plist)
    logger.debug("lag    : %s", lag)

    vinfo = 'variable not added'
    if vxlag not in f.columns:
        if root in Variable.variables:
            logger.debug("Found variable %s: ", root)
            vroot = Variable.variables[root]
            expr = vroot.expr
            if CARET in expr:
                vinfo = 'fractal deferred'
            else:
                expr_new = vsub(vxlag, expr)
                expr_split = expr_new.split(BSEP)
                expr_split = [''.join(['`', e, '`']) if e in f.columns else e for e in expr_split]
                expr_new = BSEP.join(expr_split)
                estr = "%s" % BSEP.join([''.join(['`', vxlag, '`']), ESEP, expr_new])
                logger.debug("Expression: %s", estr)
                # pandas eval
                f[vxlag] = f.eval(expr_new)
                vinfo = 'variable added'
        else:
            logger.debug("Did not find variable: %s", root)
            # Must be a function call
            func_name = root
            # Convert the parameter list and prepend the data frame
            newlist = []
            for p in plist:
                try:
                    newlist.append(int(p))
                except:
                    try:
                        newlist.append(float(p))
                    except:
                        newlist.append(p)
            newlist.insert(0, f)
            # Find the module and function
            module = None
            if vfuncs:
                for m in vfuncs:
                    funcs = vfuncs[m]
                    if func_name in funcs:
                        module = m
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
                    try:
                        ap_module = import_module('alphapy.transforms')
                        func = getattr(ap_module, func_name)
                    except:
                        func = None
            if func:
                # Create the variable by calling the function
                f[v] = func(*newlist)
                vinfo = 'function called'
            elif func_name not in dir(builtins):
                module_error = "*** Could not find module to execute function: {} ***".format(func_name)
                logger.error(module_error)
                sys.exit(module_error)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
        vinfo = 'variable lagged'
    # output frame and execution status
    return f, vinfo

    
#
# Function vexec_fractal
#

def vexec_fractal(f, v, vfuncs=None):
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
    info : str
        String describing the outcome:
            ``'variable added'``
            ``'function called'``
            ``'variable lagged'``
            ``'fractal deferred'``

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """

    vxlag, root, valias, plist, lag = vparse(v)
    logger.debug("vexec  : %s", v)
    logger.debug("vxlag  : %s", vxlag)
    logger.debug("root   : %s", root)
    logger.debug("valias : %s", valias)
    logger.debug("plist  : %s", plist)
    logger.debug("lag    : %s", lag)

    vinfo = 'variable not added'
    if vxlag not in f.columns:
        if root in Variable.variables:
            logger.debug("Found variable %s: ", root)
            vroot = Variable.variables[root]
            expr = vroot.expr
            if CARET in expr:
                vinfo = 'fractal deferred'
            else:
                expr_new = vsub(vxlag, expr)
                expr_split = expr_new.split(BSEP)
                expr_split = [''.join(['`', e, '`']) if e in f.columns else e for e in expr_split]
                expr_new = BSEP.join(expr_split)
                estr = "%s" % BSEP.join([''.join(['`', vxlag, '`']), ESEP, expr_new])
                logger.debug("Expression: %s", estr)
                # pandas eval
                f[vxlag] = f.eval(expr_new)
                vinfo = 'variable added'
        else:
            logger.debug("Did not find variable: %s", root)
            # Must be a function call
            func_name = root
            # Convert the parameter list and prepend the data frame
            newlist = []
            for p in plist:
                try:
                    newlist.append(int(p))
                except:
                    try:
                        newlist.append(float(p))
                    except:
                        newlist.append(p)
            newlist.insert(0, f)
            # Find the module and function
            module = None
            if vfuncs:
                for m in vfuncs:
                    funcs = vfuncs[m]
                    if func_name in funcs:
                        module = m
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
                    try:
                        ap_module = import_module('alphapy.transforms')
                        func = getattr(ap_module, func_name)
                    except:
                        func = None
            if func:
                # Create the variable by calling the function
                f[v] = func(*newlist)
                vinfo = 'function called'
            elif func_name not in dir(builtins):
                module_error = "*** Could not find module to execute function: {} ***".format(func_name)
                logger.error(module_error)
                sys.exit(module_error)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
        vinfo = 'variable lagged'
    # output frame and execution status
    return f, vinfo


#
# Function vfrac
#

def vfrac(f, c, fractal, func):
    r"""Transform a feature based on a higher-order fractal.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the base fractal column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    fractal : str
        Pandas offset alias.
    func : str
        The Pandas function to apply.

    Returns
    -------
    new_column : pandas.Series
        The series containing the fractal value.

    """

    if func == 'first':
        new_column = f.groupby(pd.Grouper(freq=fractal))[c].transform(lambda x: x.expanding().apply(lambda x: x.values[0]))
    elif func == 'last':
        new_column = f.groupby(pd.Grouper(freq=fractal))[c].transform(lambda x: x.expanding().apply(lambda x: x.values[-1]))
    else:
        try:
            new_column = f.groupby(pd.Grouper(freq=fractal))[c].transform(func)
        except:
            logger.info("Could not apply function %s for fractal %s", func, fractal)
    return new_column


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

    gsubject = group.space.subject
    gschema = group.space.schema
    symbols = [item.lower() for item in group.members]

    # Extract market specification fields

    fractals = market_specs['fractals']
    features = market_specs['features']
    ohlc_map = market_specs['ohlc_map']

    # Separate the features into single-fractal and multi-fractal features.

    features_sf = [f for f in features if CARET not in f]
    features_mf = [f for f in features if CARET in f]

    # Initialize list of dataframes, function dictionary, and possibly OHLC mapping values

    dffs = []
    func_dict = {'open'  : 'first',
                 'high'  : 'cummax',
                 'low'   : 'cummin',
                 'close' : 'last'}
    if ohlc_map:
        new_names = [x+'0' for x in ohlc_map.keys()]

    # Apply the variables to each frame

    for symbol in symbols:
        logger.info("Applying Variables to %s", symbol.upper())
        # apply variables to each of the fractals
        dfs = []
        for fractal in fractals:
            logger.debug("Fractal: %s", fractal)           
            fspace = Space(gsubject, gschema, fractal)
            fname = frame_name(symbol.lower(), fspace)
            if fname in Frame.frames:
                df = Frame.frames[fname].df
                if not df.empty:
                    # Remap OHLC values if specified
                    if ohlc_map:
                        for v in ohlc_map.keys():
                            df, _ = vexec(df, ohlc_map[v])
                        df.rename(columns=dict(zip(ohlc_map.keys(), new_names)), inplace=True)
                        df.rename(columns=dict(zip(ohlc_map.values(), ohlc_map.keys())), inplace=True)
                    # create the features in the dataframe
                    for vname in features_sf:
                        # get all the precedent variables
                        logger.debug("%s Variable: %s.%s", symbol.upper(), fractal, vname)
                        allv = vtree(vname)
                        for v in allv:
                            df, vinfo = vexec(df, v, vfuncs)
                            if vinfo == 'fractal deferred':
                                features_mf.append(v)
                else:
                    raise RuntimeError("Empty Dataframe for %s [%s]" % (symbol, fractal))
            else:
                raise RuntimeError("Dataframe Not Found for %s [%s]" % (symbol, fractal))
            # rename the columns
            df = df.add_prefix(fractal + PSEP)
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
                # create cumulative fractal variables
                for j in range(indexf):
                    for field in func_dict.keys():
                        col1name = '.'.join([fractals[0], field])
                        col2name = ''.join([col1name, 'f', str(j+1)])
                        dfj[col2name] = vfrac(dfj, col1name, fractals[indexf], func_dict[field])
            else:
                dfj = df
        # apply fractal features
        dfj = vexec_fractal(dfj, features_mf, fractals)
        # add the symbol
        colsym = 'symbol'
        dfj[colsym] = symbol
        first_col = dfj.pop(colsym)
        dfj.insert(0, colsym, first_col)
        # append frame to list of dataframes
        dffs.append(dfj)
    # return all of the dataframes
    return dffs
