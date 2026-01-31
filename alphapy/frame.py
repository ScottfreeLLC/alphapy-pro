################################################################################
#
# Package   : AlphaPy
# Module    : frame
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
# Imports
#

from alphapy.globals import PSEP, SSEP, USEP

import logging
import pandas as pd
import polars as pl


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function frame_name
#

def frame_name(name, space):
    r"""Get the frame name for the given name and space.

    Parameters
    ----------
    name : str
        Group name.
    space : alphapy.Space
        Context or namespace for the given group name.

    Returns
    -------
    fname : str
        Frame name.

    Examples
    --------

    >>> fname = frame_name('tech', Space('stock', 'prices', '1d'))
    # 'tech_stock_prices_1d'

    """
    return USEP.join([name, space.subject, space.source, space.fractal])


#
# Class Frame
#

class Frame(object):
    """Create a new Frame that points to a dataframe in memory. All
    frames are stored in ``Frame.frames``. Names must be unique.

    Parameters
    ----------
    name : str
        Frame key.
    space : alphapy.Space
        Namespace of the given frame.
    df : polars.DataFrame
        The contents of the actual dataframe.

    Attributes
    ----------
    frames : dict
        Class variable for storing all known frames

    Examples
    --------

    >>> Frame('tech', Space('stock', 'prices', '5m'), df)

    """

    # class variable to track all frames

    frames = {}

    # __init__

    def __init__(self,
                 name,
                 space,
                 df):
        # Accept both polars and pandas DataFrames
        df_class = df.__class__.__name__
        if df_class == 'DataFrame':
            fn = frame_name(name, space)
            if not fn in Frame.frames:
                self.name = name
                self.space = space
                self.df = df
                # add frame to frames list
                Frame.frames[fn] = self
            else:
                logger.debug("Frame %s already exists" % fn)
        else:
            logger.info("df must be of type DataFrame, got %s", df_class)

    # __str__

    def __str__(self):
        return frame_name(self.name, self.space)


#
# Function read_frame
#

def read_frame(directory, filename, extension, separator, index_col=False, dtype=None):
    r"""Read a delimiter-separated file into a Polars DataFrame.

    Parameters
    ----------
    directory : str
        Full directory specification.
    filename : str
        Name of the file to read, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    index_col : str, optional
        Column to use as the row labels (ignored in Polars).
    dtype : dict, optional
        Dictionary specifying data types for columns.

    Returns
    -------
    df : polars.DataFrame
        The Polars dataframe loaded from the file location. If the file
        cannot be located, then an empty DataFrame is returned.

    """
    file_only = PSEP.join([filename, extension])
    file_spec = SSEP.join([directory, file_only])
    logger.info("Loading data from %s", file_spec)
    try:
        # Convert dtype dict to polars schema if provided
        schema_overrides = None
        if dtype:
            type_map = {
                'int': pl.Int64,
                'int64': pl.Int64,
                'int32': pl.Int32,
                'float': pl.Float64,
                'float64': pl.Float64,
                'str': pl.Utf8,
                'string': pl.Utf8,
                'bool': pl.Boolean,
            }
            schema_overrides = {
                k: type_map.get(str(v).lower(), pl.Utf8)
                for k, v in dtype.items()
            }

        df = pl.read_csv(
            file_spec,
            separator=separator,
            schema_overrides=schema_overrides,
            infer_schema_length=10000,
        )
    except Exception as e:
        df = pl.DataFrame()
        logger.info("Could not find or access %s: %s", file_spec, e)
    return df


#
# Function write_frame
#

def write_frame(df, directory, filename, extension, separator, tag='',
                index=False, index_label=None, columns=None):
    r"""Write a dataframe into a delimiter-separated file.

    Parameters
    ----------
    df : polars.DataFrame
        The Polars dataframe to save to a file.
    directory : str
        Full directory specification.
    filename : str
        Name of the file to write, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    tag : str, optional
        An additional tag to add to the file name.
    index : bool, optional
        If True, write row index (pandas only, Polars has no indices).
    index_label : str, optional
        Column label for index column (pandas only).
    columns : list, optional
        A list of column names to write.

    Returns
    -------
    None : None

    """

    if tag != '':
        filename = USEP.join([filename, tag])
    file_only = PSEP.join([filename, extension])
    file_all = SSEP.join([directory, file_only])

    logger.info("Writing data frame to %s", file_all)
    try:
        # Handle both pandas and Polars DataFrames
        if isinstance(df, pd.DataFrame):
            # Select columns if specified
            if columns:
                df = df[columns]
            df.to_csv(file_all, sep=separator, index=index, index_label=index_label)
        else:
            # Polars DataFrame
            if columns:
                df = df.select(columns)
            df.write_csv(file_all, separator=separator)
    except Exception as e:
        logger.info("Could not write data frame to %s: %s", file_all, e)


#
# Function load_frames
#

def load_frames(group, directory, extension, separator, splits=False):
    r"""Read a group of dataframes into memory.

    Parameters
    ----------
    group : alphapy.Group
        The collection of frames to be read into memory.
    directory : str
        Full directory specification.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    splits : bool, optional
        If ``True``, then all the members of the group are stored in
        separate files corresponding with each member. If ``False``,
        then the data are stored in a single file.

    Returns
    -------
    all_frames : list
        The list of Polars dataframes loaded from the file location. If
        the files cannot be located, then an empty list is returned.

    """
    logger.info("Loading frames from %s", directory)
    gname = group.name
    gspace = group.space
    # If this is a group analysis, then consolidate the frames.
    # Otherwise, the frames are already aggregated.
    all_frames = []
    if splits:
        gnames = [item.lower() for item in group.members]
        for gn in gnames:
            fname = frame_name(gn, gspace)
            if fname in Frame.frames:
                logger.info("Joining Frame %s", fname)
                df = Frame.frames[fname].df
            else:
                logger.info("Data Frame for %s not found", fname)
                # read file for corresponding frame
                logger.info("Load Data Frame %s from file", fname)
                df = read_frame(directory, fname, extension, separator)
            # add this frame to the consolidated frame list
            if not df.is_empty():
                # set the name
                df = df.with_columns(pl.lit(gn).alias("tag_id"))
                # Move tag_id to first column
                cols = ["tag_id"] + [c for c in df.columns if c != "tag_id"]
                df = df.select(cols)
                all_frames.append(df)
            else:
                logger.debug("Empty Data Frame for: %s", gn)
    else:
        # no splits, so use data from consolidated files
        fname = frame_name(gname, gspace)
        df = read_frame(directory, fname, extension, separator)
        if not df.is_empty():
            all_frames.append(df)
    return all_frames


#
# Function dump_frames
#

def dump_frames(group, directory, extension, separator):
    r"""Save a group of data frames to disk.

    Parameters
    ----------
    group : alphapy.Group
        The collection of frames to be saved to the file system.
    directory : str
        Full directory specification.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.

    Returns
    -------
    None : None

    """
    logger.info("Dumping frames from %s", directory)
    gnames = [item.lower() for item in group.members]
    gspace = group.space
    for gn in gnames:
        fname = frame_name(gn, gspace)
        if fname in Frame.frames:
            logger.info("Writing Data Frame for %s", fname)
            df = Frame.frames[fname].df
            write_frame(df, directory, fname, extension, separator)
        else:
            logger.info("Data Frame for %s not found", fname)


#
# Function sequence_frame
#

def sequence_frame(df, target, date_id, forecast_period=1, n_lags=1, leaders=[], group_id=None):
    """
    Create sequences of lagging and leading values, with lagging applied within groups.

    Parameters
    ----------
    df : polars.DataFrame
        The original dataframe.
    target : str
        The target variable for prediction.
    date_id : str
        The datetime column.
    forecast_period : int
        The period for forecasting the target of the analysis.
    n_lags : int
        The number of lagged rows for prediction.
    leaders : list
        The features that are contemporaneous with the target.
    group_id : str, optional
        The grouping column.

    Returns
    -------
    new_frame : polars.DataFrame
        The transformed dataframe with variable sequences.
    """

    logger.info(f"Sequencing frame for target {target}")

    # Exclude non-relevant columns from lagging
    exclude_cols = [target, date_id] + leaders
    if group_id:
        exclude_cols.append(group_id)
    lag_cols = [col for col in df.columns if col not in exclude_cols]

    # Build lag expressions
    lag_exprs = []
    for i in range(1, n_lags + 1):
        for col in lag_cols:
            if group_id:
                lag_exprs.append(
                    pl.col(col).shift(i).over(group_id).alias(f"{col}[-{i}]")
                )
            else:
                lag_exprs.append(
                    pl.col(col).shift(i).alias(f"{col}[-{i}]")
                )

    # Build target shift expression
    target_expr = pl.col(target).shift(1 - forecast_period).alias(target)

    # Combine all transformations
    result = df.with_columns(lag_exprs + [target_expr])

    return result
