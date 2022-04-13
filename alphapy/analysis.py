################################################################################
#
# Package   : AlphaPy
# Module    : analysis
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

from alphapy.alphapy_main import main_pipeline
from alphapy.frame import write_frame
from alphapy.globals import LOFF, ROFF, SSEP, USEP
from alphapy.utilities import subtract_days

from datetime import timedelta
import logging
import pandas as pd
from pandas.tseries.offsets import BDay


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function analysis_name
#

def analysis_name(gname, target):
    r"""Get the name of the analysis.

    Parameters
    ----------
    gname : str
        Group name.
    target : str
        Target of the analysis.

    Returns
    -------
    name : str
        Value for the corresponding key.

    """
    name = USEP.join([gname, target])
    return name


#
# Class Analysis
#

class Analysis(object):
    """Create a new analysis for a group. All analyses are stored
    in ``Analysis.analyses``. Duplicate keys are not allowed.

    Parameters
    ----------
    model : alphapy.Model
        Model object for the analysis.
    group : alphapy.Group
        The group of members in the analysis.

    Attributes
    ----------
    Analysis.analyses : dict
        Class variable for storing all known analyses

    """

    analyses = {}

    # __new__

    def __new__(cls,
                model,
                group):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        if not an in Analysis.analyses:
            return super(Analysis, cls).__new__(cls)
        else:
            logger.info("Analysis %s already exists", an)

    # function __init__

    def __init__(self,
                 model,
                 group):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        # initialize analysis
        self.name = an
        self.model = model
        self.group = group
        # add analysis to analyses list
        Analysis.analyses[an] = self

    # __str__

    def __str__(self):
        return self.name


#
# Function run_analysis
#

def run_analysis(analysis, dfs, fractals, system_specs, forecast_period, predict_history):
    r"""Run an analysis for a given model and group.

    First, the data are loaded for each member of the analysis group.
    Then, the target value is lagged for the ``forecast_period``. Each frame
    is split along the ``predict_date`` from the ``analysis``, and finally
    the train and test files are generated.

    Parameters
    ----------
    analysis : alphapy.Analysis
        The analysis to run.
    dfs : list
        The list of pandas dataframes to analyze.
    system_specs : dict
        The system specifications containing the signals.
    fractals : list
        List of Pandas offset aliases.
    forecast_period : int
        The period for forecasting the target of the analysis.
    predict_history : int
        The number of periods required for lookback calculations.

    Returns
    -------
    analysis : alphapy.Analysis
        The completed analysis.

    """

    # Unpack analysis
    model = analysis.model

    # Unpack model data

    test_file = model.test_file
    train_file = model.train_file

    # Unpack model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    predict_date = model.specs['predict_date']
    predict_mode = model.specs['predict_mode']
    separator = model.specs['separator']
    target = model.specs['target']
    train_date = model.specs['train_date']

    # Unpack system specifications

    buysignal = system_specs['buysignal']
    sellsignal = system_specs['sellsignal']

    # Calculate split date

    logger.info("Analysis Dates")
    split_date = subtract_days(predict_date, predict_history)

    # Create dataframes

    if predict_mode:
        # create predict frame
        logger.info("Split Date for Prediction Mode: %s", split_date)
        predict_frame = pd.DataFrame()
    else:
        # create train and test frames
        logger.info("Split Date for Training Mode: %s", predict_date)
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()

    #
    # We are creating a target variable based on whether the trade was successful.
    # If the trade is profitable, then the target is 1 else 0.
    #
    # For long signals, the ROI must be greater than 0.
    # For short signals, the ROI must be less than 0.
    #

    for df in dfs:
        # subset each individual frame and add to the master frame
        symbol = df['symbol'].iloc[0]
        first_date = df.index[0]
        last_date = df.index[-1]
        logger.info("Analyzing %s from %s to %s", symbol.upper(), first_date, last_date)
        # shift ROI column back by the number of forecast periods
        target_roi = USEP.join(['roi', str(forecast_period), fractals[0]])
        df[target_roi] = df[target_roi].shift(-forecast_period)
        # filter for signal
        df_signal = pd.DataFrame()
        if buysignal:
            col_buy = USEP.join([buysignal, fractals[0]])
            df_buy = df[df[col_buy] == True]
            df_buy[target] = df_buy[target_roi] > 0.0
            df_buy.drop(columns=[target_roi], inplace=True)
            df_signal = pd.concat([df_signal, df_buy])
        if sellsignal:
            col_sell = USEP.join([sellsignal, fractals[0]])
            df_sell = df[df[col_sell] == True]
            df_sell[target] = df_sell[target_roi] < 0.0
            df_sell.drop(columns=[target_roi], inplace=True)
            df_signal = pd.concat([df_signal, df_sell])
        # get frame subsets
        if predict_mode:
            new_predict = df_signal.loc[(df_signal.index >= split_date) & (df_signal.index <= last_date)]
            if len(new_predict) > 0:
                predict_frame = predict_frame.append(new_predict)
            else:
                logger.info("%s Prediction Frame has zero rows. Check prediction date.", symbol.upper())
        else:
            # split data into train and test
            new_train = df_signal.loc[(df_signal.index >= train_date) & (df_signal.index < predict_date)]
            if not new_train.empty:
                # check if target column has NaN values
                nan_count = new_train[target].isnull().sum()
                if nan_count > 0:
                    logger.info("%s has %d train records with a NaN target.", symbol.upper(), nan_count)
                # drop records with NaN values in target column
                new_train = new_train.dropna(subset=[target])
                train_frame = train_frame.append(new_train)
                # get test frame
                new_test = df_signal.loc[(df_signal.index >= predict_date) & (df_signal.index <= last_date)]
                if not new_test.empty:
                    # check if target column has NaN values
                    nan_count = new_test[target].isnull().sum()
                    forecast_check = forecast_period - 1
                    if nan_count != forecast_check:
                        logger.info("%s has %d test records with a NaN target.", symbol.upper(), nan_count)
                    # drop records with NaN values in target column
                    new_test = new_test.dropna(subset=[target])
                    # append selected records to the test frame
                    test_frame = test_frame.append(new_test)
                else:
                    logger.info("%s Testing Frame has zero rows. Check prediction date.", symbol.upper())
            else:
                logger.info("%s Training Frame has zero rows. Check data source.", symbol.upper())

    # Convert column names from special characters

    def new_col_name(col_name):
        start = col_name.find(LOFF)
        end = col_name.find(ROFF)
        lag_string = col_name[start:end+1]
        lag_value = lag_string[1:-1]
        if lag_value:
            new_name = ''.join([col_name.replace(lag_string, ''), '_lag', lag_value])
        else:
            new_name = col_name
        return new_name

    new_columns = [new_col_name(x) for x in train_frame.columns]
    train_frame.columns = new_columns
    if not test_frame.empty:
        test_frame.columns = new_columns

    # Write out the frames for input into the AlphaPy pipeline

    directory = SSEP.join([directory, 'input'])
    if predict_mode:
        # write out the predict frame
        write_frame(test_frame, directory, test_file, extension, separator,
                    index=True, index_label='date')
    else:
        # write out the train and test frames
        write_frame(train_frame, directory, train_file, extension, separator,
                    index=True, index_label='date')
        write_frame(test_frame, directory, test_file, extension, separator,
                    index=True, index_label='date')

    # Run the AlphaPy pipeline

    logger.info('*'*80)
    logger.info("Running AlphaPy")
    logger.info('*'*80)
    analysis.model = main_pipeline(model)

    # Return the analysis
    return analysis
