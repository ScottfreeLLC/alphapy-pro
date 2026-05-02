################################################################################
#
# Package   : AlphaPy
# Module    : data
# Created   : July 11, 2013
#
# Copyright 2019 ScottFree Analytics LLC
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
from alphapy.globals import datasets
from alphapy.globals import ModelType
from alphapy.globals import Partition
from alphapy.globals import SSEP
from alphapy.globals import WILDCARD
from alphapy.space import Space

import logging
import numpy as np
import polars as pl
import re
from sklearn.preprocessing import LabelEncoder
import sys


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_data
#

def get_data(model, partition):
    r"""Get data for the given partition.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    df_X : polars.DataFrame
        The feature set.
    df_y : polars.DataFrame
        The array of target values, if available.

    """

    logger.info("Loading Data")

    # Extract the model data

    directory = model.specs['directory']
    live_results = model.specs['live_results']
    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    features = model.specs['features']
    model_type = model.specs['model_type']
    separator = model.specs['separator']
    target = model.specs['target']

    # Initialize X and y

    df_X = pl.DataFrame()
    df_y = pl.DataFrame()

    # Read in the file

    filename = datasets[partition]
    data_dir = SSEP.join([directory, 'data'])
    df = read_frame(data_dir, filename, extension, separator)
    if df.is_empty():
        input_dir = SSEP.join([run_dir, 'input'])
        df = read_frame(input_dir, filename, extension, separator)

    # Get features and target

    if not df.is_empty():
        if target in df.columns:
            logger.info("Found target %s in data frame", target)
            # check if target column has NaN values
            nan_count = df[target].null_count()
            logger.info("Found %d records with NaN target values", nan_count)
            # drop NA targets
            if not live_results or partition == Partition.train:
                df = df.drop_nulls(subset=[target])
                if nan_count > 0:
                    logger.info("Dropped %d records with NaN target values", nan_count)
            # assign the target column to y
            df_y = df.select(target)
            # encode label only for classification or system
            if model_type == ModelType.classification:
                y = LabelEncoder().fit_transform(df_y[target].to_numpy())
                df_y = pl.DataFrame({target: y})
            # drop the target from the original frame
            df = df.drop(target)
        else:
            logger.info("Target %s not found in %s", target, partition)
        # Extract features
        if features == WILDCARD:
            df_X = df
        else:
            df_X = df.select(features)

    # Labels are returned usually only for training data
    return df_X, df_y


#
# Function shuffle_data
#

def shuffle_data(model):
    r"""Randomly shuffle the training data.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the shuffled data.

    """

    # Extract model parameters.

    seed = model.specs['seed']
    shuffle = model.specs['shuffle']

    # Extract model data.

    X_train = model.X_train

    # Shuffle data

    if shuffle:
        logger.info("Shuffling Training Data")
        np.random.seed(seed)
        np.random.shuffle(X_train)
        model.X_train = X_train
    else:
        logger.info("Skipping Shuffling")

    return model

