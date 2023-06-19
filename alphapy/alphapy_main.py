################################################################################
#
# Package   : AlphaPy
# Module    : alphapy_main
# Created   : July 11, 2013
#
# Copyright 2022 ScottFree Analytics LLC
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

import argparse
import logging
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
import sys
import yaml

from alphapy.alias import Alias
from alphapy.data import get_data
from alphapy.data import sample_data
from alphapy.data import shuffle_data
from alphapy.estimators import get_estimators
from alphapy.estimators import scorers
from alphapy.features import apply_transforms
from alphapy.features import create_crosstabs
from alphapy.features import create_features
from alphapy.features import create_interactions
from alphapy.features import drop_features
from alphapy.features import remove_lv_features
from alphapy.features import save_features
from alphapy.features import select_features_lofo
from alphapy.features import select_features_univariate
from alphapy.frame import write_frame
from alphapy.globals import SSEP, USEP
from alphapy.globals import ModelType
from alphapy.globals import Partition
from alphapy.group import Group
from alphapy.model import first_fit
from alphapy.model import generate_metrics
from alphapy.model import get_model_config
from alphapy.model import load_feature_map
from alphapy.model import load_predictor
from alphapy.model import make_predictions
from alphapy.model import Model
from alphapy.model import select_best_model
from alphapy.model import predict_blend
from alphapy.model import save_feature_map
from alphapy.model import save_metrics
from alphapy.model import save_predictions
from alphapy.model import save_predictor
from alphapy.model import time_series_model
from alphapy.optimize import hyper_grid_search
from alphapy.optimize import rfecv_search
from alphapy.plots import generate_plots
from alphapy.utilities import datetime_stamp
from alphapy.utilities import most_recent_file
from alphapy.variables import Variable


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_alphapy_config
#

def get_alphapy_config(alphapy_root):
    r"""Read the configuration file for AlphaPy.

    Parameters
    ----------
    alphapy_root : str
        The root directory for AlphaPy.

    Returns
    -------
    specs : dict
        The parameters for controlling AlphaPy.

    """

    logger.info('*'*80)
    logger.info("AlphaPy Configuration")

    # Read AlphaPy configuration file

    full_path = SSEP.join([alphapy_root, 'config', 'alphapy.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    specs['alphapy_root'] = alphapy_root

    #
    # Section: groups
    #

    full_path = SSEP.join([alphapy_root, 'config', 'groups.yml'])
    with open(full_path, 'r') as ymlfile:
        group_specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger.info("Creating Groups")
    try:
        for g in group_specs.keys():
            Group(g)
            Group.groups[g].add(group_specs[g])
            logger.info("Added Group: %s", g)
    except:
        raise ValueError("No Groups Found")

    #
    # Section: aliases
    #

    full_path = SSEP.join([alphapy_root, 'config', 'variables.yml'])
    with open(full_path, 'r') as ymlfile:
        var_specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger.info("Creating Aliases")
    try:
        for k, v in list(var_specs['aliases'].items()):
            Alias(k, v)
    except:
        raise ValueError("No Aliases Found")

    #
    # Section: variables
    #

    logger.info("Creating Variables")
    try:
        for k, v in list(var_specs['variables'].items()):
            Variable(k, v)
    except:
        raise ValueError("No Variables Found")

    #
    # Section: sources
    #

    full_path = SSEP.join([alphapy_root, 'config', 'sources.yml'])
    with open(full_path, 'r') as ymlfile:
        data_sources = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger.info("Getting Data Sources")
    try:
        specs['sources'] = data_sources
    except:
        raise ValueError("No Data Sources Found")

    # Set API Key environment variables

    for key in data_sources:
        key_dict = data_sources[key]
        if 'api_key' in key_dict and 'api_key_name' in key_dict and key_dict['api_key_name']:
            os.environ[key_dict['api_key_name']] = key_dict['api_key']
        if 'directory' in key_dict:
            dir = key_dict['directory']
            dir_exists = os.path.isdir(dir)
            if dir_exists:
                specs['data_dir'] = dir
            else:
                raise ValueError(f"Directory {dir} does not exist")

    #
    # Section: systems
    #

    full_path = SSEP.join([alphapy_root, 'config', 'systems.yml'])
    with open(full_path, 'r') as ymlfile:
        trading_systems = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger.info("Getting Trading Systems")
    try:
        specs['systems'] = trading_systems
    except:
        raise ValueError("No Trading Systems Found")

    #
    # Log the AlphaPy parameters
    #

    logger.info('ALPHAPY PARAMETERS:')
    for spec in specs.keys():
        logger.info('%s: %s', spec, specs[spec])

    # AlphaPy Specifications
    return specs


#
# Function training_pipeline
#

def training_pipeline(alphapy_specs, model):
    r"""AlphaPy Training Pipeline

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    model : alphapy.Model
        The model object for controlling the pipeline.

    Returns
    -------
    model : alphapy.Model
        The final results are stored in the model object.

    Raises
    ------
    KeyError
        If the number of columns of the train and test data do not match,
        then this exception is raised.

    """

    logger.info("Training Pipeline")

    # Unpack the model specifications

    run_dir = model.specs['run_dir']
    drop = model.specs['drop']
    extension = model.specs['extension']
    fs_lofo = model.specs['fs_lofo']
    fs_univariate = model.specs['fs_univariate']
    group_id = model.specs['rank_group_id']
    grid_search = model.specs['grid_search']
    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    sampling = model.specs['sampling']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    separator = model.specs['separator']
    shuffle = model.specs['shuffle']
    split = model.specs['split']
    target = model.specs['target']
    ts_date = model.specs['ts_date_index']
    ts_option = model.specs['ts_option']

    # Get train and test data

    X_train, y_train = get_data(model, Partition.train)
    print(X_train.shape, y_train.shape)
    X_test, y_test = get_data(model, Partition.test)
    print(X_test.shape, y_test.shape)

    # If there is no test partition, then we will split the train partition

    if X_test.empty:
        logger.info("No Test Data Found")
        if ts_option:
            logger.info("Splitting Training Data for Time Series")
            df_train = pd.concat([X_train, y_train], axis=1)
            df_sorted = df_train.sort_values(by=[ts_date])
            split_index = int((1.0 - split) * df_sorted.shape[0])
            split_date = df_sorted.iloc[split_index][ts_date]
            df_train = df_sorted[df_sorted[ts_date] <= split_date].reset_index()
            y_train = pd.DataFrame(df_train[target], columns=[target])
            X_train = df_train.drop(columns=[target])
            df_test = df_sorted[df_sorted[ts_date] > split_date].reset_index()
            y_test = pd.DataFrame(df_test[target], columns=[target])
            X_test = df_test.drop(columns=[target])
        else:
            logger.info("Splitting Training Data")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=split, random_state=seed, shuffle=shuffle)

    # Save original train/test data

    model.df_X_train = X_train
    model.df_y_train = y_train
    model.df_X_test = X_test
    model.df_y_test = y_test
    model = save_features(model, X_train, X_test, y_train, y_test)

    # Save train/test groups

    if group_id:
        train_counts = X_train.groupby(group_id).agg(['count'])
        model.groups_train = train_counts[train_counts.columns[0]].values
        del train_counts
        test_counts = X_test.groupby(group_id).agg(['count'])
        model.groups_test = test_counts[test_counts.columns[0]].values
        del test_counts

    # Determine if there are any test labels

    if not y_test.empty:
        logger.info("Test Labels Found")
        model.test_labels = True
    else:
        logger.info("Test Labels Not Found")

    # Log feature statistics

    logger.info("Original Feature Statistics")
    logger.info("Number of Training Rows    : %d", X_train.shape[0])
    logger.info("Number of Training Columns : %d", X_train.shape[1])
    if model_type == ModelType.classification:
        uv, uc = np.unique(y_train, return_counts=True)
        logger.info("Unique Training Values for %s : %s", target, uv)
        logger.info("Unique Training Counts for %s : %s", target, uc)
    logger.info("Number of Testing Rows     : %d", X_test.shape[0])
    logger.info("Number of Testing Columns  : %d", X_test.shape[1])
    if model_type == ModelType.classification and model.test_labels:
        uv, uc = np.unique(y_test, return_counts=True)
        logger.info("Unique Testing Values for %s : %s", target, uv)
        logger.info("Unique Testing Counts for %s : %s", target, uc)

    # Merge training and test data

    if X_train.shape[1] == X_test.shape[1]:
        split_point = X_train.shape[0]
        X_all = pd.concat([X_train, X_test])
    else:
        raise IndexError("The number of training and test columns [%d, %d] must match." %
                         (X_train.shape[1], X_test.shape[1]))

    # Apply transforms to the feature matrix
    X_all = apply_transforms(model, X_all)

    # Drop features
    X_all = drop_features(X_all, drop)

    # Save the train and test files with extracted and dropped features

    data_dir = SSEP.join([run_dir, 'input'])
    # train data
    df_train = X_all.iloc[:split_point, :]
    df_train[target] = y_train
    write_frame(df_train, data_dir, model.train_file, extension, separator, index=False)
    # test data
    df_test = X_all.iloc[split_point:, :]
    if model.test_labels:
        df_test[target] = y_test
    write_frame(df_test, data_dir, model.test_file, extension, separator, index=False)

    # Create crosstabs for any categorical features

    if model_type == ModelType.classification:
        create_crosstabs(model, target)

    # Create initial features

    X_all = create_features(model, X_all, X_train, X_test, y_train)
    X_train, X_test = np.array_split(X_all, [split_point])
    model = save_features(model, X_train, X_test)

    # Generate interactions

    X_all = create_interactions(model, X_all)
    X_train, X_test = np.array_split(X_all, [split_point])
    model = save_features(model, X_train, X_test)

    # Remove low-variance features

    X_all = remove_lv_features(model, X_all)
    X_train, X_test = np.array_split(X_all, [split_point])
    model = save_features(model, X_train, X_test)

    # Shuffle the data [if specified]
    model = shuffle_data(model)

    # Oversampling or Undersampling [if specified]

    if model_type == ModelType.classification:
        if sampling:
            model = sample_data(model)
        else:
            logger.info("Skipping Sampling")

    # Perform feature selection, independent of algorithm

    if fs_univariate:
        model = select_features_univariate(model)

    # Get the available classifiers and regressors 

    logger.info("Getting All Estimators")
    estimators = get_estimators(alphapy_specs, model)

    # Get the available scorers

    if scorer not in scorers:
        raise KeyError(f"Scorer function {scorer} not found")

    # Model Loop

    logger.info("Selecting Models")

    for algo in model.algolist:
        logger.info("Algorithm: %s", algo)
        # select estimator
        try:
            estimator = estimators[algo]
            est = estimator.estimator
        except KeyError:
            est = None
            logger.info("Algorithm %s not found", algo)
        if est is not None:
            # select LOFO features
            if fs_lofo:
                select_features_lofo(model, algo, est)
            # run classic train/test model pipeline
            model = first_fit(model, algo, est)
            # recursive feature elimination
            if rfe:
                has_coef = hasattr(est, "coef_")
                has_fimp = hasattr(est, "feature_importances_")
                if has_coef or has_fimp:
                    model = rfecv_search(model, algo)
                else:
                    logger.info("No RFE Available for %s", algo)
            # grid search
            if grid_search:
                model = hyper_grid_search(model, estimator)
            # predictions
            model = make_predictions(model, algo)
            # walk-forward time series
            if ts_option and not shuffle:
                time_series_model(model, algo)

    # Create a blended estimator

    if len(model.algolist) > 1:
        model = predict_blend(model)

    #
    # Generate metrics, get the best estimator, generate plots, and save the model.
    #

    def generate_results(model, partition):
        model = generate_metrics(model, partition)
        model = select_best_model(model, partition)
        generate_plots(alphapy_specs, model, partition)
        model = save_predictions(model, partition)

    partition = Partition.train
    generate_results(model, partition)

    partition = Partition.test
    if model.test_labels:
        generate_results(model, partition)
    else:
        model = save_predictions(model, partition)

    if ts_option:
        partition = Partition.train_ts
        generate_results(model, partition)
        partition = Partition.test_ts
        if model.test_labels:
            generate_results(model, partition)
        else:
            model = save_predictions(model, partition)

    # Save the model and metrics

    save_predictor(model, 'BEST')
    save_feature_map(model)
    save_metrics(model)

    # Return the model
    return model


#
# Function prediction_pipeline
#

def prediction_pipeline(alphapy_specs, model):
    r"""AlphaPy Prediction Pipeline

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    model : alphapy.Model
        The model object for controlling the pipeline.

    Returns
    -------
    None : None

    Notes
    -----
    The saved model is loaded from disk, and predictions are made
    on the new testing data.

    """

    logger.info("Predict Mode")

    # Unpack the model specifications

    run_dir = model.specs['run_dir']
    drop = model.specs['drop']
    fs_lofo = model.specs['fs_lofo']
    fs_univariate = model.specs['fs_univariate']
    model_type = model.specs['model_type']
    rfe = model.specs['rfe']

    # Get all data. We need original train and test for encodings.

    X_train, y_train = get_data(model, Partition.train)

    partition = Partition.test
    X_predict, _ = get_data(model, partition)

    # Load feature_map
    model = load_feature_map(model, run_dir)

    # Log feature statistics

    logger.info("Feature Statistics")
    logger.info("Number of Prediction Rows    : %d", X_predict.shape[0])
    logger.info("Number of Prediction Columns : %d", X_predict.shape[1])

    # Apply transforms to the feature matrix
    X_all = apply_transforms(model, X_predict)

    # Drop features
    X_all = drop_features(X_all, drop)

    # Create initial features
    X_all = create_features(model, X_all, X_train, X_predict, y_train)

    # Generate interactions
    X_all = create_interactions(model, X_all)

    # Remove low-variance features
    X_all = remove_lv_features(model, X_all)

    # Load the LOFO support vector, if any

    if fs_lofo:
        logger.info("Getting LOFO Support")
        try:
            support = model.feature_map['support_lofo', model.best_algo]
            X_all = X_all[:, support]
            logger.info("New Feature Count : %d", X_all.shape[1])
        except:
            logger.info("No LOFO Support")

    # Load the univariate support vector, if any

    if fs_univariate:
        logger.info("Getting Univariate Support")
        try:
            support = model.feature_map['support_uni']
            X_all = X_all[:, support]
            logger.info("New Feature Count : %d", X_all.shape[1])
        except:
            logger.info("No Univariate Support")

    # Load the RFE support vector, if any

    if rfe:
        logger.info("Getting RFE Support")
        try:
            support = model.feature_map['rfe_support']
            X_all = X_all[:, support]
            logger.info("New Feature Count : %d", X_all.shape[1])
        except:
            logger.info("No RFE Support")

    # Load predictor
    predictor = load_predictor(run_dir)

    # Make predictions

    logger.info("Making Predictions")
    tag = 'BEST'
    model.preds[(tag, partition)] = predictor.predict(X_all)
    if model_type == ModelType.classification:
        model.probas[(tag, partition)]  = predictor.predict_proba(X_all)[:, 1]

    # Save predictions
    save_predictions(model, tag, partition)

    # Return the model
    return model


#
# Function main_pipeline
#

def main_pipeline(alphapy_specs, model):
    r"""AlphaPy Main Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model specifications for the pipeline.

    Returns
    -------
    model : alphapy.Model
        The final model.

    """

    logger.info('*'*80)
    logger.info("AlphaPy Model Pipeline")
    logger.info('*'*80)

    # Extract any model specifications
    predict_mode = model.specs['predict_mode']

    # Prediction Only or Calibration

    if predict_mode:
        model = prediction_pipeline(alphapy_specs, model)
    else:
        model = training_pipeline(alphapy_specs, model)

    # Return the completed model
    return model


#
# Function main
#

def main(args=None):
    r"""AlphaPy Main Program

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the model configuration.
    (4) Create the model object.
    (5) Call the main AlphaPy pipeline.

    """

    # Argument Parsing

    parser = argparse.ArgumentParser(description="AlphaPy Parser")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--predict', dest='predict_mode', action='store_true')
    parser.add_argument('--train', dest='predict_mode', action='store_false')
    parser.set_defaults(predict_mode=False)
    parser.add_argument('--rundir', dest='run_dir',
                        help="run directory is in the format: run_YYYYMMDD_hhmmss",
                        required=False)
    args = parser.parse_args()

    # Logging

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alphapy.log", filemode='a', level=log_level,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("AlphaPy Start")
    logger.info('*'*80)

    # Read AlphaPy root directory

    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)

    # Read the AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

    # Read the model configuration file

    _, model_specs = get_model_config()
    model_specs['predict_mode'] = args.predict_mode

    # If not in prediction mode, then create the training infrastructure.

    if not model_specs['predict_mode']:
        # create the directory infrastructure if necessary
        output_dirs = ['config', 'data', 'runs']
        for od in output_dirs:
            output_dir = SSEP.join([model_specs['directory'], od])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # create the run directory
        dt_stamp = datetime_stamp()
        run_dir_name = USEP.join(['run', dt_stamp])
        run_dir = SSEP.join([model_specs['directory'], 'runs', run_dir_name])
        os.makedirs(run_dir)
        model_specs['run_dir'] = run_dir
        # create the subdirectories of the runs directory
        sub_dirs = ['config', 'input', 'model', 'output', 'plots']
        for sd in sub_dirs:
            output_dir = SSEP.join([run_dir, sd])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # copy the model file to the config directory
        filename = 'model.yml'
        src_file = SSEP.join([model_specs['directory'], 'config', filename])
        dst_file = SSEP.join([run_dir, 'config', filename])
        shutil.copyfile(src_file, dst_file)
    else:
        run_dir = args.run_dir if args.run_dir else None
        if not run_dir:
            # get latest directory
            search_dir = SSEP.join([model_specs['directory'], 'runs'])
            run_dir = most_recent_file(search_dir, 'run_*')
    model_specs['run_dir'] = run_dir

    # Create a model from the arguments

    logger.info("Creating Model")
    model = Model(model_specs)

    # Start the pipeline

    logger.info("Calling Pipeline")
    model = main_pipeline(alphapy_specs, model)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("AlphaPy End")
    logger.info('*'*80)

    # Return the model
    return model


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
