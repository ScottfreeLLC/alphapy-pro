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

import optuna
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
# Suppress sklearn numerical warnings during optimization (benign)
warnings.filterwarnings('ignore', message='.*overflow.*matmul.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*divide by zero.*matmul.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value.*matmul.*', category=RuntimeWarning)


#
# Imports
#

import argparse
import logging
import numpy as np
import os
import polars as pl
import shutil
from sklearn.model_selection import train_test_split
import sys
import yaml

from alphapy.data import get_data
from alphapy.data import shuffle_data
from alphapy.estimators import get_estimators
from alphapy.estimators import scorers
from alphapy.estimators import is_window_algo
from alphapy.sequence import append_windows
from alphapy.sequence import history_tail
from alphapy.features import apply_transforms
from alphapy.features import create_crosstabs
from alphapy.features import drop_features
from alphapy.features import FeaturePipeline
from alphapy.features import save_features
from alphapy.features import select_features_lofo
from alphapy.features import select_features_univariate
from alphapy.frame import sequence_frame
from alphapy.frame import write_frame
from alphapy.globals import SSEP, USEP
from alphapy.globals import ModelType
from alphapy.globals import Partition, datasets
from alphapy.inference import BLEND
from alphapy.inference import load_run
from alphapy.model import first_fit
from alphapy.model import generate_metrics
from alphapy.model import get_model_config
from alphapy.model import make_predictions
from alphapy.model import Model
from alphapy.model import select_best_model
from alphapy.model import predict_blend
from alphapy.model import save_feature_map
from alphapy.model import save_metrics
from alphapy.model import save_predictions
from alphapy.model import save_estimators
from alphapy.model import save_feature_pipeline
from alphapy.model import save_predictor
from alphapy.optimize import hyper_grid_search
from alphapy.optimize import rfecv_search
from alphapy.plots import generate_plots
from alphapy.utilities import datetime_stamp
from alphapy.utilities import most_recent_file


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
        specs = yaml.safe_load(ymlfile)
    specs['alphapy_root'] = alphapy_root

    logger.info('ALPHAPY PARAMETERS:')
    for spec in specs.keys():
        logger.info('%s: %s', spec, specs[spec])

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
    fs_lofo_algorithms = model.specs.get('fs_lofo_algorithms', [])
    fs_univariate = model.specs['fs_univariate']
    group_id = model.specs['rank_group_id']
    grid_search = model.specs['grid_search']
    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    separator = model.specs['separator']
    shuffle = model.specs['shuffle']
    split = model.specs['split']
    target = model.specs['target']
    ts_date = model.specs['ts_date_index']
    ts_forecast = model.specs['ts_forecast']
    ts_group_id = model.specs['ts_group_id']
    ts_leaders = model.specs['ts_leaders']
    ts_n_lags = model.specs['ts_n_lags']
    ts_option = model.specs['ts_option']

    # Get train and test data

    X_train, y_train = get_data(model, Partition.train)
    X_test, y_test = get_data(model, Partition.test)

    # If there is no test partition, then we will split the train partition

    if X_test.is_empty():
        logger.info("No Test Data Found")
        if ts_option:
            logger.info("Splitting Training Data for Time Series")
            df_train = pl.concat([X_train, y_train], how="horizontal")
            df_sorted = df_train.sort(ts_date)
            df_sorted = sequence_frame(df_sorted, target, ts_date,
                                       forecast_period=ts_forecast,
                                       n_lags=ts_n_lags,
                                       leaders=ts_leaders,
                                       group_id=ts_group_id)
            split_index = int((1.0 - split) * df_sorted.shape[0])
            split_date = df_sorted[split_index, ts_date]
            df_train = df_sorted.filter(pl.col(ts_date) <= split_date)
            y_train = df_train.select(target)
            X_train = df_train.drop(target)
            df_test = df_sorted.filter(pl.col(ts_date) > split_date)
            y_test = df_test.select(target)
            X_test = df_test.drop(target)
        else:
            logger.info("Splitting Training Data")
            # sklearn returns numpy arrays, preserve column info
            columns_X = X_train.columns
            columns_y = y_train.columns
            X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
                X_train.to_numpy(), y_train.to_numpy(),
                test_size=split, random_state=seed, shuffle=shuffle)
            X_train = pl.DataFrame(X_train_np, schema=columns_X)
            X_test = pl.DataFrame(X_test_np, schema=columns_X)
            y_train = pl.DataFrame(y_train_np, schema=columns_y)
            y_test = pl.DataFrame(y_test_np, schema=columns_y)

    # Save original train/test data

    model.df_X_train = X_train
    cv_blocks_column = model.specs.get('cv_blocks_column')
    if cv_blocks_column:
        model.cv_blocks_train = X_train[cv_blocks_column].to_numpy()
    model.df_y_train = y_train
    model.df_X_test = X_test
    model.df_y_test = y_test
    model = save_features(model, X_train, X_test, y_train, y_test)

    # Save train/test groups

    if group_id:
        train_counts = X_train.group_by(group_id).len()
        model.groups_train = train_counts["len"].to_numpy()
        del train_counts
        test_counts = X_test.group_by(group_id).len()
        model.groups_test = test_counts["len"].to_numpy()
        del test_counts

    # Determine if there are any test labels

    if not y_test.is_empty():
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
    if (model_type == ModelType.classification) and model.test_labels:
        uv, uc = np.unique(y_test, return_counts=True)
        logger.info("Unique Testing Values for %s : %s", target, uv)
        logger.info("Unique Testing Counts for %s : %s", target, uc)

    # Merge training and test data

    if X_train.shape[1] == X_test.shape[1]:
        split_point = X_train.shape[0]
        # Align test schema to train schema to avoid type mismatches
        # This handles cases where Polars infers different types for small test sets
        train_schema = X_train.schema
        X_test_aligned = X_test.cast({col: dtype for col, dtype in train_schema.items()})
        X_all = pl.concat([X_train, X_test_aligned])
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
    df_train = X_all[:split_point].clone()
    df_train = pl.concat([df_train, y_train], how="horizontal")
    write_frame(df_train, data_dir, model.train_file, extension, separator, index=False)
    # test data
    df_test = X_all[split_point:]
    if model.test_labels:
        df_test = pl.concat([df_test, y_test], how="horizontal")
    write_frame(df_test, data_dir, model.test_file, extension, separator, index=False)

    # Create crosstabs for any categorical features

    if model_type == ModelType.classification:
        create_crosstabs(model, target)

    # Create features, interactions and the low-variance mask. Every step that learns from data is fitted on the
    # training rows only; the test rows go through the fitted steps exactly as rows scored later will.

    pipeline = FeaturePipeline(model.specs, model.df_X_train.schema)
    X_train = pipeline.fit_transform(model, X_all[:split_point], y_train)
    X_test = pipeline.transform(X_all[split_point:])
    X_all = np.vstack([X_train, X_test])
    model.feature_pipeline = pipeline
    model = save_features(model, X_train, X_test)

    # Append the raw window block [if the sequence section is on]; tabular algorithms never see it

    if model.specs.get('sequence'):
        X_all = append_windows(model, X_all)
        X_train, X_test = np.array_split(X_all, [split_point])
        model = save_features(model, X_train, X_test)
        pipeline.history = history_tail(model.df_X_train.to_pandas(), model.specs['sequence'])

    # Shuffle the data [if specified]
    model = shuffle_data(model)

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

    failed_algos = []
    # Tabular algorithms first: window algorithms load torch, after which LightGBM's multithreaded cross-validation
    # can segfault (two OpenMP runtimes).
    for algo in sorted(model.algolist, key=is_window_algo):
        logger.info("Algorithm: %s", algo)
        # select estimator
        try:
            estimator = estimators[algo]
            est = estimator.estimator
        except KeyError:
            est = None
            logger.info("Algorithm %s not found", algo)
            failed_algos.append(algo)
        if est is not None and is_window_algo(algo):
            # A window algorithm that cannot run (its features over max_gb, missing channels) is skipped, not fatal
            try:
                model = first_fit(model, algo, est)
                model = make_predictions(model, algo)
            except (MemoryError, ValueError, KeyError, RuntimeError) as e:
                logger.error("Algorithm %s failed and is skipped: %s", algo, e)
                failed_algos.append(algo)
                for store in (model.estimators, model.uncalibrated, model.fnames_algo):
                    store.pop(algo, None)
            continue
        if est is not None:
            # select LOFO features
            if fs_lofo and (not fs_lofo_algorithms or algo in fs_lofo_algorithms):
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

    # Remove algorithms that failed to load or run, and say so plainly
    for algo in failed_algos:
        model.algolist.remove(algo)
        logger.info("Removed unavailable algorithm %s from model", algo)
    if failed_algos:
        logger.warning("Algorithms not in this run's results: %s", ", ".join(failed_algos))
    if not model.algolist:
        raise RuntimeError(f"No algorithm could run (requested: {', '.join(failed_algos)}); see the log for each")

    # Create a blended estimator

    if len(model.algolist) > 1:
        model = predict_blend(model)

    #
    # Generate metrics, get the best estimator, generate plots, and save the model.
    #

    def generate_results(model, partition):
        model, metrics_generated = generate_metrics(model, partition)
        if metrics_generated:
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

    # Save the model and metrics

    save_predictor(model, 'BEST')
    save_estimators(model)
    save_feature_pipeline(model)
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
    model : alphapy.Model
        The model object with the predictions.

    Notes
    -----
    The trained run in ``run_dir`` scores the test partition's rows through ``alphapy.inference``: every algorithm
    and the blend, with the feature steps as fitted on the run's training rows. Window algorithms read each group's
    last training rows as history, as training did. The predictions are written to ``output/ranked_predict`` in the
    format of training's ``ranked_test`` (the best algorithm's columns are ``best``); training's own files are kept.

    """

    logger.info("Predict Mode")

    # Unpack the model specifications

    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    separator = model.specs['separator']

    # Load the trained run and the rows to score

    run = load_run(run_dir)
    partition = Partition.test
    X_predict, y_predict = get_data(model, partition)
    if X_predict.is_empty():
        raise ValueError("Predict mode found no rows in the test partition")
    logger.info("Number of Prediction Rows    : %d", X_predict.shape[0])
    logger.info("Number of Prediction Columns : %d", X_predict.shape[1])

    # Score every algorithm and the blend

    logger.info("Making Predictions")
    preds, probas = run.score(X_predict, history=run.training_tail)

    # Write the ranked predictions in training's format

    best = run.manifest['best_algorithm']
    best = BLEND if best == 'BLEND' else best
    tags = ['BEST'] + ([BLEND] if run.blend else []) + run.algorithms
    df_master = pl.concat([X_predict, y_predict], how="horizontal") if not y_predict.is_empty() else X_predict
    label = datasets[partition]
    for tag in tags:
        column = best if tag == 'BEST' else tag
        df_master = df_master.with_columns(pl.Series(USEP.join(['pred', label, tag.lower()]),
                                                     preds[column].to_numpy()))
        if model_type == ModelType.classification:
            df_master = df_master.with_columns(pl.Series(USEP.join(['prob', label, tag.lower()]),
                                                         probas[column].to_numpy()))
    kind = 'prob' if model_type == ModelType.classification else 'pred'
    df_master = df_master.sort(USEP.join([kind, label, 'best']), descending=True)
    write_frame(df_master, SSEP.join([run_dir, 'output']), 'ranked_predict', extension, separator)

    # Return the model
    model.preds = {(tag, partition): preds[best if tag == 'BEST' else tag].to_numpy() for tag in tags}
    if model_type == ModelType.classification:
        model.probas = {(tag, partition): probas[best if tag == 'BEST' else tag].to_numpy() for tag in tags}
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

    # Fixing mutually exclusive group setup
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--predict', dest='predict_mode', action='store_true', help="Enable predict mode")
    mode_group.add_argument('--train', dest='predict_mode', action='store_false', help="Enable train mode (default)")
    parser.set_defaults(predict_mode=False)

    parser.add_argument('--rundir', dest='run_dir',
                        help="Run directory is in the format: run_YYYYMMDD_hhmmss",
                        required=False)
    args = parser.parse_args()

    # Logging Configuration

    # Set log level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(filename="alphapy.log", filemode='a', level=log_level,
                        format="[%(asctime)s] %(levelname)s\t%(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S')

    # Configure console handler to output to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)  # Adjust console log level based on argument
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                                datefmt='%Y/%m/%d %H:%M:%S'))

    # Add the handler to the root logger
    logging.getLogger().addHandler(console_handler)

    logger = logging.getLogger(__name__)

    # Example logging messages based on command line options
    logger.debug("Debug mode is on.")
    if args.predict_mode:
        logger.info("Running in predict mode.")
    else:
        logger.info("Running in train mode.")
    if args.run_dir:
        logger.info(f"Using run directory: {args.run_dir}")

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
        elif not os.path.isdir(run_dir):
            # a run name such as run_YYYYMMDD_hhmmss, under the project's runs directory
            run_dir = SSEP.join([model_specs['directory'], 'runs', run_dir])
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


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
