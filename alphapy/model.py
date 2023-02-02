################################################################################
#
# Package   : AlphaPy
# Module    : model
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

from alphapy.estimators import scorers
from alphapy.estimators import xgb_score_map
from alphapy.features import feature_scorers
from alphapy.frame import write_frame
from alphapy.globals import Encoders
from alphapy.globals import ModelType
from alphapy.globals import Objective
from alphapy.globals import Partition, datasets
from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import SamplingMethod
from alphapy.globals import Scalers
from alphapy.utilities import most_recent_file

from copy import copy
from datetime import datetime
import joblib

import logging
import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import coverage_error
from sklearn.metrics import dcg_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Model
#
# model unifies algorithms and we use hasattr to list the available attrs for each
# algorithm so users can query an algorithm and get the list of attributes
#

class Model:
    """Create a new model.

    Parameters
    ----------
    specs : dict
        The model specifications obtained by reading the ``model.yml``
        file.

    Attributes
    ----------
    specs : dict
        The model specifications.
    df_X_train : pandas.DataFrame
        Original train features.
    df_y_train : pandas.Series
        Original train target.
    df_X_test  : pandas.DataFrame
        Original test features.
    df_y_test  : pandas.Series
        Original test target.
    X_train : pandas.DataFrame
        Selected train features in matrix format.
    y_train : pandas.Series
        Training labels in vector format.
    groups_train : numpy.array
        Group sizes for the training data.
    X_test  : pandas.DataFrame
        Selected test features in matrix format.
    y_test  : pandas.Series
        Test labels in vector format.
    groups_test : numpy.array
        Groups sizes for the test data.
    X_train_ts  : pandas.DataFrame
        Time series training frame.
    y_train_ts  : pandas.Series
        Time series target.
    algolist : list
        Algorithms to use in training.
    estimators : dict
        Dictionary of estimators (key: algorithm)
    importances : dict
        Feature Importances (key: algorithm)
    coefs : dict
        Coefficients, if applicable (key: algorithm)
    support : dict
        Support Vectors, if applicable (key: algorithm)
    preds : dict
        Predictions or labels (keys: algorithm, partition)
    probas : dict
        Probabilities from classification (keys: algorithm, partition)
    metrics : dict
        Model evaluation metrics (keys: algorith, partition, metric)

    Raises
    ------
    KeyError
        Model specs must include the key *algorithms*, which is
        stored in ``algolist``.

    """

    # __init__

    def __init__(self,
                 specs):
        # specifications
        self.specs = specs
        # data in memory
        self.df_X_train = None
        self.df_y_train = None
        self.df_X_test = None
        self.df_y_test = None
        # transformed train/test
        self.X_train = None
        self.y_train = None
        self.groups_train = None
        self.X_test = None
        self.y_test = None
        self.groups_test = None
        # time series train/test
        self.X_train_ts = None
        self.y_train_ts = None
        # test labels
        self.test_labels = False
        # datasets
        self.train_file = datasets[Partition.train]
        self.test_file = datasets[Partition.test]
        # algorithms
        try:
            self.algolist = self.specs['algorithms']
        except:
            raise KeyError("Model specs must include the key: algorithms")
        self.best_algo = None
        # feature names
        self.feature_names = []
        # feature map
        self.feature_map = {}
        # Key: (algorithm)
        self.estimators = {}
        self.importances = {}
        self.coefs = {}
        self.support = {}
        self.fnames_algo = {}
        self.lofo_df = {}
        # Keys: (algorithm, partition)
        self.preds = {}
        self.probas = {}
        # Keys: (algorithm, partition, metric)
        self.metrics = {}

    # __str__

    def __str__(self):
        return self.name

    # __getnewargs__

    def __getnewargs__(self):
        return (self.specs,)


#
# Function get_model_config
#

def get_model_config(directory='.'):
    r"""Read in the configuration file for AlphaPy.

    Parameters
    ----------
    directory : str
        The directory specifying the location of the configuration file.

    Returns
    -------
    cfg : dict
        The original configuration specifications.
    specs : dict
        The parameters for controlling AlphaPy.

    Raises
    ------
    ValueError
        Unrecognized value of a ``model.yml`` field.

    """

    logger.info("Model Configuration")

    # Read the configuration file

    full_path = SSEP.join([directory, 'config', 'model.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: project [this section must be first]

    project_section = cfg['project']
    specs['directory'] = project_section['directory']
    specs['extension'] = project_section['file_extension']
    specs['live_results'] = project_section['live_results']
    specs['submission_file'] = project_section['submission_file']
    specs['submit_probas'] = project_section['submit_probas']

    # Section: data

    data_section = cfg['data']
    specs['drop'] = data_section['drop']
    specs['features'] = data_section['features']
    specs['sentinel'] = data_section['sentinel']
    specs['separator'] = data_section['separator']
    specs['shuffle'] = data_section['shuffle']
    specs['split'] = data_section['split']
    specs['allow_na_targets'] = data_section['allow_na_targets']
    # sampling
    specs['sampling'] = data_section['sampling']['option']
    # determine whether or not sampling method is valid
    samplers = {x.name: x.value for x in SamplingMethod}
    sampling_method = data_section['sampling']['method']
    if sampling_method in samplers:
        specs['sampling_method'] = SamplingMethod(samplers[sampling_method])
    else:
        raise ValueError("model.yml data:sampling:method %s unrecognized" %
                         sampling_method)
    # sampling ratio
    specs['sampling_ratio'] = data_section['sampling']['ratio']

    # Section: features

    # clustering
    features_section = cfg['features']
    specs['clustering'] = features_section['clustering']['option']
    specs['cluster_min'] = features_section['clustering']['minimum']
    specs['cluster_max'] = features_section['clustering']['maximum']
    specs['cluster_inc'] = features_section['clustering']['increment']
    # counts
    specs['counts'] = features_section['counts']['option']
    # encoding
    specs['rounding'] = features_section['encoding']['rounding']
    # determine whether or not encoder is valid
    encoders = {x.name: x.value for x in Encoders}
    encoder = features_section['encoding']['type']
    if encoder in encoders:
        specs['encoder'] = Encoders(encoders[encoder])
    else:
        raise ValueError("model.yml features:encoding:type %s unrecognized" % encoder)
    # factors
    specs['factors'] = features_section['factors']
    # interactions
    specs['interactions'] = features_section['interactions']['option']
    specs['isample_pct'] = features_section['interactions']['sampling_pct']
    specs['poly_degree'] = features_section['interactions']['poly_degree']
    # isomap
    specs['isomap'] = features_section['isomap']['option']
    specs['iso_components'] = features_section['isomap']['components']
    specs['iso_neighbors'] = features_section['isomap']['neighbors']
    # LOFO
    specs['fs_lofo'] = features_section['lofo']['option']
    # log transformation
    specs['log_transform'] = features_section['log_transform']['option']
    specs['pvalue_level'] = features_section['log_transform']['pvalue_level']
    # low-variance features
    specs['lv_remove'] = features_section['variance']['option']
    specs['lv_threshold'] = features_section['variance']['threshold']
    # NumPy
    specs['numpy'] = features_section['numpy']['option']
    # pca
    specs['pca'] = features_section['pca']['option']
    specs['pca_min'] = features_section['pca']['minimum']
    specs['pca_max'] = features_section['pca']['maximum']
    specs['pca_inc'] = features_section['pca']['increment']
    specs['pca_whiten'] = features_section['pca']['whiten']
    # Scaling
    specs['scaler_option'] = features_section['scaling']['option']
    # determine whether or not scaling type is valid
    scaler_types = {x.name: x.value for x in Scalers}
    scaler_type = features_section['scaling']['type']
    if scaler_type in scaler_types:
        specs['scaler_type'] = Scalers(scaler_types[scaler_type])
    else:
        raise ValueError("model.yml features:scaling:type %s unrecognized" % scaler_type)
    # SciPy
    specs['scipy'] = features_section['scipy']['option']
    # text
    specs['ngrams_max'] = features_section['text']['ngrams']
    specs['vectorize'] = features_section['text']['vectorize']
    # t-sne
    specs['tsne'] = features_section['tsne']['option']
    specs['tsne_components'] = features_section['tsne']['components']
    specs['tsne_learn_rate'] = features_section['tsne']['learning_rate']
    specs['tsne_perplexity'] = features_section['tsne']['perplexity']
    # univariate
    specs['fs_univariate'] = features_section['univariate']['option']
    specs['fs_uni_pct'] = features_section['univariate']['percentage']
    specs['fs_uni_grid'] = features_section['univariate']['uni_grid']
    score_func = features_section['univariate']['score_func']
    if score_func in feature_scorers:
        specs['fs_uni_score_func'] = feature_scorers[score_func]
    else:
        raise ValueError("model.yml features:univariate:score_func %s unrecognized" %
                         score_func)

    # Section: model

    model_section = cfg['model']
    specs['algorithms'] = model_section['algorithms']
    specs['cv_folds'] = model_section['cv_folds']
    # determine whether or not model type is valid
    model_types = {x.name: x.value for x in ModelType}
    model_type = model_section['type']
    if model_type in model_types:
        specs['model_type'] = ModelType(model_types[model_type])
    else:
        raise ValueError("model.yml model:type %s unrecognized" % model_type)
    # end of model type
    specs['n_estimators'] = model_section['estimators']
    specs['scorer'] = model_section['scoring_function']
    # calibration
    specs['calibration'] = model_section['calibration']['option']
    specs['cal_type'] = model_section['calibration']['type']
    # grid search
    specs['grid_search'] = model_section['grid_search']['option']
    specs['gs_iters'] = model_section['grid_search']['iterations']
    specs['gs_random'] = model_section['grid_search']['random']
    specs['gs_sample'] = model_section['grid_search']['subsample']
    specs['gs_sample_pct'] = model_section['grid_search']['sampling_pct']
    # ranking
    specs['rank_group_id'] = model_section['ranking']['group_id']
    specs['rank_group_size'] = model_section['ranking']['group_size']
    # rfe
    specs['rfe'] = model_section['rfe']['option']
    specs['rfe_step'] = model_section['rfe']['step']
    # target
    specs['target'] = model_section['target']
    # time series
    specs['ts_option'] = model_section['time_series']['option']
    specs['ts_backtests'] = model_section['time_series']['backtests']
    specs['ts_date_index'] = model_section['time_series']['date_index']
    # forecast window
    specs['ts_forecast'] = model_section['time_series']['forecast']
    # derivation (rolling) window
    specs['ts_window'] = model_section['time_series']['window']

    if specs['ts_option'] and specs['shuffle']:
        logger.info("Time Series is enabled, turning off Shuffling")
        specs['shuffle'] = False

    # Section: pipeline

    pipeline_section = cfg['pipeline']
    specs['n_jobs'] = pipeline_section['number_jobs']
    specs['seed'] = pipeline_section['seed']
    specs['verbosity'] = pipeline_section['verbosity']

    # Section: plots

    plots_section = cfg['plots']
    specs['calibration_plot'] = plots_section['calibration']
    specs['confusion_matrix'] = plots_section['confusion_matrix']
    specs['importances'] = plots_section['importances']
    specs['learning_curve'] = plots_section['learning_curve']
    specs['roc_curve'] = plots_section['roc_curve']

    # Section: transforms

    try:
        specs['transforms'] = cfg['transforms']
    except:
        specs['transforms'] = None
        logger.info("No transforms Found")

    # Section: xgboost

    specs['esr'] = cfg['xgboost']['stopping_rounds']

    # Log the configuration parameters

    logger.info('MODEL PARAMETERS:')
    logger.info('algorithms        = %s', specs['algorithms'])
    logger.info('allow_na_targets  = %r', specs['allow_na_targets'])
    logger.info('calibration       = %r', specs['calibration'])
    logger.info('cal_type          = %s', specs['cal_type'])
    logger.info('calibration_plot  = %r', specs['calibration'])
    logger.info('clustering        = %r', specs['clustering'])
    logger.info('cluster_inc       = %d', specs['cluster_inc'])
    logger.info('cluster_max       = %d', specs['cluster_max'])
    logger.info('cluster_min       = %d', specs['cluster_min'])
    logger.info('confusion_matrix  = %r', specs['confusion_matrix'])
    logger.info('counts            = %r', specs['counts'])
    logger.info('cv_folds          = %d', specs['cv_folds'])
    logger.info('directory         = %s', specs['directory'])
    logger.info('extension         = %s', specs['extension'])
    logger.info('drop              = %s', specs['drop'])
    logger.info('encoder           = %r', specs['encoder'])
    logger.info('esr               = %d', specs['esr'])
    logger.info('factors           = %s', specs['factors'])
    logger.info('features [X]      = %s', specs['features'])
    logger.info('fs_lofo           = %r', specs['fs_lofo'])
    logger.info('fs_univariate     = %r', specs['fs_univariate'])
    logger.info('fs_uni_grid       = %s', specs['fs_uni_grid'])
    logger.info('fs_uni_pct        = %d', specs['fs_uni_pct'])
    logger.info('fs_uni_score_func = %s', specs['fs_uni_score_func'])
    logger.info('grid_search       = %r', specs['grid_search'])
    logger.info('gs_iters          = %d', specs['gs_iters'])
    logger.info('gs_random         = %r', specs['gs_random'])
    logger.info('gs_sample         = %r', specs['gs_sample'])
    logger.info('gs_sample_pct     = %f', specs['gs_sample_pct'])
    logger.info('importances       = %r', specs['importances'])
    logger.info('interactions      = %r', specs['interactions'])
    logger.info('isomap            = %r', specs['isomap'])
    logger.info('iso_components    = %d', specs['iso_components'])
    logger.info('iso_neighbors     = %d', specs['iso_neighbors'])
    logger.info('isample_pct       = %d', specs['isample_pct'])
    logger.info('learning_curve    = %r', specs['learning_curve'])
    logger.info('live_results      = %r', specs['live_results'])
    logger.info('log_transform     = %r', specs['log_transform'])
    logger.info('lv_remove         = %r', specs['lv_remove'])
    logger.info('lv_threshold      = %f', specs['lv_threshold'])
    logger.info('model_type        = %r', specs['model_type'])
    logger.info('n_estimators      = %d', specs['n_estimators'])
    logger.info('n_jobs            = %d', specs['n_jobs'])
    logger.info('ngrams_max        = %d', specs['ngrams_max'])
    logger.info('numpy             = %r', specs['numpy'])
    logger.info('pca               = %r', specs['pca'])
    logger.info('pca_inc           = %d', specs['pca_inc'])
    logger.info('pca_max           = %d', specs['pca_max'])
    logger.info('pca_min           = %d', specs['pca_min'])
    logger.info('pca_whiten        = %r', specs['pca_whiten'])
    logger.info('poly_degree       = %d', specs['poly_degree'])
    logger.info('pvalue_level      = %f', specs['pvalue_level'])
    logger.info('rank_group_id     = %s', specs['rank_group_id'])
    logger.info('rank_group_size   = %s', specs['rank_group_size'])
    logger.info('rfe               = %r', specs['rfe'])
    logger.info('rfe_step          = %d', specs['rfe_step'])
    logger.info('roc_curve         = %r', specs['roc_curve'])
    logger.info('rounding          = %d', specs['rounding'])
    logger.info('sampling          = %r', specs['sampling'])
    logger.info('sampling_method   = %r', specs['sampling_method'])
    logger.info('sampling_ratio    = %f', specs['sampling_ratio'])
    logger.info('scaler_option     = %r', specs['scaler_option'])
    logger.info('scaler_type       = %r', specs['scaler_type'])
    logger.info('scipy             = %r', specs['scipy'])
    logger.info('scorer            = %s', specs['scorer'])
    logger.info('seed              = %d', specs['seed'])
    logger.info('sentinel          = %d', specs['sentinel'])
    logger.info('separator         = %s', specs['separator'])
    logger.info('shuffle           = %r', specs['shuffle'])
    logger.info('split             = %f', specs['split'])
    logger.info('submission_file   = %s', specs['submission_file'])
    logger.info('submit_probas     = %r', specs['submit_probas'])
    logger.info('target [y]        = %s', specs['target'])
    logger.info('transforms        = %s', specs['transforms'])
    logger.info('ts_option         = %r', specs['ts_option'])
    logger.info('ts_backtests      = %d', specs['ts_backtests'])
    logger.info('ts_date_index     = %s', specs['ts_date_index'])
    logger.info('ts_forecast       = %d', specs['ts_forecast'])
    logger.info('ts_window         = %d', specs['ts_window'])
    logger.info('tsne              = %r', specs['tsne'])
    logger.info('tsne_components   = %d', specs['tsne_components'])
    logger.info('tsne_learn_rate   = %f', specs['tsne_learn_rate'])
    logger.info('tsne_perplexity   = %f', specs['tsne_perplexity'])
    logger.info('vectorize         = %r', specs['vectorize'])
    logger.info('verbosity         = %d', specs['verbosity'])

    # Specifications to create the model
    return cfg, specs


#
# Function load_predictor
#

def load_predictor(directory):
    r"""Load the model predictor from storage. By default, the
    most recent model is loaded into memory.

    Parameters
    ----------
    directory : str
        Full directory specification of the predictor's location.

    Returns
    -------
    predictor : function
        The scoring function.

    """

    # Locate the model Pickle or HD5 file

    search_dir = SSEP.join([directory, 'model'])
    file_name = most_recent_file(search_dir, 'model_*.*')

    # Load the model from the file

    file_ext = file_name.split(PSEP)[-1]
    if file_ext == 'pkl' or file_ext == 'h5':
        logger.info("Loading model predictor from %s", file_name)
        # load the model predictor
        if file_ext == 'pkl':
            predictor = joblib.load(file_name)
        elif file_ext == 'h5':
            predictor = load_model(file_name)
    else:
        logging.error("Could not find model predictor in %s", search_path)

    # Return the model predictor
    return predictor


#
# Function save_predictor
#

def save_predictor(model, tag):
    r"""Save the model predictor to the run directory.

    Parameters
    ----------
    model : alphapy.Model
        The model object that contains the best estimator.
    tag : str
        A unique identifier for a model algorithm.

    Returns
    -------
    None : None

    """

    logger.info("Saving Model Predictor")

    # Extract model parameters.
    run_dir = model.specs['run_dir']

    # Get the best predictor
    predictor = model.estimators[tag]

    # Save model object

    if 'KERAS' in model.best_algo:
        filename = PSEP.join(['model', 'h5'])
        full_path = SSEP.join([run_dir, 'model', filename])
        logger.info("Writing model predictor to %s", full_path)
        predictor.model.save(full_path)
    else:
        filename = PSEP.join(['model', 'pkl'])
        full_path = SSEP.join([run_dir, 'model', filename])
        logger.info("Writing model predictor to %s", full_path)
        joblib.dump(predictor, full_path)


#
# Function load_feature_map
#

def load_feature_map(model, directory):
    r"""Load the feature map from storage. By default, the
    most recent feature map is loaded into memory.

    Parameters
    ----------
    model : alphapy.Model
        The model object to contain the feature map.
    directory : str
        Full directory specification of the feature map's location.

    Returns
    -------
    model : alphapy.Model
        The model object containing the feature map.

    """

    # Locate the feature map and load it

    try:
        search_dir = SSEP.join([directory, 'model'])
        file_name = most_recent_file(search_dir, 'feature_map_*.pkl')
        logger.info("Loading feature map from %s", file_name)
        # load the feature map
        feature_map = joblib.load(file_name)
        model.feature_map = feature_map
    except:
        logging.error("Could not find feature map in %s", search_dir)

    # Return the model with the feature map
    return model


#
# Function save_feature_map
#

def save_feature_map(model):
    r"""Save the feature map to disk.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing the feature map.

    Returns
    -------
    None : None

    """

    logger.info("Saving Feature Map")

    # Extract model parameters.
    run_dir = model.specs['run_dir']

    # Create full path name.

    filename = PSEP.join(['feature_map', 'pkl'])
    full_path = SSEP.join([run_dir, 'model', filename])

    # Save model object

    logger.info("Writing feature map to %s", full_path)
    joblib.dump(model.feature_map, full_path)


#
# Function first_fit
#

def first_fit(model, algo, est):
    r"""Fit the model before optimization.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    algo : str
        Abbreviation of the algorithm to run.
    est : alphapy.Estimator
        The estimator to fit.

    Returns
    -------
    model : alphapy.Model
        The model object with the initial estimator.

    Notes
    -----
    AlphaPy fits an initial model because the user may choose to get
    a first score without any additional feature selection or grid
    search. XGBoost is a special case because it has the advantage
    of an ``eval_set`` and ``early_stopping_rounds``, which can
    speed up the estimation phase.

    """

    logger.info("Fitting Initial Model")

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    esr = model.specs['esr']
    model_type = model.specs['model_type']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    shuffle = model.specs['shuffle']
    split = model.specs['split']
    ts_option = model.specs['ts_option']
    verbosity = model.specs['verbosity']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train
    groups_train = model.groups_train

    # Fit the initial model.

    algo_xgb = 'XGB' in algo

    if model_type == ModelType.letor:
        est.fit(X_train, y_train, group=groups_train)
    elif algo_xgb and scorer in xgb_score_map:
        if ts_option:
            shuffle_flag = False
        else:
            shuffle_flag = shuffle
        X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=split,
                                          random_state=seed, shuffle=shuffle_flag)
        eval_set = [(X1, y1), (X2, y2)]
        eval_metric = xgb_score_map[scorer]
        est.fit(X1, y1.values.ravel(), eval_set=eval_set, eval_metric=eval_metric,
                early_stopping_rounds=esr)
    else:
        est.fit(X_train, y_train.values.ravel())

    # Get the initial scores

    if model_type != ModelType.letor:
        logger.info("Cross-Validation")
        scores = cross_val_score(est, X_train, y_train.values.ravel(), scoring=scorer,
                                 cv=cv_folds, n_jobs=n_jobs, verbose=verbosity)
        logger.info("Cross-Validation Scores: %s", scores)

    # Store the estimator
    model.estimators[algo] = est

    # Copy feature name master into feature names per algorithm
    model.fnames_algo[algo] = model.feature_names

    # Record importances and coefficients if necessary.

    if hasattr(est, "feature_importances_"):
        model.importances[algo] = est.feature_importances_

    if hasattr(est, "coef_"):
        model.coefs[algo] = est.coef_

    # Save the estimator in the model and return the model
    return model


#
# Function time_series_model
#

def time_series_model(model, algo):
    r"""Train a model using a walk-backward time series technique.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    algo : str
        Abbreviation of the algorithm to run.

    Returns
    -------
    model : alphapy.Model
        The model object with the predictions.

    Notes
    -----
    We use a training window (derivation window) and a test window
    (forecast window) to make predictions incrementally. This technique
    is in contrast to the traditional train/test split, where the model
    does not predict using the most recent data.

    """

    logger.info("Walk-Backward Time Series Model")

    # Extract model parameters.

    model_type = model.specs['model_type']
    target = model.specs['target']
    ts_backtests = model.specs['ts_backtests']
    ts_date_index = model.specs['ts_date_index']
    ts_forecast = model.specs['ts_forecast']
    ts_window = model.specs['ts_window']

    # Extract model data.

    if model.test_labels:
        df = pd.concat([model.df_X_test[ts_date_index], pd.DataFrame(model.X_test), model.y_test], axis=1)
    else:
        df = pd.concat([model.df_X_train[ts_date_index], pd.DataFrame(model.X_train), model.y_train], axis=1)
    est = model.estimators[algo]

    # Sort train and test by ascending date

    df.sort_values(by=[ts_date_index], inplace=True)
    df_y = df[[ts_date_index, target]]
    df_X = df.drop(columns=[target])

    # Walk forward through the training set, incrementally adding predictions

    dates_ts = df_X[ts_date_index]
    _, date_index = np.unique(dates_ts, return_index=True)
    
    if len(date_index) < ts_window:
        raise ValueError("Window Length Exceeded. Maximum Window is: %d", len(date_index))

    first_date = dates_ts.iloc[date_index[0]]
    last_date = dates_ts.iloc[date_index[-1]]
    test_date = last_date
    train2_date = dates_ts.iloc[date_index[-ts_forecast - 1]]
    train1_date = dates_ts.iloc[date_index[-ts_forecast - ts_window]]

    all_indices = []
    all_preds = []
    all_probas = []

    niters = 1
    walk_backward = True

    while walk_backward and niters <= ts_backtests:
        logger.info("%d: Train: [%s, %s], Test: [%s]",
                    niters, train1_date, train2_date, test_date)
        # define train and prediction datasets
        df_X_sub = df_X[(df_X[ts_date_index] >= train1_date) & (df_X[ts_date_index] <= train2_date)]
        df_y_sub = df_y[(df_y[ts_date_index] >= train1_date) & (df_y[ts_date_index] <= train2_date)]
        # fit the model
        est.fit(df_X_sub.drop(columns=[ts_date_index]), df_y_sub[target])
        # make walk-forward predictions
        df_pred_X = df_X[df_X[ts_date_index] == test_date]
        df_pred_y = df_y[df_y[ts_date_index] == test_date]
        preds = est.predict(df_pred_X.drop(columns=[ts_date_index]))
        if model_type == ModelType.classification:
            probas = est.predict_proba(df_pred_X.drop(columns=[ts_date_index]))[:, 1]
        # save actuals and predicted
        all_indices.extend(df_pred_y.index)
        all_preds.extend(preds)
        if model_type == ModelType.classification:
            all_probas.extend(probas)
        if train1_date > first_date:
            # next iteration
            next_index = -niters - ts_forecast
            test_date = dates_ts.iloc[date_index[-niters - 1]]
            train2_date = dates_ts.iloc[date_index[next_index - 1]]
            train1_date = dates_ts.iloc[date_index[next_index - ts_window]]
            niters += 1
        else:
            walk_backward = False

    # Store the time series dataframes and training predictions

    model.X_train_ts = model.df_X_train.loc[all_indices]
    model.y_train_ts = model.df_y_train.loc[all_indices]

    model.preds[(algo, Partition.train_ts)] = all_preds
    if model_type == ModelType.classification:
        model.probas[(algo, Partition.train_ts)] = all_probas

    # Fit on the most recent train data and make test predictions

    logger.info("Time Series Test Predictions")
    train_date = dates_ts.iloc[date_index[-ts_window]]
    df_X_sub = df_X[df_X[ts_date_index] >= train_date].drop(columns=[ts_date_index]).values
    df_y_sub = df_y[df_y[ts_date_index] >= train_date][target].values

    est = model.estimators[algo]
    est.fit(df_X_sub, df_y_sub)

    partition = Partition.test_ts
    model.preds[(algo, partition)] = est.predict(model.X_test)
    if model_type == ModelType.classification:
        model.probas[(algo, partition)] = est.predict_proba(model.X_test)[:, 1]

    # Return the model
    return model


#
# Function make_predictions
#

def make_predictions(model, algo):
    r"""Make predictions for the training and testing data.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    algo : str
        Abbreviation of the algorithm to make predictions.

    Returns
    -------
    model : alphapy.Model
        The model object with the predictions.

    Notes
    -----
    For classification, calibration is a precursor to making the
    actual predictions. In this case, AlphaPy predicts both labels
    and probabilities. For regression, real values are predicted.

    """

    logger.info("Final Model Predictions for %s", algo)

    # Extract model parameters.

    calibrate = model.specs['calibration']
    cal_type = model.specs['cal_type']
    cv_folds = model.specs['cv_folds']
    model_type = model.specs['model_type']

    # Get the estimator

    est = model.estimators[algo]

    # Extract model data

    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
        X_test = model.X_test[:, support]
    except:
        X_train = model.X_train
        X_test = model.X_test

    y_train = model.y_train
    groups_train = model.groups_train
    groups_test = model.groups_test

    # Calibration

    if model_type == ModelType.classification:
        if calibrate:
            logger.info("Calibrating Classifier")
            est = CalibratedClassifierCV(est, cv=cv_folds, method=cal_type)
            est.fit(X_train, y_train.values.ravel())
            model.estimators[algo] = est
            logger.info("Calibration Complete")
        else:
            logger.info("Skipping Calibration")

    # Make predictions on original training and test data.

    logger.info("Making Predictions")
    model.preds[(algo, Partition.train)] = est.predict(X_train)
    model.preds[(algo, Partition.test)] = est.predict(X_test)
    if model_type == ModelType.classification:
        model.probas[(algo, Partition.train)] = est.predict_proba(X_train)[:, 1]
        model.probas[(algo, Partition.test)] = est.predict_proba(X_test)[:, 1]
    logger.info("Predictions Complete")

    # Return the model
    return model


#
# Function predict_blend
#

def predict_blend(model):
    r"""Make blended predictions.

    Parameters
    ----------
    model : alphapy.Model
        The model object.

    Returns
    -------
    model : alphapy.Model
        The model object with the blended predictions.

    Notes
    -----
    Currently, we simply average the predictions. Previously, for classification,
    AlphaPy usesd logistic regression for creating a blended model. For regression,
    ridge regression was applied.

    """

    logger.info('='*80)
    logger.info("Blended Predictions")

    # Extract model parameters.
    model_type = model.specs['model_type']

    # Set the tag.
    blend_tag = 'BLEND'

    # Iterate through the partitions, averaging predictions for each one.

    start_time = datetime.now()
    logger.info("Blending Start: %s", start_time)

    for partition in datasets.keys():
        pred_set = [model.preds[key] for key, _ in model.preds.items() if partition in key]
        model.preds[(blend_tag, partition)] = np.round(np.mean(pred_set, axis=0), 0).astype(int)
        if model_type == ModelType.classification:
            proba_set = [model.probas[key] for key, _ in model.probas.items() if partition in key]
            model.probas[(blend_tag, partition)] = np.mean(proba_set, axis=0)
            model.preds[(blend_tag, partition)] = np.round(model.probas[(blend_tag, partition)], 0).astype(int)

    # Return the model with blended predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Blending Complete: %s", time_taken)

    return model


#
# Function select_best_model
#

def select_best_model(model, partition):
    r"""Select the best model based on score.

    Parameters
    ----------
    model : alphapy.Model
        The model object with all of the estimators.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    model : alphapy.Model
        The model object with the best estimator.

    Notes
    -----
    Best model selection is based on a scoring function. If the
    objective is to minimize (e.g., negative log loss), then we
    select the model with the algorithm that has the lowest score.
    If the objective is to maximize, then we select the algorithm
    with the highest score (e.g., AUC).
    For multiple algorithms, AlphaPy always creates a blended model.
    Therefore, the best algorithm that is selected could actually
    be the blended model itself.

    """

    logger.info('='*80)
    logger.info("Selecting Best Model for partition: %s" % partition)

    # Define model tags

    best_tag = 'BEST'
    blend_tag = 'BLEND'

    # Extract model parameters.

    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    scorer = model.specs['scorer']

    # Initialize best parameters.

    maximize = True if scorers[scorer][1] == Objective.maximize else False
    if maximize:
        best_score = -sys.float_info.max
    else:
        best_score = sys.float_info.max

    # Initialize the model selection process.

    start_time = datetime.now()
    logger.info("Best Model Selection Start: %s", start_time)

    # Iterate through the models, getting the best score for each one.

    for index, algorithm in enumerate(model.algolist):
        logger.info("Scoring %s Model", algorithm)
        top_score = model.metrics[(algorithm, partition, scorer)]
        if index > 0:
            # objective is to either maximize or minimize score
            if maximize:
                if top_score > best_score:
                    best_score = top_score
                    best_algo = algorithm
            else:
                if top_score < best_score:
                    best_score = top_score
                    best_algo = algorithm
        else:
            best_score = top_score
            best_algo = algorithm

    # Record predictions of best estimator

    logger.info("Best Model is %s with a %s score of %.4f", best_algo, scorer, best_score)
    model.best_algo = best_algo
    model.estimators[best_tag] = model.estimators[best_algo]
    model.preds[(best_tag, partition)] = model.preds[(best_algo, partition)]
    if model_type == ModelType.classification:
        model.probas[(best_tag, partition)] = model.probas[(best_algo, partition)]

    # Record support vector for any recursive feature elimination

    if rfe and 'XGB' not in best_algo:
        try:
            model.feature_map['rfe_support'] = model.support[best_algo]
        except:
            # no RFE support for best algorithm
            pass

    # Return the model with best estimator and predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Best Model Selection Complete: %s", time_taken)

    return model


#
# Function generate_metrics
#

def generate_metrics(model, partition):
    r"""Generate model evaluation metrics for all estimators.

    Parameters
    ----------
    model : alphapy.Model
        The model object with stored predictions.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    model : alphapy.Model
        The model object with the completed metrics.

    Notes
    -----
    AlphaPy takes a brute-force approach to calculating each metric.
    It calls every scikit-learn function without exception. If the
    calculation fails for any reason, then the evaluation will still
    continue without error.

    References
    ----------
    For more information about model evaluation and the associated metrics,
    refer to [EVAL]_.

    .. [EVAL] http://scikit-learn.org/stable/modules/model_evaluation.html

    """

    logger.info('='*80)
    logger.info("Metrics for: %s", partition)

    # Extract model parameters

    model_type = model.specs['model_type']

    # Extract model data

    if partition == Partition.train:
        expected = model.y_train
        groups = model.groups_train
    elif partition == Partition.test:
        expected = model.y_test
        groups = model.groups_test
    elif partition == Partition.train_ts:
        expected = model.y_train_ts
    elif partition == Partition.test_ts:
        expected = model.y_test
    else:
        raise ValueError("Invalid Partition: %s", partition)

    # Kendall Tau Function
    
    def kendall_tau(expected, predicted):
        group_indices = list(np.cumsum(groups))
        group_indices.insert(0, 0)
        scores = []
        for i, _ in enumerate(groups):
            group_exp = expected[group_indices[i]:group_indices[i+1]]
            group_pred = predicted[group_indices[i]:group_indices[i+1]]
            kt_values = stats.kendalltau(group_exp, group_pred)
            kt_score = kt_values.correlation
            if not np.isnan(kt_score):
                scores.append(kt_score)
        return np.mean(scores)

    # Generate Metrics

    if not expected.empty:
        algolist = copy(model.algolist)
        if len(algolist) > 1:
            algolist.append('BLEND')
        # get the metrics for each algorithm
        for algo in algolist:
            # get predictions for the given algorithm
            predicted = model.preds[(algo, partition)]
            # Classification Metrics
            if model_type == ModelType.classification:
                probas = model.probas[(algo, partition)]
                try:
                    model.metrics[(algo, partition, 'accuracy')] = accuracy_score(expected, predicted)
                except:
                    logger.info("Accuracy Score not calculated")
                try:
                    model.metrics[(algo, partition, 'average_precision')] = average_precision_score(expected, probas)
                except:
                    logger.info("Average Precision Score not calculated")
                try:
                    model.metrics[(algo, partition, 'balanced_accuracy')] = balanced_accuracy_score(expected, predicted)
                except:
                    logger.info("Balanced Accuracy Score not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_brier_score')] = brier_score_loss(expected, probas)
                except:
                    logger.info("Brier Score not calculated")
                try:
                    model.metrics[(algo, partition, 'matthews_corrcoef')] = matthews_corrcoef(expected, predicted)
                except:
                    logger.info("Matthews Correlation Coefficient not calculated")
                try:
                    model.metrics[(algo, partition, 'cohen_kappa')] = cohen_kappa_score(expected, predicted)
                except:
                    logger.info("Cohen's Kappa Score not calculated")
                try:
                    model.metrics[(algo, partition, 'confusion_matrix')] = confusion_matrix(expected, predicted)
                except:
                    logger.info("Confusion Matrix not calculated")
                try:
                    model.metrics[(algo, partition, 'f1')] = f1_score(expected, predicted)
                except:
                    logger.info("F1 Score not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_log_loss')] = log_loss(expected, probas)
                except:
                    logger.info("Log Loss not calculated")
                try:
                    model.metrics[(algo, partition, 'precision')] = precision_score(expected, predicted)
                except:
                    logger.info("Precision Score not calculated")
                try:
                    model.metrics[(algo, partition, 'recall')] = recall_score(expected, predicted)
                except:
                    logger.info("Recall Score not calculated")
                try:
                    fpr, tpr, _ = roc_curve(expected, probas)
                    model.metrics[(algo, partition, 'roc_auc')] = auc(fpr, tpr)
                except:
                    logger.info("ROC AUC Score not calculated")
            # Regression Metrics
            elif model_type == ModelType.regression:
                try:
                    model.metrics[(algo, partition, 'explained_variance')] = explained_variance_score(expected, predicted)
                except:
                    logger.info("Explained Variance Score not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_mean_absolute_error')] = mean_absolute_error(expected, predicted)
                except:
                    logger.info("Mean Absolute Error not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_mean_absolute_percentage_error')] = mean_absolute_percentage_error(expected, predicted)
                except:
                    logger.info("Mean Absolute Percentage Error not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_median_absolute_error')] = median_absolute_error(expected, predicted)
                except:
                    logger.info("Median Absolute Error not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_mean_squared_error')] = mean_squared_error(expected, predicted)
                except:
                    logger.info("Mean Squared Error not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_mean_squared_log_error')] = mean_squared_log_error(expected, predicted)
                except:
                    logger.info("Mean Squared Log Error not calculated")
                try:
                    model.metrics[(algo, partition, 'r2')] = r2_score(expected, predicted)
                except:
                    logger.info("R-Squared Score not calculated")
            # Letor Metrics
            elif model_type == ModelType.letor:
                try:
                    model.metrics[(algo, partition, 'dcg_score')] = dcg_score([expected], [predicted])
                except:
                    logger.info("DCG Score not calculated")
                try:
                    model.metrics[(algo, partition, 'kendall_tau')] = kendall_tau(expected, predicted)
                except:
                    logger.info("Kendall Tau not calculated")
                try:
                    model.metrics[(algo, partition, 'ndcg_score')] = ndcg_score([expected], [predicted])
                except:
                    logger.info("NDCG Score not calculated")
        # log the metrics for each algorithm
        for algo in algolist:
            logger.info('-'*80)
            logger.info("Algorithm: %s", algo)
            metrics = [(k[2], v) for k, v in list(model.metrics.items()) if k[0] == algo and k[1] == partition]
            for key, value in sorted(metrics):
                svalue = str(value)
                svalue.replace('\n', ' ')
                logger.info("%s: %s", key, svalue)
    else:
        logger.info("No labels for generating %s metrics", partition)

    return model


#
# Function save_predictions
#

def save_predictions(model, partition):
    r"""Save the predictions to files.

    Parameters
    ----------
    model : alphapy.Model
        The model object to save.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    model : alphapy.Model
        The model object with the blended estimator.

    Notes
    -----

    The following components are extracted from the model object
    and saved to disk:

    * Model predictor (via joblib/pickle)
    * Feature Map (via joblib/pickle)

    """

    logger.info('='*80)
    logger.info("Saving Predictions for partition: %s" % partition)

    # Extract model parameters.

    directory = model.specs['directory']
    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    separator = model.specs['separator']
    submission_file = model.specs['submission_file']
    submit_probas = model.specs['submit_probas']

    # Specify input and output directories

    data_dir = SSEP.join([directory, 'data'])
    input_dir = SSEP.join([run_dir, 'input'])
    output_dir = SSEP.join([run_dir, 'output'])

    # Join train and test files

    df_master = pd.DataFrame()
    if partition == Partition.train:
        df_master = pd.concat([model.df_X_train, model.df_y_train], axis=1)
    elif partition == Partition.test or partition == Partition.test_ts:
        if model.test_labels:
            df_master = pd.concat([model.df_X_test, model.df_y_test], axis=1)
        else:
            df_master = model.df_X_test
    elif partition == Partition.train_ts:
        df_master = pd.concat([model.X_train_ts, model.y_train_ts], axis=1)
    else:
        raise ValueError("Invalid Partition: %s", partition)

    # Iterate through tags, including algorithms.

    logger.info("Adding Prediction Columns")

    tag_list = []

    best_tag = 'BEST'
    condition1 = (partition == Partition.train or partition == Partition.train_ts)
    condition2 = (partition == Partition.test or partition == Partition.test_ts) and model.test_labels
    if condition1 or condition2:
        sort_tag = best_tag
    else:
        sort_tag = model.best_algo
    tag_list.append(sort_tag)

    blend_tag = 'BLEND'
    if len(model.algolist) > 1:
        tag_list.append(blend_tag)

    tag_list.extend(model.algolist)

    for tag_id in tag_list:
        pred_name = USEP.join(['pred', datasets[partition], tag_id.lower()])
        df_master[pred_name] = model.preds[(tag_id, partition)]
        if model_type == ModelType.classification:
            prob_name = USEP.join(['prob', datasets[partition], tag_id.lower()])
            df_master[prob_name] = model.probas[(tag_id, partition)]

    # Save ranked predictions

    logger.info("Saving Ranked Predictions")
    if model_type == ModelType.classification:
        prob_name = USEP.join(['prob', datasets[partition], sort_tag.lower()])
        df_master.sort_values(prob_name, ascending=False, inplace=True)
    else:
        pred_name = USEP.join(['pred', datasets[partition], sort_tag.lower()])
        df_master.sort_values(pred_name, ascending=False, inplace=True)
    output_file = USEP.join(['ranked', datasets[partition]])
    write_frame(df_master, output_dir, output_file, extension, separator)

    # Generate submission file

    if submission_file and partition == Partition.train:
        sample_spec = PSEP.join([submission_file, extension])
        sample_input = SSEP.join([data_dir, sample_spec])
        df_sub = pd.read_csv(sample_input)
        if submit_probas and model_type == ModelType.classification:
            df_sub[df_sub.columns[1]] = model.probas[(model.best_algo, Partition.test)]
        else:
            df_sub[df_sub.columns[1]] = model.preds[(model.best_algo, Partition.test)]
        submission_spec = PSEP.join(['submission', extension])
        submission_output = SSEP.join([output_dir, submission_spec])
        logger.info("Saving Submission to %s", submission_output)
        df_sub.to_csv(submission_output, index=False)

    # Return model
    return model


#
# Function save_metrics
#

def save_metrics(model):
    r"""Save the model metrics to a file.

    Parameters
    ----------
    model : alphapy.Model
        The model object to save.

    Returns
    -------
    model : alphapy.Model
        The model object with the blended estimator.

    """

    # Extract model parameters.

    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    separator = model.specs['separator']

    # Specify output directory and file

    output_dir = SSEP.join([run_dir, 'model'])
    output_file = 'model_metrics'

    # Create metrics dataframe

    keys = [list(k) for (k, _) in model.metrics.items()]
    values = [v for (_, v) in model.metrics.items()]
    df = pd.DataFrame(keys, columns=['algo', 'partition', 'metric'])
    df['value'] = values

    # Save model metrics

    logger.info("Saving Model Metrics")
    write_frame(df, output_dir, output_file, extension, separator)

    # Return model
    return model
