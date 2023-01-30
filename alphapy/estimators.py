################################################################################
#
# Package   : AlphaPy
# Module    : estimators
# Created   : July 11, 2013
#
# Copyright 2023 ScottFree Analytics LLC
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

import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import yaml


from alphapy.globals import ModelType
from alphapy.globals import Objective
from alphapy.globals import SSEP


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Define scorers
#

scorers = {'accuracy'                              : (ModelType.classification, Objective.maximize),
           'average_precision'                     : (ModelType.classification, Objective.maximize),
           'balanced_accuracy'                     : (ModelType.classification, Objective.maximize),
           'neg_brier_score'                       : (ModelType.classification, Objective.minimize),
           'f1'                                    : (ModelType.classification, Objective.maximize),
           'f1_macro'                              : (ModelType.classification, Objective.maximize),
           'f1_micro'                              : (ModelType.classification, Objective.maximize),
           'f1_samples'                            : (ModelType.classification, Objective.maximize),
           'f1_weighted'                           : (ModelType.classification, Objective.maximize),
           'neg_log_loss'                          : (ModelType.classification, Objective.minimize),
           'precision'                             : (ModelType.classification, Objective.maximize),
           'recall'                                : (ModelType.classification, Objective.maximize),
           'roc_auc'                               : (ModelType.classification, Objective.maximize),
           'coverage_error'                        : (ModelType.letor,          Objective.minimize),
           'dcg_score'                             : (ModelType.letor,          Objective.maximize),
           'label_ranking_average_precision_score' : (ModelType.letor,          Objective.maximize),
           'label_ranking_loss'                    : (ModelType.letor,          Objective.minimize),
           'ndcg_score'                            : (ModelType.letor,          Objective.maximize),
           'explained_variance'                    : (ModelType.regression,     Objective.maximize),
           'neg_mean_absolute_error'               : (ModelType.regression,     Objective.minimize),
           'neg_mean_absolute_percentage_error'    : (ModelType.regression,     Objective.minimize),
           'neg_mean_squared_error'                : (ModelType.regression,     Objective.minimize),
           'neg_mean_squared_log_error'            : (ModelType.regression,     Objective.minimize),
           'neg_median_absolute_error'             : (ModelType.regression,     Objective.minimize),
           'r2'                                    : (ModelType.regression,     Objective.maximize)}


#
# Define XGB scoring map
#

xgb_score_map = {'ndcg_score'              : 'ndcg',
                 'neg_log_loss'            : 'logloss',
                 'neg_mean_absolute_error' : 'mae',
                 'neg_mean_squared_error'  : 'rmse',
                 'precision'               : 'map',
                 'roc_auc'                 : 'auc'}


#
# Class Estimator
#

class Estimator:
    """Store information about each estimator.

    Parameters
    ----------
    algorithm : str
        Abbreviation representing the given algorithm.
    model_type : enum ModelType
        The machine learning task for this algorithm.
    estimator : function
        A scikit-learn, TensorFlow, or XGBoost function.
    grid : dict
        The dictionary of hyperparameters for grid search.

    """

    # __new__

    def __new__(cls,
                algorithm,
                model_type,
                estimator,
                grid):
        return super(Estimator, cls).__new__(cls)

    # __init__

    def __init__(self,
                 algorithm,
                 model_type,
                 estimator,
                 grid):
        self.algorithm = algorithm.upper()
        self.model_type = model_type
        self.estimator = estimator
        self.grid = grid

    # __str__

    def __str__(self):
        return self.name


#
# Define estimator map
#

estimator_map = {'AB'     : AdaBoostClassifier,
                 'GB'     : GradientBoostingClassifier,
                 'GBR'    : GradientBoostingRegressor,
                 'KNN'    : KNeighborsClassifier,
                 'KNR'    : KNeighborsRegressor,
                 'LOGR'   : LogisticRegression,
                 'LR'     : LinearRegression,
                 'LSVC'   : LinearSVC,
                 'LSVM'   : SVC,
                 'NB'     : MultinomialNB,
                 'RBF'    : SVC,
                 'RF'     : RandomForestClassifier,
                 'RFR'    : RandomForestRegressor,
                 'SVM'    : SVC,
                 'XT'     : ExtraTreesClassifier,
                 'XTR'    : ExtraTreesRegressor
                }


#
# Find optional packages
#

def find_optional_packages():
    r"""Find optional machine learning packages.

    Parameters
    ----------
    
    None

    Returns
    -------
    
    None

    """

    module_name = 'xgboost'
    try:
        import xgboost as xgb
        estimator_map['XGB'] = xgb.XGBClassifier
        estimator_map['XGBM'] = xgb.XGBClassifier
        estimator_map['XGBR'] = xgb.XGBRegressor
        estimator_map['XGRK'] = xgb.XGBRanker
    except ModuleNotFoundError:
        logger.info("Cannot load %s", module_name)

    module_name = 'lightgbm'
    try:
        import lightgbm as lgb
        estimator_map['LGB'] = lgb.LGBMClassifier
        estimator_map['LGBR'] = lgb.LGBMRegressor
    except ModuleNotFoundError:
        logger.info("Cannot load %s", module_name)

    module_name = 'catboost'
    try:
        import catboost as catb
        estimator_map['CATB'] = catb.CatBoostClassifier
        estimator_map['CATBR'] = catb.CatBoostRegressor
    except ModuleNotFoundError:
        logger.info("Cannot load %s", module_name)

    return


#
# Function get_algos_config
#

def get_algos_config(cfg_dir):
    r"""Read the algorithms configuration file.

    Parameters
    ----------
    cfg_dir : str
        The directory where the configuration file ``algos.yml``
        is stored.

    Returns
    -------
    specs : dict
        The specifications for determining which algorithms to run.

    """

    logger.info("Algorithm Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'algos.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Find optional packages

    find_optional_packages()

    # Ensure each algorithm has required keys

    minimum_keys = ['model_type', 'params', 'grid']
    for algo in specs:
        required_keys = minimum_keys
        algo_keys = list(specs[algo].keys())
        if set(algo_keys) != set(required_keys):
            logger.warning("Algorithm %s has the wrong keys %s",
                           algo, required_keys)
            logger.warning("Keys found instead: %s", algo_keys)
        else:
            # determine whether or not model type is valid
            model_types = {x.name: x.value for x in ModelType}
            model_type = specs[algo]['model_type']
            if model_type in model_types:
                specs[algo]['model_type'] = ModelType(model_types[model_type])
            else:
                raise ValueError("algos.yml model:type %s unrecognized" % model_type)

    # Algorithm Specifications
    return specs


#
# Function get_estimators
#

def get_estimators(alphapy_specs, model):
    r"""Define all the AlphaPy estimators based on the contents
    of the ``algos.yml`` file.

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    model : alphapy.Model
        The model object containing global AlphaPy parameters.

    Returns
    -------
    estimators : dict
        All of the estimators required for running the pipeline.

    """

    # Extract model data

    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Initialize estimator dictionary
    estimators = {}

    # Global parameter substitution fields

    ps_fields = {'n_estimators' : 'n_estimators',
                 'iterations'   : 'n_estimators',
                 'n_jobs'       : 'n_jobs',
                 'nthread'      : 'n_jobs',
                 'thread_count' : 'n_jobs',
                 'seed'         : 'seed',
                 'random_state' : 'seed',
                 'random_seed'  : 'seed',
                 'verbosity'    : 'verbosity',
                 'verbose'      : 'verbosity'}

    # Get algorithm specifications

    config_dir = SSEP.join([alphapy_specs['alphapy_root'], 'config'])
    algo_specs = get_algos_config(config_dir)

    # Create estimators for all of the algorithms

    for algo in algo_specs:
        model_type = algo_specs[algo]['model_type']
        params = algo_specs[algo]['params']
        for param in params:
            if param in ps_fields and isinstance(param, str):
                algo_specs[algo]['params'][param] = eval(ps_fields[param])
        try:
            algo_found = True
            func = estimator_map[algo]
        except Exception:
            algo_found = False
            logger.info("Algorithm %s not found (check package installation)", algo)
        if algo_found:
            est = func(**params)
            grid = algo_specs[algo]['grid']
            estimators[algo] = Estimator(algo, model_type, est, grid)

    # return the entire classifier list
    return estimators
