################################################################################
#
# Package   : AlphaPy
# Module    : optimize
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
# Imports
#

from alphapy.globals import ModelType

from contextlib import contextmanager
from datetime import datetime
import itertools
import logging
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna_integration.sklearn import OptunaSearchCV
import os
import sys
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from time import time


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Helper function to get valid __init__ params
#

def get_valid_init_params(estimator):
    """Get params that are actually in the estimator's __init__ signature."""
    import inspect
    try:
        sig = inspect.signature(estimator.__class__.__init__)
        return set(sig.parameters.keys()) - {'self'}
    except (ValueError, TypeError):
        return set()


#
# Context manager to suppress stderr (for C++ library warnings)
#

@contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)


#
# Function convert_to_optuna_distributions
#

def convert_to_optuna_distributions(grid):
    r"""Convert a grid of parameter lists to Optuna distributions.

    Parameters
    ----------
    grid : dict
        Dictionary mapping parameter names to lists of values.

    Returns
    -------
    distributions : dict
        Dictionary mapping parameter names to Optuna distributions.

    """
    distributions = {}
    for param, values in grid.items():
        if isinstance(values, list) and len(values) > 0:
            # Use CategoricalDistribution for lists of values
            distributions[param] = CategoricalDistribution(values)
        elif hasattr(values, '__iter__') and not isinstance(values, str):
            # Handle numpy arrays or other iterables
            distributions[param] = CategoricalDistribution(list(values))
        else:
            # Single value - wrap in list for CategoricalDistribution
            distributions[param] = CategoricalDistribution([values])
    return distributions


#
# Function rfecv_search
#

def rfecv_search(model, algo):
    r"""Return the best feature set using recursive feature elimination
    with cross-validation.

    Parameters
    ----------
    model : alphapy.Model
        The model object with RFE parameters.
    algo : str
        Abbreviation of the algorithm to run.

    Returns
    -------
    model : alphapy.Model
        The model object with the RFE support vector and the best
        estimator.

    Notes
    -----
    If a scoring function is available, then AlphaPy can perform RFE
    with Cross-Validation (CV), as in this function; otherwise, it just
    does RFE without CV.

    References
    ----------
    For more information about Recursive Feature Elimination,
    refer to [RFECV]_.

    .. [RFECV] http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    """

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    n_jobs = model.specs['n_jobs']
    rfe_step = model.specs['rfe_step']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination
    # Clone estimator and set to single-threaded to avoid nested parallelism with RFECV
    rfe_est = clone(estimator)
    thread_params = ['n_jobs', 'nthread', 'thread_count', 'num_threads']
    if isinstance(rfe_est, Pipeline):
        inner_est = rfe_est.named_steps.get('estimator')
        if inner_est:
            valid_params = get_valid_init_params(inner_est)
            for param in thread_params:
                if param in valid_params:
                    rfe_est.set_params(**{f'estimator__{param}': 1})
    else:
        valid_params = get_valid_init_params(rfe_est)
        for param in thread_params:
            if param in valid_params:
                rfe_est.set_params(**{param: 1})

    logger.info("Recursive Feature Elimination with CV")
    rfecv = RFECV(rfe_est, step=rfe_step, cv=cv_folds,
                  scoring=scorer, verbose=verbosity, n_jobs=n_jobs)
    start = time()
    selector = rfecv.fit(X_train, y_train.values.ravel())
    logger.info("RFECV took %.2f seconds for step %d and %d folds",
                (time() - start), rfe_step, cv_folds)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algo, selector.n_features_, selector.ranking_)

    # Record the new estimator, support vector, feature names, and importances

    best_estimator = selector.estimator_
    model.estimators[algo] = best_estimator
    model.support[algo] = selector.support_
    fnames_algo = model.fnames_algo[algo]
    model.fnames_algo[algo] = list(itertools.compress(fnames_algo, selector.support_))
    if hasattr(best_estimator, "feature_importances_"):
        model.importances[algo] = best_estimator.feature_importances_

    # Return the model with the support vector

    return model


#
# Function grid_report
#

def grid_report(results, n_top=3):
    r"""Report the top grid search scores.

    Parameters
    ----------
    results : dict of numpy arrays
        Mean test scores for each grid search iteration.
    n_top : int, optional
        The number of grid search results to report.

    Returns
    -------
    None : None

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logger.info("Model with rank: {0}".format(i))
            logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
            logger.info("Parameters: {0}".format(results['params'][candidate]))


#
# Function hyper_grid_search
#

def hyper_grid_search(model, estimator):
    r"""Hyperparameter optimization using Optuna.

    Parameters
    ----------
    model : alphapy.Model
        The model with training data and configuration.
    estimator : alphapy.Estimator
        The estimator to optimize.

    Returns
    -------
    model : alphapy.Model
        Model with optimized estimator.

    Notes
    -----
    This function uses Optuna's OptunaSearchCV for hyperparameter
    optimization, which provides more efficient search than
    traditional grid search or random search.

    References
    ----------
    For more information about Optuna, refer to [OPTUNA]_.

    .. [OPTUNA] https://optuna.readthedocs.io/

    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    algo = estimator.algorithm
    scorer = model.specs['scorer']
    cv_folds = model.specs['cv_folds']
    n_trials = model.specs.get('optuna_trials', 100)
    timeout = model.specs.get('optuna_timeout', None)

    logger.info("Optuna optimization for %s (%d trials)", algo, n_trials)

    grid = estimator.grid
    if not grid:
        logger.info("No grid parameters for %s, skipping optimization", algo)
        return model

    # Convert grid to Optuna distributions
    param_distributions = convert_to_optuna_distributions(grid)

    # Get the estimator
    est = model.estimators[algo]

    # Extract model data
    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
    except:
        X_train = model.X_train
    y_train = model.y_train

    # Clone estimator and set to single-threaded to avoid nested parallelism with Optuna
    optuna_est = clone(est)
    thread_params = ['n_jobs', 'nthread', 'thread_count', 'num_threads']

    # Handle Pipeline estimators - need to prefix params correctly
    if isinstance(optuna_est, Pipeline):
        # For pipelines, prefix with the estimator step name
        prefixed_params = {}
        for k, v in param_distributions.items():
            prefixed_params[f'estimator__{k}'] = v
        param_distributions = prefixed_params
        # Set estimator threading to 1
        inner_est = optuna_est.named_steps.get('estimator')
        if inner_est:
            valid_params = get_valid_init_params(inner_est)
            for param in thread_params:
                if param in valid_params:
                    optuna_est.set_params(**{f'estimator__{param}': 1})
    else:
        # Set estimator threading to 1
        valid_params = get_valid_init_params(optuna_est)
        for param in thread_params:
            if param in valid_params:
                optuna_est.set_params(**{param: 1})

    n_jobs = model.specs['n_jobs']
    search = OptunaSearchCV(
        estimator=optuna_est,
        param_distributions=param_distributions,
        cv=cv_folds,
        scoring=scorer,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        verbose=0
    )

    # Convert to numpy if needed
    X_train_np = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
    y_train_np = y_train.to_numpy().ravel() if hasattr(y_train, 'to_numpy') else np.ravel(y_train)

    start = time()
    search.fit(X_train_np, y_train_np)
    logger.info("Optuna optimization took %.2f seconds", time() - start)

    logger.info("Best score for %s: %.4f", algo, search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    estimator.estimator = search.best_estimator_
    model.estimators[algo] = search.best_estimator_

    return model
