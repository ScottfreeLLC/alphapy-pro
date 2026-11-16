################################################################################
#
# Package   : AlphaPy
# Module    : features
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

from alphapy.estimators import worker_jobs
from alphapy.sequence import tabular
from alphapy.globals import BSEP, LOFF, NULLTEXT
from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import Encoders
from alphapy.globals import ModelType
from alphapy.globals import Scalers
from alphapy.variables import Variable
from alphapy.variables import vparse

import category_encoders as ce
from importlib import import_module
import itertools
import lofo
import logging
import numpy as np
import os
import pandas as pd
import polars as pl
import re
import scipy.stats as sps
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.manifold import Isomap
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import sys
from types import SimpleNamespace


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Define feature scoring functions
#

feature_scorers = {'f_classif'    : f_classif,
                   'chi2'         : chi2,
                   'f_regression' : f_regression,
                   'SelectKBest'  : SelectKBest,
                   'SelectFpr'    : SelectFpr,
                   'SelectFdr'    : SelectFdr,
                   'SelectFwe'    : SelectFwe}


#
# Define Encoder map
#

encoder_map = {Encoders.backdiff     : ce.BackwardDifferenceEncoder,
               Encoders.basen        : ce.BaseNEncoder,
               Encoders.binary       : ce.BinaryEncoder,
               Encoders.catboost     : ce.CatBoostEncoder,
               Encoders.hashing      : ce.HashingEncoder,
               Encoders.helmert      : ce.HelmertEncoder,
               Encoders.jstein       : ce.JamesSteinEncoder,
               Encoders.leaveone     : ce.LeaveOneOutEncoder,
               Encoders.mestimate    : ce.MEstimateEncoder,
               Encoders.onehot       : ce.OneHotEncoder,
               Encoders.ordinal      : ce.OrdinalEncoder,
               Encoders.polynomial   : ce.PolynomialEncoder,
               Encoders.sum          : ce.SumEncoder,
               Encoders.target       : ce.TargetEncoder,
               Encoders.woe          : ce.WOEEncoder}


#
# Function apply_transform
#

def apply_transform(fname, df, fparams):
    r"""Apply a transform function to a column of the dataframe.

    Parameters
    ----------
    fname : str
        Name of the column to be treated in the dataframe ``df``.
    df : pandas.DataFrame
        Dataframe containing the column ``fname``.
    fparams : list
        The module, function, and parameter list of the transform
        function

    Returns
    -------
    new_features : pandas.DataFrame
        The set of features after applying a transform function.

    """
    # Extract the transform parameter list
    module = fparams[0]
    func_name = fparams[1]
    plist = fparams[2:]
    # Append to system path
    sys.path.append(os.getcwd())
    # Import the external transform function
    ext_module = import_module(module)
    func = getattr(ext_module, func_name)
    # Prepend the parameter list with the data frame and feature name
    plist.insert(0, fname)
    plist.insert(0, df)
    # Apply the transform
    logger.info("Applying function %s from module %s to feature %s",
                func_name, module, fname)
    return func(*plist)


#
# Function apply_transforms
#

def apply_transforms(model, X):
    r"""Apply special functions to the original features.

    Parameters
    ----------
    model : alphapy.Model
        Model specifications indicating any transforms.
    X : pandas.DataFrame
        Combined train and test data, or just prediction data.

    Returns
    -------
    all_features : pandas.DataFrame
        All features, including transforms.

    Raises
    ------
    IndexError
        The number of transform rows must match the number of
        rows in ``X``.

    """

    # Extract model parameters
    transforms = model.specs['transforms']

    # Log input parameters

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])

    # Iterate through columns, dispatching and transforming each feature.

    logger.info("Applying transforms")
    all_features = X

    if transforms:
        for fname in transforms:
            # find feature series
            fcols = []
            for col in X.columns:
                if col.split(LOFF)[0] == fname:
                    fcols.append(col)
            # get lag values
            lag_values = []
            for item in fcols:
                _, _, _, _, lag = vparse(item)
                lag_values.append(lag)
            # apply transform to the most recent value
            if lag_values:
                f_latest = fcols[lag_values.index(min(lag_values))]
                features = apply_transform(f_latest, X, transforms[fname])
                if features is not None:
                    if features.shape[0] == X.shape[0]:
                        if hasattr(all_features, "to_pandas"):
                            # X is polars; coerce features to polars too
                            features_pl = pl.from_pandas(features) if isinstance(features, pd.DataFrame) else features
                            all_features = pl.concat([all_features, features_pl], how="horizontal")
                        else:
                            all_features = pd.concat([all_features, features], axis=1)
                    else:
                        raise IndexError("The number of transform rows [%d] must match X [%d]" %
                                         (features.shape[0], X.shape[0]))
                else:
                    logger.info("Could not apply transform for feature %s", fname)
            else:
                logger.info("Feature %s is missing for transform", fname)
    else:
        logger.info("No transforms Specified")

    logger.info("New Feature Count : %d", all_features.shape[1])

    # Return all transformed training and test features
    return all_features


#
# Function float_factor
#

def float_factor(x, rounding):
    r"""Convert a floating point number to a factor.

    Parameters
    ----------
    x : float
        The value to convert to a factor.
    rounding : int
        The number of places to round.

    Returns
    -------
    ffactor : int
        The resulting factor.

    """
    num2str = '{0:.{1}f}'.format
    fstr = re.sub("[^0-9]", "", num2str(x, rounding))
    ffactor = int(fstr) if len(fstr) > 0 else 0
    return ffactor


#
# Function create_crosstabs
#

def create_crosstabs(model, target):
    r"""Create cross-tabulations for categorical variables.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing the data.
    target : str
        The target variable to cross-tabulate.

    Returns
    -------
    model : alphapy.Model
        The model object with the updated feature map.

    """

    logger.info("Creating Cross-Tabulations")

    # Extract model data
    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters

    factors = model.specs['factors']

    # Iterate through columns, dispatching and transforming each feature.

    crosstabs = {}
    for fname in X_train.columns:
        if fname in factors:
            logger.info("Creating crosstabs for feature %s", fname)
            # Convert to pandas for crosstab
            x_col = X_train[fname].to_pandas() if hasattr(X_train[fname], 'to_pandas') else X_train[fname]
            y_col = y_train[target].to_pandas() if hasattr(y_train[target], 'to_pandas') else y_train[target]
            ct_factor = pd.crosstab(x_col, y_col).apply(lambda r : r / r.sum(), axis=1)
            crosstabs[fname] = ct_factor

    # Save crosstabs to the feature map

    model.feature_map['crosstabs'] = crosstabs
    return model


#
# Feature pipeline
#

# The model specifications the feature steps read; the pipeline keeps its own copy, so a saved run needs no model.yml.
PIPELINE_SPECS = ('clustering', 'cluster_inc', 'cluster_max', 'cluster_min', 'counts', 'drop', 'encoder', 'factors',
                  'interactions', 'isample_pct', 'isomap', 'iso_components', 'iso_neighbors', 'log_transform',
                  'lv_remove', 'lv_threshold', 'model_type', 'n_jobs', 'ngrams_max', 'numpy', 'pca', 'pca_inc',
                  'pca_max', 'pca_min', 'pca_whiten', 'poly_degree', 'pvalue_level', 'rounding', 'scaler_option',
                  'scaler_type', 'scipy', 'seed', 'sentinel', 'transforms', 'tsne', 'vectorize')


def column_values(feature, dtype):
    r"""One column as a 2-D float or integer array, with infinity treated as missing for floating point columns."""
    values = feature.to_numpy().reshape(-1, 1)
    if dtype.is_float():
        values = values.astype(np.float64, copy=True)
        values[np.isinf(values)] = np.nan
    return values


def new_imputer(dtype, sentinel):
    r"""The imputer for a numerical column: the median for floating point values, the most frequent value for signed
    integers, zero for unsigned integers and ``sentinel`` for anything else except Booleans, which are not imputed."""
    if dtype.is_float():
        return SimpleImputer(missing_values=np.nan, strategy='median')
    if dtype.is_signed_integer():
        return SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    if dtype.is_unsigned_integer():
        return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=np.uint32(0))
    if dtype == pl.Boolean:
        return None
    return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=sentinel)


def finite_scaled(features, scaler):
    r"""``features`` with missing and infinite values set to zero, scaled by ``scaler``, then cleaned again."""
    clean = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(scaler.transform(clean), nan=0.0, posinf=0.0, neginf=0.0)


class FeaturePipeline:
    r"""The feature steps of a run, fitted on its training rows only and applied unchanged to any other rows.

    Training fits every step that learns from data on the training rows (imputation values, the log-transform
    decision, the scaler, factor encoders, text codes, the NumPy and SciPy row statistics' imputer and scaler,
    clustering, PCA, isomap, the interaction selection and scaler, and the low-variance mask), then turns its test
    rows into features with :meth:`transform`, the same method that scores rows later. Column routing (factor,
    numerical or text) and the decision to keep a column are made once, on the training rows, so a small frame (one
    row with hour 0) produces the same columns as a large one.

    Until this pipeline existed, these steps were fitted on training and test rows together, so training read
    statistics of its test rows. Runs with any of these steps on now give different (and honest) results.

    Parameters
    ----------
    specs : dict
        The model specifications; the pipeline keeps the ones it reads (``PIPELINE_SPECS``).
    raw_schema : dict
        Training's raw columns (as loaded, before transforms and drops) and their Polars types, in order.

    """

    def __init__(self, specs, raw_schema):
        self.specs = {key: specs.get(key) for key in PIPELINE_SPECS}
        self.raw_schema = dict(raw_schema)
        self.input_schema = None        # the columns after transforms and drops, with their types
        self.columns = []               # how each base column is turned into features
        self.fitted = {}                # every other fitted step, by name
        self.poly_support = None
        self.lv_support = None
        self.feature_names = []
        self.history = None             # each group's last training rows, for window algorithms (sequence section)

    # raw rows

    def conform(self, frame):
        r"""``frame`` (pandas or Polars) as training loaded its rows: its raw columns by name, in training's order,
        with training's types. A frame missing a column, or whose values cannot take training's types, is refused;
        other columns (the target, say) are left out."""
        if not isinstance(frame, pl.DataFrame):
            frame = pl.from_pandas(pd.DataFrame(frame).reset_index(drop=True))
        missing = [c for c in self.raw_schema if c not in frame.columns]
        if missing:
            raise ValueError(f"The frame lacks columns the run was trained on: {missing}")
        try:
            return frame.select(list(self.raw_schema)).cast(self.raw_schema)
        except pl.exceptions.PolarsError as e:
            raise ValueError(f"The frame's values do not fit the run's column types: {e}") from e

    def prepare(self, frame):
        r"""Apply the run's transforms and drops to conformed raw rows, as training did before creating features."""
        return drop_features(apply_transforms(SimpleNamespace(specs=self.specs), frame), self.specs['drop'])

    # features

    def fit_transform(self, model, X, y_train):
        r"""Fit every step on the training rows ``X`` (transformed and dropped, as :meth:`prepare` returns them) and
        return their features. Records the feature names, interaction support and low-variance mask in ``model``."""
        if self.specs['tsne']:
            raise ValueError("t-SNE cannot place rows it was not fitted on (scikit-learn's TSNE has no transform), so "
                             "a run with t-SNE features could score its test rows only by fitting on them. "
                             "Turn features.tsne off.")
        y = y_train.to_numpy().ravel() if hasattr(y_train, 'to_numpy') else np.asarray(y_train).ravel()
        self.input_schema = dict(X.schema)
        logger.info("Original Features : %s", X.columns)
        logger.info("Feature Count     : %d", X.shape[1])
        X = self._counted(X)
        logger.info("Creating Base Features")
        self.columns, blocks, names = [], [], []
        for i, fname in enumerate(X.columns):
            step, values = self._fit_column(i + 1, fname, X, y)
            self.columns.append(step)
            if step['keep']:
                blocks.append(values)
                names.extend(step['names'])
            else:
                logger.info("Feature %d: %s is null (fractal length spans data)", i + 1, fname)
        features, derived = self._features(blocks, fit=True)
        names.extend(derived)
        features = self._interactions(features, names, y, fit=True)
        model.feature_names = names
        if self.poly_support is not None:
            model.feature_map['poly_support'] = self.poly_support
        features = remove_lv_features(model, features)
        self.lv_support = model.feature_map.get('lv_support') if self.specs['lv_remove'] else None
        self.feature_names = list(model.feature_names)
        return features

    def transform(self, X):
        r"""The features of any rows (transformed and dropped, as :meth:`prepare` returns them), with every step as
        fitted on the training rows. Each row's features depend on that row alone."""
        if list(X.columns) != list(self.input_schema):
            raise ValueError(f"The rows' columns {list(X.columns)} differ from training's {list(self.input_schema)}")
        if X.height == 0:
            return np.empty((0, len(self.feature_names)))
        X = self._counted(X.cast(self.input_schema))
        blocks = [self._apply_column(step, X) for step in self.columns if step['keep']]
        features, _ = self._features(blocks, fit=False)
        features = self._interactions(features, None, None, fit=False)
        if self.lv_support is not None:
            features = features[:, self.lv_support]
        return features

    # steps

    def _counted(self, X):
        r"""Add the count features (row by row) and turn object columns into text."""
        if self.specs['counts']:
            X = X.with_columns(pl.sum_horizontal(pl.all().is_not_null()).alias('nan_count'))
            numeric_cols = [c for c in X.columns if X[c].dtype.is_numeric()]
            if numeric_cols:
                X = X.with_columns([pl.sum_horizontal((pl.col(c) == i).cast(pl.Int32) for c in numeric_cols)
                                    .alias(USEP.join(['count', str(i)])) for i in range(10)])
        for col in X.columns:
            if X[col].dtype == pl.Object:
                X = X.with_columns(pl.Series(col, [str(v) if v is not None else None for v in X[col].to_list()])
                                   .cast(pl.Utf8))
        return X

    def _fit_column(self, fnum, fname, X, y):
        r"""Decide how training's column ``fname`` becomes features, fit that on the training rows, and return the
        decision and the training rows' values."""
        factors = self.specs['factors']
        dtype = X[fname].dtype
        nunique = X[fname].n_unique()
        step = {'name': fname, 'dtype': dtype}
        if factors and fname in factors:
            logger.info("Feature %d: %s is a factor of type %s with %d unique values", fnum, fname, dtype, nunique)
            try:
                encoder = encoder_map[self.specs['encoder']](cols=[fname])
            except KeyError:
                raise ValueError("Unknown Encoder %s" % self.specs['encoder'])
            step |= {'kind': 'factor', 'encoder': encoder, 'float': dtype.is_float()}
            values = np.asarray(encoder.fit_transform(self._factor_frame(step, X), y))
            step['names'] = list(encoder.get_feature_names())
        elif dtype.is_numeric():
            logger.info("Feature %d: %s is a numerical feature of type %s with %d unique values",
                        fnum, fname, dtype, nunique)
            imputer = new_imputer(dtype, self.specs['sentinel'])
            values = column_values(X[fname], dtype)
            if imputer is not None:
                values = imputer.fit_transform(values)
            log = False
            if self.specs['log_transform'] and np.all(values > 0):
                _, pvalue = sps.normaltest(values)
                if pvalue <= self.specs['pvalue_level']:
                    logger.info("Feature %d: %s is not normally distributed [p-value: %f]", fnum, fname, pvalue)
                    values, log = np.log(values), True
            step |= {'kind': 'numeric', 'imputer': imputer, 'log': log,
                     'names': [USEP.join([fname, 'log']) if log else fname]}
        elif dtype == pl.Utf8 or dtype == pl.String:
            logger.info("Feature %d: %s is a text feature with %d unique values", fnum, fname, nunique)
            if self.specs['vectorize']:
                logger.warning("Feature %d: %s => text vectorization is not available; factorizing", fnum, fname)
            text = X[fname].fill_null(NULLTEXT).to_list()
            step |= {'kind': 'text', 'codes': {v: k for k, v in enumerate(pd.unique(np.asarray(text, dtype=object)))},
                     'names': [USEP.join([fname, 'factor'])]}
            values = self._apply_column(step, X)
        else:
            raise TypeError("Base Feature Error with unrecognized type %s" % dtype)
        step['keep'] = bool(values.shape[0] == X.height and values.any())
        return step, values

    def _factor_frame(self, step, X):
        fname = step['name']
        feature = X[fname].to_pandas()
        if step['float']:
            feature = feature.apply(float_factor, args=[self.specs['rounding']])
        return pd.DataFrame({fname: feature})

    def _apply_column(self, step, X):
        r"""One base column's features for any rows, as fitted on the training rows."""
        kind = step['kind']
        if kind == 'factor':
            return np.asarray(step['encoder'].transform(self._factor_frame(step, X)))
        if kind == 'text':                          # a value training never saw gets -1
            text = pd.Series(X[step['name']].fill_null(NULLTEXT).to_list(), dtype=object)
            return text.map(step['codes']).fillna(-1).astype(np.int64).to_numpy()
        values = column_values(X[step['name']], step['dtype'])
        if step['imputer'] is not None:
            values = step['imputer'].transform(values)
        if step['log']:
            if not np.all(values > 0):
                raise ValueError(f"Column {step['name']} was log-transformed in training, where all its values were "
                                 "positive, but these rows have a value at or below zero")
            values = np.log(values)
        return values

    def _step(self, name, make, X, fit, method='transform'):
        r"""Fit a new step on training rows (``fit``) or apply the fitted one."""
        if fit:
            self.fitted[name] = make()
            if method == 'transform':
                return self.fitted[name].fit_transform(X)
            self.fitted[name].fit(X)
        return getattr(self.fitted[name], method)(X)

    def _features(self, blocks, fit):
        r"""The scaled base features and the features derived from them (NumPy and SciPy row statistics,
        clusters, PCA and isomap), with the derived features' names."""
        specs = self.specs
        n_rows = blocks[0].shape[0] if blocks else 0
        base = np.column_stack(blocks) if blocks else np.zeros((n_rows, 0))
        logger.info("New Feature Count : %d", base.shape[1])
        if specs['scaler_option']:
            makers = {Scalers.standard: StandardScaler, Scalers.minmax: MinMaxScaler}
            if specs['scaler_type'] in makers:
                logger.info("Scaling Base Features")
                base = self._step('scaler', makers[specs['scaler_type']], base, fit)
            else:
                logger.info("Unrecognized scaler: %s", specs['scaler_type'])
        else:
            logger.info("Skipping Scaling")
        blocks, names = [base], []
        if specs['numpy']:
            logger.info("Creating NumPy Features")
            stats = np.column_stack([base.sum(axis=1), base.mean(axis=1), base.std(axis=1), base.var(axis=1)])
            stats[np.isinf(stats)] = np.nan
            stats = self._step('numpy_imputer', lambda: SimpleImputer(strategy='median'), stats, fit)
            blocks.append(self._step('numpy_scaler', StandardScaler, stats, fit))
            names.extend(['sum', 'mean', 'std', 'var'])
        if specs['scipy']:
            logger.info("Creating SciPy Features")
            stats = scipy_row_statistics(base)
            stats[np.isinf(stats)] = np.nan
            stats = self._step('scipy_imputer', lambda: SimpleImputer(strategy='median'), stats, fit)
            blocks.append(self._step('scipy_scaler', StandardScaler, stats, fit))
            names.extend(['sp_geometric_mean', 'sp_kurtosis', 'sp_kurtosis_test', 'sp_normal_test', 'sp_skew',
                          'sp_skew_test', 'sp_variation', 'sp_signal_to_noise', 'sp_standard_error_of_mean'])
        if specs['clustering'] or specs['pca'] or specs['isomap']:
            clean = np.nan_to_num(base.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if fit:
                self.fitted['reduction_scaler'] = StandardScaler().fit(clean)
            scaled = finite_scaled(base, self.fitted['reduction_scaler'])
        if specs['clustering']:
            logger.info("Creating Clustering Features")
            for k in range(specs['cluster_min'], specs['cluster_max'] + 1, specs['cluster_inc']):
                make = lambda k=k: MiniBatchKMeans(n_clusters=k, random_state=specs['seed'], init='random')
                blocks.append(self._step(f'cluster_{k}', make, scaled, fit, method='predict').reshape(-1, 1))
                names.append(USEP.join(['cluster', str(k)]))
        if specs['pca']:
            logger.info("Creating PCA Features")
            for n in range(specs['pca_min'], specs['pca_max'] + 1, specs['pca_inc']):
                make = lambda n=n: PCA(n_components=n, whiten=specs['pca_whiten'])
                blocks.append(self._step(f'pca_{n}', make, scaled, fit))
                names.extend(USEP.join(['pca', str(n), str(j + 1)]) for j in range(n))
        if specs['isomap']:
            logger.info("Creating Isomap Features")
            make = lambda: Isomap(n_neighbors=specs['iso_neighbors'], n_components=specs['iso_components'],
                                  n_jobs=specs['n_jobs'])
            blocks.append(self._step('isomap', make, scaled, fit))
            names.extend(USEP.join(['isomap', str(i + 1)]) for i in range(specs['iso_components']))
        features = np.column_stack(blocks)
        logger.info("New Feature Count : %d", features.shape[1])
        return features, names

    def _interactions(self, features, names, y, fit):
        r"""Append the scaled products of the features the training rows' univariate test selected."""
        specs = self.specs
        if not specs['interactions']:
            logger.info("Skipping Interactions")
            return features
        if fit:
            logger.info("Generating Polynomial Features")
            if specs['model_type'] == ModelType.regression:
                selector = SelectPercentile(f_regression, percentile=specs['isample_pct'])
            elif specs['model_type'] == ModelType.classification:
                selector = SelectPercentile(f_classif, percentile=specs['isample_pct'])
            else:
                raise TypeError("Unknown model type when creating interactions")
            self.poly_support = selector.fit(features, y).get_support()
        make = lambda: PolynomialFeatures(interaction_only=True, degree=specs['poly_degree'], include_bias=False)
        products = self._step('poly', make, features[:, self.poly_support], fit)
        if fit:
            names.extend(self.fitted['poly'].get_feature_names_out())
        products = self._step('poly_scaler', StandardScaler, products, fit)
        logger.info("Polynomial Feature Count : %d", products.shape[1])
        return np.hstack((features, products))


def scipy_row_statistics(features):
    r"""Each row's geometric mean, kurtosis, kurtosis test, normality test, skew, skew test, variation,
    signal-to-noise ratio (mean over standard deviation, zero when the deviation is zero) and standard error."""
    mean, sd = features.mean(axis=1), features.std(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        signal_to_noise = np.where(sd == 0, 0.0, mean / sd)
    return np.column_stack((sps.gmean(features, axis=1), sps.kurtosis(features, axis=1),
                            sps.kurtosistest(features, axis=1)[0], sps.normaltest(features, axis=1)[0],
                            sps.skew(features, axis=1), sps.skewtest(features, axis=1)[0],
                            sps.variation(features, axis=1), signal_to_noise, sps.sem(features, axis=1)))


#
# Function select_features_lofo
#

def select_features_lofo(model, algo, est):
    r"""Select LOFO features.

    Parameters
    ----------
    model : alphapy.Model
        Model object with the feature selection specifications.
    algo : str
        Abbreviation of the algorithm to run.
    est : alphapy.Estimator
        The estimator to fit.

    Returns
    -------
    model : alphapy.Model
        Model object with the revised number of features.

    References
    ----------
    You can find more information on LOFO feature selection here [LOFO]_.

    .. [LOFO] http://https://github.com/aerdem4/lofo-importance

    """

    logger.info("LOFO Feature Selection for %s", algo)

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    fs_univariate = model.specs['fs_univariate']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    target = model.specs['target']
    
    # Filter features with preliminary univariate support.

    if fs_univariate:
        support_uni = model.feature_map['support_uni']
        X_fs = tabular(model, model.X_train)[:, support_uni].copy()
        fnames = list(itertools.compress(model.feature_names, support_uni))
    else:
        X_fs = tabular(model, model.X_train).copy()
        fnames = model.feature_names

    # Construct the LOFO dataset.

    X_df = pd.DataFrame(X_fs, columns=fnames)
    # Convert y_train to pandas if it's a Polars DataFrame (lofo requires pandas)
    y_train_pd = y_train.to_pandas() if hasattr(y_train, 'to_pandas') else y_train
    df = pd.concat([X_df, y_train_pd], axis=1)
    dataset = lofo.Dataset(df=df, target=target, features=fnames)

    # Calculate the feature importances.
    # Clone estimator and reduce threading to avoid contention with LOFO parallelism

    lofo_est = clone(est)
    thread_params = ['n_jobs', 'nthread', 'thread_count']

    def get_valid_init_params(estimator):
        """Get params that are actually in the estimator's __init__ signature."""
        import inspect
        try:
            sig = inspect.signature(estimator.__class__.__init__)
            return set(sig.parameters.keys()) - {'self'}
        except (ValueError, TypeError):
            return set()

    if isinstance(lofo_est, Pipeline):
        inner_est = lofo_est.named_steps.get('estimator')
        if inner_est:
            valid_params = get_valid_init_params(inner_est)
            for param in thread_params:
                if param in valid_params:
                    lofo_est.set_params(**{f'estimator__{param}': 1})
    else:
        valid_params = get_valid_init_params(lofo_est)
        for param in thread_params:
            if param in valid_params:
                lofo_est.set_params(**{param: 1})

    cv = KFold(n_splits=cv_folds, shuffle=False)
    # Suppress LOFO's generic multithreading warning and stdout noise from multiprocessing
    import warnings
    import sys
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*multithreaded.*', category=UserWarning)
        lofo_imp = lofo.LOFOImportance(dataset, model=lofo_est, cv=cv, scoring=scorer,
                                       n_jobs=worker_jobs(n_jobs, lofo_est))
    # Suppress None outputs from LOFO's multiprocessing (redirect at fd level for child processes)
    stdout_fd = sys.stdout.fileno()
    old_stdout_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    try:
        importance_df = lofo_imp.get_importance()
    finally:
        os.dup2(old_stdout_fd, stdout_fd)
        os.close(old_stdout_fd)
        os.close(devnull)
    importance_df.sort_values('importance_mean', ascending=False, inplace=True)
    
    # Filter the feature names and store.
    
    importance_df = importance_df[importance_df['importance_mean'] > 0.0]
    feature_names_lofo = importance_df['feature'].tolist()
    model.lofo_df[algo] = importance_df.copy()

    # Create the support vector

    support = []
    for fname in model.feature_names:
        support.append(any(fname == x for x in feature_names_lofo))

    # Record the support vector

    logger.info("Saving LOFO Support")
    model.feature_map['support_lofo', algo] = support

    # Count the number of new features.

    logger.info("Old Feature Count : %d", X_fs.shape[1])
    logger.info("New Feature Count : %d", len(feature_names_lofo))

    # Return the modified model
    return model


#
# Function select_features_univariate
#

def select_features_univariate(model):
    r"""Select features with univariate selection.

    Parameters
    ----------
    model : alphapy.Model
        Model object with the feature selection specifications.

    Returns
    -------
    model : alphapy.Model
        Model object with the revised number of features.

    References
    ----------
    You can find more information on univariate feature selection here [UNI]_.

    .. [UNI] http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """

    logger.info("Univariate Feature Selection")

    # Extract model data.

    X_train = tabular(model, model.X_train)       # never the raw window block
    y_train = model.y_train

    # Extract model parameters.

    fs_uni_pct = model.specs['fs_uni_pct']
    fs_uni_score_func = model.specs['fs_uni_score_func']

    # Select top features based on percentile.

    fs = SelectPercentile(score_func=fs_uni_score_func,
                          percentile=fs_uni_pct)

    # Perform feature selection and get the support mask
    y_train_np = y_train.to_numpy().ravel() if hasattr(y_train, 'to_numpy') else y_train.ravel()
    fsfit = fs.fit(X_train, y_train_np)
    support = fsfit.get_support()

    # Record the support vector

    logger.info("Saving Univariate Support")
    model.feature_map['support_uni'] = support

    # Count the number of new features.

    X_train_new = X_train[:, support]
    logger.info("Old Feature Count : %d", X_train.shape[1])
    logger.info("New Feature Count : %d", X_train_new.shape[1])

    # Mask the feature names and test that feature and name lengths are equal

    feature_names = list(itertools.compress(model.feature_names, support))
    assert X_train_new.shape[1] == len(feature_names), "Mismatched Features and Names"
    
    # Propagate features name to each algorithm.
    
    for algo in model.algolist:
        model.fnames_algo[algo] = feature_names

    # Return the modified model
    return model


#
# Function save_features
#

def save_features(model, X_train, X_test, y_train=None, y_test=None):
    r"""Save new features to the model.

    Parameters
    ----------
    model : alphapy.Model
        Model object with train and test data.
    X_train : numpy array
        Training features.
    X_test : numpy array
        Testing features.
    y_train : numpy array
        Training labels.
    y_test : numpy array
        Testing labels.

    Returns
    -------
    model : alphapy.Model
        Model object with new train and test data.

    """

    logger.info("Saving New Features in Model")

    model.X_train = X_train
    model.X_test = X_test
    if y_train is not None:
        model.y_train = y_train
    if y_test is not None:
        model.y_test = y_test

    return model


#
# Function drop_features
#

def drop_features(X, drop):
    r"""Drop any specified features.

    Parameters
    ----------
    X : polars.DataFrame
        The dataframe containing the features.
    drop : list
        The list of features to remove from ``X``.

    Returns
    -------
    X : polars.DataFrame
        The dataframe without the dropped features.

    """
    drop_cols = []
    if drop:
        for d in drop:
            for col in X.columns:
                if col.split(LOFF)[0] == d:
                    drop_cols.append(col)
        logger.info("Dropping Features: %s", drop_cols)
        logger.info("Original Feature Count : %d", X.shape[1])
        # Filter to only columns that exist (ignore missing)
        existing_cols = [c for c in drop_cols if c in X.columns]
        if existing_cols:
            X = X.drop(existing_cols)
        logger.info("Reduced Feature Count  : %d", X.shape[1])
    return X


#
# Function remove_lv_features
#

def remove_lv_features(model, X):
    r"""Remove low-variance features, fitting the variance mask on ``X`` (the training rows) and recording it in
    ``model.feature_map['lv_support']``.

    Parameters
    ----------
    model : alphapy.Model
        Model specifications for removing features.
    X : numpy array
        The feature matrix.

    Returns
    -------
    X_reduced : numpy array
        The reduced feature matrix.

    References
    ----------
    You can find more information on low-variance feature selection here [LV]_.

    .. [LV] http://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold

    """

    logger.info("Removing Low-Variance Features")

    # Extract model parameters

    lv_remove = model.specs['lv_remove']
    lv_threshold = model.specs['lv_threshold']

    # Remove low-variance features

    if lv_remove:
        logger.info("Low-Variance Threshold  : %.2f", lv_threshold)
        logger.info("Original Feature Count  : %d", X.shape[1])
        selector = VarianceThreshold(threshold=lv_threshold)
        selector.fit(X)
        support = selector.get_support()
        model.feature_map['lv_support'] = support
        X_reduced = X[:, support]
        feature_names = list(itertools.compress(model.feature_names, support))
        model.feature_names = feature_names       # later stages read the reduced names
        for algo in model.algolist:
            model.fnames_algo[algo] = feature_names
        logger.info("Reduced Feature Count   : %d", X_reduced.shape[1])
        assert X_reduced.shape[1] == len(feature_names), "Mismatched Features and Names"
    else:
        X_reduced = X
        logger.info("Skipping Low-Variance Features")

    return X_reduced
