import polars as pl
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from alphapy.globals import ModelType
from alphapy.model import first_fit
from alphapy.model import Model


class ShapeRecordingClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def fit(self, X, y):
        self.n_features_seen_ = X.shape[1]
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        return [self.classes_[0]] * X.shape[0]


def make_model():
    model = Model(
        {
            "algorithms": ["TEST"],
            "cv_folds": 2,
            "esr": 10,
            "model_type": ModelType.classification,
            "n_jobs": 1,
            "scorer": "accuracy",
            "seed": 13,
            "shuffle": False,
            "split": 0.25,
            "ts_option": False,
            "verbosity": 0,
        }
    )
    model.X_train = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 1, 0, 0, 1, 0],
        }
    )
    model.y_train = pl.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
    model.feature_names = ["a", "b", "c"]
    return model


def test_first_fit_applies_univariate_support():
    model = make_model()
    model.feature_map["support_uni"] = [True, False, True]

    model = first_fit(model, "TEST", ShapeRecordingClassifier())

    assert model.estimators["TEST"].n_features_seen_ == 2
    assert model.fnames_algo["TEST"] == ["a", "c"]


def test_first_fit_prefers_lofo_support_over_univariate_support():
    model = make_model()
    model.feature_map["support_uni"] = [True, True, True]
    model.feature_map["support_lofo", "TEST"] = [False, True, False]

    model = first_fit(model, "TEST", ShapeRecordingClassifier())

    assert model.estimators["TEST"].n_features_seen_ == 1
    assert model.fnames_algo["TEST"] == ["b"]


def test_first_fit_uses_univariate_support_when_lofo_selects_no_features():
    model = make_model()
    model.feature_map["support_uni"] = [True, False, True]
    model.feature_map["support_lofo", "TEST"] = [False, False, False]

    model = first_fit(model, "TEST", ShapeRecordingClassifier())

    assert model.estimators["TEST"].n_features_seen_ == 2
    assert model.fnames_algo["TEST"] == ["a", "c"]
