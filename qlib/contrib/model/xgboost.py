# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter


class XGBModel(Model, FeatureInt):
    """XGBModel Model"""

    def __init__(self, **kwargs):
        self._params = {}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("XGBoost doesn't support multi-label training")

        x_train_values = np.asarray(x_train.values, dtype=np.float32)
        x_valid_values = np.asarray(x_valid.values, dtype=np.float32)
        x_train_values[~np.isfinite(x_train_values)] = np.nan
        x_valid_values[~np.isfinite(x_valid_values)] = np.nan

        train_mask = np.isfinite(y_train_1d)
        valid_mask = np.isfinite(y_valid_1d)
        if not train_mask.all():
            x_train_values = x_train_values[train_mask]
            y_train_1d = y_train_1d[train_mask]
        if not valid_mask.all():
            x_valid_values = x_valid_values[valid_mask]
            y_valid_1d = y_valid_1d[valid_mask]

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        if w_train is not None and not train_mask.all():
            w_train = np.asarray(w_train)[train_mask]
        if w_valid is not None and not valid_mask.all():
            w_valid = np.asarray(w_valid)[valid_mask]

        dtrain = xgb.DMatrix(x_train_values, label=y_train_1d, weight=w_train)
        dvalid = xgb.DMatrix(x_valid_values, label=y_valid_1d, weight=w_valid)
        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test_values = np.asarray(x_test.values, dtype=np.float32)
        x_test_values[~np.isfinite(x_test_values)] = np.nan
        return pd.Series(self.model.predict(xgb.DMatrix(x_test_values)), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)
