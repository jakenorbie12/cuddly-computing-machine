import pandas as pd

from darts.models import LightGBMModel
from darts.models import ExponentialSmoothing as ExponentialSmoothingModel
from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor


class Splitter():
    def __init__(self) -> None:
        pass

    def split_data(self, data_X, data_Y):
        data = pd.concat([data_X, data_Y], axis=1)
        data_subsets = {}
        subset_name_definer = "{family}-{store_number}"
        for family in data['family'].unique():
            for store_number in data['store_nbr'].unique():
                subset_definition = subset_name_definer.format(
                    family=family,
                    store_number=store_number,
                )
                data_subset = data.loc[
                    (data.family == family) &
                    (data.store_nbr == store_number)
                ]
                data_subset_timeseries = TimeSeries.from_dataframe(
                    data_subset,
                    time_col='date',
                    value_cols='sales',
                    fill_missing_dates=True,
                    freq='D',
                )
                data_subsets[subset_definition] = fill_missing_values(
                    data_subset_timeseries
                )
        return data_subsets

    def fit_split_data(self, model, split_data):
        subset_models = {}
        for subset, subset_data in split_data.items():
            subset_model = model.fit(subset_data)
            subset_models[subset] = subset_model
        return subset_models

    def forecast_split_data(self, split_models, split_X):
        Y = pd.DataFrame()
        for subset, X in split_X.items():
            y_pred = pd.DataFrame(
                split_models[subset].predict(len(X.index)),
                index=X.index,
                columns=['sales'],
            )
            y_pred = y_pred.stack().squeeze()
            Y = pd.concat([Y, y_pred])
        return Y


class LinearRegressor():
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit_data(self, X, Y) -> None:
        self.model.fit(X, Y)

    def forecast_data(self, X):
        Y = self.model.predict(X)
        Y[Y < 0] = 0
        return Y


class BoostedHybrid():
    def __init__(self,
                 model_1=LinearRegression(),
                 model_2=XGBRegressor()) -> None:
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

    def fit_data(self, X_2, Y):
        dp = DeterministicProcess(index=X_2.index, order=1)
        X_1 = dp.in_sample()

        self.model_1.fit(X_1, Y)
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=Y.columns,
        )
        y_residuals = Y - y_pred
        y_residuals = y_residuals.stack().squeeze()

        self.model_2.fit(X_2, y_residuals)
        self.y_columns = Y.columns
        self.y_pred = y_pred
        self.y_residuals = y_residuals

    def forecast_data(self, X_2):
        dp = DeterministicProcess(index=X_2.index, order=1)
        X_1 = dp.in_sample()

        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()

        y_pred += self.model_2.predict(X_2)
        y_pred[y_pred < 0] = 0
        return y_pred.unstack()


class ExponentialSmoothing():
    def __init__(self, seasonal_periods="infer") -> None:
        if seasonal_periods == "infer":
            self.model = ExponentialSmoothingModel()
        else:
            self.model = ExponentialSmoothingModel(
                seasonal_periods=seasonal_periods
            )
        self.splitter = Splitter()

    def fit_data(self, X, Y) -> None:
        split_data = self.splitter.split_data(X, Y)
        self.split_models = self.splitter.fit_split_data(self.model,
                                                         split_data,
                                                         )

    def forecast_data(self, X):
        split_test_data = self.splitter.split_data(X, None)
        Y = self.splitter.forecast_split_data(self.split_models,
                                              split_test_data)
        Y[Y < 0] = 0
        return Y


class LightGBM():

    def __init__(self, lags=14) -> None:
        self.model = LightGBMModel(lags=lags)
        self.splitter = Splitter()

    def fit_data(self, X, Y) -> None:
        split_data = self.splitter.split_data(X, Y)
        self.split_models = self.splitter.fit_split_data(self.model,
                                                         split_data,
                                                         )

    def forecast_data(self, X):
        split_test_data = self.splitter.split_data(X, None)
        Y = self.splitter.forecast_split_data(self.split_models,
                                              split_test_data)
        Y[Y < 0] = 0
        return Y
