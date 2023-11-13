import pandas as pd
import joblib
import json

from src.architectures.model_architectures import (
    LinearRegressor,
    BoostedHybrid,
    ExponentialSmoothing,
    LightGBM,
)


model_config_file = open('config/model_configs.json')
model_configs = json.load(model_config_file)
forecasting_model_preset = model_configs['forecasting-model']
forecasting_model_configs = model_configs[forecasting_model_preset]
match forecasting_model_preset:
    case 'Linear Regression':
        time_series_forecasting_model = LinearRegressor
    case 'Boosted Hybrid':
        time_series_forecasting_model = BoostedHybrid
    case 'Exponential Smoothing':
        time_series_forecasting_model = ExponentialSmoothing
    case 'LightGBM':
        time_series_forecasting_model = LightGBM
    case _:
        time_series_forecasting_model = LinearRegressor

time_series_forecasting_model = time_series_forecasting_model(
    **forecasting_model_configs,
)

train_X = pd.read_csv('./data/processed/train_X.csv')
train_Y = pd.read_csv('./data/processed/train_Y.csv')
time_series_forecasting_model.fit_data(train_X, train_Y[['sales']])

joblib.dump(time_series_forecasting_model,
            f'src/models/{model_configs["model-name"]}.pkl')
