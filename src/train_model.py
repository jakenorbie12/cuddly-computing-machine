import pandas as pd
import joblib

from src.architectures.model_architectures import *

train_X = pd.read_csv('./data/processed/train_X.csv')
train_Y = pd.read_csv('./data/processed/train_Y.csv')
time_series_forecasting_model = LinearRegressor()
time_series_forecasting_model.fit_data(train_X, train_Y[['sales']])

joblib.dump(time_series_forecasting_model, 'src/models/model.pkl')
