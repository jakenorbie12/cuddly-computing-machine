import joblib
import pandas as pd

from src.utils import evaluate_error

train_X = pd.read_csv('./data/processed/train_X.csv')
train_Y = pd.read_csv('./data/processed/train_Y.csv')
time_series_forecasting_model = joblib.load('./src/models/model.pkl')

Y_pred = time_series_forecasting_model.forecast_data(train_X)
error = evaluate_error(train_Y[['sales']], Y_pred)
print(f'Training root mean square log error is: {error:.2f}')
