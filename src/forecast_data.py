import numpy as np
import pandas as pd
import joblib

from src.architectures.model_architectures import *

test_X = pd.read_csv('./data/processed/test_X.csv')
test_ids = test_X[['id']]
testing_X = test_X.drop(columns=['id'])
time_series_forecasting_model = joblib.load('./src/models/model.pkl')
test_Y = time_series_forecasting_model.forecast_data(testing_X)
final_prediction = pd.concat([test_ids, pd.Series(np.squeeze(test_Y))], axis=1)
final_prediction.to_csv('./data/final/test_Y.csv')
