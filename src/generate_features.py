import json

from src.architectures.feature_generator_architectures import *


data_config_file = open('config/data_configs.json')
data_configs = json.load(data_config_file)
feature_generator_preset = data_configs['feature-generator']
match feature_generator_preset:
    case 'Base':
        feature_generator = BaseFeaturesGenerator()
    case _:
        feature_generator = BaseFeaturesGenerator()

feature_generator.preprocess_data()
train_X, train_Y = feature_generator.get_train_data()
test_X = feature_generator.get_test_data()

train_X.to_csv('./data/processed/train_X.csv')
train_Y.to_csv('./data/processed/train_Y.csv')
test_X.to_csv('./data/processed/test_X.csv')
