from src.architectures.feature_generator_architectures import *

feature_generator = BaseFeaturesGenerator()
feature_generator.preprocess_data()
train_X, train_Y = feature_generator.get_train_data()
test_X = feature_generator.get_test_data()

train_X.to_csv('./data/processed/train_X.csv')
train_Y.to_csv('./data/processed/train_Y.csv')
test_X.to_csv('./data/processed/test_X.csv')
