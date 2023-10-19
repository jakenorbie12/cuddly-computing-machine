import pandas as pd

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


class BaseFeaturesGenerator():
    def __init__(self) -> None:
        train_data_dir = Path('./data/original/train.csv')
        test_data_dir = Path('./data/original/test.csv')
        training_datatype_map = {
            'store_nbr': 'uint8',
            'family': 'category',
            'sales': 'float32',
            'onpromotion': 'uint64',
        }

        self.train_df = pd.read_csv(
            train_data_dir,
            dtype=training_datatype_map,
            parse_dates=['date'],
        )
        self.test_df = pd.read_csv(
            test_data_dir,
            dtype=training_datatype_map,
            parse_dates=['date'],
        )

    def preprocess_data(self) -> None:
        self.train_df['time'] = (self.train_df['date'] -
                                 min(self.train_df['date'])).dt.days
        self.train_df = pd.concat(
            [self.train_df, pd.get_dummies(self.train_df['family'])], axis=1
            )
        self.train_df = pd.concat(
            [self.train_df, pd.get_dummies(self.train_df['store_nbr'])], axis=1
            )
        self.train_df = self.train_df.drop(columns=[
                                                    'id',
                                                    'date',
                                                    'store_nbr',
                                                    'family',
                                                    ])

        self.train_X = self.train_df.drop(columns=['sales'])
        self.train_Y = self.train_df[['sales']]

        self.test_df['time'] = (self.test_df['date'] -
                                min(self.test_df['date'])).dt.days
        self.test_df = pd.concat(
            [self.test_df, pd.get_dummies(self.test_df['family'])], axis=1
            )
        self.test_df = pd.concat(
            [self.test_df, pd.get_dummies(self.test_df['store_nbr'])], axis=1
            )
        self.test_X = self.test_df.drop(columns=[
                                                    'date',
                                                    'store_nbr',
                                                    'family',
                                                    ])

    def get_train_data(self):
        return (self.train_X, self.train_Y)

    def get_test_data(self):
        return self.test_X
