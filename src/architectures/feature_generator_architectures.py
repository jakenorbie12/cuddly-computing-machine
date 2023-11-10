import json
import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path


class DataLoader():
    def __init__(self) -> None:
        self.train_data_dir = Path('./data/original/train.csv')
        self.test_data_dir = Path('./data/original/test.csv')
        self.oil_data_dir = Path('./data/original/oil.csv')
        self.events_data_dir = Path('./data/original/holidays_events.csv')

        self.training_datatype_map = {
            'store_nbr': 'uint8',
            'family': 'category',
            'sales': 'float32',
            'onpromotion': 'uint8',
        }
        self.testing_datatype_map = {
            'store_nbr': 'uint8',
            'family': 'category',
            'onpromotion': 'uint8',
        }

    def load_train_df(self) -> pd.DataFrame:
        return pd.read_csv(
            self.train_data_dir,
            dtype=self.training_datatype_map,
            parse_dates=['date'],
        )

    def load_test_df(self) -> pd.DataFrame:
        return pd.read_csv(
            self.test_data_dir,
            dtype=self.testing_datatype_map,
            parse_dates=['date'],
        )

    def load_oil_df(self) -> pd.DataFrame:
        oil_df = pd.read_csv(
            self.oil_data_dir,
            parse_dates=['date'],
        )
        oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(0.0)
        return oil_df

    def load_events_df(self) -> pd.DataFrame:
        events_df = pd.read_csv(
            self.events_data_dir,
            parse_dates=['date'],
        )
        events_df = events_df.loc[
            (events_df.transferred == False) &
            ((events_df.type == 'Holiday') |
             (events_df.type == 'Additional') |
             (events_df.type == 'Bridge')) &
            (events_df.locale != 'Local')
        ]
        events_df = events_df.drop_duplicates(subset='date')
        events_df = events_df[['date', 'type']]
        return events_df


class FeaturesGenerator(ABC):
    def __init__(self) -> None:
        self.dataloader = DataLoader()
        self.train_df = self.dataloader.load_train_df()
        self.test_df = self.dataloader.load_test_df()

        data_splitting_models = ['Exponential-Smoothing', 'lightGBM']
        model_configs_dir = Path('./config/model_configs.json')
        with open(model_configs_dir, 'r') as fp:
            forecasting_model = json.load(fp)['forecasting-model']
            if forecasting_model in data_splitting_models:
                self.needs_data_splitting = 1
            else:
                self.needs_data_splitting = 0

    @abstractmethod
    def preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def add_feature_time(self, df) -> pd.DataFrame:
        df['time'] = (df['date'] - min(df['date'])).dt.days
        return df

    def add_feature_family(self, df) -> pd.DataFrame:
        if not self.needs_data_splitting:
            return pd.concat(
                [df, pd.get_dummies(df['family'])],
                axis=1,
            )
        else:
            return df

    def add_feature_store_number(self, df) -> pd.DataFrame:
        if not self.needs_data_splitting:
            return pd.concat(
                [df, pd.get_dummies(df['store_nbr'], prefix='s_nbr')],
                axis=1,
            )
        else:
            return df

    def add_feature_oil_price(self, df, oil_df) -> pd.DataFrame:
        df = df.join(
            oil_df.set_index('date'), on='date', validate='m:1'
        )
        df['dcoilwtico'] = df['dcoilwtico'].fillna(0.0)
        return df

    def add_feature_holidays(self, df, holiday_df) -> pd.DataFrame:
        df = df.join(
            holiday_df.set_index('date'), on='date', validate='m:1'
        )
        df = pd.concat(
            [df,
             pd.get_dummies(df['type'])
             ],
            axis=1
        )
        df = df.drop(columns=['type'])
        return df

    def add_feature_day_of_week(self, df) -> pd.DataFrame:
        return pd.concat(
            [df,
             pd.get_dummies(df['date'].dt.dayofweek, prefix='DoW')
             ],
            axis=1
        )

    def add_feature_day_of_month(self, df) -> pd.DataFrame:
        return pd.concat(
            [df,
             pd.get_dummies(df['date'].dt.day, prefix='DoM')
             ],
            axis=1
        )

    def add_feature_earthquake_relevancy(self, df) -> pd.DataFrame:
        df.loc[
            (pd.Timestamp(2016, 4, 12) <= df['date']) &
            (pd.Timestamp(2016, 4, 15) >= df['date']), 'before_EQ'
         ] = 1
        df['before_EQ'] = df['before_EQ'].fillna(0)
        df.loc[
            (pd.Timestamp(2016, 4, 16) <= df['date']) &
            (pd.Timestamp(2016, 4, 26) >= df['date']), 'after_EQ'
         ] = 1
        df['after_EQ'] = df['after_EQ'].fillna(0)
        return df

    def drop_unneeded_columns(self, df):
        return df.drop(columns=[
            'date',
            'store_nbr',
            'family',
        ])

    def return_training_data(self, train_df):
        X = train_df.drop(columns=['id', 'sales'])
        Y = train_df[['sales']]
        return X, Y

    def return_testing_data(self, test_df):
        X = test_df
        return X

    @abstractmethod
    def preprocess_data(self) -> None:
        self.train_df = self.preprocessing_pipeline(self.train_df)
        self.train_X, self.train_Y = self.return_training_data(self.train_df)

        self.test_df = self.preprocessing_pipeline(self.test_df)
        self.test_X = self.return_testing_data(self.test_df)

    @abstractmethod
    def get_train_data(self):
        return (self.train_X, self.train_Y)

    @abstractmethod
    def get_test_data(self):
        return self.test_X


class SimpleFeaturesGenerator(FeaturesGenerator):
    def __init__(self) -> None:
        super().__init__()

    def preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_feature_time(df)
        df = self.add_feature_family(df)
        df = self.add_feature_store_number(df)
        df = self.drop_unneeded_columns(df)
        return df

    def preprocess_data(self) -> None:
        return super().preprocess_data()

    def get_train_data(self):
        return super().get_train_data()

    def get_test_data(self):
        return super().get_test_data()


class AllFeaturesGenerator(FeaturesGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.oil_df = self.dataloader.load_oil_df()
        self.events_df = self.dataloader.load_events_df()

    def preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_feature_time(df)
        df = self.add_feature_family(df)
        df = self.add_feature_store_number(df)
        df = self.add_feature_oil_price(df, self.oil_df)
        df = self.add_feature_holidays(df, self.events_df)
        df = self.add_feature_day_of_week(df)
        df = self.add_feature_day_of_month(df)
        df = self.add_feature_earthquake_relevancy(df)
        df = self.drop_unneeded_columns(df)
        return df

    def preprocess_data(self) -> None:
        return super().preprocess_data()

    def get_train_data(self):
        return super().get_train_data()

    def get_test_data(self):
        return super().get_test_data()


class CustomFeaturesGenerator(FeaturesGenerator):
    def __init__(self) -> None:
        super().__init__()

        feature_configs_dir = Path('./config/data_configs.json')
        with open(feature_configs_dir, 'r') as fp:
            self.features_configs = json.load(fp)['custom-feature-generator']

    def preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.features_configs['index']:
            df = self.add_feature_time(df)
        if self.features_configs['family']:
            df = self.add_feature_family(df)
        if self.features_configs['store-number']:
            df = self.add_feature_store_number(df)
        if self.features_configs['oil-price']:
            self.oil_df = self.dataloader.load_oil_df()
            df = self.add_feature_oil_price(df, self.oil_df)
        if self.features_configs['holidays-and-events']:
            self.events_df = self.dataloader.load_events_df()
            df = self.add_feature_holidays(df, self.events_df)
        if self.features_configs['day-of-week']:
            df = self.add_feature_day_of_week(df)
        if self.features_configs['day-of-month']:
            df = self.add_feature_day_of_month(df)
        if self.features_configs['earthquake']:
            df = self.add_feature_earthquake_relevancy(df)
        df = self.drop_unneeded_columns(df)
        return df

    def preprocess_data(self) -> None:
        return super().preprocess_data()

    def get_train_data(self):
        return super().get_train_data()

    def get_test_data(self):
        return super().get_test_data()
