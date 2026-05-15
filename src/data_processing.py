import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_store = feature_store
        logger.info("Data Processing initialized.")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info(f"Data loaded successfully from {self.train_data_path} and {self.test_data_path}.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(str(e), sys)

    def preprocess_data(self):
        try:
            self.data.drop_duplicates(inplace=True)
            self.test_data.drop_duplicates(inplace=True)

            for df in [self.data, self.test_data]:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)

            self.X_train = self.data.drop(columns=["Outcome"])
            self.y_train = self.data["Outcome"]

            self.X_test = self.test_data.drop(columns=["Outcome"])
            self.y_test = self.test_data["Outcome"]

            logger.info("Data preprocessing done.")
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException(str(e), sys)

    def scale_data(self):
        try:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            logger.info("Data scaling done.")
        except Exception as e:
            logger.error(f"Error while scaling data: {e}")
            raise CustomException(str(e), sys)

    def store_features_in_redis(self):
        try:
            feature_cols = self.data.drop(columns=["Outcome"]).columns.tolist()
            batch_data = {}

            for idx, row in self.data.iterrows():
                features = {col: row[col] for col in feature_cols}
                features["Outcome"] = row["Outcome"]
                batch_data[idx] = features

            self.feature_store.store_batch_features(batch_data)
            logger.info("Features stored in Redis feature store.")
        except Exception as e:
            logger.error(f"Error while storing features in Redis: {e}")
            raise CustomException(str(e), sys)

    def retrieve_feature_from_redis(self, row_index):
        features = self.feature_store.get_features(row_index)
        if features:
            return features
        return None

    def run(self):
        try:
            logger.info("Data Processing Pipeline started.")
            self.load_data()
            self.preprocess_data()
            self.scale_data()
            self.store_features_in_redis()
            logger.info("Data Processing Pipeline finished.")
        except Exception as e:
            logger.error(f"Error in Data Processing Pipeline: {e}")
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()

    print(data_processor.retrieve_feature_from_redis(0))