import pandas as pd
import pickle
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, feature_store: RedisFeatureStore, model_save_path="artifacts/models/"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info("Model Training initialized.")

    def load_data_from_redis(self):
        try:
            row_indexes = self.feature_store.get_all_entity_ids()

            data = []
            for idx in row_indexes:
                features = self.feature_store.get_features(idx)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"Features not found for index: {idx}")

            logger.info(f"{len(data)} records loaded from Redis.")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error while loading data from Redis: {e}")
            raise CustomException(str(e), sys)

    def prepare_data(self):
        try:
            df = self.load_data_from_redis()

            df = df.fillna(df.median(numeric_only=True))

            X = df.drop(columns=["Outcome"])
            y = df["Outcome"]

            logger.info("Data preparation completed.")
            return X, y
        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys)

    def train_and_evaluate(self, X, y):
        try:
            self.model = LogisticRegression()
            self.model.fit(X, y)

            train_acc = accuracy_score(y, self.model.predict(X))
            logger.info(f"Train Accuracy: {train_acc}")

            logger.info(f"Confusion Matrix:\n{confusion_matrix(y, self.model.predict(X))}")
            logger.info(f"Classification Report:\n{classification_report(y, self.model.predict(X))}")

            self.save_model()

            return train_acc
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException(str(e), sys)

    def save_model(self):
        try:
            model_path = os.path.join(self.model_save_path, "logistic_regression_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved at {model_path}")
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException(str(e), sys)

    def run(self):
        try:
            logger.info("Model Training Pipeline started.")
            X, y = self.prepare_data()
            self.train_and_evaluate(X, y)
            logger.info("Model Training Pipeline finished.")
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline: {e}")
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()