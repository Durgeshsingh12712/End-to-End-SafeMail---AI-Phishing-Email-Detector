import os
import sys
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from emailDetector.entity.config_entity import ModelTrainerConfig
from emailDetector.entity.artifact_entity import ModelTrainerArtifact
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.utils.utils import save_object, load_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting Model Training")

            # Load Data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            # Load Vectorizer
            vectorizer = load_object(file_path = self.config.vectorizer_path)

            # Prepare Data
            X_train = train_df['text']
            y_train = train_df['target']
            X_test = test_df['text']
            y_test = test_df['target']

            # Transform Text data
            X_train_transformed = vectorizer.transform(X_train)
            X_test_transformed = vectorizer.transform(X_test)

            # Train Model 
            model = MultinomialNB(alpha=0.1, fit_prior=True)
            model.fit(X_train_transformed, y_train)

            # Make Predictions
            train_preds = model.predict(X_train_transformed)
            test_preds = model.predict(X_test_transformed)

            # Calculate Accuracy
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)

            logger.info(f"Training Accuracy: {train_accuracy:.4f}")
            logger.info(f"Testing Accuracy: {test_accuracy:.4f}")

            #Save Model
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            save_object(file_path = model_path, obj= model)

            logger.info("Model Training Completed Successfully")

            return ModelTrainerArtifact(trained_model_path= model_path)
        
        except Exception as e:
            raise EmailDetectionException(e, sys)
