import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from emailDetector.entity.config_entity import DataTransformationConfig
from emailDetector.entity.artifact_entity import DataTransformationArtifact
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.utils.utils import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer(self):
        try:
            vectorizer = TfidfVectorizer(
                # Vocabulary Parameters
                min_df=2,
                max_df=0.95,
                max_features=10000,

                # Text Preprocessing
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',

                # N-gram Paramenter
                ngram_range=(1, 2),

                # TF-IDF Parameters
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,

                # Performance
                dtype=np.float32
            )

            logger.info("TF-IDF vectorizer configured successfully")
            return vectorizer
        except Exception as e:
            raise EmailDetectionException(e, sys)
        
    def get_feature_stats(self, vectorizer: TfidfVectorizer, X_transformed) -> dict:
        """Get Statistics about the transformed Feature"""
        try:
            stats = {
                'vocabulary_size': len(vectorizer.vocabulary_),
                'feature_metrix_shape': X_transformed.shape,
                'sparsity': 1.0 - (X_transformed.nnz / (X_transformed.shape[0] * X_transformed.shape[1])),
                'top_features': list(vectorizer.get_feature_names_out()[:10]) if hasattr(vectorizer, 'get_feature_name_out') else[]
            }

            logger.info(f"Feature Statistics: {stats}")
            return stats
        
        except Exception as e:
            logger.warning(f"Could not compute feature statistics: {e}")
            return {}
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting Data Transformation")

            # Load Data
            df = pd.read_csv(self.config.data_path)

            # Separate Features and Target
            X = df['Message']
            y = df['Category'].astype(int)

            # Split Data

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )

            # Initiatelize and fit Vectorizer
            vectorizer = self.get_data_transformer()
            X_train_transformed = vectorizer.fit_transform(X_train)
            X_test_transformed = vectorizer.transform(X_test)

            self.get_feature_stats(vectorizer, X_train_transformed)

            # Save train and test data
            train_df = pd.DataFrame({
                'text': X_train.values,
                'target': y_train.values
            })

            test_df = pd.DataFrame({
                'text': X_test.values,
                'target': y_test.values
            })

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            # Save Vectorizer
            save_object(
                file_path = self.config.vectorizer_path,
                obj = vectorizer
            )

            logger.info("Data Transformation completed Successfully")

            return DataTransformationArtifact(
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
                vectorizer_path=self.config.vectorizer_path
            )
        
        except Exception as e:
            raise EmailDetectionException(e, sys)