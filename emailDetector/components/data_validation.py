import os
import sys
import pandas as pd
from pathlib import Path

from emailDetector.entity.config_entity import DataValidationConfig
from emailDetector.entity.artifact_entity import DataValidationArtifact
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            
            # Read data
            data_path = os.path.join(self.config.unzip_data_dir, "email.csv")
            if not os.path.exists(data_path):
                logger.error(f"Data file not found at {data_path}")
                return False
                
            df = pd.read_csv(data_path, encoding="ISO-8859-1")
            all_cols = list(df.columns)
            
            # Check if required columns exist
            all_schema = self.config.required_columns
            
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    logger.error(f"Required column {col} not found in dataset")
                    return validation_status
            
            validation_status = True
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            logger.info("Data validation completed successfully")
            return validation_status
            
        except Exception as e:
            raise EmailDetectionException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")
            validation_status = self.validate_all_columns()
            
            return DataValidationArtifact(validation_status=validation_status)
            
        except Exception as e:
            raise EmailDetectionException(e, sys)
        