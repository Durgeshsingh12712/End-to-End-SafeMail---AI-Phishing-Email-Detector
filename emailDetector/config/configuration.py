import sys
from emailDetector.constants.constants import *
from emailDetector.utils.utils import read_yaml, create_directories
from emailDetector.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
)
from emailDetector.exception.exception import EmailDetectionException

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([self.config.artifacts_root])
        except Exception as e:
            raise EmailDetectionException(e, sys)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.data_ingestion
            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                source_url = config.source_url,
                local_data_file = config.local_data_file,
                unzip_dir = config.unzip_dir,
                raw_data_path = config.raw_data_path
            )

            return data_ingestion_config
        except Exception as e:
            raise EmailDetectionException(e, sys)
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config.data_validation
            create_directories([config.root_dir])
            
            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                unzip_data_dir=config.unzip_data_dir,
                status_file=config.status_file,
                required_columns=config.required_columns
            )
            return data_validation_config
        except Exception as e:
            raise EmailDetectionException(e, sys)