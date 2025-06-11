import sys
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

from emailDetector.components.data_ingestion import DataIngestion
from emailDetector.components.data_validation import DataValidation

from emailDetector.config.configuration import ConfigurationManager

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info(">>>>>>> Training Pipeline Started <<<<<<<")
            config_manager = ConfigurationManager()

            # Data Ingestion
            logger.info(">>>>>>> Stage 01 : Data Ingestion <<<<<<<")
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            print(data_ingestion_artifact)
            logger.info(">>>>>>> Stage 01 : Data Ingestion Completed <<<<<<<")

            # Data Validation
            logger.info(">>>>>>> Stage 02 : Data Validation <<<<<<<")
            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            print(data_validation_artifact)
            logger.info(">>>>>> Stage 02: Data Validation Completed <<<<<<")


        except Exception as e:
            logger.error("Training Pipeline Failed")
            raise EmailDetectionException(e, sys)
        
