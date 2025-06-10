import sys
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

from emailDetector.components.data_ingestion import DataIngestion


from emailDetector.config.configuration import ConfigurationManager

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info(">>>>>>> Training Pipeline Started <<<<<<<")

            # Data Ingestion
            logger.info(">>>>>>> Stage 01 : Data Ingestion <<<<<<<")
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            print(data_ingestion_artifact)
            logger.info(">>>>>>> Stage 01 : Data Ingestion Completed <<<<<<<")

        except Exception as e:
            logger.error("Training Pipeline Failed")
            raise EmailDetectionException(e, sys)
        
