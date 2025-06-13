import sys
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

from emailDetector.components.data_ingestion import DataIngestion
from emailDetector.components.data_validation import DataValidation
from emailDetector.components.data_transformation import DataTransformation
from emailDetector.components.model_trainer import ModelTrainer
from emailDetector.components.model_evaluation import ModelEvaluation

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

            # Data Transformaion
            logger.info(">>>>>>> Stage 03 : Data Transformation <<<<<<<")
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            print(data_transformation_artifact)
            logger.info(">>>>>>> Stage 03 : Data Transformation Completed <<<<<<<")

            # Model Trainer
            logger.info(">>>>>>> Stage 04 : Model Trainer <<<<<<<")
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config= model_trainer_config)
            model_trainer_artifact = model_trainer.train()
            print(model_trainer_artifact)
            logger.info(">>>>>>> Stage 04 : Model Trainer Completed <<<<<<<<")

            # Model Evaluation
            logger.info(">>>>>>> Stage 05 : Model Evaluation <<<<<<<")
            model_evaluation_config = config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_artifact = model_evaluation.evaluate_model()
            print(model_evaluation_artifact)
            logger.info(">>>>>>> Stage 0 : Model Evaluation Completed Successfully <<<<<<<")

            logger.info(">>>>>>> Training Pipeline Completed Successfully <<<<<<<")


        except Exception as e:
            logger.error("Training Pipeline Failed")
            raise EmailDetectionException(e, sys)
        
