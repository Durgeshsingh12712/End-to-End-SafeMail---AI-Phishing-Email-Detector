import sys
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        logger.info("Starting The Training Pipeline")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logger.info("Training Pipeline Completed Successfully")

    except Exception as e:
        logger.error("Training Pipeline Failed")
        raise EmailDetectionException(e, sys)