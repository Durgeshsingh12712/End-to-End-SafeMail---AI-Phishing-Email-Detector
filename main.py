import sys
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

if __name__ == "__main__":
    try:
        result = 10 / 0
    except Exception as e:
        raise EmailDetectionException(e, sys)