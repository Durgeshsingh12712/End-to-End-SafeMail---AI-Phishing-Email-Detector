from emailDetector.logging.logger import logger

def division(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        logger.error(f"Division by zero error: {e}")
        return None
if __name__ == "__main__":

    result = division(1, 0)  
    if result is not None:
        logger.info(f"Result of division: {result}")
        print(f"Result: {result}")
    else:
        print("Division failed due to division by zero.")