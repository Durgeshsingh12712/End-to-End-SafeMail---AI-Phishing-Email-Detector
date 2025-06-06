import sys
from emailDetector.logging.logger import logger

def error_message_detail(error, error_detail: sys):
    """
    Create detailed error message with file name and line number
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class EmailDetectionException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)
        
    def __str__(self):
        return self.error_message