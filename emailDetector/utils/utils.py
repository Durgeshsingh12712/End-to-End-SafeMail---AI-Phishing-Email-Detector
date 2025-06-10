import os
import sys
import yaml
import pickle
import json
import joblib
import pandas as pd
from pathlib import Path
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError

from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise EmailDetectionException(e, sys)

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        raise EmailDetectionException(e, sys)

@ensure_annotations
def save_object(file_path: str, obj):
    """
    Save object as pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logger.info(f"Object saved at {file_path}")
    except Exception as e:
        raise EmailDetectionException(e, sys)

@ensure_annotations
def load_object(file_path: str):
    """
    Load object from pickle file
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise EmailDetectionException(e, sys)

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"json file saved at: {path}")
    except Exception as e:
        raise EmailDetectionException(e, sys)

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    """
    try:
        with open(path) as f:
            content = json.load(f)
        
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise EmailDetectionException(e, sys)