import os
import sys
import zipfile
import urllib.request as request
import pandas as pd
from pathlib import Path

from emailDetector.entity.config_entity import DataIngestionConfig
from emailDetector.entity.artifact_entity import DataIngestionArtifact
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url = self.config.source_url,
                    filename = self.config.local_data_file
                )
                logger.info(f"Downloaded file: {filename} with info: {headers} ")
            else:
                logger.info("File already exists")
        except Exception as e:
            raise EmailDetectionException(e, sys)
    
    def extract_zip_file(self):
        """
        Extract Zip file into Data Directory
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info("Extracted Zip file Successfully")
        except Exception as e:
            raise EmailDetectionException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting Data Ingestion")
            self.download_file()
            self.extract_zip_file()

            # Load the Data
            df = pd.read_csv(self.config.raw_data_path, encoding="ISO-8859-1")

            # Drop Unnecessary Column
            columns_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
            df = df.drop(columns_to_remove, axis= 1, errors='ignore')

            # Drop Duplicates
            df = df.drop_duplicates()

            # Convert Lables to Numerical
            df.loc[df["Category"] == "phishing", "Category"] = 0
            df.loc[df["Category"] == "safe", "Category"] = 1

            # Save Preprocess Data
            df.to_csv(self.config.raw_data_path, index=False)

            logger.info("Data Ingestion Completed Successfully")
            return DataIngestionArtifact(raw_data_path=self.config.raw_data_path)
        
        except Exception as e:
            raise EmailDetectionException(e, sys)