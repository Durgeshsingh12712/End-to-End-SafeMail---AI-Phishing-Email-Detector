from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    raw_data_path: Path

@dataclass
class DataValidationArtifact:
    validation_status: bool

@dataclass
class DataTransformationArtifact:
    train_data_path: Path
    test_data_path: Path
    vectorizer_path: Path
    
@dataclass
class ModelTrainerArtifact:
    trained_model_path: Path