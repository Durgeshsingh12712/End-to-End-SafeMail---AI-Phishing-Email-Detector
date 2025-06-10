from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    raw_data_path: Path