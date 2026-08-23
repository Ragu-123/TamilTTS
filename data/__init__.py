"""TamilTTSv2 data package."""
from data.dataset import DirectParquetTamilDataset, build_tamil_datasets, tamil_tts_collate_fn

__all__ = ["DirectParquetTamilDataset", "tamil_tts_collate_fn", "build_tamil_datasets"]
