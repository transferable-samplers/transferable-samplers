"""
Backward compatibility module for TensorDataset.
The functionality has been merged into PeptideDataset in peptides_dataset.py.
"""

from src.data.datasets.peptides_dataset import PeptideDataset

TensorDataset = PeptideDataset
