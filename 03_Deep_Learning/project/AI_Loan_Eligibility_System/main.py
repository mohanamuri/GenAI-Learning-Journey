"""
Project Entry Point
"""

from config.config import DATASET_PATH
from data.dataset import load_dataset
from utils.preprocessing import normalize_features

dataset = load_dataset(DATASET_PATH)

print("\nFirst Five Records")
print(dataset.head())

X = dataset.drop("approved", axis=1)

X = normalize_features(X)

print("\nNormalized Features")
print(X.head())