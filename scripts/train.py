"""CLI script to run model training and tuning (Stage 6)."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.fraud.models.train_lgbm import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
