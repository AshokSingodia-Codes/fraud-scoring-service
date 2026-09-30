"""CLI script to run full pipeline evaluation (Stages 7, 8, 9)."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from scripts.evaluate_pipeline import run_pipeline_evaluation

if __name__ == "__main__":
    run_pipeline_evaluation()
