#!/usr/bin/env bash
# Download script for Kaggle IEEE-CIS Fraud Detection dataset (Stage 1)
set -e

mkdir -p data/raw
echo "Downloading IEEE-CIS dataset via Kaggle CLI..."
kaggle competitions download -c ieee-fraud-detection -p data/raw

if [ -f data/raw/ieee-fraud-detection.zip ]; then
    echo "Unzipping dataset files..."
    unzip -o data/raw/ieee-fraud-detection.zip -d data/raw/
fi

echo "Data download completed. Files in data/raw/:"
ls -lh data/raw/
