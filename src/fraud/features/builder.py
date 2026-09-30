"""Feature engineering module for real-time fraud scoring.

Implements FeatureBuilder which fits transformation state (frequency maps,
categorical categories) strictly on training data and applies consistent
transformations at training and serving time.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FREQ_COLS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "card1_addr1",
]

FREQ_ENCODED_NAMES = [f"{col}_fq" for col in FREQ_COLS]

ID_COLS = [f"id_0{i}" if i < 10 else f"id_{i}" for i in range(1, 39)]


class FeatureBuilder:
    """Transforms raw transaction DataFrames into engineered feature matrices."""

    def __init__(self, drop_high_missing: bool = False, missing_threshold: float = 0.90):
        self.drop_high_missing = drop_high_missing
        self.missing_threshold = missing_threshold
        self.is_fitted = False

        self.freq_maps: dict[str, dict[Any, int]] = {}
        self.cat_categories: dict[str, list[Any]] = {}
        self.dropped_cols: list[str] = []
        self.final_feature_names: list[str] = []
        self.feature_dtypes: dict[str, str] = {}

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn frequency maps, categorical dtypes, and column dropping logic from train data strictly."""
        train_df = train_df.copy()

        # Build card1_addr1 key for train
        c1 = train_df["card1"].astype(str) if "card1" in train_df else ""
        a1 = train_df["addr1"].astype(str) if "addr1" in train_df else ""
        train_df["card1_addr1"] = c1 + "_" + a1

        # Learn frequency maps
        self.freq_maps = {}
        for col in FREQ_COLS:
            if col in train_df.columns:
                counts = train_df[col].value_counts(dropna=True).to_dict()
                self.freq_maps[col] = counts
            else:
                self.freq_maps[col] = {}

        # Column dropping logic if specified
        cols_to_check = [c for c in train_df.columns if c not in ["TransactionID", "isFraud"]]
        if self.drop_high_missing:
            missing_ratios = train_df[cols_to_check].isnull().mean()
            self.dropped_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        else:
            self.dropped_cols = []

        # Transform train to capture categorical category lists strictly from train
        sample_transformed = self._transform_raw(train_df)

        # Capture categorical category lists strictly from train
        cat_cols = [
            c for c in sample_transformed.columns
            if pd.api.types.is_object_dtype(sample_transformed[c])
            or pd.api.types.is_categorical_dtype(sample_transformed[c])
            or pd.api.types.is_string_dtype(sample_transformed[c])
        ]
        self.cat_categories = {}
        for col in cat_cols:
            cats = [str(x) for x in sample_transformed[col].dropna().unique() if str(x) not in ["nan", "None", "<NA>", ""]]
            self.cat_categories[col] = sorted(cats)

        # Record final feature names and dtypes after transformation
        sample_final = self._apply_categoricals(sample_transformed.iloc[:10])
        self.final_feature_names = sample_final.columns.tolist()
        self.feature_dtypes = {c: str(sample_final[c].dtype) for c in self.final_feature_names}

        self.is_fitted = True
        return self

    def _transform_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal feature computation on input DataFrame."""
        col_dict = {}

        # 1. Base time features
        if "TransactionDT" in df.columns:
            dt = df["TransactionDT"].fillna(0)
            col_dict["day_index"] = (dt // 86400).astype("float32")
            col_dict["hour_of_day"] = ((dt // 3600) % 24).astype("float32")
            col_dict["day_of_week"] = ((dt // 86400) % 7).astype("float32")

        # 2. Amount features
        if "TransactionAmt" in df.columns:
            amt = df["TransactionAmt"].fillna(0).astype("float32")
            col_dict["log_TransactionAmt"] = np.log1p(np.maximum(amt, 0)).astype("float32")
            col_dict["amt_cents"] = (amt % 1.0).astype("float32")
            col_dict["amt_is_round"] = ((amt % 1.0) == 0).astype("float32")

        # 3. Email features
        p_email = df["P_emaildomain"].astype(str) if "P_emaildomain" in df.columns else pd.Series("", index=df.index)
        r_email = df["R_emaildomain"].astype(str) if "R_emaildomain" in df.columns else pd.Series("", index=df.index)

        p_split = p_email.str.split(".", n=1, expand=True)
        col_dict["P_emaildomain_provider"] = p_split[0].replace({"nan": None, "": None})
        col_dict["P_emaildomain_suffix"] = p_split[1].replace({"nan": None, "": None}) if p_split.shape[1] > 1 else None

        r_split = r_email.str.split(".", n=1, expand=True)
        col_dict["R_emaildomain_provider"] = r_split[0].replace({"nan": None, "": None})
        col_dict["R_emaildomain_suffix"] = r_split[1].replace({"nan": None, "": None}) if r_split.shape[1] > 1 else None

        p_str = df["P_emaildomain"].astype(str) if "P_emaildomain" in df.columns else pd.Series("", index=df.index)
        r_str = df["R_emaildomain"].astype(str) if "R_emaildomain" in df.columns else pd.Series("", index=df.index)

        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            valid_match = (
                df["P_emaildomain"].notna()
                & df["R_emaildomain"].notna()
                & (p_str == r_str)
                & (~p_str.isin(["nan", "None", ""]))
            ).astype("float32")
        else:
            valid_match = pd.Series(0.0, index=df.index, dtype="float32")
        col_dict["email_domain_match"] = valid_match

        # 4. Card & address combination
        c1 = df["card1"].astype(str) if "card1" in df.columns else pd.Series("", index=df.index)
        a1 = df["addr1"].astype(str) if "addr1" in df.columns else pd.Series("", index=df.index)
        df_card1_addr1 = c1 + "_" + a1

        # 5. Frequency encodings
        for col in FREQ_COLS:
            fq_col_name = f"{col}_fq"
            freq_map = self.freq_maps.get(col, {})
            if col == "card1_addr1":
                series = df_card1_addr1
            elif col in df.columns:
                series = df[col]
            else:
                series = pd.Series(index=df.index, dtype=object)

            mapped = series.map(freq_map).fillna(0).astype("float32")
            col_dict[fq_col_name] = mapped

        # 6. Identity missingness flags
        present_id_cols = [c for c in ID_COLS if c in df.columns]
        if present_id_cols:
            col_dict["id_missing_count"] = df[present_id_cols].isnull().sum(axis=1).astype("float32")
            col_dict["has_identity"] = (df[present_id_cols].notnull().any(axis=1)).astype("float32")
        else:
            col_dict["id_missing_count"] = float(len(ID_COLS))
            col_dict["has_identity"] = 0.0

        # 7. Copy over remaining raw features (excluding dropped, ID, target, and handled)
        ignore_cols = set(
            ["TransactionID", "isFraud", "TransactionDT"]
            + self.dropped_cols
        )
        for col in df.columns:
            if col not in ignore_cols and col not in col_dict:
                col_dict[col] = df[col]

        return pd.DataFrame(col_dict, index=df.index)

    def _apply_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast categorical columns using stored categories and ensure no object dtypes remain."""
        df = df.copy()
        for col in df.columns:
            if col in self.cat_categories:
                cats = self.cat_categories[col]
                series_str = df[col].astype(str).replace({"nan": None, "None": None, "<NA>": None})
                df[col] = pd.Categorical(series_str, categories=cats, ordered=False)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("float32")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into final feature matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureBuilder must be fit before transform can be called.")

        transformed = self._transform_raw(df)

        # Fast align columns with final_feature_names
        missing_cols = [c for c in self.final_feature_names if c not in transformed.columns]
        if missing_cols:
            missing_df = pd.DataFrame(np.nan, index=transformed.index, columns=missing_cols)
            transformed = pd.concat([transformed, missing_df], axis=1)

        transformed = self._apply_categoricals(transformed)
        transformed = transformed[self.final_feature_names]
        return transformed

    def save(self, joblib_path: str | Path, spec_path: str | Path) -> None:
        """Save feature builder model and JSON specification."""
        joblib_path = Path(joblib_path)
        spec_path = Path(spec_path)
        joblib_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, joblib_path)
        spec = {
            "final_feature_names": self.final_feature_names,
            "feature_dtypes": self.feature_dtypes,
            "dropped_cols": self.dropped_cols,
            "freq_cols": FREQ_COLS,
            "num_features": len(self.final_feature_names),
        }
        with spec_path.open("w", encoding="utf-8") as f:
            import json
            json.dump(spec, f, indent=2)

    @classmethod
    def load(cls, joblib_path: str | Path) -> "FeatureBuilder":
        """Load feature builder instance from disk."""
        return joblib.load(joblib_path)
