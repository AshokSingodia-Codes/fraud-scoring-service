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
            or isinstance(sample_transformed[c].dtype, pd.CategoricalDtype)
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
        c1 = df["card1"].astype(str).fillna("nan") if "card1" in df.columns else pd.Series("nan", index=df.index)
        a1 = df["addr1"].astype(str).fillna("nan") if "addr1" in df.columns else pd.Series("nan", index=df.index)
        df_card1_addr1 = c1 + "_" + a1
        col_dict["card1_addr1"] = df_card1_addr1

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
        missing_id_cols_count = len(ID_COLS) - len(present_id_cols)
        if present_id_cols:
            col_dict["id_missing_count"] = (df[present_id_cols].isnull().sum(axis=1) + missing_id_cols_count).astype("float32")
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

    def _init_fast_structures(self) -> None:
        """Initialize fast lookup indices and mappings for single-row scoring."""
        if getattr(self, "_initialized_fast", False):
            return
        self._col_to_idx = {name: i for i, name in enumerate(self.final_feature_names)}
        self._cat_to_code = {
            col: {cat: float(code) for code, cat in enumerate(cats)}
            for col, cats in self.cat_categories.items()
        }
        self._ignore_cols = set(
            ["TransactionID", "isFraud", "TransactionDT"] + self.dropped_cols
            + ["hour_of_day", "day_of_week", "log_TransactionAmt", "amt_cents", "amt_is_round"]
        )
        self._initialized_fast = True

    def transform_dict_to_numpy(
        self, payload: dict[str, Any], out: np.ndarray | None = None
    ) -> np.ndarray:
        """Fast-path feature extraction mapping a transaction dictionary to a 2D float32 numpy row.

        Reuses a preallocated numpy buffer, avoiding DataFrame overhead and per-column loops.
        """
        self._init_fast_structures()
        num_cols = len(self.final_feature_names)
        if out is None:
            out = np.full((1, num_cols), np.nan, dtype=np.float32)
        else:
            out.fill(np.nan)

        col_to_idx = self._col_to_idx
        cat_to_code = self._cat_to_code

        # 1. Base time features
        dt = payload.get("TransactionDT")
        if dt is not None and not pd.isna(dt):
            if "hour_of_day" in col_to_idx:
                out[0, col_to_idx["hour_of_day"]] = np.float32((dt // 3600) % 24)
            if "day_of_week" in col_to_idx:
                out[0, col_to_idx["day_of_week"]] = np.float32((dt // 86400) % 7)

        # 2. Amount features
        amt = payload.get("TransactionAmt")
        if amt is not None and not pd.isna(amt):
            amt_f = np.float32(amt)
            if "log_TransactionAmt" in col_to_idx:
                out[0, col_to_idx["log_TransactionAmt"]] = np.log1p(np.maximum(amt_f, 0.0, dtype=np.float32))
            if "amt_cents" in col_to_idx:
                out[0, col_to_idx["amt_cents"]] = np.float32(amt_f % 1.0)
            if "amt_is_round" in col_to_idx:
                out[0, col_to_idx["amt_is_round"]] = np.float32(1.0 if (amt_f % 1.0 == 0) else 0.0)

        # 3. Email features
        p_email = str(payload.get("P_emaildomain", ""))
        r_email = str(payload.get("R_emaildomain", ""))

        p_prov, p_suff = None, None
        if p_email and p_email not in ["nan", "None", ""]:
            parts = p_email.split(".", 1)
            p_prov = parts[0]
            if len(parts) > 1:
                p_suff = parts[1]

        r_prov, r_suff = None, None
        if r_email and r_email not in ["nan", "None", ""]:
            parts = r_email.split(".", 1)
            r_prov = parts[0]
            if len(parts) > 1:
                r_suff = parts[1]

        if "P_emaildomain_provider" in col_to_idx and p_prov in cat_to_code.get("P_emaildomain_provider", {}):
            out[0, col_to_idx["P_emaildomain_provider"]] = cat_to_code["P_emaildomain_provider"][p_prov]
        if "P_emaildomain_suffix" in col_to_idx and p_suff in cat_to_code.get("P_emaildomain_suffix", {}):
            out[0, col_to_idx["P_emaildomain_suffix"]] = cat_to_code["P_emaildomain_suffix"][p_suff]
        if "R_emaildomain_provider" in col_to_idx and r_prov in cat_to_code.get("R_emaildomain_provider", {}):
            out[0, col_to_idx["R_emaildomain_provider"]] = cat_to_code["R_emaildomain_provider"][r_prov]
        if "R_emaildomain_suffix" in col_to_idx and r_suff in cat_to_code.get("R_emaildomain_suffix", {}):
            out[0, col_to_idx["R_emaildomain_suffix"]] = cat_to_code["R_emaildomain_suffix"][r_suff]

        if "email_domain_match" in col_to_idx:
            p_has = p_email and p_email not in ["nan", "None", ""]
            r_has = r_email and r_email not in ["nan", "None", ""]
            match = 1.0 if (p_has and r_has and p_email == r_email) else 0.0
            out[0, col_to_idx["email_domain_match"]] = np.float32(match)

        # 4. Card & address combination
        c1_val = payload.get("card1")
        if c1_val is None or pd.isna(c1_val) or str(c1_val) in ["nan", "None", "<NA>", ""]:
            c1_str = "nan"
        else:
            c1_str = str(c1_val)

        a1_val = payload.get("addr1")
        if a1_val is None or pd.isna(a1_val) or str(a1_val) in ["nan", "None", "<NA>", ""]:
            a1_str = "nan"
        else:
            try:
                a1_str = str(float(a1_val))
            except (ValueError, TypeError):
                a1_str = str(a1_val)
        c1_a1_str = f"{c1_str}_{a1_str}"

        if "card1_addr1" in col_to_idx:
            c1_a1_raw = str(payload.get("card1_addr1")) if "card1_addr1" in payload else c1_a1_str
            if c1_a1_raw in cat_to_code.get("card1_addr1", {}):
                out[0, col_to_idx["card1_addr1"]] = cat_to_code["card1_addr1"][c1_a1_raw]

        # 5. Frequency encodings
        for col, freq_map in self.freq_maps.items():
            fq_name = f"{col}_fq"
            if fq_name in col_to_idx:
                if col == "card1_addr1":
                    out[0, col_to_idx[fq_name]] = np.float32(freq_map.get(c1_a1_str, 0.0))
                else:
                    val = payload.get(col)
                    out[0, col_to_idx[fq_name]] = np.float32(freq_map.get(val, 0.0)) if (val is not None and not pd.isna(val)) else np.float32(0.0)

        # 6. Identity missingness flags
        has_id = 0.0
        missing_count = 0
        for ic in ID_COLS:
            ival = payload.get(ic)
            if ival is not None and not pd.isna(ival):
                has_id = 1.0
            else:
                missing_count += 1
        if "id_missing_count" in col_to_idx:
            out[0, col_to_idx["id_missing_count"]] = np.float32(missing_count)
        if "has_identity" in col_to_idx:
            out[0, col_to_idx["has_identity"]] = np.float32(has_id)

        # 7. Remaining raw features
        ignore_cols = self._ignore_cols
        for k, v in payload.items():
            if k in ignore_cols:
                continue
            if k in col_to_idx:
                idx = col_to_idx[k]
                if k in cat_to_code:
                    s_val = str(v)
                    if s_val in cat_to_code[k]:
                        out[0, idx] = cat_to_code[k][s_val]
                else:
                    try:
                        out[0, idx] = np.float32(v) if (v is not None and not pd.isna(v)) else np.nan
                    except (ValueError, TypeError):
                        out[0, idx] = np.nan

        return out

    def _apply_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast categorical columns using stored categories and ensure float32 numeric columns."""
        for col, cats in self.cat_categories.items():
            if col in df.columns:
                series_str = df[col].astype(str).replace({"nan": None, "None": None, "<NA>": None})
                series_str = series_str.where(series_str.isin(cats), None)
                df[col] = pd.Categorical(series_str, categories=cats, ordered=False)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into final feature matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureBuilder must be fit before transform can be called.")

        transformed = self._transform_raw(df)
        transformed = transformed.reindex(columns=self.final_feature_names, fill_value=np.nan)
        transformed = self._apply_categoricals(transformed)
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
