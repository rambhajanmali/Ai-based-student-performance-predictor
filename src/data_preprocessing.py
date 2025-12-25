"""
Reusable data preprocessing pipeline for the student performance dataset.

Goals
- Safe loading and cleaning of raw data
- Consistent categorical encoding for train/inference
- Clear separation of features/target and train/test splits

Scope
- No model training or visualization here.
- Uses relative paths via config to remain portable.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_RAW_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


class DataPreprocessor:
    """Provides consistent preprocessing for training and inference."""

    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        feature_columns_path: str = "models/feature_columns.json",
    ) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns: list[str] | None = None  # set after encoding
        self.feature_columns_path = feature_columns_path

    def load_data(self, filepath: str = DATA_RAW_PATH) -> pd.DataFrame:
        """Load raw CSV data; raise if missing for early fail."""
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. Place student-mat.csv under data/raw."
            )
        df = pd.read_csv(filepath, sep=';')

        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values (numeric median, categorical mode)."""
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(exclude="number").columns

        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        for col in cat_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        return df

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target with validation."""
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    def encode_categoricals(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One-hot encode categoricals using train categories; align test columns."""
        cat_cols = X_train.select_dtypes(exclude="number").columns

        X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        self.feature_columns = X_train_enc.columns.tolist()

        X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        X_test_enc = X_test_enc.reindex(columns=self.feature_columns, fill_value=0)
        return X_train_enc, X_test_enc

    def save_feature_columns(self, path: str | Path | None = None) -> None:
        """Persist ordered feature column names for inference alignment."""
        if self.feature_columns is None:
            raise ValueError("feature_columns is not set; run encode_categoricals first")

        target_path = Path(path or self.feature_columns_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train-test split with fixed randomness for reproducibility."""
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=None,
        )

    def process(
        self, filepath: str = DATA_RAW_PATH
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full preprocessing pipeline.

        Steps: load -> impute -> split X/y -> train-test split -> one-hot encode.
        Returns: X_train, X_test, y_train, y_test ready for model training.
        """

        df = self.load_data(filepath)
        df = self.handle_missing(df)
        X, y = self.split_features_target(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_enc, X_test_enc = self.encode_categoricals(X_train, X_test)
        self.save_feature_columns()
        return X_train_enc, X_test_enc, y_train, y_test


def preprocess_dataset(
    filepath: str = DATA_RAW_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience function to run preprocessing without instantiating the class."""

    preprocessor = DataPreprocessor()
    return preprocessor.process(filepath)


if __name__ == "__main__":
    # Minimal smoke test (will raise if data missing)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.process()
    print("Preprocessing complete")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
