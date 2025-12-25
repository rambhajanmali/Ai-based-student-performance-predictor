"""
Exploratory Data Analysis (EDA)

Dataset: UCI Student Performance (Portuguese secondary school) — student-mat.csv.
Brief: 395 records with socio-demographic, school-related, and behavioral features.
Target: `G3` — final grade on a 0–20 scale. Auxiliary grades `G1` and `G2` are prior periods.

Purpose: Provide fast, reproducible inspection of structure and basic statistics.
This module is designed for reuse in preprocessing and modeling workflows.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATA_RAW_PATH, TARGET_COLUMN


class DataExplorer:
    """
    Exploratory utilities for student performance data.
    Handles data loading, validation, and statistical summaries.
    """
    
    def __init__(self, filepath: str) -> None:
        """Initialize with dataset path."""
        self.filepath = filepath
        self.df: pd.DataFrame | None = None
        self.target_col: str = TARGET_COLUMN
        
    def load_dataset(self) -> bool:
        """Load dataset from CSV with basic validation."""
        try:
            self.df = pd.read_csv(self.filepath)
            print("Dataset loaded successfully")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return False
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return False
    
    def display_shape(self) -> None:
        """Display dataset dimensions (rows, columns)."""
        if self.df is None:
            print("No dataset loaded")
            return
        
        rows, cols = self.df.shape
        print("\nDataset Shape:")
        print(f"   Rows: {rows} | Columns: {cols}")
    
    def display_columns(self) -> None:
        """Display all column names with index."""
        if self.df is None:
            return
        
        print(f"\nColumn Names ({len(self.df.columns)} total):")
        for idx, col in enumerate(self.df.columns, 1):
            print(f"   {idx:2d}. {col}")
    
    def display_data_types(self) -> None:
        """Display data types of all columns."""
        if self.df is None:
            return
        
        print("\nData Types:")
        dtype_info = self.df.dtypes
        for col, dtype in dtype_info.items():
            print(f"   {col:20s} -> {str(dtype)}")
    
    def display_first_rows(self, n: int = 5) -> None:
        """Display first n rows of the dataset."""
        if self.df is None:
            return
        
        print(f"\nFirst {n} Rows:")
        print(self.df.head(n).to_string())
    
    def display_target_info(self) -> None:
        """Display statistics and validation for the target variable (`G3`)."""
        if self.df is None or self.target_col not in self.df.columns:
            print(f"Target column '{self.target_col}' not found")
            return
        
        target_data = self.df[self.target_col]
        
        print(f"\nTarget Variable: '{self.target_col}' (final grade, 0–20)")
        print(f"   Range: {target_data.min()} - {target_data.max()}")
        print(f"   Mean: {target_data.mean():.2f}")
        print(f"   Median: {target_data.median():.2f}")
        print(f"   Std Dev: {target_data.std():.2f}")
        print(f"   Missing Values: {target_data.isna().sum()}")
    
    def display_missing_values(self) -> None:
        """Display count and percentage of missing values."""
        if self.df is None:
            return
        
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("\nNo missing values detected")
            return
        
        print("\nMissing Values:")
        missing_pct = (missing / len(self.df)) * 100
        for col in missing[missing > 0].index:
            print(f"   {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    
    def display_statistical_summary(self) -> None:
        """Display statistical summary of numerical columns."""
        if self.df is None:
            return
        
        print("\nStatistical Summary (numerical columns):")
        print(self.df.describe().to_string())
    
    def get_feature_types(self) -> dict:
        """Categorize features into numerical, categorical, and counts."""
        if self.df is None:
            return {}
        
        numerical = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = self.df.select_dtypes(include=['object']).columns.tolist()
        
        return {
            'numerical': numerical,
            'categorical': categorical,
            'total_features': len(self.df.columns) - 1  # Exclude target
        }

    def dataset_brief(self) -> str:
        """Short dataset description suitable for reports/documentation."""
        return (
            "UCI Student Performance (Mathematics) dataset: 395 students, "
            "features spanning demographics (e.g., age, sex), prior grades (G1, G2), "
            "school engagement (study time, failures, absences), and social context. "
            "Target `G3` is the final grade on a 0–20 scale."
        )
    
    def run_full_eda(self) -> bool:
        """Execute complete exploratory data analysis."""
        print("\n" + "="*70)
        print("Exploratory Data Analysis (EDA)")
        print("="*70)
        
        if not self.load_dataset():
            return False
        
        self.display_shape()
        self.display_columns()
        self.display_data_types()
        self.display_first_rows(5)
        self.display_missing_values()
        self.display_target_info()
        self.display_statistical_summary()
        
        feature_types = self.get_feature_types()
        print("\nFeature Categories:")
        print(f"   Numerical Features: {len(feature_types['numerical'])}")
        print(f"   Categorical Features: {len(feature_types['categorical'])}")
        
        print("\n" + "="*70)
        print("EDA complete")
        print("="*70)
        
        return True


# Entry point for running EDA
if __name__ == "__main__":
    # Initialize explorer with path from config
    explorer = DataExplorer(DATA_RAW_PATH)
    
    # Run full exploratory data analysis
    explorer.run_full_eda()
