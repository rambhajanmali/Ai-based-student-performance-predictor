"""
Synthetic Dataset Generator
Creates a realistic student performance dataset for the project
Based on UCI Student Performance Dataset structure (student-mat.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_student_dataset():
    """
    Generate synthetic student performance dataset matching UCI structure.
    This includes key features and the target variable (G3 - Final Grade).
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of records
    n_records = 395
    
    # Generate synthetic data with realistic distributions
    data = {
        # Demographics
        'school': np.random.choice(['GP', 'MS'], n_records, p=[0.7, 0.3]),
        'sex': np.random.choice(['F', 'M'], n_records, p=[0.5, 0.5]),
        'age': np.random.randint(15, 23, n_records),
        
        # Family information
        'Pstatus': np.random.choice(['T', 'A'], n_records, p=[0.75, 0.25]),  # Parental cohabitation
        'Medu': np.random.choice([0, 1, 2, 3, 4], n_records),  # Mother's education
        'Fedu': np.random.choice([0, 1, 2, 3, 4], n_records),  # Father's education
        'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_records),
        'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_records),
        'traveltime': np.random.randint(1, 5, n_records),  # Travel time to school
        'famsize': np.random.choice(['LE3', 'GT3'], n_records),  # Family size
        'Pfinance': np.random.choice(['low', 'high'], n_records, p=[0.4, 0.6]),  # Family financial status
        
        # Academic information
        'reason': np.random.choice(['course', 'home', 'reputation', 'other'], n_records),
        'paid': np.random.choice(['yes', 'no'], n_records, p=[0.3, 0.7]),  # Extra paid classes
        'activities': np.random.choice(['yes', 'no'], n_records, p=[0.4, 0.6]),  # Extracurricular activities
        'nursery': np.random.choice(['yes', 'no'], n_records),  # Attended nursery school
        'higher': np.random.choice(['yes', 'no'], n_records, p=[0.8, 0.2]),  # Wants higher education
        'internet': np.random.choice(['yes', 'no'], n_records, p=[0.75, 0.25]),  # Home internet
        'romantic': np.random.choice(['yes', 'no'], n_records, p=[0.5, 0.5]),  # In romantic relationship
        
        # Academic performance (predictors)
        'Medu_numeric': np.random.choice([0, 1, 2, 3, 4], n_records),  # Mother education (numeric)
        'Fedu_numeric': np.random.choice([0, 1, 2, 3, 4], n_records),  # Father education (numeric)
        'Dalc': np.random.randint(1, 6, n_records),  # Workday alcohol consumption
        'Walc': np.random.randint(1, 6, n_records),  # Weekend alcohol consumption
        'absences': np.random.randint(0, 33, n_records),  # School absences
        'G1': np.random.randint(3, 21, n_records),  # First period grade
        'G2': np.random.randint(3, 21, n_records),  # Second period grade
        
        # Study time
        'studytime': np.random.randint(1, 5, n_records),  # Weekly study hours
        
        # Failures
        'failures': np.random.choice([0, 1, 2, 3, 4], n_records, p=[0.65, 0.20, 0.10, 0.03, 0.02]),
        
        # Other factors
        'goout': np.random.randint(1, 6, n_records),  # Frequency of going out
        'health': np.random.randint(1, 6, n_records),  # Health status
        'freetime': np.random.randint(1, 6, n_records),  # Free time after school
        'sleep': np.random.randint(1, 6, n_records),  # Sleep quality (1-5)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable (G3 - Final Grade) with realistic correlation
    # G3 is influenced by G1, G2, study time, failures, etc.
    G3 = (
        0.3 * df['G1'] + 
        0.3 * df['G2'] + 
        0.1 * df['studytime'] * 2 +
        0.1 * (df['Medu'] + df['Fedu']) +
        -0.15 * df['failures'] * 2 +
        -0.05 * df['Dalc'] +
        -0.05 * df['Walc'] +
        -0.02 * df['absences'] +
        np.random.normal(0, 1.5, n_records)
    ).astype(int)
    
    # Clip to valid grade range (0-20)
    df['G3'] = np.clip(G3, 0, 20)
    
    return df


def save_dataset(df, output_path):
    """Save dataset to CSV"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset generated and saved to {output_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SYNTHETIC STUDENT PERFORMANCE DATASET GENERATOR")
    print("="*70)
    
    # Generate dataset
    print("\n📊 Generating synthetic student performance data...")
    df = generate_student_dataset()
    
    # Save dataset
    output_path = 'data/raw/student-mat.csv'
    save_dataset(df, output_path)
    
    print("\n✓ Dataset ready for exploratory data analysis!")
    print("  Run: python src/eda.py")
    print("="*70)
