"""
Dataset Download Script
Downloads the UCI Student Performance Dataset (student-mat.csv)
"""

import urllib.request
import os
from pathlib import Path


def download_dataset(url, output_path):
    """
    Download dataset from UCI repository
    
    Args:
        url (str): URL to download from
        output_path (str): Local file path to save dataset
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"✓ Dataset already exists at {output_path}")
        return True
    
    try:
        print(f"📥 Downloading dataset from UCI repository...")
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, output_path)
        
        # Verify file size
        file_size = os.path.getsize(output_path) / 1024  # Size in KB
        print(f"✓ Dataset downloaded successfully")
        print(f"   Location: {output_path}")
        print(f"   File Size: {file_size:.2f} KB")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {str(e)}")
        return False


if __name__ == "__main__":
    # UCI ML Repository URL for student performance dataset (alternative mirrors)
    DATASET_URLS = [
        'https://raw.githubusercontent.com/amankharwal/student-performance-prediction/main/student-mat.csv',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip',
    ]
    OUTPUT_PATH = 'data/raw/student-mat.csv'
    DATASET_URL = DATASET_URLS[0]  # Use primary mirror
    
    print("\n" + "="*70)
    print("  UCI STUDENT PERFORMANCE DATASET DOWNLOADER")
    print("="*70)
    
    success = download_dataset(DATASET_URL, OUTPUT_PATH)
    
    if success:
        print("\n✓ Ready for exploratory data analysis!")
        print("  Run: python src/eda.py")
    else:
        print("\n✗ Download failed. Please check your internet connection.")
    
    print("="*70)
