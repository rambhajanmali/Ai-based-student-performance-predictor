"""
Utility Module
Helper functions for the project
"""

import os
import json


def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Directory created: {path}")


def file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath)


def load_config(config_path):
    """Load configuration from JSON file"""
    if file_exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def save_config(config_dict, config_path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Config saved to {config_path}")


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
