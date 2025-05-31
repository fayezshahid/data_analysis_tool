"""
DatasetManager - Handles all dataset loading, storage, and management operations.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .constants import (
    DEFAULT_DATASET_DIR, 
    DEFAULT_METADATA_DIR, 
    SUPPORTED_FORMATS
)


class DatasetManager:
    """Manages dataset loading, storage, and metadata operations."""
    
    def __init__(self, dataset_dir: str = None, metadata_dir: str = None):
        """
        Initialize the DatasetManager.
        
        Args:
            dataset_dir: Directory to store datasets
            metadata_dir: Directory to store metadata files
        """
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.metadata_dir = metadata_dir or DEFAULT_METADATA_DIR
        self.metadata_file = os.path.join(self.metadata_dir, "datasets.json")
        
        # In-memory storage for loaded datasets
        self.datasets = {}
        
        # For undo implementation
        self.undo_stacks = {}  # Key: dataset_name, Value: List[pd.DataFrame]

        # Initialize directories and load existing metadata
        self._initialize_directories()
        self.metadata = self._load_metadata()
    
    def _initialize_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.dataset_dir, self.metadata_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create empty metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load metadata file. Creating new one. Error: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save current metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print(f"Error saving metadata: {e}")
    
    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from file path."""
        return os.path.splitext(file_path.lower())[1]
    
    def _validate_file_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        extension = self._get_file_extension(file_path)
        return extension in SUPPORTED_FORMATS
    
    def _copy_dataset_file(self, source_path: str, dataset_name: str) -> str:
        """Copy dataset file to managed storage directory."""
        extension = self._get_file_extension(source_path)
        target_filename = f"{dataset_name}{extension}"
        target_path = os.path.join(self.dataset_dir, target_filename)
        
        try:
            shutil.copy2(source_path, target_path)
            return target_path
        except IOError as e:
            raise IOError(f"Failed to copy dataset file: {e}")
    
    def load_dataset(self, file_path: str, dataset_name: str) -> bool:
        """
        Load a dataset from a CSV file.
        
        Args:
            file_path: Path to the source CSV file
            dataset_name: User-defined name for the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate inputs
        if not dataset_name or not dataset_name.strip():
            print("Error: Dataset name cannot be empty.")
            return False
        
        dataset_name = dataset_name.strip()
        
        if dataset_name in self.metadata:
            print(f"Error: Dataset '{dataset_name}' already exists. Use a different name or remove the existing dataset first.")
            return False
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return False
        
        if not self._validate_file_format(file_path):
            print(f"Error: Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            return False
        
        try:
            # Load the dataset to validate it
            df = pd.read_csv(file_path)
            
            if df.empty:
                print("Error: The dataset is empty.")
                return False
            
            # Copy file to managed storage
            stored_path = self._copy_dataset_file(file_path, dataset_name)
            
            # Store dataset in memory
            self.datasets[dataset_name] = df
            
            # Update metadata
            self.metadata[dataset_name] = {
                "file_path": stored_path,
                "name": dataset_name,
                "rows": len(df),
                "columns": len(df.columns),
                "loaded_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "column_names": list(df.columns),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Save metadata
            self._save_metadata()
            
            print(f"Dataset '{dataset_name}' loaded successfully.")
            print(f"Shape: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def list_datasets(self) -> Dict:
        """
        List all loaded datasets with basic information.
        
        Returns:
            Dict: Dictionary containing dataset information
        """
        if not self.metadata:
            print("No datasets currently loaded.")
            return {}
        
        print("\nLoaded datasets:")
        print("-" * 60)
        
        for i, (name, info) in enumerate(self.metadata.items(), 1):
            print(f"{i}. {name}")
            print(f"   Rows: {info['rows']}, Columns: {info['columns']}")
            print(f"   Loaded: {info['loaded_date']}")
            print(f"   Columns: {', '.join(info['column_names'][:5])}" + 
                  ("..." if len(info['column_names']) > 5 else ""))
            print()
        
        return self.metadata
    
    def view_dataset(self, dataset_name: str, n_rows: int = 5) -> bool:
        """
        Display the first N rows of a dataset.
        
        Args:
            dataset_name: Name of the dataset to view
            n_rows: Number of rows to display
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._dataset_exists(dataset_name):
            return False
        
        try:
            # Load dataset if not in memory
            df = self._get_dataset(dataset_name)
            
            print(f"\nDisplaying the first {min(n_rows, len(df))} rows of '{dataset_name}':")
            print("-" * 80)
            
            # Display dataset info
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns)}")
            print("-" * 80)
            
            # Display data
            print(df.head(n_rows).to_string(index=True))
            print("-" * 80)
            
            return True
            
        except Exception as e:
            print(f"Error viewing dataset: {e}")
            return False
    
    def remove_dataset(self, dataset_name: str) -> bool:
        """
        Remove a dataset from the inventory.
        
        Args:
            dataset_name: Name of the dataset to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._dataset_exists(dataset_name):
            return False
        
        try:
            # Get file path
            file_path = self.metadata[dataset_name]["file_path"]
            
            # Remove from memory
            if dataset_name in self.datasets:
                del self.datasets[dataset_name]
            
            # Remove file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[dataset_name]
            
            # Save updated metadata
            self._save_metadata()
            
            print(f"Dataset '{dataset_name}' removed successfully.")
            return True
            
        except Exception as e:
            print(f"Error removing dataset: {e}")
            return False
    
    def _dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists in the inventory."""
        if dataset_name not in self.metadata:
            print(f"Error: Dataset '{dataset_name}' not found.")
            print("Use 'list' command to see available datasets.")
            return False
        return True
    
    def _get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get dataset from memory or load from file."""
        # If dataset is in memory, return it
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        # Load from file
        file_path = self.metadata[dataset_name]["file_path"]
        df = pd.read_csv(file_path)
        
        # Store in memory for future use
        self.datasets[dataset_name] = df
        
        return df
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get metadata information for a specific dataset."""
        if dataset_name in self.metadata:
            return self.metadata[dataset_name]
        return None
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.metadata.keys())
    
    def reload_metadata(self):
        """Reload metadata from file (useful for debugging)."""
        self.metadata = self._load_metadata()
        print("Metadata reloaded from file.")