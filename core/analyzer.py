"""
DataAnalyzer - Handles data exploration, analysis, and cleaning operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

class DataAnalyzer:
    """Handles data analysis, exploration, and cleaning operations."""
    
    def __init__(self, data_manager):
        """
        Initialize the DataAnalyzer with a reference to DatasetManager.
        
        Args:
            data_manager: Instance of DatasetManager
        """
        self.data_manager = data_manager
    
    def summary_statistics(self, dataset_name: str) -> bool:
        """
        Calculate and display summary statistics for numerical columns.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            numerical_cols = self._get_numerical_columns(df)
            
            if not numerical_cols:
                print(f"No numerical columns found in dataset '{dataset_name}'.")
                return False
            
            print(f"\nSummary Statistics for '{dataset_name}':")
            print("=" * 60)
            
            for col in numerical_cols:
                print(f"\n{col}:")
                print("-" * 40)
                
                series = df[col].dropna()  # Remove NaN values for calculations
                
                if len(series) == 0:
                    print("  No valid data (all values are missing)")
                    continue
                
                # Calculate statistics
                stats = {
                    'Count': len(series),
                    'Mean': series.mean(),
                    'Median': series.median(),
                    'Mode': series.mode().iloc[0] if not series.mode().empty else 'N/A',
                    'Std Dev': series.std(),
                    'Min': series.min(),
                    'Max': series.max(),
                    'Range': series.max() - series.min(),
                    'Q1 (25%)': series.quantile(0.25),
                    'Q3 (75%)': series.quantile(0.75)
                }
                
                # Format and display statistics
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if stat_name == 'Count':
                            print(f"  {stat_name}: {int(value)}")
                        elif abs(value) >= 1000:
                            print(f"  {stat_name}: {value:,.2f}")
                        else:
                            print(f"  {stat_name}: {value:.2f}")
                    else:
                        print(f"  {stat_name}: {value}")
            
            return True
            
        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            return False
    
    def missing_data_report(self, dataset_name: str) -> bool:
        """
        Generate a report on missing data in the dataset.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            print(f"\nMissing Data Report for '{dataset_name}':")
            print("=" * 60)
            
            # Calculate missing data statistics
            total_cells = df.shape[0] * df.shape[1]
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            # Overall statistics
            total_missing = missing_counts.sum()
            overall_percentage = (total_missing / total_cells) * 100
            
            print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Total cells: {total_cells:,}")
            print(f"Total missing values: {total_missing:,} ({overall_percentage:.2f}%)")
            print()
            
            # Per-column statistics
            has_missing = False
            print("Missing data by column:")
            print("-" * 60)
            print(f"{'Column':<20} {'Missing':<10} {'Percentage':<12} {'Data Type'}")
            print("-" * 60)
            
            for col in df.columns:
                missing_count = missing_counts[col]
                missing_pct = missing_percentages[col]
                dtype = str(df[col].dtype)
                
                if missing_count > 0:
                    has_missing = True
                    print(f"{col:<20} {missing_count:<10} {missing_pct:<12.2f}% {dtype}")
                else:
                    print(f"{col:<20} {'0':<10} {'0.00':<12}% {dtype}")
            
            if not has_missing:
                print("\n✓ No missing values found in the dataset!")
            else:
                print(f"\n⚠ {missing_counts[missing_counts > 0].count()} columns have missing values")
            
            return True
            
        except Exception as e:
            print(f"Error generating missing data report: {e}")
            return False
    
    def frequency_counts(self, dataset_name: str, column: str = None) -> bool:
        """
        Display frequency counts for categorical columns or a specific column.
        
        Args:
            dataset_name: Name of the dataset to analyze
            column: Specific column to analyze (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            if column:
                # Analyze specific column
                if column not in df.columns:
                    print(f"Error: Column '{column}' not found in dataset.")
                    return False
                
                self._display_column_frequencies(df, column)
            else:
                # Analyze all categorical columns
                categorical_cols = self._get_categorical_columns(df)
                
                if not categorical_cols:
                    print(f"No categorical columns found in dataset '{dataset_name}'.")
                    return False
                
                print(f"\nFrequency Counts for '{dataset_name}':")
                print("=" * 60)
                
                for col in categorical_cols:
                    self._display_column_frequencies(df, col)
                    print()
            
            return True
            
        except Exception as e:
            print(f"Error calculating frequency counts: {e}")
            return False
    
    def _display_column_frequencies(self, df: pd.DataFrame, column: str, max_display: int = 10):
        """Display frequency counts for a specific column."""
        print(f"\n{column}:")
        print("-" * 40)
        
        # Calculate frequencies
        value_counts = df[column].value_counts(dropna=False)
        total_count = len(df)
        
        print(f"Total values: {total_count}")
        print(f"Unique values: {df[column].nunique()}")
        
        if df[column].isnull().sum() > 0:
            print(f"Missing values: {df[column].isnull().sum()}")
        
        print("\nTop values:")
        for i, (value, count) in enumerate(value_counts.head(max_display).items()):
            percentage = (count / total_count) * 100
            value_str = str(value) if pd.notna(value) else '<Missing>'
            print(f"  {value_str:<20} {count:>6} ({percentage:>5.1f}%)")
        
        if len(value_counts) > max_display:
            remaining = len(value_counts) - max_display
            print(f"  ... and {remaining} more values")

    def _push_undo_stack(self, dataset_name: str):
        if dataset_name not in self.data_manager.undo_stacks:
            self.data_manager.undo_stacks[dataset_name] = []
        # Push a copy of the current dataset to the stack
        self.data_manager.undo_stacks[dataset_name].append(self.data_manager.datasets[dataset_name].copy())
    
    def remove_duplicates(self, dataset_name: str) -> bool:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            dataset_name: Name of the dataset to clean
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            original_count = len(df)
            
            # Remove duplicates
            df_cleaned = df.drop_duplicates()
            duplicate_count = original_count - len(df_cleaned)
            
            if duplicate_count == 0:
                print(f"No duplicate rows found in dataset '{dataset_name}'.")
                return True
            
            # Create backup before modifying the dataset
            self._push_undo_stack(dataset_name)

            # Update the dataset in memory
            self.data_manager.datasets[dataset_name] = df_cleaned
            
            # Update metadata
            metadata = self.data_manager.metadata[dataset_name]
            metadata['rows'] = len(df_cleaned)
            self.data_manager._save_metadata()

            # Save back to CSV file
            filepath = self.data_manager.metadata[dataset_name].get('file_path')
            if filepath:
                df_cleaned.to_csv(filepath, index=False)
                print("Changes saved to disk.")
            else:
                print("Warning: Dataset file path not found; changes not saved to disk.")
            
            print(f"Removed {duplicate_count} duplicate rows from '{dataset_name}'.")
            print(f"Dataset now has {len(df_cleaned)} rows (was {original_count}).")
            
            return True
            
        except Exception as e:
            print(f"Error removing duplicates: {e}")
            return False
    
    def handle_missing_values(self, dataset_name: str, method: str = "remove") -> bool:
        """
        Handle missing values in the dataset.
        
        Args:
            dataset_name: Name of the dataset to clean
            method: Method to handle missing values ('remove', 'fill_mean', 'fill_mode')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            original_count = len(df)
            missing_before = df.isnull().sum().sum()
            
            if missing_before == 0:
                print(f"No missing values found in dataset '{dataset_name}'.")
                return True
            
            if method == "remove":
                df_cleaned = df.dropna()
                rows_removed = original_count - len(df_cleaned)
                print(f"Removed {rows_removed} rows with missing values.")
                
            elif method == "fill_mean":
                df_cleaned = df.copy()
                numerical_cols = self._get_numerical_columns(df_cleaned)
                
                for col in numerical_cols:
                    mean_value = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(mean_value)
                
                print(f"Filled missing values in numerical columns with mean values.")
                
            elif method == "fill_mode":
                df_cleaned = df.copy()
                categorical_cols = self._get_categorical_columns(df_cleaned)
                
                for col in categorical_cols:
                    mode_value = df_cleaned[col].mode()
                    if not mode_value.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
                
                print(f"Filled missing values in categorical columns with mode values.")
                
            else:
                print(f"Error: Unknown method '{method}'.")
                return False
            
            # Create backup before modifying the dataset
            self._push_undo_stack(dataset_name)

            # Update the dataset
            self.data_manager.datasets[dataset_name] = df_cleaned
            
            # Update metadata
            metadata = self.data_manager.metadata[dataset_name]
            metadata['rows'] = len(df_cleaned)
            self.data_manager._save_metadata()

            # Save back to CSV file
            filepath = self.data_manager.metadata[dataset_name].get('file_path')
            if filepath:
                df_cleaned.to_csv(filepath, index=False)
                print("Changes saved to disk.")
            else:
                print("Warning: Dataset file path not found; changes not saved to disk.")
            
            missing_after = df_cleaned.isnull().sum().sum()
            print(f"Missing values reduced from {missing_before} to {missing_after}.")
            print(f"Dataset now has {len(df_cleaned)} rows.")
            
            return True
            
        except Exception as e:
            print(f"Error handling missing values: {e}")
            return False
        
    def _get_filtered_df(self, df, column, condition, value):
        if condition == '>':
            return df[df[column] > value]
        elif condition == '<':
            return df[df[column] < value]
        elif condition == '>=':
            return df[df[column] >= value]
        elif condition == '<=':
            return df[df[column] <= value]
        elif condition == '==':
            return df[df[column] == value]
        elif condition == '!=':
            return df[df[column] != value]
        elif condition == 'contains':
            return df[df[column].astype(str).str.contains(str(value), na=False)]
        else:
            raise ValueError(f"Unknown condition '{condition}'")

    def filter_data_and_display(self, dataset_name: str, column: str, condition: str, value: Any) -> bool:
        """
        Filter dataset based on a condition.
        
        Args:
            dataset_name: Name of the dataset to filter
            column: Column to filter on
            condition: Condition ('>', '<', '>=', '<=', '==', '!=', 'contains')
            value: Value to compare against
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            if column not in df.columns:
                print(f"Error: Column '{column}' not found in dataset.")
                return False
            
            # Apply filter based on condition
            df_filtered = self._get_filtered_df(df, column, condition, value)

            filtered_count = len(df_filtered)
            
            if filtered_count == 0:
                print(f"Warning: Filter resulted in empty dataset.")
                return False
            
            print(f"\nFiltered results for: {column} {condition} {value}\n")
            print(df_filtered.to_string(index=False))  # or .head(n) for limit
            
            return True
            
        except Exception as e:
            print(f"Error filtering data: {e}")
            return False

    def filter_data_and_modify(self, dataset_name: str, column: str, condition: str, value: Any) -> bool:
        """
        Filter dataset based on a condition.
        
        Args:
            dataset_name: Name of the dataset to filter
            column: Column to filter on
            condition: Condition ('>', '<', '>=', '<=', '==', '!=', 'contains')
            value: Value to compare against
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            if column not in df.columns:
                print(f"Error: Column '{column}' not found in dataset.")
                return False
            
            original_count = len(df)
            
            # Apply filter based on condition
            df_filtered = self._get_filtered_df(df, column, condition, value)

            filtered_count = len(df_filtered)
            
            if filtered_count == 0:
                print(f"Warning: Filter resulted in empty dataset.")
                return False
            
            # Create backup before modifying the dataset
            self._push_undo_stack(dataset_name)

            # Update the dataset
            self.data_manager.datasets[dataset_name] = df_filtered
            
            # Update metadata
            metadata = self.data_manager.metadata[dataset_name]
            metadata['rows'] = filtered_count
            self.data_manager._save_metadata()

            # Save back to CSV file
            filepath = self.data_manager.metadata[dataset_name].get('file_path')
            if filepath:
                df_filtered.to_csv(filepath, index=False)
                print("Changes saved to disk.")
            else:
                print("Warning: Dataset file path not found; changes not saved to disk.")
            
            print(f"Filter applied: {column} {condition} {value}")
            print(f"Dataset filtered from {original_count} to {filtered_count} rows.")
            
            return True
            
        except Exception as e:
            print(f"Error filtering data: {e}")
            return False
    
    def undo_last_clean(self, dataset_name: str) -> bool:
        """
        Restore the last dataset version from the undo stack.
        """
        if dataset_name not in self.data_manager.undo_stacks or not self.data_manager.undo_stacks[dataset_name]:
            print(f"No undo available for dataset '{dataset_name}'.")
            return False

        # Pop the last version from the stack
        previous_df = self.data_manager.undo_stacks[dataset_name].pop()
        self.data_manager.datasets[dataset_name] = previous_df

        # Update metadata
        self.data_manager.metadata[dataset_name]['rows'] = len(previous_df)
        self.data_manager._save_metadata()

        # Save back to CSV file
        filepath = self.data_manager.metadata[dataset_name].get('file_path')
        if filepath:
            previous_df.to_csv(filepath, index=False)
            print("Changes saved to disk.")
        else:
            print("Warning: Dataset file path not found; changes not saved to disk.")

        print(f"Undo successful. Dataset '{dataset_name}' restored to previous state.")
        return True

    
    def _get_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numerical columns in the dataframe."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of categorical/text columns in the dataframe."""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _validate_dataset(self, dataset_name: str) -> bool:
        """Validate that the dataset exists and can be accessed."""
        return self.data_manager._dataset_exists(dataset_name)
    
    def get_dataset_overview(self, dataset_name: str) -> bool:
        """
        Display a comprehensive overview of the dataset.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            print(f"\nDataset Overview: '{dataset_name}'")
            print("=" * 60)
            
            # Basic info
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Column types
            print(f"\nColumn Types:")
            numerical_cols = self._get_numerical_columns(df)
            categorical_cols = self._get_categorical_columns(df)
            
            print(f"  Numerical: {len(numerical_cols)} columns")
            print(f"  Categorical: {len(categorical_cols)} columns")
            
            # Missing data summary
            missing_total = df.isnull().sum().sum()
            if missing_total > 0:
                print(f"\nMissing Values: {missing_total} total")
            else:
                print(f"\nMissing Values: None")
            
            # Sample data
            print(f"\nFirst 3 rows:")
            print("-" * 40)
            print(df.head(3).to_string())
            
            return True
            
        except Exception as e:
            print(f"Error generating dataset overview: {e}")
            return False