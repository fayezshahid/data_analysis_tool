"""
DataVisualizer - Handles visualization generation and report creation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .constants import DEFAULT_OUTPUT_DIR


class DataVisualizer:
    """Handles data visualization and report generation."""
    
    def __init__(self, data_manager, plots_dir: str = None, reports_dir: str = None):
        """
        Initialize the DataVisualizer.
        
        Args:
            data_manager: Instance of DatasetManager
            plots_dir: Directory to save plots
            reports_dir: Directory to save reports
        """
        self.data_manager = data_manager
        self.plots_dir = plots_dir or os.path.join(DEFAULT_OUTPUT_DIR, "plots")
        self.reports_dir = reports_dir or os.path.join(DEFAULT_OUTPUT_DIR, "reports")
        
        # Track generated plots and reports for each dataset
        self.generated_plots = {}
        self.generated_reports = {}
        
        # Initialize directories
        self._initialize_directories()
        
        # Set up matplotlib style
        self._setup_plot_style()
    
    def _initialize_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.plots_dir, self.reports_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def _setup_plot_style(self):
        """Set up matplotlib and seaborn styling."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
    
    def generate_histogram(self, dataset_name: str, column: str, bins: int = 30) -> bool:
        """
        Generate a histogram for a numerical column.
        
        Args:
            dataset_name: Name of the dataset
            column: Column to plot
            bins: Number of bins for histogram
            
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
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                print(f"Error: Column '{column}' is not numerical. Use bar chart instead.")
                return False
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            
            # Remove missing values for plotting
            data = df[column].dropna()
            
            if len(data) == 0:
                print(f"Error: No valid data in column '{column}' (all values are missing).")
                return False
            
            # Generate histogram
            plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {column} - {dataset_name}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add statistics to plot
            mean_val = data.mean()
            median_val = data.median()
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            plt.legend()
            
            # Save plot
            filename = f"{dataset_name}_{column}_histogram.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Track generated plot
            if dataset_name not in self.generated_plots:
                self.generated_plots[dataset_name] = []
            self.generated_plots[dataset_name].append({
                'type': 'histogram',
                'column': column,
                'filename': filename,
                'filepath': filepath,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Histogram for '{column}' generated and saved as '{filename}'.")
            return True
            
        except Exception as e:
            print(f"Error generating histogram: {e}")
            return False
    
    def generate_bar_chart(self, dataset_name: str, column: str, max_categories: int = 20) -> bool:
        """
        Generate a bar chart for a categorical column.
        
        Args:
            dataset_name: Name of the dataset
            column: Column to plot
            max_categories: Maximum number of categories to display
            
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
            
            # Get value counts
            value_counts = df[column].value_counts().head(max_categories)
            
            if len(value_counts) == 0:
                print(f"Error: No data available in column '{column}'.")
                return False
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            
            bars = plt.bar(range(len(value_counts)), value_counts.values, 
                          color='lightcoral', alpha=0.7, edgecolor='black')
            
            plt.title(f'Bar Chart of {column} - {dataset_name}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(value), ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"{dataset_name}_{column}_bar_chart.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Track generated plot
            if dataset_name not in self.generated_plots:
                self.generated_plots[dataset_name] = []
            self.generated_plots[dataset_name].append({
                'type': 'bar_chart',
                'column': column,
                'filename': filename,
                'filepath': filepath,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Bar chart for '{column}' generated and saved as '{filename}'.")
            if len(df[column].value_counts()) > max_categories:
                print(f"Note: Only top {max_categories} categories are displayed.")
            
            return True
            
        except Exception as e:
            print(f"Error generating bar chart: {e}")
            return False
    
    def generate_scatter_plot(self, dataset_name: str, x_column: str, y_column: str) -> bool:
        """
        Generate a scatter plot for two numerical columns.
        
        Args:
            dataset_name: Name of the dataset
            x_column: Column for x-axis
            y_column: Column for y-axis
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            
            # Validate columns exist
            for col in [x_column, y_column]:
                if col not in df.columns:
                    print(f"Error: Column '{col}' not found in dataset.")
                    return False
            
            # Check if columns are numerical
            for col in [x_column, y_column]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"Error: Column '{col}' is not numerical.")
                    return False
            
            # Remove rows with missing values in either column
            clean_df = df[[x_column, y_column]].dropna()
            
            if len(clean_df) == 0:
                print(f"Error: No complete data pairs available for columns '{x_column}' and '{y_column}'.")
                return False
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(clean_df[x_column], clean_df[y_column], alpha=0.6, color='darkblue')
            plt.title(f'Scatter Plot: {x_column} vs {y_column} - {dataset_name}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = clean_df[x_column].corr(clean_df[y_column])
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
            
            # Save plot
            filename = f"{dataset_name}_{x_column}_vs_{y_column}_scatter.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Track generated plot
            if dataset_name not in self.generated_plots:
                self.generated_plots[dataset_name] = []
            self.generated_plots[dataset_name].append({
                'type': 'scatter_plot',
                'columns': [x_column, y_column],
                'filename': filename,
                'filepath': filepath,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Scatter plot for '{x_column}' vs '{y_column}' generated and saved as '{filename}'.")
            print(f"Correlation coefficient: {correlation:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error generating scatter plot: {e}")
            return False
    
    def generate_box_plot(self, dataset_name: str, column: str, group_by: str = None) -> bool:
        """
        Generate a box plot for a numerical column, optionally grouped by a categorical column.
        
        Args:
            dataset_name: Name of the dataset
            column: Numerical column to plot
            group_by: Optional categorical column to group by
            
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
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                print(f"Error: Column '{column}' is not numerical.")
                return False
            
            plt.figure(figsize=(10, 6))
            
            if group_by:
                if group_by not in df.columns:
                    print(f"Error: Group column '{group_by}' not found in dataset.")
                    return False
                
                # Grouped box plot
                df_clean = df[[column, group_by]].dropna()
                groups = df_clean[group_by].unique()
                
                if len(groups) > 20:
                    print(f"Warning: Too many groups ({len(groups)}). Showing top 20.")
                    top_groups = df_clean[group_by].value_counts().head(20).index
                    df_clean = df_clean[df_clean[group_by].isin(top_groups)]
                
                sns.boxplot(data=df_clean, x=group_by, y=column)
                plt.title(f'Box Plot of {column} by {group_by} - {dataset_name}')
                plt.xticks(rotation=45, ha='right')
                filename = f"{dataset_name}_{column}_by_{group_by}_boxplot.png"
            else:
                # Simple box plot
                data = df[column].dropna()
                plt.boxplot(data, tick_labels=[column])
                plt.title(f'Box Plot of {column} - {dataset_name}')
                plt.ylabel(column)
                filename = f"{dataset_name}_{column}_boxplot.png"
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Track generated plot
            if dataset_name not in self.generated_plots:
                self.generated_plots[dataset_name] = []
            self.generated_plots[dataset_name].append({
                'type': 'box_plot',
                'column': column,
                'group_by': group_by,
                'filename': filename,
                'filepath': filepath,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Box plot for '{column}' generated and saved as '{filename}'.")
            return True
            
        except Exception as e:
            print(f"Error generating box plot: {e}")
            return False
    
    def generate_report(self, dataset_name: str, include_plots: bool = True) -> bool:
        """
        Generate a comprehensive report for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            include_plots: Whether to include plot references in report
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_dataset(dataset_name):
            return False
        
        try:
            df = self.data_manager._get_dataset(dataset_name)
            metadata = self.data_manager.get_dataset_info(dataset_name)
            
            # Generate report content
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"DATA ANALYSIS REPORT: {dataset_name}")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Dataset Overview
            report_lines.append("1. DATASET OVERVIEW")
            report_lines.append("-" * 40)
            report_lines.append(f"Dataset Name: {dataset_name}")
            report_lines.append(f"File Path: {metadata.get('file_path', 'N/A')}")
            report_lines.append(f"Loaded Date: {metadata.get('loaded_date', 'N/A')}")
            report_lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            report_lines.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            report_lines.append("")
            
            # Column Information
            report_lines.append("2. COLUMN INFORMATION")
            report_lines.append("-" * 40)
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            report_lines.append(f"Total Columns: {len(df.columns)}")
            report_lines.append(f"Numerical Columns: {len(numerical_cols)}")
            report_lines.append(f"Categorical Columns: {len(categorical_cols)}")
            report_lines.append("")
            
            report_lines.append("Column Details:")
            for col in df.columns:
                col_type = "Numerical" if col in numerical_cols else "Categorical"
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                report_lines.append(f"  {col}: {col_type}, Missing: {missing_count} ({missing_pct:.1f}%)")
            report_lines.append("")
            
            # Missing Data Summary
            report_lines.append("3. MISSING DATA SUMMARY")
            report_lines.append("-" * 40)
            total_missing = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (total_missing / total_cells) * 100
            
            report_lines.append(f"Total Missing Values: {total_missing:,} ({missing_pct:.2f}%)")
            
            if total_missing > 0:
                report_lines.append("Columns with Missing Data:")
                missing_by_col = df.isnull().sum()
                for col, count in missing_by_col[missing_by_col > 0].items():
                    pct = (count / len(df)) * 100
                    report_lines.append(f"  {col}: {count} ({pct:.1f}%)")
            else:
                report_lines.append("No missing values found.")
            report_lines.append("")
            
            # Numerical Columns Statistics
            if numerical_cols:
                report_lines.append("4. NUMERICAL COLUMNS STATISTICS")
                report_lines.append("-" * 40)
                
                for col in numerical_cols:
                    series = df[col].dropna()
                    if len(series) > 0:
                        report_lines.append(f"{col}:")
                        report_lines.append(f"  Count: {len(series)}")
                        report_lines.append(f"  Mean: {series.mean():.2f}")
                        report_lines.append(f"  Median: {series.median():.2f}")
                        report_lines.append(f"  Std Dev: {series.std():.2f}")
                        report_lines.append(f"  Min: {series.min():.2f}")
                        report_lines.append(f"  Max: {series.max():.2f}")
                        report_lines.append("")
            
            # Categorical Columns Summary
            if categorical_cols:
                report_lines.append("5. CATEGORICAL COLUMNS SUMMARY")
                report_lines.append("-" * 40)
                
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    most_common = df[col].value_counts().head(1)
                    report_lines.append(f"{col}:")
                    report_lines.append(f"  Unique Values: {unique_count}")
                    if not most_common.empty:
                        report_lines.append(f"  Most Common: {most_common.index[0]} ({most_common.iloc[0]} times)")
                    report_lines.append("")
            
            # Generated Plots
            if include_plots and dataset_name in self.generated_plots:
                report_lines.append("6. GENERATED VISUALIZATIONS")
                report_lines.append("-" * 40)
                
                for plot_info in self.generated_plots[dataset_name]:
                    report_lines.append(f"Plot Type: {plot_info['type'].replace('_', ' ').title()}")
                    if 'column' in plot_info:
                        report_lines.append(f"Column: {plot_info['column']}")
                    elif 'columns' in plot_info:
                        report_lines.append(f"Columns: {', '.join(plot_info['columns'])}")
                    report_lines.append(f"File: {plot_info['filename']}")
                    report_lines.append(f"Generated: {plot_info['generated_at']}")
                    report_lines.append("")
            
            # Footer
            report_lines.append("=" * 80)
            report_lines.append("End of Report")
            report_lines.append("=" * 80)
            
            # Save report
            filename = f"{dataset_name}_report.txt"
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write('\n'.join(report_lines))
            
            # Track generated report
            if dataset_name not in self.generated_reports:
                self.generated_reports[dataset_name] = []
            self.generated_reports[dataset_name].append({
                'filename': filename,
                'filepath': filepath,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Report for '{dataset_name}' generated and saved as '{filename}'.")
            return True
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return False
    
    def list_generated_plots(self, dataset_name: str = None):
        """List all generated plots for a dataset or all datasets."""
        if dataset_name:
            if dataset_name in self.generated_plots:
                print(f"\nGenerated plots for '{dataset_name}':")
                for plot in self.generated_plots[dataset_name]:
                    print(f"  - {plot['filename']} ({plot['type']}) - {plot['generated_at']}")
            else:
                print(f"No plots generated for dataset '{dataset_name}'.")
        else:
            if self.generated_plots:
                print("\nAll generated plots:")
                for ds_name, plots in self.generated_plots.items():
                    print(f"\n{ds_name}:")
                    for plot in plots:
                        print(f"  - {plot['filename']} ({plot['type']}) - {plot['generated_at']}")
            else:
                print("No plots have been generated yet.")
    
    def _validate_dataset(self, dataset_name: str) -> bool:
        """Validate that the dataset exists and can be accessed."""
        return self.data_manager._dataset_exists(dataset_name)