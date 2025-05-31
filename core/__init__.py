"""
Core package for Data Analysis Assistant
Contains the main components for dataset management, analysis, and visualization.
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Data Analysis Assistant Team"
__description__ = "A command-line tool for data management, analysis, and visualization"

# Import main classes for easy access
from .data_manager import DatasetManager
from .analyzer import DataAnalyzer
from .visualizer import DataVisualizer
from .cli_interface import CLIInterface

# Define what gets imported when using "from core import *"
__all__ = [
    'DatasetManager',
    'DataAnalyzer', 
    'DataVisualizer',
    'CLIInterface'
]

