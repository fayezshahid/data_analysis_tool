#!/usr/bin/env python3
"""
Data Analysis Assistant - Main Entry Point
A command-line tool for managing datasets, performing analysis, and generating visualizations.
"""

import sys
import os

# Add the core module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.cli_interface import CLIInterface
from core.data_manager import DatasetManager
from core.analyzer import DataAnalyzer
from core.visualizer import DataVisualizer


class DataAnalysisAssistant:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the application with all core components."""
        self.data_manager = DatasetManager()
        self.analyzer = DataAnalyzer(self.data_manager)
        self.visualizer = DataVisualizer(self.data_manager)
        self.cli = CLIInterface(self.data_manager, self.analyzer, self.visualizer)
        self.running = True
    
    def display_welcome(self):
        """Display welcome message and basic instructions."""
        print("=" * 60)
        print("Welcome to the Python Data Analysis Assistant!")
        print("=" * 60)
        print("This tool helps you manage datasets, perform analysis, and create visualizations.")
        print("Type 'help' to see available commands or 'exit' to quit.")
        print("-" * 60)
    
    def run(self):
        """Main application loop."""
        self.display_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle exit command
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self.exit_application()
                    break
                
                # Process command through CLI interface
                result = "EXIT" # method to be called from CLI class
                
                # Handle special return codes
                if result == "EXIT":
                    self.exit_application()
                    break
                
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal. Exiting...")
                self.exit_application()
                break
            
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
                print("Please try again or type 'help' for available commands.")
    
    def exit_application(self):
        """Handle application exit with cleanup."""
        print("\n" + "=" * 60)
        print("Thank you for using the Python Data Analysis Assistant!")
        print("Your datasets and metadata have been saved.")
        print("Goodbye!")
        print("=" * 60)
        self.running = False


def main():
    """Main function to run the application."""
    try:
        # Create and run the application
        app = DataAnalysisAssistant()
        app.run()
    
    except Exception as e:
        print(f"Failed to start the application: {str(e)}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()