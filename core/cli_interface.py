"""
CLIInterface - Command-line interface handler for the Data Analysis Assistant.
"""

import os
import shlex
from typing import Dict, List, Optional, Tuple


class CLIInterface:
    """Handles command-line interface interactions and command processing."""
    
    def __init__(self, data_manager, analyzer, visualizer):
        """
        Initialize the CLI interface with core components.
        
        Args:
            data_manager: Instance of DatasetManager
            analyzer: Instance of DataAnalyzer
            visualizer: Instance of DataVisualizer
        """
        self.data_manager = data_manager
        self.analyzer = analyzer
        self.visualizer = visualizer
        
        # Command mapping
        self.commands = {
            'help': self.show_help,
            'load': self.handle_load_command,
            'list': self.handle_list_command,
            'view': self.handle_view_command,
            'remove': self.handle_remove_command,
            'analyze': self.handle_analyze_command,
            'clean': self.handle_clean_command,
            'visualize': self.handle_visualize_command,
            'report': self.handle_report_command,
            'plots': self.handle_plots_command,
            'info': self.handle_info_command,
            'overview': self.handle_overview_command,
            'exit': self.handle_exit_command,
            'quit': self.handle_exit_command,
            'clear': self.handle_clear_command
        }
        
        # Analysis sub-commands
        self.analyze_commands = {
            'stats': 'summary_statistics',
            'missing': 'missing_data_report',
            'frequency': 'frequency_counts',
            'filter': 'filter_data_and_display',
            'overview': 'get_dataset_overview'
        }
        
        # Visualization sub-commands
        self.visualize_commands = {
            'hist': 'generate_histogram',
            'histogram': 'generate_histogram',
            'bar': 'generate_bar_chart',
            'scatter': 'generate_scatter_plot',
            'box': 'generate_box_plot'
        }
        
        # Clean sub-commands
        self.clean_commands = {
            'duplicates': 'remove_duplicates',
            'missing': 'handle_missing_values',
            'filter': 'filter_data_and_modify',
            'undo': 'undo_last_clean'
        }
    
    def process_command(self, user_input: str) -> Optional[str]:
        """
        Process user input and execute the appropriate command.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Optional[str]: Special return codes (like "EXIT") or None
        """
        try:
            # Parse command and arguments
            parts = shlex.split(user_input)
            if not parts:
                return None
            
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            # Execute command
            if command in self.commands:
                return self.commands[command](args)
            else:
                print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                return None
                
        except ValueError as e:
            print(f"Invalid command format: {e}")
            print("Use quotes around arguments containing spaces.")
            return None
        except Exception as e:
            print(f"Error processing command: {e}")
            return None
    
    def show_help(self, args: List[str] = None) -> None:
        """Display help information for all commands or a specific command."""
        if args and len(args) > 0:
            self._show_specific_help(args[0])
        else:
            self._show_general_help()
    
    def _show_general_help(self):
        """Display general help with all available commands."""
        print("\n" + "=" * 70)
        print("DATA ANALYSIS ASSISTANT - COMMAND REFERENCE")
        print("=" * 70)
        
        help_sections = [
            ("DATASET MANAGEMENT", [
                ("load <file_path> <dataset_name>", "Load a CSV file as a dataset"),
                ("list", "List all loaded datasets"),
                ("view <dataset_name> [rows]", "View first N rows of a dataset (default: 5)"),
                ("remove <dataset_name>", "Remove a dataset from inventory"),
                ("info <dataset_name>", "Show dataset metadata information")
            ]),
            
            ("DATA ANALYSIS", [
                ("analyze stats <dataset_name>", "Calculate summary statistics"),
                ("analyze missing <dataset_name>", "Generate missing data report"),
                ("analyze frequency <dataset_name> [column]", "Show frequency counts"),
                ("analyze filter <dataset_name> <column> <condition> <value>", "Filter data by condition"),
                ("overview <dataset_name>", "Show comprehensive dataset overview")
            ]),
            
            ("DATA CLEANING", [
                ("clean duplicates <dataset_name>", "Remove duplicate rows"),
                ("clean missing <dataset_name> <method>", "Handle missing values (remove/fill_mean/fill_mode)"),
                ("clean filter <dataset_name> <column> <condition> <value>", "Filter data by condition and apply changes"),
                ("clean undo <dataset_name>", "Undo the last cleaning operation")
            ]),
            
            ("VISUALIZATION", [
                ("visualize hist <dataset_name> <column> [bins]", "Generate histogram"),
                ("visualize bar <dataset_name> <column>", "Generate bar chart"),
                ("visualize scatter <dataset_name> <x_col> <y_col>", "Generate scatter plot"),
                ("visualize box <dataset_name> <column> [group_by]", "Generate box plot")
            ]),
            
            ("REPORTING", [
                ("report <dataset_name>", "Generate comprehensive report"),
                ("plots [dataset_name]", "List generated plots")
            ]),
            
            ("GENERAL", [
                ("help [command]", "Show help (general or for specific command)"),
                ("clear", "Clear the screen"),
                ("exit/quit", "Exit the application")
            ])
        ]
        
        for section_name, commands in help_sections:
            print(f"\n{section_name}:")
            print("-" * len(section_name))
            for cmd, desc in commands:
                print(f"  {cmd:<35} {desc}")
        
        print("\n" + "=" * 70)
        print("Tips:")
        print("- Use quotes around file paths or names with spaces")
        print("- Commands are case-insensitive")
        print("- Use 'help <command>' for detailed help on specific commands")
        print("=" * 70)
    
    def _show_specific_help(self, command: str):
        """Show detailed help for a specific command."""
        detailed_help = {
            'load': {
                'usage': 'load <file_path> <dataset_name>',
                'description': 'Load a CSV file into the system',
                'examples': [
                    'load data/sales.csv sales_data',
                    'load "/path/with spaces/data.csv" my_dataset'
                ],
                'notes': ['Only CSV files are currently supported',
                         'Dataset names must be unique']
            },
            'analyze': {
                'usage': 'analyze <subcommand> <dataset_name> [options]',
                'description': 'Perform various data analysis operations',
                'subcommands': {
                    'stats': 'Calculate summary statistics for numerical columns',
                    'missing': 'Generate report on missing data',
                    'frequency': 'Show frequency counts for categorical columns'
                },
                'examples': [
                    'analyze stats sales_data',
                    'analyze missing customer_data',
                    'analyze frequency sales_data category'
                ]
            },
            'visualize': {
                'usage': 'visualize <plot_type> <dataset_name> <column(s)> [options]',
                'description': 'Generate various types of plots',
                'subcommands': {
                    'hist': 'Generate histogram for numerical column',
                    'bar': 'Generate bar chart for categorical column',
                    'scatter': 'Generate scatter plot for two numerical columns',
                    'box': 'Generate box plot for numerical column'
                },
                'examples': [
                    'visualize hist sales_data price 20',
                    'visualize bar sales_data category',
                    'visualize scatter sales_data price quantity',
                    'visualize box sales_data price category'
                ]
            },
            'clean': {
                'usage': 'clean <operation> <dataset_name> [options]',
                'description': 'Clean and preprocess data',
                'subcommands': {
                    'duplicates': 'Remove duplicate rows',
                    'missing': 'Handle missing values (remove, fill_mean, fill_mode)',
                    'filter': 'Filter data based on conditions'
                },
                'examples': [
                    'clean duplicates sales_data',
                    'clean missing sales_data remove',
                    'clean filter sales_data price > 100'
                ]
            }
        }
        
        if command in detailed_help:
            info = detailed_help[command]
            print(f"\n{'='*50}")
            print(f"HELP: {command.upper()}")
            print(f"{'='*50}")
            print(f"Usage: {info['usage']}")
            print(f"Description: {info['description']}")
            
            if 'subcommands' in info:
                print("\nSubcommands:")
                for subcmd, desc in info['subcommands'].items():
                    print(f"  {subcmd:<12} {desc}")
            
            if 'examples' in info:
                print("\nExamples:")
                for example in info['examples']:
                    print(f"  {example}")
            
            if 'notes' in info:
                print("\nNotes:")
                for note in info['notes']:
                    print(f"  - {note}")
            
            print("="*50)
        else:
            print(f"No detailed help available for '{command}'.")
            print("Use 'help' to see all available commands.")
    
    def handle_load_command(self, args: List[str]) -> None:
        """Handle dataset loading command."""
        if len(args) < 2:
            print("Usage: load <file_path> <dataset_name>")
            print("Example: load data/sales.csv sales_data")
            return
        
        file_path = args[0]
        dataset_name = args[1]
        
        print(f"Loading dataset from '{file_path}' as '{dataset_name}'...")
        success = self.data_manager.load_dataset(file_path, dataset_name)
        
        if success:
            print("✓ Dataset loaded successfully!")
            print("Use 'view' command to preview the data.")
    
    def handle_list_command(self, args: List[str]) -> None:
        """Handle list datasets command."""
        self.data_manager.list_datasets()
    
    def handle_view_command(self, args: List[str]) -> None:
        """Handle view dataset command."""
        if len(args) < 1:
            print("Usage: view <dataset_name> [rows]")
            print("Example: view sales_data 10")
            return
        
        dataset_name = args[0]
        n_rows = 5
        
        if len(args) > 1:
            try:
                n_rows = int(args[1])
                if n_rows <= 0:
                    print("Number of rows must be positive.")
                    return
            except ValueError:
                print("Invalid number of rows. Using default (5).")
        
        self.data_manager.view_dataset(dataset_name, n_rows)
    
    def handle_remove_command(self, args: List[str]) -> None:
        """Handle remove dataset command."""
        if len(args) < 1:
            print("Usage: remove <dataset_name>")
            print("Example: remove sales_data")
            return
        
        dataset_name = args[0]
        
        # Confirm removal
        confirm = input(f"Are you sure you want to remove dataset '{dataset_name}'? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Removal cancelled.")
            return
        
        success = self.data_manager.remove_dataset(dataset_name)
        if success:
            print("✓ Dataset removed successfully!")
    
    def handle_analyze_command(self, args: List[str]) -> None:
        """Handle data analysis commands."""
        if len(args) < 2:
            print("Usage: analyze <subcommand> <dataset_name> [options]")
            print("Subcommands: stats, missing, frequency")
            print("Examples:")
            print("  analyze stats sales_data")
            print("  analyze missing customer_data")
            print("  analyze frequency sales_data category")
            return
        
        subcommand = args[0]
        dataset_name = args[1]
        
        if subcommand not in self.analyze_commands:
            print(f"Unknown analyze subcommand: '{subcommand}'")
            print(f"Available subcommands: {', '.join(self.analyze_commands.keys())}")
            return
        
        method_name = self.analyze_commands[subcommand]
        method = getattr(self.analyzer, method_name)
        
        try:
            if subcommand == 'frequency' and len(args) > 2:
                # Frequency analysis with specific column
                column = args[2]
                method(dataset_name, column)
            elif subcommand == 'filter' and len(args) >= 5:
                # Data filtering
                column = args[2]
                condition = args[3]
                value = args[4]
                
                # Try to convert value to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
                
                method(dataset_name, column, condition, value)
            else:
                method(dataset_name)
        except Exception as e:
            print(f"Error during analysis: {e}")
    
    def handle_clean_command(self, args: List[str]) -> None:
        """Handle data cleaning commands."""
        if len(args) < 2:
            print("Usage: clean <operation> <dataset_name> [options]")
            print("Operations: duplicates, missing, filter")
            print("Examples:")
            print("  clean duplicates sales_data")
            print("  clean missing sales_data remove")
            print("  clean filter sales_data price > 100")
            return
        
        subcommand = args[0]
        dataset_name = args[1]
        
        if subcommand not in self.clean_commands:
            print(f"Unknown clean operation: '{subcommand}'")
            print(f"Available operations: {', '.join(self.clean_commands.keys())}")
            return
        
        method_name = self.clean_commands[subcommand]
        method = getattr(self.analyzer, method_name)
        
        try:
            if subcommand == 'missing' and len(args) > 2:
                # Missing value handling with method
                clean_method = args[2]
                if clean_method not in ['remove', 'fill_mean', 'fill_mode']:
                    print("Invalid method. Use: remove, fill_mean, or fill_mode")
                    return
                method(dataset_name, clean_method)
            
            elif subcommand == 'filter' and len(args) >= 5:
                # Data filtering
                column = args[2]
                condition = args[3]
                value = args[4]
                
                # Try to convert value to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
                
                method(dataset_name, column, condition, value)
            
            else:
                method(dataset_name)
                
        except Exception as e:
            print(f"Error during cleaning: {e}")
    
    def handle_visualize_command(self, args: List[str]) -> None:
        """Handle visualization commands."""
        if len(args) < 3:
            print("Usage: visualize <plot_type> <dataset_name> <column(s)> [options]")
            print("Plot types: hist, bar, scatter, box")
            print("Examples:")
            print("  visualize hist sales_data price")
            print("  visualize bar sales_data category")
            print("  visualize scatter sales_data price quantity")
            print("  visualize box sales_data price category")
            return
        
        plot_type = args[0]
        dataset_name = args[1]
        
        if plot_type not in self.visualize_commands:
            print(f"Unknown plot type: '{plot_type}'")
            print(f"Available plot types: {', '.join(self.visualize_commands.keys())}")
            return
        
        method_name = self.visualize_commands[plot_type]
        method = getattr(self.visualizer, method_name)
        
        try:
            if plot_type in ['hist', 'histogram']:
                column = args[2]
                bins = 30
                if len(args) > 3:
                    try:
                        bins = int(args[3])
                    except ValueError:
                        print("Invalid bins value. Using default (30).")
                method(dataset_name, column, bins)
            
            elif plot_type == 'bar':
                column = args[2]
                method(dataset_name, column)
            
            elif plot_type == 'scatter':
                if len(args) < 4:
                    print("Scatter plot requires two columns: x_column y_column")
                    return
                x_column = args[2]
                y_column = args[3]
                method(dataset_name, x_column, y_column)
            
            elif plot_type == 'box':
                column = args[2]
                group_by = args[3] if len(args) > 3 else None
                method(dataset_name, column, group_by)
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
    
    def handle_report_command(self, args: List[str]) -> None:
        """Handle report generation command."""
        if len(args) < 1:
            print("Usage: report <dataset_name>")
            print("Example: report sales_data")
            return
        
        dataset_name = args[0]
        print(f"Generating comprehensive report for '{dataset_name}'...")
        
        success = self.visualizer.generate_report(dataset_name)
        if success:
            print("✓ Report generated successfully!")
    
    def handle_plots_command(self, args: List[str]) -> None:
        """Handle list plots command."""
        if len(args) == 0:
            self.visualizer.list_generated_plots()
        else:
            dataset_name = args[0]
            self.visualizer.list_generated_plots(dataset_name)
    
    def handle_info_command(self, args: List[str]) -> None:
        """Handle dataset info command."""
        if len(args) < 1:
            print("Usage: info <dataset_name>")
            print("Example: info sales_data")
            return
        
        dataset_name = args[0]
        info = self.data_manager.get_dataset_info(dataset_name)
        
        if info:
            print(f"\nDataset Information: '{dataset_name}'")
            print("-" * 40)
            print(f"File Path: {info['file_path']}")
            print(f"Loaded Date: {info['loaded_date']}")
            print(f"Rows: {info['rows']:,}")
            print(f"Columns: {info['columns']}")
            print(f"Column Names: {', '.join(info['column_names'])}")
        else:
            print(f"Dataset '{dataset_name}' not found.")
    
    def handle_overview_command(self, args: List[str]) -> None:
        """Handle dataset overview command."""
        if len(args) < 1:
            print("Usage: overview <dataset_name>")
            print("Example: overview sales_data")
            return
        
        dataset_name = args[0]
        self.analyzer.get_dataset_overview(dataset_name)
    
    def handle_clear_command(self, args: List[str]) -> None:
        """Handle clear screen command."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Screen cleared.")
    
    def handle_exit_command(self, args: List[str]) -> str:
        """Handle exit command."""
        return "EXIT"
    
    def validate_dataset_exists(self, dataset_name: str) -> bool:
        """
        Validate that a dataset exists in the system.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            bool: True if dataset exists, False otherwise
        """
        return self.data_manager._dataset_exists(dataset_name)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return self.data_manager.get_available_datasets()
    
    def suggest_command(self, invalid_command: str) -> None:
        """Suggest similar commands for invalid input."""
        # Simple suggestion based on command similarity
        suggestions = []
        for cmd in self.commands.keys():
            if invalid_command in cmd or cmd in invalid_command:
                suggestions.append(cmd)
        
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}?")
    
    def autocomplete_dataset_name(self, partial_name: str) -> List[str]:
        """Return dataset names that match partial input."""
        available = self.get_available_datasets()
        return [name for name in available if name.startswith(partial_name)]