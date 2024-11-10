import os
import sys
from typing import List, Dict, Callable
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel

class TerminalMenu:
    def __init__(self):
        self.console = Console()
        self.default_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data"
        )
        os.makedirs(self.default_data_dir, exist_ok=True)
        self.menu_options = {
            '1': 'Train new model',
            '2': 'Analyze text',
            '3': 'Evaluate model',
            '4': 'Generate training data',
            '5': 'View model metrics',
            '6': 'Test model',
            '7': 'Restore model from checkpoint',
            'q': 'Quit'
        }
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_header(self):
        self.clear_screen()
        self.console.print(Panel(
            "[bold blue]Sentiment Analysis System[/bold blue]\n"
            "[cyan]Vietnamese-English Text Analysis Tool[/cyan]",
            expand=False
        ))

    def display_menu(self) -> str:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="dim")
        table.add_column("Description")
        table.add_column("Command Example", style="green")

        table.add_row(
            "1", 
            "Train New Model",
            "python main.py --mode train --language vi"
        )
        table.add_row(
            "2", 
            "Analyze Text (Predict)",
            "python main.py --mode predict --language vi --input data.csv --output results.csv"
        )
        table.add_row(
            "3", 
            "Evaluate Model",
            "python main.py --mode evaluate --language vi --input test.csv"
        )
        table.add_row(
            "4", 
            "Generate Training Data",
            "Generate synthetic samples (specify count)"
        )
        table.add_row(
            "5", 
            "View Model Performance",
            "Display current model metrics"
        )
        table.add_row(
            "6", 
            "Test Model",
            "Test the model with custom text"
        )
        table.add_row(
            "7", 
            "Restore Model from Checkpoint",
            "Restore model from saved checkpoint"
        )
        table.add_row(
            "q", 
            "Quit",
            "Exit the application"
        )

        self.console.print(table)
        
        choice = Prompt.ask(
            "\n[yellow]Choose an option[/yellow]",
            choices=["1", "2", "3", "4", "5", "6", "7", "q"],
            default="q"
        )
        return choice

    def get_language_choice(self) -> str:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Language")
        table.add_column("Code")
        
        table.add_row("Vietnamese", "vi")
        table.add_row("English", "en")
        
        self.console.print(table)
        
        return Prompt.ask(
            "\n[yellow]Select language[/yellow]",
            choices=["vi", "en"],
            default="vi"
        )

    def get_file_path(self, file_type: str) -> str:
        """Get and validate file path"""
        default_path = os.path.join(self.default_data_dir, f"{file_type}.csv")
        while True:
            path = Prompt.ask(
                f"\n[yellow]Enter {file_type} file path[/yellow]",
                default=default_path
            )
            
            # Ensure directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except Exception as e:
                    self.console.print(f"[red]Error creating directory: {e}[/red]")
                    continue
            
            return path

    def get_sample_count(self) -> int:
        """Get the number of samples to generate"""
        while True:
            try:
                count = IntPrompt.ask(
                    "\n[yellow]Enter number of samples to generate[/yellow]",
                    default=1000,
                    show_default=True
                )
                if 100 <= count <= 50000:
                    return count
                self.console.print("[red]Please enter a number between 100 and 50000[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def display_progress(self, message: str):
        self.console.print(f"[bold blue]>>> {message}...[/bold blue]")

    def display_result(self, success: bool, message: str):
        style = "green" if success else "red"
        self.console.print(f"[{style}]{message}[/{style}]")

    def wait_for_user(self):
        self.console.print("\n[yellow]Press Enter to continue...[/yellow]")
        input()

    def get_test_text(self):
        """Get test text from user"""
        print("\nEnter text to test (or 'q' to quit):")
        return input("> ").strip()

    def display_sentiment_result(self, text, sentiment, confidence):
        """Display sentiment analysis result"""
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        print("\nResults:")
        print("-" * 50)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment_map[sentiment]}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50)