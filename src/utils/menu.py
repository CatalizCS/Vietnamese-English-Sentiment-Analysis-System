import os
import sys
from typing import List, Dict, Callable
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel


class TerminalMenu:
    def __init__(self, config=None):
        """Initialize TerminalMenu with configuration"""
        from rich.console import Console
        self.console = Console()
        self.config = config  # Store config object
        self.default_data_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
        )
        os.makedirs(self.default_data_dir, exist_ok=True)
        self.menu_options = {
            "1": "Train new model",
            "2": "Analyze text",
            "3": "Evaluate model",
            "4": "Generate training data",
            "5": "View model metrics",
            "6": "Test model",
            "7": "Restore model from checkpoint",
            "8": "Data collection",
            "9": "Data preprocessing",
            "10": "Model optimization",
            "11": "Export results",
            "12": "API Server",
            "q": "Quit",
        }

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def display_header(self):
        self.clear_screen()
        self.console.print(
            Panel(
                "[bold blue]Sentiment Analysis System[/bold blue]\n"
                "[cyan]Vietnamese-English Text Analysis Tool[/cyan]",
                expand=False,
            )
        )

    def display_menu(self) -> str:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="dim")
        table.add_column("Description")
        table.add_column("Command Example", style="green")

        table.add_row(
            "1", "Train New Model", "python main.py --mode train --language vi"
        )
        table.add_row(
            "2",
            "Analyze Text (Predict)",
            "python main.py --mode predict --language vi --input data.csv --output results.csv",
        )
        table.add_row(
            "3",
            "Evaluate Model",
            "python main.py --mode evaluate --language vi --input test.csv",
        )
        table.add_row(
            "4", "Generate Training Data", "Generate synthetic samples (specify count)"
        )
        table.add_row("5", "View Model Performance", "Display current model metrics")
        table.add_row("6", "Test Model", "Test the model with custom text")
        table.add_row(
            "7", "Restore Model from Checkpoint", "Restore model from saved checkpoint"
        )
        table.add_row("8", "Data Collection", "Collect data from various sources")
        table.add_row("9", "Data Preprocessing", "Clean and prepare collected data")
        table.add_row("10", "Model Optimization", "Tune model parameters")
        table.add_row("11", "Export Results", "Export analysis results and reports")
        table.add_row("12", "API Server", "Start/Stop REST API Server")
        table.add_row("q", "Quit", "Exit the application")

        self.console.print(table)

        choice = Prompt.ask(
            "\n[yellow]Choose an option[/yellow]",
            choices=list(self.menu_options.keys()),
            default="q",
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
            "\n[yellow]Select language[/yellow]", choices=["vi", "en"], default="vi"
        )

    def get_file_path(self, file_type: str) -> str:
        """Get and validate file path"""
        default_path = os.path.join(self.default_data_dir, f"{file_type}.csv")
        while True:
            path = Prompt.ask(
                f"\n[yellow]Enter {file_type} file path[/yellow]", default=default_path
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
                    show_default=True,
                )
                if 100 <= count <= 50000:
                    return count
                self.console.print(
                    "[red]Please enter a number between 100 and 50000[/red]"
                )
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def get_custom_sample_count(self) -> dict:
        """Get custom sample counts for each sentiment category"""
        counts = {}

        self.console.print("\n[cyan]Enter number of samples for each category:[/cyan]")

        categories = {
            "positive": "Tích cực (Positive)",
            "neutral": "Trung tính (Neutral)",
            "negative": "Tiêu cực (Negative)",
        }

        for key, label in categories.items():
            while True:
                try:
                    count = IntPrompt.ask(
                        f"\n[yellow]Số mẫu {label}[/yellow]",
                        default=100,
                        show_default=True,
                    )
                    if 0 <= count <= 10000:
                        counts[key] = count
                        break
                    self.console.print("[red]Vui lòng nhập số từ 0 đến 10000[/red]")
                except ValueError:
                    self.console.print("[red]Vui lòng nhập số hợp lệ[/red]")

        return counts

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
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print("\nResults:")
        print("-" * 50)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment_map[sentiment]}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50)

    def display_emotion_result(self, text, emotion_result):
        """Display emotion analysis results with proper config access"""
        try:
            self.console.print("\n[bold cyan]Analysis Results:[/bold cyan]")
            self.console.print(f"Text: {text}")
            
            if emotion_result:
                # Get sentiment label
                sentiment = emotion_result.get('sentiment')
                sentiment_conf = emotion_result.get('sentiment_confidence', 0)
                
                sentiment_label = "Tích cực" if sentiment == 2 else "Tiêu cực" if sentiment == 0 else "Trung tính"
                self.console.print(f"\nCảm xúc chung: {sentiment_label}")
                self.console.print(f"Độ tin cậy: {sentiment_conf:.2f}")
                
                # Display detailed emotion
                emotion = emotion_result.get('emotion', '')
                emotion_vi = emotion_result.get('emotion_vi', '')
                emoji = emotion_result.get('emotion_emoji', '')
                emotion_conf = emotion_result.get('emotion_confidence', 0)
                
                self.console.print(f"\nBiểu cảm chi tiết: {emotion_vi} {emoji}")
                self.console.print(f"Độ tin cậy: {emotion_conf:.2f}")
                
                # Display emotion scores if available
                if emotion_result.get('emotion_scores'):
                    self.console.print("\nĐiểm số các biểu cảm:")
                    for emotion, score in emotion_result['emotion_scores'].items():
                        if score > 0:
                            self.console.print(f"{emotion}: {score:.2f}")
            else:
                self.console.print("[red]Không thể phân tích cảm xúc[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error displaying results: {str(e)}[/red]")

    def display_data_collection_menu(self):
        """Display data collection options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Collect from Google Play")
        table.add_row("2", "Collect from Shopee")
        table.add_row("3", "Collect from Facebook")
        table.add_row("4", "Import from CSV/Excel")
        table.add_row("5", "Scrape from websites")
        table.add_row("b", "Back to main menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select data collection method[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_preprocessing_menu(self):
        """Display preprocessing options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Clean text data")
        table.add_row("2", "Remove duplicates")
        table.add_row("3", "Balance dataset")
        table.add_row("4", "Filter by criteria")
        table.add_row("5", "Augment data")
        table.add_row("b", "Back to main menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select preprocessing action[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_optimization_menu(self):
        """Display model optimization options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Hyperparameter tuning")
        table.add_row("2", "Feature selection")
        table.add_row("3", "Cross validation")
        table.add_row("4", "Model ensemble")
        table.add_row("5", "Performance analysis")
        table.add_row("b", "Back to main menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select optimization method[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_export_menu(self):
        """Display export options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Export predictions")
        table.add_row("2", "Export model metrics")
        table.add_row("3", "Generate report")
        table.add_row("4", "Export visualizations")
        table.add_row("5", "Export model")
        table.add_row("b", "Back to main menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select export option[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_api_menu(self):
        """Display enhanced API server options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Start API Server")
        table.add_row("2", "Stop API Server")
        table.add_row("3", "View API Status")
        table.add_row("4", "Configure API Settings")
        table.add_row("5", "Test API Endpoints") 
        table.add_row("6", "View Server Logs")
        table.add_row("7", "Monitor Metrics")
        table.add_row("8", "Dashboard")
        table.add_row("9", "Export Data")
        table.add_row("b", "Back to main menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select API option[/yellow]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "b"],
            default="b",
        )

    def display_api_test_menu(self):
        """Display API testing options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Test single text prediction")
        table.add_row("2", "Test batch prediction")
        table.add_row("3", "Test health check")
        table.add_row("b", "Back to API menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select test option[/yellow]",
            choices=["1", "2", "3", "b"],
            default="b",
        )

    def get_api_endpoint(self):
        """Get API endpoint from user"""
        return Prompt.ask(
            "\n[yellow]Enter API endpoint[/yellow]", default="http://localhost:8000"
        )

    def display_api_response(self, response_data):
        """Display API response in a formatted way"""
        if isinstance(response_data, dict):
            self.console.print("\n[cyan]API Response:[/cyan]")
            for key, value in response_data.items():
                self.console.print(f"[green]{key}:[/green] {value}")
        elif isinstance(response_data, list):
            self.console.print("\n[cyan]API Response (Batch):[/cyan]")
            for item in response_data:
                self.console.print("\n[green]Item:[/green]")
                for key, value in item.items():
                    self.console.print(f"[green]{key}:[/green] {value}")

    def display_detailed_test_menu(self):
        """Display detailed testing options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "Test với văn bản đơn lẻ")
        table.add_row("2", "Test với tập dữ liệu mẫu")
        table.add_row("3", "Test hiệu năng (Performance)")
        table.add_row("4", "Test độ chính xác (Accuracy)")
        table.add_row("5", "Test khả năng chịu tải (Load)")
        table.add_row("b", "Quay lại")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Chọn loại test[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_performance_metrics(self, metrics: dict):
        """Display detailed performance metrics"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        self.console.print("\n[cyan]Performance Metrics:[/cyan]")
        self.console.print(table)

    def get_test_batch_size(self) -> int:
        """Get batch size for testing"""
        while True:
            try:
                size = IntPrompt.ask("\n[yellow]Enter batch size[/yellow]", default=10)
                if 1 <= size <= 1000:
                    return size
                self.console.print(
                    "[red]Please enter a number between 1 and 1000[/red]"
                )
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def display_dashboard_menu(self):
        """Display dashboard management options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "View Live Metrics")
        table.add_row("2", "View Historical Data")
        table.add_row("3", "Configure Alerts")
        table.add_row("4", "Export Metrics")
        table.add_row("5", "System Status")
        table.add_row("b", "Back to API menu")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select dashboard option[/yellow]",
            choices=["1", "2", "3", "4", "5", "b"],
            default="b",
        )

    def display_metrics_summary(self, metrics):
        """Display metrics summary in a formatted table"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value")

        # Add rows for each metric
        table.add_row("Uptime", metrics["uptime"])
        table.add_row("Total Requests", str(metrics["total_requests"]))
        table.add_row("Total Errors", str(metrics["total_errors"]))
        table.add_row("Memory Usage", f"{metrics['current_memory_usage']}%")
        table.add_row("CPU Usage", f"{metrics['current_cpu_usage']}%")
        
        # Model status
        for lang, status in metrics["active_models"].items():
            table.add_row(
                f"{lang.upper()} Model",
                "[green]Active[/green]" if status else "[red]Inactive[/red]"
            )

        self.console.print("\n[bold]API Metrics Summary[/bold]")
        self.console.print(table)

    def display_logs_menu(self):
        """Display enhanced log viewing options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "View latest logs")
        table.add_row("2", "View initialization logs")
        table.add_row("3", "View request logs")
        table.add_row("4", "Filter by path")
        table.add_row("5", "Filter by status code")
        table.add_row("6", "Export logs")
        table.add_row("b", "Back")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select option[/yellow]",
            choices=["1", "2", "3", "4", "5", "6", "b"],
            default="b"
        )

    def get_log_lines(self):
        """Get number of log lines to display"""
        return IntPrompt.ask(
            "\n[yellow]Enter number of lines[/yellow]",
            default=50
        )

    def get_log_level(self):
        """Get log level to filter"""
        return Prompt.ask(
            "\n[yellow]Enter log level[/yellow]",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "all"],
            default="all"
        )

    def get_log_time(self):
        """Get time filter for logs"""
        hours = IntPrompt.ask(
            "\n[yellow]Show logs from last N hours[/yellow]",
            default=24
        )
        time_ago = datetime.now() - timedelta(hours=hours)
        return time_ago.isoformat()

    def display_logs(self, logs: List[str]):
        """Display log entries in a formatted table"""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            wrap=True,
            width=self.console.width
        )
        table.add_column("Time", style="cyan")
        table.add_column("Level", style="yellow")
        table.add_column("Message")

        for log in logs:
            try:
                parts = log.split(None, 2)
                if len(parts) >= 3:
                    time, level, msg = parts
                    table.add_row(time, level, msg.strip())
            except:
                table.add_row("", "", log.strip())

        self.console.print(table)

    def display_logs_menu(self):
        """Display enhanced log viewing options"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option")
        table.add_column("Description")

        table.add_row("1", "View Latest Logs")
        table.add_row("2", "View Init Logs")
        table.add_row("3", "View Request Logs")
        table.add_row("4", "Filter by Path")
        table.add_row("5", "Filter by Status Code")
        table.add_row("6", "Filter by Time")
        table.add_row("7", "Filter by Level")
        table.add_row("8", "Search Logs")
        table.add_row("9", "Export Logs")
        table.add_row("b", "Back")

        self.console.print(table)
        return Prompt.ask(
            "\n[yellow]Select option[/yellow]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "b"],
            default="b"
        )

    def get_log_filters(self):
        """Get log filter parameters"""
        filters = {}
        
        # Get log type
        filters["type"] = Prompt.ask(
            "Log type",
            choices=["all", "init", "request"],
            default="all"
        )
        
        # Get path filter if requested
        if Prompt.ask("Filter by path?", choices=["y", "n"], default="n") == "y":
            filters["path"] = self.console.input("Enter path (e.g., /predict): ")
        
        # Get status code if requested
        if Prompt.ask("Filter by status code?", choices=["y", "n"], default="n") == "y":
            filters["status_code"] = IntPrompt.ask("Enter status code (e.g., 200): ")
        
        # Get number of lines
        filters["lines"] = IntPrompt.ask("Number of lines to show", default=50)
        
        return filters

    def format_log_entry(self, log: str) -> str:
        """Format log entry for display"""
        try:
            # Parse log entry parts
            parts = log.split(" ", 3)
            timestamp = parts[0]
            level = parts[1]
            message = parts[2:]

            # Color based on log level
            level_colors = {
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "DEBUG": "blue"
            }
            color = level_colors.get(level.strip("[]"), "white")

            # Format with color
            return f"[cyan]{timestamp}[/cyan] [{color}]{level}[/{color}] {''.join(message)}"
        except:
            return log

    def display_filtered_logs(self, logs: List[str], filters: dict):
        """Display filtered logs with formatting"""
        if not logs:
            self.console.print("[yellow]No logs found matching filters[/yellow]")
            return

        # Show active filters
        self.console.print("\n[bold cyan]Active Filters:[/bold cyan]")
        for key, value in filters.items():
            if value is not None and value != "all":
                self.console.print(f"[green]{key}:[/green] {value}")

        # Show logs
        self.console.print("\n[bold cyan]Log Entries:[/bold cyan]")
        for log in logs:
            self.console.print(self.format_log_entry(log))

    def get_log_search_params(self):
        """Get log search parameters"""
        params = {}
        params["keyword"] = self.console.input("[yellow]Enter search keyword: [/yellow]")
        
        if Prompt.ask("Add date filter?", choices=["y", "n"], default="n") == "y":
            params["from_date"] = self.console.input("[yellow]From date (YYYY-MM-DD): [/yellow]")
            params["to_date"] = self.console.input("[yellow]To date (YYYY-MM-DD): [/yellow]")
            
        if Prompt.ask("Add more filters?", choices=["y", "n"], default="n") == "y":
            params["level"] = Prompt.ask(
                "Log level", 
                choices=["ERROR", "WARNING", "INFO", "DEBUG", "all"],
                default="all"
            )
            params["path"] = self.console.input("[yellow]Filter by path (optional): [/yellow]")
            
        return params

    def get_metrics_filter(self):
        """Get metrics filter parameters"""
        filters = {}
        
        # Time range
        filters["time_range"] = Prompt.ask(
            "Time range",
            choices=["1h", "6h", "24h", "7d", "30d"],
            default="24h"
        )
        
        # Metrics type
        filters["type"] = Prompt.ask(
            "Metrics type",
            choices=["performance", "errors", "models", "all"],
            default="all"
        )
        
        # Aggregation
        if Prompt.ask("Add aggregation?", choices=["y", "n"], default="n") == "y":
            filters["aggregation"] = Prompt.ask(
                "Aggregation period",
                choices=["1min", "5min", "1h", "1d"],
                default="1h"
            )
            
        return filters

    def display_metrics_summary(self, metrics: dict):
        """Display metrics summary"""
        if not metrics:
            self.console.print("[yellow]No metrics data available[/yellow]")
            return
            
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value")

        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            # Add color formatting based on thresholds
            if "error" in key.lower():
                color = "red" if value > 0 else "green"
                formatted_value = f"[{color}]{formatted_value}[/{color}]"
            elif "usage" in key.lower():
                if value > 90:
                    formatted_value = f"[red]{formatted_value}%[/red]"
                elif value > 70:
                    formatted_value = f"[yellow]{formatted_value}%[/yellow]"
                else:
                    formatted_value = f"[green]{formatted_value}%[/green]"
                    
            table.add_row(key, formatted_value)
            
        self.console.print(table)

def get_more_test_samples():
    """Get additional test samples"""
    return [
        ("Sản phẩm rất chất lượng, đóng gói cẩn thận", 2),
        ("Giao hàng chậm, thái độ phục vụ kém", 0), 
        ("Hàng tạm được, giá hơi cao", 1),
        ("Tuyệt vời, sẽ ủng hộ shop dài dài", 2),
        ("Thất vọng về chất lượng sản phẩm", 0),
        ("Hàng đúng như mô tả", 1)
    ]

def get_dashboard_config():
    """Get dashboard configuration options"""
    config = {}
    config['update_interval'] = IntPrompt.ask(
        "Update interval (seconds)",
        default=5
    )
    config['chart_history'] = IntPrompt.ask(
        "Number of data points to show",
        default=50
    )
    config['alert_threshold'] = IntPrompt.ask(
        "Error rate alert threshold (%)",
        default=10
    )
    return config

def get_log_export_format():
    """Get log export format"""
    return Prompt.ask(
        "Export format",
        choices=["txt", "json", "csv"],
        default="txt"
    )

# ...existing code...
