import argparse
import json
from logging import config
import subprocess
from anyio import open_process
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
import joblib
from datetime import datetime
import psutil
from rich.prompt import Prompt
import requests
import socket
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.data_collection import DataCollector
from src.utils.server_utils import force_kill_port, is_port_in_use
from src.api.app import app, start_api_server, stop_api_server, get_api_status
from src.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureExtractor
from src.models.model_trainer import EnhancedModelTrainer
from src.models.model_predictor import SentimentPredictor
from src.utils.evaluation import ModelEvaluator
from src.utils.logger import Logger
from src.utils.menu import TerminalMenu
from scripts.generate_training_data import TrainingDataGenerator
from src.visualization import ModelVisualizer

# Add logger initialization at the top level
logger = Logger(__name__).logger


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict", "evaluate"],
        help="Mode of operation: train, predict, or evaluate",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "vi"],
        help="Language for sentiment analysis (en/vi)",
    )
    parser.add_argument(
        "--input", type=str, help="Input file path for prediction or evaluation"
    )
    parser.add_argument(
        "--output", type=str, help="Output file path for saving results"
    )
    return parser.parse_args()


def train(language: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting training for {language} language")

    try:
        # Initialize components with proper error handling
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(language, config)
        feature_extractor = None

        try:
            feature_extractor = FeatureExtractor(language, config)
            if feature_extractor is None:
                raise ValueError("Failed to initialize feature extractor")
        except Exception as fe:
            logger.error(f"Feature extractor initialization failed: {str(fe)}")
            raise

        # initialize model trainer
        model_trainer = EnhancedModelTrainer(language, config)
        if feature_extractor:
            model_trainer.feature_extractor = feature_extractor

        # Load and validate data with error checks
        df = data_loader.load_data(language)
        if df is None or df.empty:
            raise ValueError("No data loaded")

        processed_df = preprocessor.preprocess(df)
        if processed_df is None or processed_df.empty:
            raise ValueError("No valid samples after preprocessing")

        # Split and extract features with validation
        train_df, test_df = data_loader.split_data(processed_df)
        if train_df.empty or test_df.empty:
            raise ValueError("Error in train-test split")

        X_train, y_train = data_loader.get_features_and_labels(train_df)
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("No training samples available")

        # Record start time
        start_time = datetime.now()

        # Train model
        model = model_trainer.train_with_grid_search(X_train, y_train)

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate on test set
        X_test, y_test = data_loader.get_features_and_labels(test_df)
        X_test_features = feature_extractor.extract_features(X_test)

        # Get final metrics
        final_metrics = {
            "test_score": model["rf"].score(X_test_features, y_test),
            "feature_importance": getattr(model["rf"], "feature_importances_", None),
            "training_time": training_time,
        }

        # Save final model with metrics
        model_trainer.save_final_model(model, final_metrics)

        logger.info("Training completed successfully")
        logger.info(f"Total training time: {training_time:.2f} seconds")
        return model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Load last checkpoint if available
        checkpoint_dir = os.path.join(config.DATA_DIR, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(
                [
                    f
                    for f in os.listdir(checkpoint_dir)
                    if f.startswith(f"{language}_checkpoint")
                ]
            )
            if checkpoints:
                latest_checkpoint = joblib.load(
                    os.path.join(checkpoint_dir, checkpoints[-1])
                )
                logger.info(f"Loaded latest checkpoint from {checkpoints[-1]}")
                return latest_checkpoint["model_state"]
        raise e


def predict(language: str, input_file: str, output_file: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting prediction for {language} language")

    # Initialize components
    predictor = SentimentPredictor(language, config)
    preprocessor = DataPreprocessor(language, config)
    feature_extractor = FeatureExtractor(language, config)

    # Load and process input data
    df = pd.read_csv(input_file)
    processed_df = preprocessor.preprocess(df)
    features = feature_extractor.extract_features(processed_df["cleaned_text"])

    # Make predictions
    predictions = predictor.predict(features)
    df["sentiment"] = predictions
    df.to_csv(output_file, index=False)

    logger.info(f"Predictions saved to {output_file}")


def evaluate(language: str, input_file: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting evaluation for {language} language")

    # Initialize components
    predictor = SentimentPredictor(language, config)
    preprocessor = DataPreprocessor(language, config)
    feature_extractor = FeatureExtractor(language, config)
    evaluator = ModelEvaluator(language)

    # Load and process test data
    df = pd.read_csv(input_file)
    processed_df = preprocessor.preprocess(df)
    features = feature_extractor.extract_features(processed_df["cleaned_text"])

    # Make predictions and evaluate
    predictions = predictor.predict(features)
    probabilities = predictor.predict_proba(features)
    results = evaluator.evaluate(processed_df["label"], predictions, probabilities)

    logger.info("Evaluation results:")
    logger.info(results["classification_report"])


def validate_paths(input_file: str = None, output_file: str = None):
    """Validate input and output paths"""
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


def test_model(language: str, config: Config):
    """Interactive model testing"""
    logger = Logger(__name__).logger
    menu = TerminalMenu()  # Move menu initialization to top

    try:
        # Initialize components
        predictor = SentimentPredictor(language, config)
        preprocessor = DataPreprocessor(language, config)
        feature_extractor = FeatureExtractor(language, config)

        while True:
            try:
                # Get test text
                test_text = menu.get_test_text()
                if test_text.lower() == "q":
                    break

                # Create DataFrame with single text
                df = pd.DataFrame({"text": [test_text]})

                # Process text
                processed_df = preprocessor.preprocess(df)
                if processed_df.empty:
                    menu.display_result(False, "Text preprocessing failed")
                    continue

                # Extract features
                features = feature_extractor.extract_features(
                    processed_df["cleaned_text"]
                )
                if features is None or features.size == 0:
                    menu.display_result(False, "Feature extraction failed")
                    continue

                # Get predictions with emotion analysis
                emotion_result = predictor.predict_emotion(features, test_text)

                # Display results
                menu.display_emotion_result(test_text, emotion_result)

            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                menu.display_result(False, f"Error: {str(e)}")

    except Exception as e:
        logger.error(f"Test model error: {str(e)}")
        menu.display_result(False, f"Error: {str(e)}")

    return


def restore_model(language: str, config: Config):
    """Restore model from checkpoint"""
    logger = Logger(__name__).logger
    trainer = EnhancedModelTrainer(language, config)

    try:
        # List available checkpoints
        checkpoints = trainer.list_checkpoints()
        if not checkpoints:
            logger.warning("No checkpoints available")
            return None

        # Display available checkpoints
        print("\nAvailable checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. {cp['filename']}")
            print(f"   Timestamp: {cp['timestamp']}")
            print(f"   Epoch: {cp['epoch']}")
            print(f"   Score: {cp['metrics']:.4f if cp['metrics'] else 'N/A'}")

        # Get user choice
        choice = input(
            "\nEnter checkpoint number to restore (or press Enter for latest): "
        )
        if choice.strip():
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_name = checkpoints[idx]["filename"]
            else:
                raise ValueError("Invalid checkpoint number")
        else:
            checkpoint_name = None

        # Restore model
        model, metrics = trainer.restore_from_checkpoint(checkpoint_name)
        if model is not None:
            logger.info("Model restored successfully")
            logger.info(f"Model metrics: {metrics}")
            return model

    except Exception as e:
        logger.error(f"Error restoring model: {str(e)}")
        return None


def display_metrics(language: str, config: Config):
    """Display current model metrics and performance visualization"""
    logger = Logger(__name__).logger
    model_path = os.path.join(
        config.DATA_DIR, "models", f"{language}_sentiment_model.pkl"
    )
    metrics_img_path = os.path.join(
        config.DATA_DIR, "metrics", f"{language}_model_metrics.png"
    )
    os.makedirs(os.path.dirname(metrics_img_path), exist_ok=True)

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {language}")

        # Load model info
        model_info = joblib.load(model_path)
        print("\nModel Info Structure:", model_info.keys())  # Debug info

        # Print basic info
        print("\n=== Model Performance Metrics ===")
        print(f"Language: {language.upper()}")
        print(f"Timestamp: {model_info.get('config', {}).get('timestamp', 'N/A')}")

        # Extract metrics with proper validation
        if "metrics" not in model_info:
            print("No metrics found in model_info")
            print("Available keys:", model_info.keys())
            raise ValueError("No metrics found in model")

        metrics = model_info["metrics"]
        print("\nMetrics Structure:", metrics.keys())  # Debug info

        # Extract scores and times
        scores = []
        times = []
        model_names = []

        if isinstance(metrics, dict):
            model_data = metrics.get("models", {})
            if not model_data:
                model_data = {"base_model": metrics}

            print("\nProcessing models:", list(model_data.keys()))  # Debug info

            for model_name, model_metrics in model_data.items():
                if isinstance(model_metrics, dict):
                    # Try to get score from various possible keys
                    score = None
                    for score_key in [
                        "best_score",
                        "test_score",
                        "f1_score",
                        "accuracy",
                    ]:
                        if score_key in model_metrics:
                            score = float(model_metrics[score_key])
                            break

                    time = float(model_metrics.get("training_time", 0))

                    if score is not None:
                        scores.append(score)
                        times.append(time)
                        model_names.append(model_name)

                        # Print detailed metrics
                        print(f"\n{model_name.upper()} Model:")
                        print(f"Score: {score:.4f}")
                        print(f"Training Time: {time:.2f}s")

                        if "parameters" in model_metrics:
                            print("Parameters:")
                            for param, value in model_metrics["parameters"].items():
                                print(f"  {param}: {value}")

        if not scores:
            print("\nNo scores found in metrics structure:")
            print("Model data:", model_data)  # Debug info
            raise ValueError("No valid scores found in metrics")

        # Rest of the visualization code...
        plt.figure(figsize=(15, 8))

        # Score comparison plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(scores)), scores, color=["blue", "green", "red"])
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.title(f"Model Performance Comparison\n{language.upper()}")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # Add score labels
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.01,
                f"{score:.3f}",
                ha="center",
            )

        # Time comparison plot
        if any(times):
            plt.subplot(1, 2, 2)
            bars = plt.bar(
                range(len(times)), times, color=["lightblue", "lightgreen", "pink"]
            )
            plt.xticks(range(len(model_names)), model_names, rotation=45)
            plt.title("Training Time Comparison")
            plt.xlabel("Models")
            plt.ylabel("Time (seconds)")

            # Add time labels
            for bar, time in zip(bars, times):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    time + max(times) * 0.02,
                    f"{time:.1f}s",
                    ha="center",
                )

        plt.tight_layout()
        plt.savefig(metrics_img_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\nMetrics visualization saved to: {metrics_img_path}")

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        print(f"Error: Could not display metrics: {str(e)}")
        import traceback

        print("\nFull error traceback:")
        print(traceback.format_exc())


def handle_data_collection(menu, language, config):
    """Handle data collection menu options"""
    while True:
        choice = menu.display_data_collection_menu()
        if choice == "b":
            break

        try:
            # Get custom sample counts
            sample_counts = menu.get_custom_sample_count()
            collector = DataCollector(config)

            if choice == "1":
                app_id = menu.console.input("Enter Google Play app ID: ")
                df = collector.collect_google_play_reviews(
                    app_id, language, sample_counts=sample_counts
                )

            elif choice == "2":
                product_ids = menu.console.input(
                    "Enter Shopee product IDs (comma-separated): "
                )
                df = collector.collect_shopee_reviews(
                    product_ids.split(","), sample_counts=sample_counts
                )

            elif choice == "3":
                token = menu.console.input("Enter Facebook access token: ")
                post_ids = menu.console.input("Enter post IDs (comma-separated): ")
                df = collector.collect_facebook_comments(
                    post_ids.split(","), token, sample_counts=sample_counts
                )

            elif choice == "4":
                file_path = menu.get_file_path("input")
                df = collector.collect_manual_reviews(
                    file_path, sample_counts=sample_counts
                )

            elif choice == "5":
                url = menu.console.input("Enter website URL: ")
                df = collector.collect_from_website(url, sample_counts=sample_counts)

            if not df.empty:
                # Print collection statistics
                stats = df["sentiment"].value_counts()
                menu.console.print("\n[cyan]Collection Statistics:[/cyan]")
                menu.console.print(f"Positive samples: {stats.get(2, 0)}")
                menu.console.print(f"Neutral samples: {stats.get(1, 0)}")
                menu.console.print(f"Negative samples: {stats.get(0, 0)}")

                output_path = menu.get_file_path("output")
                collector.save_collected_data(df, output_path)
                menu.display_result(True, f"Collected {len(df)} samples total")

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")
        menu.wait_for_user()


def run_detailed_test(menu, endpoint: str, language: str):
    """Run detailed API testing"""
    try:
        # Test single text
        response = requests.post(
            f"{endpoint}/predict",
            json={"text": "Tôi rất thích sản phẩm này", "language": language},
        )

        # Calculate basic metrics
        metrics = {
            "Response Time": response.elapsed.total_seconds(),
            "Status Code": response.status_code,
            "Response Size": len(response.content),
        }

        # Test accuracy if response is successful
        if response.status_code == 200:
            result = response.json()
            metrics["Confidence"] = result.get("confidence", 0)
            metrics["Sentiment Score"] = result.get("sentiment", -1)

        menu.display_performance_metrics(metrics)
        return metrics

    except Exception as e:
        menu.display_result(False, f"Test failed: {str(e)}")
        return None


def test_api(menu, language):
    """Test API endpoints with detailed options"""
    while True:
        choice = menu.display_api_test_menu()
        if choice == "b":
            break

        try:
            endpoint = menu.get_api_endpoint()

            if choice == "1":  # Test single text prediction
                text = menu.get_test_text()
                if text.lower() == "q":
                    continue

                response = requests.post(
                    f"{endpoint}/predict", json={"text": text, "language": language}
                )
                if response.status_code == 200:
                    menu.display_api_response(response.json())
                else:
                    menu.display_result(False, f"API Error: {response.text}")

            elif choice == "2":  # Test batch prediction
                texts = []
                menu.console.print("\nEnter texts (empty line to finish):")
                while True:
                    text = input("> ").strip()
                    if not text:
                        break
                    texts.append(text)

                if texts:
                    response = requests.post(
                        f"{endpoint}/batch", json={"texts": texts, "language": language}
                    )
                    if response.status_code == 200:
                        menu.display_api_response(response.json()["results"])
                    else:
                        menu.display_result(False, f"API Error: {response.text}")

            elif choice == "3":  # Test health check
                response = requests.get(f"{endpoint}/health")
                if response.status_code == 200:
                    menu.display_api_response(response.json())
                else:
                    menu.display_result(False, f"API Error: {response.text}")

            elif choice == "4":  # Detailed testing
                while True:
                    test_choice = menu.display_detailed_test_menu()
                    if test_choice == "b":
                        break

                    if test_choice == "1":
                        # Test với văn bản đơn lẻ
                        text = menu.get_test_text()
                        if text.lower() == "q":
                            continue

                        response = requests.post(
                            f"{endpoint}/predict",
                            json={"text": text, "language": language},
                        )
                        metrics = run_detailed_test(menu, endpoint, language)
                        if response.status_code == 200:
                            menu.display_api_response(response.json())

                    elif test_choice == "2":
                        # Test với tập dữ liệu mẫu
                        batch_size = menu.get_test_batch_size()
                        test_texts = [
                            "Sản phẩm tốt",
                            "Dịch vụ kém",
                            "Bình thường",
                            # Thêm các mẫu test khác...
                        ][:batch_size]

                        response = requests.post(
                            f"{endpoint}/batch",
                            json={"texts": test_texts, "language": language},
                        )
                        if response.status_code == 200:
                            menu.display_api_response(response.json()["results"])

                    elif test_choice == "3":
                        # Test hiệu năng
                        menu.display_progress("Testing performance")
                        metrics = run_detailed_test(menu, endpoint, language)

                    elif test_choice == "4":
                        # Test độ chính xác
                        menu.display_progress("Testing accuracy")
                        # Implement accuracy testing with known samples
                        test_samples = [
                            ("Sản phẩm rất tốt, tôi rất thích", 2),  # Positive
                            ("Sản phẩm này thật tệ", 0),  # Negative
                            ("Sản phẩm tạm được", 1),  # Neutral
                        ]

                        correct = 0
                        for text, expected in test_samples:
                            response = requests.post(
                                f"{endpoint}/predict",
                                json={"text": text, "language": language},
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result["sentiment"] == expected:
                                    correct += 1

                        accuracy = correct / len(test_samples)
                        menu.display_performance_metrics({"Accuracy": accuracy})

                    elif test_choice == "5":
                        # Test khả năng chịu tải
                        menu.display_progress("Testing load capacity")
                        batch_sizes = [10, 50, 100]
                        load_metrics = {}

                        for size in batch_sizes:
                            test_texts = ["Test text"] * size
                            start_time = datetime.now()
                            response = requests.post(
                                f"{endpoint}/batch",
                                json={"texts": test_texts, "language": language},
                            )
                            processing_time = (
                                datetime.now() - start_time
                            ).total_seconds()

                            load_metrics[f"Batch {size}"] = {
                                "Processing Time": processing_time,
                                "Time per Text": processing_time / size,
                            }

                        menu.display_performance_metrics(
                            {"Load Test Results": load_metrics}
                        )

        except requests.exceptions.ConnectionError:
            menu.display_result(False, "Could not connect to API server")
        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")

        menu.wait_for_user()


def find_free_port(start_port=7270, end_port=7280):
    """Find first available port in range"""
    import socket

    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    return None


def force_kill_port(port):
    """Force kill any process using the specified port"""
    try:
        # Get all network connections
        connections = psutil.net_connections()
        for conn in connections:
            try:
                # Check if connection is using our port
                if conn.laddr.port == port:
                    # Find and kill the process
                    try:
                        proc = psutil.Process(conn.pid)
                        proc.kill()
                        return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (AttributeError, TypeError):
                continue
        return False
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


def end_task_by_name(name: str) -> bool:
    """End task by process name"""
    try:
        killed = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                # Check process name first
                if name.lower() in proc.info["name"].lower():
                    proc.kill()
                    killed = True
                    logger.info(
                        f"Killed process: {proc.info['name']} (PID: {proc.info['pid']})"
                    )
                    continue

                # Then check cmdline if available
                cmdline = proc.info.get("cmdline")
                if cmdline:
                    if any(name.lower() in cmd.lower() for cmd in cmdline):
                        proc.kill()
                        killed = True
                        logger.info(
                            f"Killed process: {proc.info['name']} (PID: {proc.info['pid']})"
                        )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ) as e:
                logger.debug(f"Skip process: {e}")
                continue
        return killed
    except Exception as e:
        logger.error(f"Error ending task {name}: {e}")
        return False


def cleanup_api_servers():
    """Stop all running API server processes"""
    try:
        killed = False
        target_ports = [7270, 8000]  # Define target ports

        # Get all connections using target ports
        connections = psutil.net_connections()
        target_pids = set()

        for conn in connections:
            try:
                if hasattr(conn, "laddr") and conn.laddr.port in target_ports:
                    target_pids.add(conn.pid)
            except (AttributeError, TypeError):
                continue

        # Only kill Python processes that are using our target ports
        for pid in target_pids:
            try:
                proc = psutil.Process(pid)
                # Verify it's a Python process before killing
                if "python" in proc.name().lower():
                    proc.kill()
                    killed = True
                    logger.info(f"Killed Python process using port {port}: PID {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Skip process {pid}: {e}")
                continue

        # Verify ports are freed
        for port in target_ports:
            if is_port_in_use(port):
                logger.warning(f"Port {port} is still in use")

        # Small delay to ensure processes are terminated
        if killed:
            time.sleep(1)

        return True
    except Exception as e:
        logger.error(f"Error cleaning up servers: {e}")
        return False


def start_api_server(host="0.0.0.0", port=7270):
    """Start API server with improved management"""
    global server_process
    try:
        # Kill any existing process using the port
        if is_port_in_use(port):
            force_kill_port(port)
            time.sleep(1)  # Wait for port to be freed

        # Set environment variables for better stability
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = project_root

        # Start server with improved settings
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            f"--host={host}",
            f"--port={port}",
            "--reload",
            "--reload-dir",
            "src",
            "--workers",
            "1",  # Single worker for stability
            "--timeout-keep-alive",
            "30",
            "--limit-concurrency",
            "100",
            "--log-level",
            "info",
        ]

        server_process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Verify server started successfully
        time.sleep(2)
        if server_process.poll() is not None:
            stderr = server_process.stderr.read()
            raise RuntimeError(f"Server failed to start: {stderr}")

        logger.info(f"API server started on {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        return False


# Improved server stop function
def stop_api_server():
    """Stop API server with improved cleanup"""
    global server_process
    try:
        # Try graceful shutdown first
        if server_process and server_process.poll() is None:
            requests.post(f"http://localhost:{config.API_CONFIG['PORT']}/shutdown")
            time.sleep(2)

        # Force kill if still running
        if server_process and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)

        # Additional cleanup
        cleanup_api_servers()
        time.sleep(1)

        server_process = None
        return True

    except Exception as e:
        logger.error(f"Error stopping API server: {e}")
        # Force cleanup
        cleanup_api_servers()
        return False


def handle_api_server(menu, language, config):
    """Handle API server operations"""
    while True:
        choice = menu.display_api_menu()
        if choice == "b":
            break

        try:
            if choice == "1":  # Start Server
                menu.display_progress("Starting API server")
                # Get user choice for new terminal

                success = start_api_server(
                    config.API_CONFIG["HOST"], config.API_CONFIG["PORT"]
                )

                if success:
                    menu.display_result(
                        "API server started on http://localhost:{config.API_CONFIG['PORT']}"
                    )
                else:
                    menu.display_result(False, "Failed to start server")

            elif choice == "2":  # Stop Server
                menu.display_progress("Stopping API server")
                try:
                    # Try graceful shutdown first
                    response = requests.post(
                        f"http://localhost:{config.API_CONFIG['PORT']}/shutdown"
                    )
                    if response.status_code == 200:
                        menu.display_result(True, "Server stopped gracefully")
                    else:
                        # Only kill process on used port
                        if force_kill_port(config.API_CONFIG["PORT"]):
                            menu.display_result(
                                True, "Server force stopped successfully"
                            )
                        else:
                            menu.display_result(False, "Failed to stop server")
                except:
                    # If server is not responding, kill specific port
                    if force_kill_port(config.API_CONFIG["PORT"]):
                        menu.display_result(True, "Server force stopped successfully")
                    else:
                        menu.display_result(False, "Failed to stop server")

            elif choice == "3":  # View Status
                try:
                    status = get_api_status()
                    menu.console.print("\n[cyan]API Server Status:[/cyan]")
                    menu.console.print(
                        f"Running: {'Yes' if status['running'] else 'No'}"
                    )
                    menu.console.print("\nLoaded Models:")
                    for lang, loaded in status["models_loaded"].items():
                        menu.console.print(
                            f"{lang.upper()}: {'Loaded' if loaded else 'Not loaded'}"
                        )
                except Exception as e:
                    menu.display_result(False, f"Error getting status: {e}")

            elif choice == "4":  # Configure Settings
                menu.console.print("\n[cyan]API Configuration:[/cyan]")
                menu.console.print(f"Host: {config.API_CONFIG['HOST']}")
                menu.console.print(f"Port: {config.API_CONFIG['PORT']}")
                menu.console.print(f"Workers: {config.API_CONFIG['WORKERS']}")
                menu.console.print(f"Request Timeout: {config.API_CONFIG['TIMEOUT']}s")
                menu.console.print(
                    f"Max Request Size: {config.API_CONFIG['MAX_REQUEST_SIZE']/1024/1024}MB"
                )
                menu.console.print("\nConfiguration can be modified in config.py")

            elif choice == "5":  # Test API
                test_api(menu, language)

            elif choice == "6":  # View Logs
                subchoice = menu.display_logs_menu()
                if subchoice != "b":
                    params = {}

                    if subchoice in ["1", "2", "3"]:  # View logs
                        params["lines"] = menu.get_log_lines()
                        if subchoice == "2":
                            params["type"] = "init"
                        elif subchoice == "3":
                            params["type"] = "request"

                    elif subchoice == "4":  # Filter by path
                        params["path"] = menu.console.input("Enter path: ")

                    elif subchoice == "5":  # Filter by status
                        params["status_code"] = menu.console.input(
                            "Enter status code: "
                        )

                    elif subchoice == "6":  # Filter by time
                        params["since"] = menu.get_log_time()

                    elif subchoice == "7":  # Filter by level
                        params["level"] = menu.get_log_level()

                    elif subchoice == "8":  # Search logs
                        params.update(menu.get_log_search_params())

                    elif subchoice == "9":  # Export logs
                        params["export"] = True
                        params["output"] = menu.get_file_path("logs")

                    # Get and display logs
                    display_server_logs(menu, config, params)

            elif choice == "7":  # Monitor Metrics
                subchoice = menu.display_metrics_menu()
                if subchoice != "b":
                    filters = menu.get_metrics_filter()
                    metrics = ModelEvaluator(language).get_filtered_metrics(filters)
                    menu.display_metrics_summary(metrics)

            elif choice == "8":  # Open Dashboard
                import webbrowser

                port = config.API_CONFIG["PORT"]
                webbrowser.open(f"http://localhost:{port}/dashboard")

            elif choice == "9":  # Export Data
                export_path = menu.get_file_path("export")
                data = {
                    "metrics": ModelEvaluator(language).get_filtered_metrics(
                        {"time_range": "7d"}
                    ),
                    "logs": get_server_logs(config.API_CONFIG["PORT"], lines=1000),
                }
                with open(export_path, "w") as f:
                    json.dump(data, f, indent=2)
                menu.display_result(True, f"Data exported to {export_path}")

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")

        menu.wait_for_user()


def handle_preprocessing_menu(menu, language, config):
    """Handle preprocessing options"""
    while True:
        subchoice = menu.display_preprocessing_menu()
        if subchoice == "b":
            break

        try:
            if subchoice == "1":  # Clean text
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")
                preprocessor = DataPreprocessor(language, config)

                df = pd.read_csv(input_file)
                processed_df = preprocessor.preprocess(df)
                processed_df.to_csv(output_file, index=False)
                menu.display_result(True, f"Cleaned data saved to {output_file}")

            elif subchoice == "2":  # Remove duplicates
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")

                df = pd.read_csv(input_file)
                df.drop_duplicates(subset=["text"], inplace=True)
                df.to_csv(output_file, index=False)
                menu.display_result(
                    True, f"Duplicates removed and saved to {output_file}"
                )

            elif subchoice == "3":  # Balance dataset
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")

                df = pd.read_csv(input_file)
                min_samples = df["label"].value_counts().min()
                balanced_df = pd.concat(
                    [
                        df[df["label"] == label].sample(n=min_samples, random_state=42)
                        for label in df["label"].unique()
                    ]
                )
                balanced_df.to_csv(output_file, index=False)
                menu.display_result(True, f"Balanced dataset saved to {output_file}")

            elif subchoice == "4":  # Filter by criteria
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")
                min_length = menu.console.input("Enter minimum text length: ")

                df = pd.read_csv(input_file)
                filtered_df = df[df["text"].str.len() >= int(min_length)]
                filtered_df.to_csv(output_file, index=False)
                menu.display_result(True, f"Filtered data saved to {output_file}")

            elif subchoice == "5":  # Augment data
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")

                from data.data_augmentation import DataAugmentor

                augmentor = DataAugmentor(language)
                df = pd.read_csv(input_file)
                augmented_df = augmentor.augment_data(df)
                augmented_df.to_csv(output_file, index=False)
                menu.display_result(True, f"Augmented data saved to {output_file}")

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")
        menu.wait_for_user()


def handle_optimization_menu(menu, language, config):
    """Handle model optimization options"""
    while True:
        subchoice = menu.display_optimization_menu()
        if subchoice == "b":
            break

        try:
            if subchoice == "1":  # Hyperparameter tuning
                menu.display_progress("Running hyperparameter optimization...")
                trainer = EnhancedModelTrainer(language, config)
                best_params = trainer.optimize_hyperparameters()
                menu.display_result(True, f"Best parameters found: {best_params}")

            elif subchoice == "2":  # Feature selection
                menu.display_progress("Performing feature selection...")
                feature_selector = feature_selector(language, config)
                selected_features = feature_selector.select_best_features()
                menu.display_result(
                    True, f"Selected {len(selected_features)} best features"
                )

            elif subchoice == "3":  # Cross validation
                menu.display_progress("Running cross validation...")
                evaluator = ModelEvaluator(language)
                cv_scores = evaluator.cross_validate()
                menu.display_result(True, f"Cross validation scores: {cv_scores}")

            elif subchoice == "4":  # Model ensemble
                menu.display_progress("Creating model ensemble...")
                trainer = EnhancedModelTrainer(language, config)
                ensemble = trainer.create_ensemble()
                menu.display_result(True, "Model ensemble created successfully")

            elif subchoice == "5":  # Performance analysis
                menu.display_progress("Analyzing model performance...")
                evaluator = ModelEvaluator(language)
                metrics = evaluator.analyze_performance()
                menu.display_result(True, "Performance analysis complete")
                menu.display_performance_metrics(metrics)

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")
        menu.wait_for_user()


def handle_export_menu(menu, language, config):
    """Handle export options"""
    while True:
        subchoice = menu.display_export_menu()
        if subchoice == "b":
            break

        try:
            if subchoice == "1":  # Export predictions
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")
                predict(language, input_file, output_file, config)
                menu.display_result(True, f"Predictions exported to {output_file}")

            elif subchoice == "2":  # Export metrics
                output_file = menu.get_file_path("metrics")
                evaluator = ModelEvaluator(language)
                metrics = evaluator.get_all_metrics()
                evaluator.export_metrics(metrics, output_file)
                menu.display_result(True, f"Metrics exported to {output_file}")

            elif subchoice == "3":  # Generate report
                output_file = menu.get_file_path("report")
                from src.utils.report import ReportGenerator

                generator = ReportGenerator(language)
                generator.generate_report(output_file)
                menu.display_result(True, f"Report generated at {output_file}")

            elif subchoice == "4":  # Export visualizations
                output_dir = menu.get_file_path("visualizations")
                visualizer = ModelVisualizer(language)
                try:
                    visualizer.generate_plots(output_dir)
                    menu.display_result(True, f"Visualizations saved to {output_dir}")
                except Exception as e:
                    menu.display_result(
                        False, f"Error generating visualizations: {str(e)}"
                    )

            elif subchoice == "5":  # Export model
                output_file = menu.get_file_path("model")
                trainer = EnhancedModelTrainer(language, config)
                trainer.export_model(output_file)
                menu.display_result(True, f"Model exported to {output_file}")

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")
        menu.wait_for_user()


def display_server_logs(menu, config, params=None):
    """Display and filter server logs"""
    while True:
        if not params:
            subchoice = menu.display_logs_menu()
            if subchoice == "b":
                break

            params = {}
            if subchoice == "1":  # View latest logs
                params["lines"] = menu.get_log_lines()

            elif subchoice == "2":  # View init logs
                params["type"] = "init"
                params["lines"] = menu.get_log_lines()

            elif subchoice == "3":  # View request logs
                params["type"] = "request"
                params["lines"] = menu.get_log_lines()

            elif subchoice == "4":  # Filter by path
                params["path"] = menu.console.input("Enter path: ")
                params["lines"] = menu.get_log_lines()

            elif subchoice == "5":  # Filter by status
                params["status_code"] = int(menu.console.input("Enter status code: "))
                params["lines"] = menu.get_log_lines()

            elif subchoice == "6":  # Export logs
                output_file = menu.get_file_path("logs")
                params = {"lines": 1000, "output": output_file}

        try:
            # Make API request
            response = requests.get(
                f"http://localhost:{config.API_CONFIG['PORT']}/api/logs",
                params={k: v for k, v in params.items() if k != "output"},
            )

            if response.status_code == 200:
                logs = response.json()

                if "output" in params:  # Export to file
                    with open(params["output"], "w", encoding="utf-8") as f:
                        f.writelines(logs["logs"])
                    menu.display_result(True, f"Logs exported to {params['output']}")
                else:  # Display in console
                    menu.display_filtered_logs(logs["logs"], logs["filters"])
                    menu.display_result(True, f"Found {logs['total']} log entries")
                break
            else:
                menu.display_result(False, f"Failed to get logs: {response.text}")
                break

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")
            break

        menu.wait_for_user()


def get_server_logs(port, lines=1000):
    """Fetch server logs from the API"""
    try:
        response = requests.get(f"http://localhost:{port}/api/logs", params={"lines": lines})
        if response.status_code == 200:
            return response.json()["logs"]
        else:
            raise RuntimeError(f"Failed to get logs: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching server logs: {e}")
        return []

def main():
    # Only cleanup when starting the program
    cleanup_api_servers()  # This will cleanup any leftover servers from previous crashes

    menu = TerminalMenu()
    config = Config()

    while True:
        menu.display_header()
        choice = menu.display_menu()

        if choice == "q":
            break

        language = menu.get_language_choice()

        try:
            if choice == "1":
                menu.display_progress("Training new model")
                train(language, config)
                menu.display_result(True, "Model training completed successfully")
            elif choice == "2":
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")
                validate_paths(input_file, output_file)
                menu.display_progress("Analyzing text")
                predict(language, input_file, output_file, config)
                menu.display_result(True, f"Results saved to {output_file}")

            elif choice == "3":
                input_file = menu.get_file_path("test data")
                validate_paths(input_file)
                menu.display_progress("Evaluating model")
                evaluate(language, input_file, config)

            elif choice == "4":
                output_file = menu.get_file_path("output")
                validate_paths(output_file=output_file)
                num_samples = menu.get_sample_count()
                menu.display_progress(f"Generating {num_samples} training samples")

                generator = TrainingDataGenerator(language, config, num_samples)
                generator.generate_training_data(output_file)

                menu.display_result(
                    True, f"Generated {num_samples} samples successfully"
                )

            elif choice == "5":
                menu.display_progress("Loading model metrics")
                display_metrics(language, config)

            elif choice == "6":  # Add new test option
                menu.display_progress("Loading model for testing")
                test_model(language, config)

            elif choice == "7":  # Add new restore option
                menu.display_progress("Restoring model from checkpoint")
                model = restore_model(language, config)
                if model:
                    menu.display_result(True, "Model restored successfully")
                else:
                    menu.display_result(False, "Failed to restore model")

            elif choice == "8":
                handle_data_collection(menu, language, config)

            elif choice == "9":
                handle_preprocessing_menu(menu, language, config)

            elif choice == "10":
                handle_optimization_menu(menu, language, config)

            elif choice == "11":
                handle_export_menu(menu, language, config)

            elif choice == "12":  # API Server
                handle_api_server(menu, language, config)

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")

        menu.wait_for_user()


if __name__ == "__main__":
    main()
