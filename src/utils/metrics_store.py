import os
import json
import time
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
import atexit
from typing import Dict, Any

import psutil
from src.utils.server_utils import ConnectionManager
from src.utils.logger import Logger
import collections


def _convert_deque_to_list(obj):
    """Recursively convert deque objects to lists in the given object."""
    if isinstance(obj, collections.deque):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _convert_deque_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_deque_to_list(v) for v in obj]
    else:
        return obj


class MetricsStore:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = Logger(__name__).logger
        self._data = {
            "requests": deque(maxlen=100),
            "response_times": deque(maxlen=100),
            "errors": deque(maxlen=100),
            "start_time": datetime.now().isoformat(),
            "total_requests": 0,
            "total_errors": 0,
            "model_performance": {
                "vi": {
                    "loading_time": 0.0,
                    "inference_times": deque(maxlen=1000),
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                },
                "en": {
                    "loading_time": 0.0,
                    "inference_times": deque(maxlen=1000),
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                },
            },
        }
        self._file_path = "metrics.json"
        self._lock = Lock()
        self._initialized = True
        self._load_metrics()
        atexit.register(self._save_metrics)
        self.requests = deque(maxlen=1000)
        self.response_times = deque(maxlen=1000)
        self.errors = deque(maxlen=1000)
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = datetime.now()
        self.model_loading_times = {"vi": [], "en": []}
        self.inference_times = {"vi": [], "en": []}
        self.accuracy = {"vi": 0.0, "en": 0.0}
        self.precision = {"vi": 0.0, "en": 0.0}
        self.recall = {"vi": 0.0, "en": 0.0}

        self._metrics_update_interval = 1.0  # Update interval in seconds
        self._last_update = time.time()
        self._metrics_lock = Lock()

    def _load_metrics(self):
        """Load metrics from file if exists"""
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, "r") as f:
                    data = json.load(f)
                    # Convert lists back to deques
                    for key in ["requests", "response_times", "errors"]:
                        self._data[key] = deque(data[key], maxlen=100)
                    self._data["total_requests"] = data["total_requests"]
                    self._data["total_errors"] = data["total_errors"]
                    self._data["start_time"] = data["start_time"]
                    self._data["model_performance"] = data.get(
                        "model_performance", self._data["model_performance"]
                    )
        except json.JSONDecodeError as e:
            self.logger.error(f"Error loading metrics: {e}")
            # Reset metrics to default state
            self.requests = deque(maxlen=1000)
            self.response_times = deque(maxlen=1000)
            self.errors = deque(maxlen=1000)
            self.total_requests = 0
            self.total_errors = 0
            self.start_time = datetime.now()
        except Exception as e:
            self.logger.error(f"Unexpected error loading metrics: {e}")
            # Reset metrics to default state
            self.requests = deque(maxlen=1000)
            self.response_times = deque(maxlen=1000)
            self.errors = deque(maxlen=1000)
            self.total_requests = 0
            self.total_errors = 0
            self.start_time = datetime.now()

    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_to_save = _convert_deque_to_list(self._data)
            with open(self._file_path, "w") as f:
                json.dump(metrics_to_save, f)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def update_metrics(self, processing_time, is_error=False):
        """Update metrics with thread safety"""
        with self._lock:
            self._data["total_requests"] += 1
            self._data["requests"].append(datetime.now().isoformat())
            self._data["response_times"].append(processing_time)

            if is_error:
                self._data["total_errors"] += 1
                self._data["errors"].append(datetime.now().isoformat())

            # Periodically save metrics
            if self._data["total_requests"] % 100 == 0:
                self._save_metrics()
        self.requests.append(datetime.now())
        self.response_times.append(processing_time)
        if is_error:
            self.errors.append(datetime.now())
            self.total_errors += 1
        self.total_requests += 1

    def get_metrics(self):
        """Get current metrics"""
        with self._lock:
            return {
                "total_requests": self._data["total_requests"],
                "total_errors": self._data["total_errors"],
                "recent_requests": len(self._data["requests"]),
                "avg_response_time": (
                    sum(self._data["response_times"])
                    / len(self._data["response_times"])
                    if self._data["response_times"]
                    else 0
                ),
                "start_time": self._data["start_time"],
            }
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "requests_last_period": len(self.requests),
            "average_response_time": (
                sum(self.response_times) / len(self.response_times)
                if self.response_times
                else 0
            ),
            "error_rate": (
                (self.total_errors / self.total_requests) * 100
                if self.total_requests
                else 0
            ),
        }

    def clear_metrics(self):
        """Clear all metrics"""
        with self._lock:
            self._data["requests"].clear()
            self._data["response_times"].clear()
            self._data["errors"].clear()
            self._data["total_requests"] = 0
            self._data["total_errors"] = 0
            self._data["start_time"] = datetime.now().isoformat()
            self._save_metrics()

    def clear_all(self):
        """Clear all metrics and reinitialize"""
        with self._lock:
            self._data = {
                "requests": deque(maxlen=100),
                "response_times": deque(maxlen=100),
                "errors": deque(maxlen=100),
                "start_time": datetime.now().isoformat(),
                "total_requests": 0,
                "total_errors": 0,
                "model_performance": {
                    "vi": {
                        "loading_time": 0.0,
                        "inference_times": deque(maxlen=1000),
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                    },
                    "en": {
                        "loading_time": 0.0,
                        "inference_times": deque(maxlen=1000),
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                    },
                },
            }
            self._save_metrics()
        self.requests.clear()
        self.response_times.clear()
        self.errors.clear()
        self.total_requests = 0
        self.total_errors = 0
        self.model_loading_times = {"vi": [], "en": []}
        self.inference_times = {"vi": [], "en": []}
        self.accuracy = {"vi": 0.0, "en": 0.0}
        self.precision = {"vi": 0.0, "en": 0.0}
        self.recall = {"vi": 0.0, "en": 0.0}
        self.start_time = datetime.now()
        self.logger.info("All metrics have been cleared.")

    def update_model_loading_time(self, language: str, loading_time: float):
        """Update model loading time for a specific language."""
        if language in self.model_loading_times:
            self.model_loading_times[language].append(loading_time)
            self.logger.debug(f"Model loading time for {language}: {loading_time}s")
        else:
            self.logger.warning(
                f"Attempted to update loading time for unsupported language: {language}"
            )

    def update_inference_time(self, language: str, inference_time: float):
        """Update inference time for a specific language."""
        if language in self.inference_times:
            self.inference_times[language].append(inference_time)
            self.logger.debug(f"Inference time for {language}: {inference_time}s")
        else:
            self.logger.warning(
                f"Attempted to update inference time for unsupported language: {language}"
            )

    def update_ml_metrics(
        self,
        language: str,
        accuracy_score: float,
        precision_score: float,
        recall_score: float,
    ):
        """Update ML performance metrics for a specific language."""
        if language in self.accuracy:
            self.accuracy[language] = accuracy_score
            self.precision[language] = precision_score
            self.recall[language] = recall_score
            self.logger.debug(
                f"ML Metrics for {language} - Accuracy: {accuracy_score}, Precision: {precision_score}, Recall: {recall_score}"
            )
        else:
            self.logger.warning(
                f"Attempted to update ML metrics for unsupported language: {language}"
            )

    def update_processing_time(self, processing_time: float):
        """Update processing time."""
        self.response_times.append(processing_time)
        self.logger.debug(f"Processing time updated: {processing_time}s")

    def increment_total_requests(self):
        """Increment total requests counter."""
        self.total_requests += 1
        self.logger.debug(f"Total requests incremented: {self.total_requests}")

    def add_request(self, timestamp: datetime):
        """Add a new request timestamp."""
        self.requests.append(timestamp)
        self.logger.debug(f"Request added at {timestamp}")

    def add_response_time(self, response_time: float):
        """Add a new response time."""
        self.response_times.append(response_time)
        self.logger.debug(f"Response time added: {response_time}s")

    def cleanup_old_metrics(self, cutoff: datetime):
        """Remove metrics older than the cutoff time."""
        while self.requests and self.requests[0] < cutoff:
            removed_request = self.requests.popleft()
            removed_response_time = self.response_times.popleft()
            self.logger.debug(f"Old request at {removed_request} removed")
            self.logger.debug(f"Old response time {removed_response_time}s removed")

    def get_requests_per_sec(self, interval: int) -> float:
        """Calculate requests per second over the given interval."""
        cutoff = datetime.now() - timedelta(seconds=interval)
        recent_requests = [req for req in self.requests if req >= cutoff]
        requests_per_sec = len(recent_requests) / interval
        self.logger.debug(f"Requests per second: {requests_per_sec}")
        return requests_per_sec

    def get_avg_response_time(self) -> float:
        """Calculate average response time."""
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            self.logger.debug(f"Average response time: {avg_time}s")
            return avg_time
        return 0.0

    def get_error_rate(self) -> float:
        """Calculate error rate."""
        if self.requests:
            error_rate = len(self.errors) / len(self.requests)
            self.logger.debug(f"Error rate: {error_rate}")
            return error_rate
        return 0.0

    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        return {
            lang: {
                "loading_time_avg": sum(times) / len(times) if times else 0.0,
                "inference_time_avg": (
                    sum(self.inference_times[lang]) / len(self.inference_times[lang])
                    if self.inference_times[lang]
                    else 0.0
                ),
                "accuracy": self.accuracy.get(lang, 0.0),
                "precision": self.precision.get(lang, 0.0),
                "recall": self.recall.get(lang, 0.0),
            }
            for lang, times in self.model_loading_times.items()
        }

    def __getitem__(self, key):
        """Allow dictionary-like access to metrics"""
        return self._data[key]

    def __setitem__(self, key, value):
        """Allow dictionary-like setting of metrics"""
        with self._lock:
            self._data[key] = value

    async def send_metrics(self, manager: ConnectionManager, metrics: Dict[str, Any]):
        """Send metrics to all active WebSocket connections"""
        try:
            await manager.broadcast(json.dumps(metrics))
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")

    def update_system_metrics(self):
        """Update system metrics with rate limiting"""
        current_time = time.time()

        # Check if enough time has passed since last update
        with self._lock:
            if current_time - self._last_update < self._metrics_update_interval:
                return

            self._last_update = current_time

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            with self._metrics_lock:
                self.metrics["system"]["cpu_usage"].append(
                    {"timestamp": datetime.now().isoformat(), "value": cpu_percent}
                )

                self.metrics["system"]["memory_usage"].append(
                    {"timestamp": datetime.now().isoformat(), "value": memory.percent}
                )

                # Calculate requests per second
                recent_requests = list(self.metrics["requests"]["rate"])
                if recent_requests:
                    requests_per_sec = len(recent_requests) / 60.0  # Last minute
                else:
                    requests_per_sec = 0

                self.metrics["system"]["requests_per_sec"] = requests_per_sec

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def get_current_metrics(self):
        """Get current metrics with proper locks"""
        with self._metrics_lock:
            try:
                latest_metrics = {
                    "system": {
                        "cpu_usage": self._get_latest("system", "cpu_usage"),
                        "memory_usage": self._get_latest("system", "memory_usage"),
                        "error_rate": self._get_latest("system", "error_rate"),
                        "requests_per_sec": self.metrics["system"].get(
                            "requests_per_sec", 0
                        ),
                    },
                    "model": {
                        "vi": self._get_model_metrics("vi"),
                        "en": self._get_model_metrics("en"),
                    },
                    "requests": {
                        "total": self.metrics["requests"]["total"],
                        "success_rate": self._calculate_success_rate(),
                    },
                    "uptime": self._calculate_uptime(),
                }
                return latest_metrics
            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                return {}
