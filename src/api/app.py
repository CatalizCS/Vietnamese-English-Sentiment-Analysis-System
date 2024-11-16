import os
import subprocess
import sys
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import asyncio
from datetime import datetime, timedelta
import time

from h11 import ConnectionClosed

from src.utils.server_utils import force_kill_port, is_port_in_use, ConnectionManager

# Add correct project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

# Update imports to be relative to src
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import pandas as pd
from datetime import datetime
import psutil
import json
import time
from collections import deque
import asyncio
import signal
import socket
from starlette.websockets import WebSocketState, WebSocketDisconnect
from contextlib import asynccontextmanager
import pathlib

from src.config import Config
from src.features.feature_engineering import FeatureExtractor
from src.models.model_predictor import SentimentPredictor
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import Logger
from src.utils.metrics_store import MetricsStore
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Add performance optimizations
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int, period: int):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()

        # Clean old requests
        self.requests = {
            ip: reqs
            for ip, reqs in self.requests.items()
            if reqs[-1] > now - self.period
        }

        if (client_ip in self.requests):
            if (len(self.requests[client_ip]) >= self.calls):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(now)
        else:
            self.requests[client_ip] = [now]

        return await call_next(request)


# Add model caching
@lru_cache(maxsize=2)
def get_cached_model(language: str):
    """Cache model loading to improve performance"""
    return load_model(language)


# Add dependency for language validation
def validate_language(language: str = "vi"):
    if (language not in ["vi", "en"]):
        raise HTTPException(status_code=400, detail="Invalid language")
    return language


# Optimize app initialization
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for Vietnamese-English sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize config and logger
config = Config()
logger = Logger(__name__).logger

# Add performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    RateLimitMiddleware,
    calls=config.API_CONFIG["RATE_LIMIT"]["requests"],
    period=config.API_CONFIG["RATE_LIMIT"]["window"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize models dict
models = {"vi": None, "en": None}


# Pydantic models for request/response
class TextRequest(BaseModel):
    text: str
    language: str = "vi"


class BatchRequest(BaseModel):
    texts: List[str]
    language: str = "vi"


class SentimentResponse(BaseModel):
    text: str
    sentiment: int
    sentiment_label: str
    confidence: float
    emotion: Optional[Dict] = None
    processing_time: float


class ModelInfo(BaseModel):
    loaded: bool
    info: Dict[str, Any]


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    models: Dict[str, ModelInfo]


def load_model(language: str):
    """Load model for specified language and track loading time"""
    try:
        start_time = time.time()
        if (models[language] is None):
            feature_extractor = FeatureExtractor(language, config)
            predictor = SentimentPredictor(language, config)
            preprocessor = DataPreprocessor(language, config)
            models[language] = {
                "predictor": predictor,
                "extractor": feature_extractor,
                "preprocessor": preprocessor,
            }
        loading_time = time.time() - start_time
        metrics_store.update_model_loading_time(language, loading_time)
        return models[language]
    except Exception as e:
        logger.error(f"Error loading model for {language}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load model for {language}"
        )


def get_sentiment_label(sentiment: int) -> str:
    """Convert sentiment code to label"""
    return {
        0: "Negative / Tiêu cực",
        1: "Neutral / Trung tính",
        2: "Positive / Tích cực",
    }.get(sentiment, "Unknown")


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "languages": ["vi", "en"],
        "endpoints": [
            "/predict - Single text prediction",
            "/batch - Batch text prediction",
            "/health - API health check",
        ],
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


# Optimize prediction endpoint
def evaluate_model(language: str, true_labels: List[int], predictions: List[int]):
    """Evaluate model performance and update metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    metrics_store.update_ml_metrics(language, accuracy, precision, recall)

@app.post("/predict")
async def predict(request: TextRequest, language: str = Depends(validate_language)):
    """Optimized prediction endpoint"""
    start_time = time.time()

    try:
        if (not request.text.strip()):
            raise HTTPException(status_code=400, detail="Empty text")

        # Get cached model
        model = get_cached_model(language)

        # Create DataFrame with optimized memory usage
        df = pd.DataFrame({"text": [request.text]}, copy=False)

        # Process text with error handling
        try:
            processed_df = model["preprocessor"].preprocess(df)
            features = model["extractor"].extract_features(processed_df["cleaned_text"])
            prediction_start = time.time()
            result = model["predictor"].predict_emotion(features, request.text)
            inference_time = time.time() - prediction_start
            metrics_store.update_inference_time(language, inference_time)  # Updated method call

            # Example: Update ML metrics if true labels are available
            # true_label = get_true_label(request.text)  # Implement this function as needed
            # if (true_label is not None):
            #     predictions = [result["sentiment"]]
            #     true_labels = [true_label]
            #     metrics_store.update_ml_metrics(language, accuracy_score=true_label, precision_score=precision_score(true_label), recall_score=recall_score=true_label)  # Updated method call

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Processing failed")

        # Optimize response creation
        response = {
            "text": request.text,
            "sentiment": result["sentiment"],
            "confidence": float(result["sentiment_confidence"]),
            "emotion": {
                "label": result["emotion_vi"],
                "emoji": result["emotion_emoji"],
                "confidence": float(result["emotion_confidence"]),
            },
            "processing_time": time.time() - start_time,
        }

        # Update metrics asynchronously
        asyncio.create_task(metrics_store.update_processing_time(response["processing_time"]))  # Updated method call

        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_predict(request: BatchRequest):
    """Predict sentiment for multiple texts"""
    start_time = datetime.now()

    try:
        # Validate language
        if (request.language not in ["vi", "en"]):
            raise HTTPException(status_code=400, detail="Language must be 'vi' or 'en'")

        # Load model components
        model = load_model(request.language)

        # Process texts
        df = pd.DataFrame({"text": request.texts})
        processed_df = model["preprocessor"].preprocess(df)

        if (processed_df.empty):
            raise HTTPException(status_code=400, detail="Text preprocessing failed")

        # Extract features
        features = model["extractor"].extract_features(processed_df["cleaned_text"])

        # Get predictions
        results = []
        for i, text in enumerate(request.texts):
            emotion_result = model["predictor"].predict_emotion(
                features[i : i + 1], text
            )

            results.append(
                {
                    "text": text,
                    "sentiment": emotion_result["sentiment"],
                    "sentiment_label": get_sentiment_label(emotion_result["sentiment"]),
                    "confidence": float(emotion_result["sentiment_confidence"]),
                    "emotion": {
                        "label": emotion_result["emotion_vi"],
                        "emoji": emotion_result["emotion_emoji"],
                        "confidence": float(emotion_result["emotion_confidence"]),
                        "scores": emotion_result.get("emotion_scores"),
                    },
                }
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "results": results,
            "count": len(results),
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


# Add metrics optimization
async def update_metrics(processing_time: float):
    """Update metrics asynchronously"""
    try:
        metrics_store.increment_total_requests()  # Updated method call
        metrics_store.add_request(datetime.now())  # Updated method call
        metrics_store.add_response_time(processing_time)  # Updated method call

        # Cleanup old metrics
        now = datetime.now()
        cutoff = now - timedelta(days=config.METRICS_CONFIG["retention_days"])

        metrics_store.cleanup_old_metrics(cutoff)  # Updated method call

    except Exception as e:
        logger.error(f"Metrics update error: {str(e)}")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Optimized health check"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                lang: ModelInfo(
                    loaded=bool(models[lang]),
                    info=config.MODEL_INFO.get(lang, {})
                ) for lang in ["vi", "en"]
            },
            "metrics": {
                "requests": len(metrics_store["requests"]),
                "avg_response_time": (
                    sum(metrics_store["response_times"])
                    / len(metrics_store["response_times"])
                    if metrics_store["response_times"]
                    else 0
                ),
                "memory_usage": psutil.Process().memory_percent(),
            },
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    try:
        # Clear model cache
        models["vi"] = None
        models["en"] = None
        # Additional cleanup if needed
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown endpoint"""
    try:
        # Get the process ID
        pid = os.getpid()

        # Schedule shutdown after response is sent
        async def shutdown_server():
            await asyncio.sleep(1)
            # Kill the current process
            if (sys.platform == "win32"):
                subprocess.run(["taskkill", "/F", "/PID", str(pid)])
            else:
                os.kill(pid, signal.SIGTERM)

        asyncio.create_task(shutdown_server())

        return {"message": "Server shutting down..."}
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise HTTPException(status_code=500, detail="Shutdown failed")


# Server control functions
server_process = None


def start_new_terminal():
    """Start API server in a new terminal window"""
    try:
        # Get the project root directory and normalize path
        project_root = os.path.abspath(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        api_script = os.path.join(project_root, "src", "api", "app.py")

        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["PYTHONUNBUFFERED"] = "1"

        # Command to run API server
        api_command = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host=0.0.0.0",
            "--port=7270",
            "--reload",
            "--reload-dir=src",
        ]

        # Platform-specific commands
        if (sys.platform == "win32"):
            # Windows: use 'start' command
            full_command = [
                "cmd",
                "/c",
                "start",
                "cmd",
                "/k",
                "set PYTHONPATH=" + project_root + "&&" + " ".join(api_command),
            ]
            subprocess.Popen(full_command, shell=True, env=env, cwd=project_root)

        elif (sys.platform == "darwin"):
            # macOS: use AppleScript to open Terminal
            apple_script = [
                "osascript",
                "-e",
                f'tell app "Terminal" to do script "cd {project_root} && {" ".join(api_command)}"',
            ]
            subprocess.Popen(apple_script)

        else:
            # Linux: try common terminal emulators
            terminals = [
                ["gnome-terminal", "--"],
                ["xterm", "-e"],
                ["konsole", "-e"],
                ["xfce4-terminal", "--execute"],
            ]

            success = False
            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd + api_command, env=env, cwd=project_root)
                    success = True
                    break
                except FileNotFoundError:
                    continue

            if (not success):
                raise RuntimeError("No suitable terminal emulator found")

        logger.info("API server started in new terminal window")
        return True

    except Exception as e:
        logger.error(f"Failed to start API server in new terminal: {e}")
        return False


def start_api_server(host="0.0.0.0", port=7270):
    """Enhanced API server starter"""
    global server_process
    try:
        # Kill any existing process using the port
        if (is_port_in_use(port)):
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
        if (server_process.poll() is not None):
            stderr = server_process.stderr.read()
            raise RuntimeError(f"Server failed to start: {stderr}")

        logger.info(f"API server started on {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        return False


def stop_api_server():
    """Stop the API server subprocess"""
    global server_process
    try:
        if (server_process and server_process.poll() is None):
            server_process.terminate()
            server_process.wait(timeout=5)
            logger.info(
                f"API server with PID {server_process.pid} terminated successfully."
            )
        else:
            logger.info("API server process is not running.")
        server_process = None

        # Clear model cache
        models["vi"] = None
        models["en"] = None

        time.sleep(1)  # Ensure the port is freed
        return True
    except Exception as e:
        logger.error(f"Error stopping API server: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


def get_api_status():
    """Get API server status"""
    try:
        # Check if any Python processes are using our ports
        target_ports = [7270, 8000]
        connections = psutil.net_connections()
        server_running = False

        for conn in connections:
            try:
                if (hasattr(conn, "laddr") and (conn.laddr.port in target_ports)):
                    # Get process info
                    proc = psutil.Process(conn.pid)
                    if ("python" in proc.name().lower()):
                        server_running = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue

        return {
            "running": server_running,
            "models_loaded": {
                "vi": models["vi"] is not None,
                "en": models["en"] is not None,
            },
            "port": config.API_CONFIG["PORT"],
            "uptime": (
                str(datetime.now() - metrics_store["start_time"])
                if (server_running)
                else "0:00:00"
            ),
            "total_requests": metrics_store["total_requests"],
            "total_errors": metrics_store["total_errors"],
        }
    except Exception as e:
        logger.error(f"Failed to get API status: {str(e)}")
        return {
            "running": False,
            "models_loaded": {"vi": False, "en": False},
            "error": str(e),
        }


# Initialize metrics storage
metrics_store = MetricsStore()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
templates = Jinja2Templates(directory="src/api/templates")


@app.get("/dashboard")
async def dashboard(request: Request):
    """Render dashboard page"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "config": {
                "update_interval": config.DASHBOARD_CONFIG["update_interval"],
                "metrics_history": config.DASHBOARD_CONFIG["metrics_history"],
                "alert_thresholds": config.METRICS_CONFIG["alert_thresholds"],
            },
            "start_time": metrics_store["start_time"],
            "total_requests": metrics_store["total_requests"],
            "total_errors": metrics_store["total_errors"],
        },
    )


# Initialize connection manager
manager = ConnectionManager()

@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Receive messages and send metrics concurrently
            receive_task = asyncio.create_task(websocket.receive_text())
            send_task = asyncio.create_task(send_metrics(websocket))

            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if receive_task in done:
                message = receive_task.result()
                data = json.loads(message)
                
                if data.get("type") == "heartbeat":
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))
                    # Cancel send_task to prevent duplication
                    send_task.cancel()
                    continue  # Continue to the next iteration

            if send_task in done:
                # Metrics have been sent
                pass

            # Cancel pending tasks
            for task in pending:
                task.cancel()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

async def send_metrics(websocket: WebSocket):
    # Send metrics to the client
    metrics = {
        "type": "metrics",
        "cpu_usage": psutil.cpu_percent(interval=0.5),
        "memory_usage": {
            "percent": psutil.virtual_memory().percent,
            "used": psutil.virtual_memory().used,
            "total": psutil.virtual_memory().total
        },
        "requests_per_sec": metrics_store.get_requests_per_sec(config.DASHBOARD_CONFIG["update_interval"]),
        "avg_response_time": metrics_store.get_avg_response_time() if metrics_store.response_times else 0,
        "model_status": {
            lang: {
                "loaded": models[lang] is not None,
                "info": config.MODEL_INFO.get(lang, {}),
                "performance": metrics_store.get_model_performance().get(lang, {})
            } for lang in ["vi", "en"]
        },
        "total_requests": metrics_store.total_requests,
        "total_errors": metrics_store.total_errors
    }
    await websocket.send_text(json.dumps(metrics))
    await asyncio.sleep(1)

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get summary of API metrics"""
    return metrics_store.get_metrics()


# Update middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Update metrics with process time and error status
    metrics_store.update_metrics(
        process_time, 
        is_error=(response.status_code >= 400)
    )

    return response


@app.get("/api/logs")
async def get_server_logs(
    lines: int = 100,
    level: str = "all",
    since: str = None,
    type: str = "all",
    path: str = None,
    status_code: int = None,
):
    """Enhanced log retrieval with more filtering options"""
    try:
        log_file = pathlib.Path(config.LOG_FILE)
        if (not log_file.exists()):
            return {"logs": [], "message": "No log file found"}

        # Read log file
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.readlines()

        filtered_logs = []
        for log in logs:
            try:
                # Apply filters
                if (type != "all"):
                    if ((type == "init") and ("API Server" not in log)):
                        continue
                    if ((type == "request") and ("Request:" not in log)):
                        continue

                if (path and (path not in log)):
                    continue

                if (status_code and (f"Status: {status_code}" not in log)):
                    continue

                if ((level != "all") and (f"[{level.upper()}]" not in log)):
                    continue

                if (since):
                    try:
                        log_time = datetime.fromisoformat(log.split()[0])
                        since_time = datetime.fromisoformat(since)
                        if (log_time < since_time):
                            continue
                    except:
                        pass

                filtered_logs.append(log)
            except:
                continue

        # Get last N lines
        filtered_logs = filtered_logs[-lines:]

        return {
            "logs": filtered_logs,
            "total": len(filtered_logs),
            "filters": {
                "type": type,
                "level": level,
                "path": path,
                "status_code": status_code,
                "since": since,
                "lines": lines,
            },
        }
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


# Add timeout handler
async def timeout_handler():
    """Handle request timeout"""
    await asyncio.sleep(config.API_CONFIG["TIMEOUT"])
    raise HTTPException(status_code=408, detail="Request timeout")


# Add request timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Create timeout task
        timeout_task = asyncio.create_task(timeout_handler())
        # Create response task
        response_task = asyncio.create_task(call_next(request))

        done, pending = await asyncio.wait(
            [timeout_task, response_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        if (response_task in done):
            return await response_task
        else:
            raise HTTPException(status_code=408, detail="Request timeout")

    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        # Cleanup resources
        models["vi"] = None
        models["en"] = None
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )


# Add periodic health check
async def periodic_health_check():
    """Run periodic health check"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            status = await health_check()
            if (status["status"] != "healthy"):
                logger.warning("Unhealthy state detected, reloading models...")
                # Reload models
                models["vi"] = None
                models["en"] = None
        except Exception as e:
            logger.error(f"Health check error: {e}")


# Update startup event
@app.on_event("startup")
async def startup_event():
    """Enhanced startup logging and model preloading"""
    try:
        logger.info("=== API Server Starting ===")
        logger.info(f"Environment: {os.getenv('ENV', 'development')}")
        logger.info(f"Debug Mode: {app.debug}")
        logger.info(f"API Config: {json.dumps(config.API_CONFIG, indent=2)}")

        # Preload models on startup
        logger.info("Preloading models...")
        for language in ["vi", "en"]:
            try:
                logger.info(f"Loading {language.upper()} model...")
                _ = get_cached_model(language)
                logger.info(f"{language.upper()} model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load {language.upper()} model: {str(e)}")

        # Log available endpoints
        routes = []
        for route in app.routes:
            if (hasattr(route, "methods")):
                routes.append(f"{route.methods} {route.path}")
        logger.info(f"Registered Routes:\n" + "\n".join(routes))

        # Start background tasks
        asyncio.create_task(periodic_health_check())
        logger.info("Health check task started")

        # Initialize metrics
        metrics_store.clear_all()
        logger.info("Metrics store initialized")

        logger.info("=== API Server Started Successfully ===")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


# Update shutdown event to be more thorough
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Clear model cache
        models["vi"] = None
        models["en"] = None

        # Close all active connections
        for ws in manager.active_connections:
            await ws.close()
        manager.active_connections.clear()

        # Additional cleanup
        metrics_store["requests"].clear()
        metrics_store["response_times"].clear()
        metrics_store["errors"].clear()

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path
    query_params = str(request.query_params)

    # Log request
    logger.info(f"Request: {method} {path} {query_params}")

    try:
        # Get request body for POST/PUT requests
        if (method in ["POST", "PUT"]):
            body = await request.json()
            logger.info(f"Request Body: {json.dumps(body, ensure_ascii=False)}")
    except:
        pass

    try:
        response = await call_next(request)

        process_time = time.time() - start_time
        status_code = response.status_code

        # Log response
        logger.info(
            f"Response: {method} {path} - Status: {status_code} - Time: {process_time:.3f}s"
        )

        return response
    except Exception as e:
        logger.error(f"Request failed: {method} {path} - Error: {str(e)}")
        raise


if __name__ == "__main__":
    # Run with uvicorn directly when script is executed
    import uvicorn
    
    port = 7270
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=["src"],
        log_level="info",
        # Remove invalid timeout parameter
        workers=1,
        limit_concurrency=100,
        # Add valid websocket configurations
        websocket_ping_interval=20,
        websocket_ping_timeout=30,
        limit_max_requests=None
    )

# Export necessary functions
__all__ = [
    "app",
    "start_api_server",
    "stop_api_server",
    "is_port_in_use",
    "get_api_status",
    "get_cached_model",
    "validate_language",
    "update_metrics"
]
