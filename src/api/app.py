import os
import sys
import subprocess

from h11 import ConnectionClosed

# Add correct project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Update imports to be relative to src
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
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

# Initialize FastAPI app with better configuration
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for Vietnamese-English sentiment analysis",
    version="1.0.0",
    root_path="",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Initialize config and logger
config = Config()
logger = Logger(__name__).logger

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


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    models: Dict[str, bool]


def load_model(language: str):
    """Load model for specified language"""
    try:
        if models[language] is None:
            feature_extractor = FeatureExtractor(language, config)
            predictor = SentimentPredictor(language, config)
            preprocessor = DataPreprocessor(language, config)
            models[language] = {
                "predictor": predictor,
                "extractor": feature_extractor,
                "preprocessor": preprocessor,
            }
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


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest):
    """Enhanced prediction endpoint with better error handling"""
    start_time = datetime.now()

    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        model = load_model(request.language)
        if not model:
            raise HTTPException(
                status_code=503,
                detail=f"Model for language {request.language} is not available",
            )

        # Create DataFrame and process
        df = pd.DataFrame({"text": [request.text]})
        processed_df = model["preprocessor"].preprocess(df)

        if processed_df.empty:
            raise HTTPException(status_code=400, detail="Text preprocessing failed")

        # Extract features
        features = model["extractor"].extract_features(processed_df["cleaned_text"])
        if features is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")

        # Get prediction with emotion analysis
        emotion_result = model["predictor"].predict_emotion(features, request.text)
        if not emotion_result:
            raise HTTPException(status_code=500, detail="Prediction failed")

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "text": request.text,
            "sentiment": emotion_result["sentiment"],
            "sentiment_label": get_sentiment_label(emotion_result["sentiment"]),
            "confidence": float(emotion_result["sentiment_confidence"]),
            "emotion": {
                "label": emotion_result["emotion_vi"],
                "emoji": emotion_result["emotion_emoji"],
                "confidence": float(emotion_result["emotion_confidence"]),
                "scores": emotion_result.get("emotion_scores"),
            },
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
        }

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
        if request.language not in ["vi", "en"]:
            raise HTTPException(status_code=400, detail="Language must be 'vi' or 'en'")

        # Load model components
        model = load_model(request.language)

        # Process texts
        df = pd.DataFrame({"text": request.texts})
        processed_df = model["preprocessor"].preprocess(df)

        if processed_df.empty:
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


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check if models can be loaded
        vi_model = load_model("vi")
        en_model = load_model("en")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "vi": vi_model is not None and all(vi_model.values()),
                "en": en_model is not None and all(en_model.values()),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "models": {"vi": False, "en": False},
        }


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
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)])
            else:
                os.kill(pid, signal.SIGTERM)

        asyncio.create_task(shutdown_server())

        return {"message": "Server shutting down..."}
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise HTTPException(status_code=500, detail="Shutdown failed")


# Server control functions
server_process = None  # Global variable to store the server process


def start_api_server(host="0.0.0.0", port=7270):
    """Start the API server in a subprocess"""
    global server_process
    try:
        command = [
            sys.executable,  # Path to the Python executable
            "-m",
            "uvicorn",
            "src.api.app:app",
            f"--host={host}",
            f"--port={port}",
            "--reload",
        ]
        server_process = subprocess.Popen(command)
        logger.info(
            f"API server started on {host}:{port} with PID {server_process.pid}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        return False


def stop_api_server():
    """Stop the API server subprocess"""
    global server_process
    try:
        if server_process and server_process.poll() is None:
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
                if hasattr(conn, "laddr") and conn.laddr.port in target_ports:
                    # Get process info
                    proc = psutil.Process(conn.pid)
                    if "python" in proc.name().lower():
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
                if server_running
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
metrics_store = {
    "requests": deque(maxlen=config.DASHBOARD_CONFIG["metrics_history"]),
    "response_times": deque(maxlen=config.DASHBOARD_CONFIG["metrics_history"]),
    "errors": deque(maxlen=config.DASHBOARD_CONFIG["metrics_history"]),
    "start_time": datetime.now(),
    "total_requests": 0,
    "total_errors": 0,
}

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


# Add connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.keep_alive_interval = 10

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(
                f"WebSocket client connected. Total connections: {len(self.active_connections)}"
            )
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket client disconnected. Remaining connections: {len(self.active_connections)}"
            )

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


# Initialize connection manager
manager = ConnectionManager()


@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            try:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    break

                # Prepare metrics data
                current_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_usage": psutil.cpu_percent(interval=1),
                    "memory_usage": dict(psutil.virtual_memory()._asdict()),
                    "requests_per_sec": len(metrics_store["requests"])
                    / config.DASHBOARD_CONFIG["update_interval"],
                    "avg_response_time": (
                        sum(metrics_store["response_times"])
                        / len(metrics_store["response_times"])
                        if metrics_store["response_times"]
                        else 0
                    ),
                    "error_rate": len(metrics_store["errors"])
                    / max(len(metrics_store["requests"]), 1),
                    "model_status": {
                        "vi": models["vi"] is not None,
                        "en": models["en"] is not None,
                    },
                }

                await websocket.send_json(current_metrics)
                await asyncio.sleep(2)  # Update interval

                # Handle ping/pong
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    if data == "pong":
                        continue
                except asyncio.TimeoutError:
                    continue

            except WebSocketDisconnect:
                break
            except ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket communication: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        try:
            await websocket.close()
        except:
            pass


@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get summary of API metrics"""
    now = datetime.now()
    uptime = now - metrics_store["start_time"]

    return {
        "uptime": str(uptime),
        "total_requests": metrics_store["total_requests"],
        "total_errors": metrics_store["total_errors"],
        "current_memory_usage": psutil.virtual_memory().percent,
        "current_cpu_usage": psutil.cpu_percent(),
        "active_models": {
            "vi": models["vi"] is not None,
            "en": models["en"] is not None,
        },
    }


# Update middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Update metrics
    metrics_store["total_requests"] += 1
    metrics_store["requests"].append(datetime.now())
    metrics_store["response_times"].append(process_time)

    if response.status_code >= 400:
        metrics_store["total_errors"] += 1
        metrics_store["errors"].append(datetime.now())

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
        if not log_file.exists():
            return {"logs": [], "message": "No log file found"}

        # Read log file
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.readlines()

        filtered_logs = []
        for log in logs:
            try:
                # Apply filters
                if type != "all":
                    if type == "init" and "API Server" not in log:
                        continue
                    if type == "request" and "Request:" not in log:
                        continue

                if path and path not in log:
                    continue

                if status_code and f"Status: {status_code}" not in log:
                    continue

                if level != "all" and f"[{level.upper()}]" not in log:
                    continue

                if since:
                    try:
                        log_time = datetime.fromisoformat(log.split()[0])
                        since_time = datetime.fromisoformat(since)
                        if log_time < since_time:
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

        if response_task in done:
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
            if status["status"] != "healthy":
                logger.warning("Unhealthy state detected, reloading models...")
                # Reload models
                models["vi"] = None
                models["en"] = None
        except Exception as e:
            logger.error(f"Health check error: {e}")


# Update startup event
@app.on_event("startup")
async def startup_event():
    """Enhanced startup logging"""
    try:
        logger.info("=== API Server Starting ===")
        logger.info(f"Environment: {os.getenv('ENV', 'development')}")
        logger.info(f"Debug Mode: {app.debug}")
        logger.info(f"API Config: {json.dumps(config.API_CONFIG, indent=2)}")

        # Log available endpoints
        routes = []
        for route in app.routes:
            if hasattr(route, "methods"):
                routes.append(f"{route.methods} {route.path}")
        logger.info(f"Registered Routes:\n" + "\n".join(routes))

        # Start background tasks
        asyncio.create_task(periodic_health_check())
        logger.info("Health check task started")

        # Initialize metrics
        metrics_store.clear()
        metrics_store.update(
            {
                "requests": deque(maxlen=config.DASHBOARD_CONFIG["metrics_history"]),
                "response_times": deque(
                    maxlen=config.DASHBOARD_CONFIG["metrics_history"]
                ),
                "errors": deque(maxlen=config.DASHBOARD_CONFIG["metrics_history"]),
                "start_time": datetime.now(),
                "total_requests": 0,
                "total_errors": 0,
            }
        )
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
        if method in ["POST", "PUT"]:
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
        ws_max_size=1024 * 1024,  # 1MB max WebSocket message size
        ws_ping_interval=20,  # WebSocket ping interval
        ws_ping_timeout=30,  # WebSocket ping timeout
    )

# Export necessary functions
__all__ = [
    "app",
    "start_api_server",
    "stop_api_server",
    "is_port_in_use",
    "get_api_status",
]

# Add missing English emotion keywords
EMOTION_KEYWORDS = {
    "vi": {
        # ...existing Vietnamese keywords...
    },
    "en": {
        "happy": ["happy", "glad", "delighted", "joyful", "pleased"],
        "excited": ["excited", "thrilled", "enthusiastic", "eager"],
        "satisfied": ["satisfied", "content", "fulfilled", "pleased"],
        "proud": ["proud", "accomplished", "successful"],
        "neutral": ["neutral", "okay", "fine", "alright"],
        "surprised": ["surprised", "amazed", "astonished", "shocked"],
        "confused": ["confused", "puzzled", "perplexed", "uncertain"],
        "sad": ["sad", "unhappy", "depressed", "down"],
        "angry": ["angry", "mad", "furious", "irritated"],
        "disappointed": ["disappointed", "letdown", "unsatisfied"],
        "frustrated": ["frustrated", "annoyed", "upset"],
        "worried": ["worried", "anxious", "concerned", "nervous"],
    },
}
