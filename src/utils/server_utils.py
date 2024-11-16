import psutil
import socket
import time
from typing import List, Optional, Any  # Add this import
import logging
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from src.utils.logger import Logger
import json  # Add this import at the top

logger = logging.getLogger(__name__)


def force_kill_port(port: int) -> bool:
    """Force kill any process using the specified port"""
    try:
        connections = psutil.net_connections()
        for conn in connections:
            try:
                if hasattr(conn, "laddr") and conn.laddr.port == port:
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
        logger.error(f"Error killing process on port {port}: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


async def safe_send(websocket: WebSocket, message: str):
    """Safely send a message over WebSocket, handling disconnections."""
    try:
        await websocket.send_text(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected before message could be sent.")
    except Exception as e:
        logger.error(f"Error sending message over WebSocket: {e}")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.keep_alive_interval = 10

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger = Logger(__name__).logger
            logger.info(
                f"WebSocket client connected. Total connections: {len(self.active_connections)}"
            )
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            raise

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

    async def broadcast(self, message: Any):
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            elif not isinstance(message, str):
                message = str(message)

            disconnected = []
            for connection in self.active_connections:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_text(message)
                    else:
                        disconnected.append(connection)
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {e}")
                    disconnected.append(connection)

            for connection in disconnected:
                await self.disconnect(connection)
                logger.info("WebSocket client disconnected during broadcast.")
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")

    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming messages from WebSocket clients."""
        try:
            data = json.loads(message)
            if data.get("type") == "heartbeat":
                await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))
                logger.debug("Heartbeat acknowledged.")
            # Handle other message types if necessary
        except json.JSONDecodeError:
            logger.error("Received invalid JSON message.")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            # Optionally, you can close the websocket here if needed
            # await websocket.close()
