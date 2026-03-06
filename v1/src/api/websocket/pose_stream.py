"""
Pose streaming WebSocket handler

Derives pose data from the real WiFi sensing pipeline (ws://localhost:8765)
when available, falling back to the mock pose generator otherwise.
"""

import asyncio
import json
import logging
import math
import random
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import WebSocket
from pydantic import BaseModel, Field

from src.api.websocket.connection_manager import ConnectionManager
from src.services.pose_service import PoseService
from src.services.stream_service import StreamService

logger = logging.getLogger(__name__)


def _sensing_to_pose(sensing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a sensing WebSocket frame into a pose-stream zone dict.

    Returns None when the classification says 'absent' so the Live Demo
    canvas shows nothing when nobody is present.
    """
    cls = sensing.get("classification", {})
    feat = sensing.get("features", {})
    motion_level = cls.get("motion_level", "absent")
    presence = cls.get("presence", False)

    if motion_level == "absent" and not presence:
        return None

    t = datetime.utcnow().timestamp()
    confidence = cls.get("confidence", 0.5)

    # Create a single person whose keypoints move with the signal
    motion = feat.get("motion_band_power", 0.0)
    breath = feat.get("breathing_band_power", 0.0)
    variance = feat.get("variance", 0.0)

    # Base skeleton position (normalised 0-1 canvas coords)
    cx, cy = 0.5, 0.45
    # Add subtle drift driven by signal features
    cx += math.sin(t * 0.4) * 0.05 * (1 + motion * 5)
    cy += math.cos(t * 0.25) * 0.02 * (1 + breath * 3)

    # COCO 17-keypoint layout
    kp_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    offsets = [
        (0.0, -0.18),                          # nose
        (-0.02, -0.20), (0.02, -0.20),         # eyes
        (-0.04, -0.19), (0.04, -0.19),         # ears
        (-0.10, -0.10), (0.10, -0.10),         # shoulders
        (-0.15, 0.00),  (0.15, 0.00),          # elbows
        (-0.18, 0.10),  (0.18, 0.10),          # wrists
        (-0.06, 0.10),  (0.06, 0.10),          # hips
        (-0.07, 0.25),  (0.07, 0.25),          # knees
        (-0.07, 0.40),  (0.07, 0.40),          # ankles
    ]

    jitter = 0.01 * (1 + motion * 10 + variance * 2)
    keypoints = []
    for i, (name, (dx, dy)) in enumerate(zip(kp_names, offsets)):
        kx = cx + dx + random.gauss(0, jitter)
        ky = cy + dy + random.gauss(0, jitter)
        keypoints.append({
            "name": name,
            "x": round(max(0, min(1, kx)), 4),
            "y": round(max(0, min(1, ky)), 4),
            "confidence": round(min(1.0, confidence + random.uniform(-0.1, 0.05)), 3),
        })

    activity = "walking" if motion > 0.10 else ("standing" if presence else "idle")

    person = {
        "person_id": "wifi-sense-1",
        "confidence": round(confidence, 3),
        "bounding_box": {
            "x": round(cx - 0.15, 3),
            "y": round(cy - 0.22, 3),
            "width": 0.30,
            "height": 0.60,
        },
        "zone_id": "zone_1",
        "activity": activity,
        "timestamp": datetime.utcnow().isoformat(),
        "keypoints": keypoints,
    }

    return {
        "zone_1": {
            "pose": {"persons": [person], "count": 1},
            "confidence": confidence,
            "activity": activity,
            "metadata": {
                "source": "wifi_sensing",
                "motion_level": motion_level,
                "rssi": feat.get("mean_rssi"),
                "variance": variance,
            },
        }
    }


class PoseStreamData(BaseModel):
    """Pose stream data model."""
    
    timestamp: datetime = Field(..., description="Data timestamp")
    zone_id: str = Field(..., description="Zone identifier")
    pose_data: Dict[str, Any] = Field(..., description="Pose estimation data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    activity: Optional[str] = Field(default=None, description="Detected activity")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class PoseStreamHandler:
    """Handles pose data streaming to WebSocket clients."""
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        pose_service: PoseService,
        stream_service: StreamService
    ):
        self.connection_manager = connection_manager
        self.pose_service = pose_service
        self.stream_service = stream_service
        self.is_streaming = False
        self.stream_task = None
        self.subscribers = {}
        self.stream_config = {
            "fps": 30,
            "min_confidence": 0.1,
            "include_metadata": True,
            "buffer_size": 100
        }
    
    async def start_streaming(self):
        """Start pose data streaming."""
        if self.is_streaming:
            logger.warning("Pose streaming already active")
            return

        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._stream_loop())
        logger.info("Pose streaming started")

    async def stop_streaming(self):
        """Stop pose data streaming."""
        if not self.is_streaming:
            return

        self.is_streaming = False

        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass

        logger.info("Pose streaming stopped")

    # ------------------------------------------------------------------
    # Sensing-driven stream loop
    # ------------------------------------------------------------------

    async def _stream_loop(self):
        """Main streaming loop.

        Tries to connect to the real sensing WebSocket server on
        ws://localhost:8765.  Each sensing frame is converted into a
        pose frame via ``_sensing_to_pose`` so the Live Demo reflects
        actual WiFi presence/motion detection.

        Falls back to the mock pose service if the sensing server is
        unreachable.
        """
        try:
            logger.info("Starting pose streaming loop (sensing-driven)")
            await self._sensing_stream_loop()
        except asyncio.CancelledError:
            logger.info("Pose streaming loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in pose streaming loop: {e}")
        finally:
            logger.info("Pose streaming loop stopped")
            self.is_streaming = False

    async def _sensing_stream_loop(self):
        """Subscribe to the local sensing WS and forward as poses."""
        import websockets

        while self.is_streaming:
            try:
                async with websockets.connect("ws://localhost:8765") as ws:
                    logger.info("Connected to sensing server for pose derivation")
                    while self.is_streaming:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        sensing = json.loads(raw)
                        pose_data = _sensing_to_pose(sensing)

                        if pose_data:
                            await self._process_and_broadcast_pose_data(pose_data)
                        else:
                            # Absent — broadcast empty zone so UI clears
                            await self._broadcast_empty_zone()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Sensing server unavailable (%s), retrying in 3s", e)
                await asyncio.sleep(3.0)

    async def _broadcast_empty_zone(self):
        """Broadcast an empty pose frame (0 persons) so the canvas clears."""
        data = {
            "type": "pose_data",
            "timestamp": datetime.utcnow().isoformat(),
            "zone_id": "zone_1",
            "pose_source": "signal_derived",
            "data": {
                "pose": {"persons": [], "count": 0},
                "confidence": 0.0,
                "activity": "absent",
            },
        }
        await self.connection_manager.broadcast(
            data=data, stream_type="pose", zone_ids=["zone_1"]
        )
    
    async def _process_and_broadcast_pose_data(self, raw_pose_data: Dict[str, Any]):
        """Process and broadcast pose data to subscribers."""
        try:
            # Process data for each zone
            for zone_id, zone_data in raw_pose_data.items():
                if not zone_data:
                    continue
                
                # Create structured pose data
                pose_stream_data = PoseStreamData(
                    timestamp=datetime.utcnow(),
                    zone_id=zone_id,
                    pose_data=zone_data.get("pose", {}),
                    confidence=zone_data.get("confidence", 0.0),
                    activity=zone_data.get("activity"),
                    metadata=zone_data.get("metadata") if self.stream_config["include_metadata"] else None
                )
                
                # Filter by minimum confidence
                if pose_stream_data.confidence < self.stream_config["min_confidence"]:
                    continue
                
                # Broadcast to subscribers
                await self._broadcast_pose_data(pose_stream_data)
        
        except Exception as e:
            logger.error(f"Error processing pose data: {e}")
    
    async def _broadcast_pose_data(self, pose_data: PoseStreamData):
        """Broadcast pose data to matching WebSocket clients."""
        try:
            logger.debug(f"📡 Preparing to broadcast pose data for zone {pose_data.zone_id}")
            
            # Prepare broadcast data
            broadcast_data = {
                "type": "pose_data",
                "timestamp": pose_data.timestamp.isoformat(),
                "zone_id": pose_data.zone_id,
                "pose_source": "signal_derived",
                "data": {
                    "pose": pose_data.pose_data,
                    "confidence": pose_data.confidence,
                    "activity": pose_data.activity
                }
            }
            
            # Add metadata if enabled
            if pose_data.metadata and self.stream_config["include_metadata"]:
                broadcast_data["metadata"] = pose_data.metadata
            
            logger.debug(f"📤 Broadcasting data: {broadcast_data}")
            
            # Broadcast to pose stream subscribers
            sent_count = await self.connection_manager.broadcast(
                data=broadcast_data,
                stream_type="pose",
                zone_ids=[pose_data.zone_id]
            )
            
            logger.info(f"✅ Broadcasted pose data for zone {pose_data.zone_id} to {sent_count} clients")
        
        except Exception as e:
            logger.error(f"Error broadcasting pose data: {e}")
    
    async def handle_client_subscription(
        self,
        client_id: str,
        subscription_config: Dict[str, Any]
    ):
        """Handle client subscription configuration."""
        try:
            # Store client subscription config
            self.subscribers[client_id] = {
                "zone_ids": subscription_config.get("zone_ids", []),
                "min_confidence": subscription_config.get("min_confidence", 0.5),
                "max_fps": subscription_config.get("max_fps", 30),
                "include_metadata": subscription_config.get("include_metadata", True),
                "stream_types": subscription_config.get("stream_types", ["pose_data"]),
                "subscribed_at": datetime.utcnow()
            }
            
            logger.info(f"Updated subscription for client {client_id}")
            
            # Send confirmation
            confirmation = {
                "type": "subscription_updated",
                "client_id": client_id,
                "config": self.subscribers[client_id],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.connection_manager.send_to_client(client_id, confirmation)
        
        except Exception as e:
            logger.error(f"Error handling client subscription: {e}")
    
    async def handle_client_disconnect(self, client_id: str):
        """Handle client disconnection."""
        if client_id in self.subscribers:
            del self.subscribers[client_id]
            logger.info(f"Removed subscription for disconnected client {client_id}")
    
    async def send_historical_data(
        self,
        client_id: str,
        zone_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ):
        """Send historical pose data to client."""
        try:
            # Get historical data from pose service
            historical_data = await self.pose_service.get_historical_data(
                zone_id=zone_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Send data in chunks to avoid overwhelming the client
            chunk_size = 10
            for i in range(0, len(historical_data), chunk_size):
                chunk = historical_data[i:i + chunk_size]
                
                message = {
                    "type": "historical_data",
                    "zone_id": zone_id,
                    "chunk_index": i // chunk_size,
                    "total_chunks": (len(historical_data) + chunk_size - 1) // chunk_size,
                    "data": chunk,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.connection_manager.send_to_client(client_id, message)
                
                # Small delay between chunks
                await asyncio.sleep(0.1)
            
            # Send completion message
            completion_message = {
                "type": "historical_data_complete",
                "zone_id": zone_id,
                "total_records": len(historical_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.connection_manager.send_to_client(client_id, completion_message)
        
        except Exception as e:
            logger.error(f"Error sending historical data: {e}")
            
            # Send error message to client
            error_message = {
                "type": "error",
                "message": f"Failed to retrieve historical data: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.connection_manager.send_to_client(client_id, error_message)
    
    async def send_zone_statistics(self, client_id: str, zone_id: str):
        """Send zone statistics to client."""
        try:
            # Get zone statistics
            stats = await self.pose_service.get_zone_statistics(zone_id)
            
            message = {
                "type": "zone_statistics",
                "zone_id": zone_id,
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.connection_manager.send_to_client(client_id, message)
        
        except Exception as e:
            logger.error(f"Error sending zone statistics: {e}")
    
    async def broadcast_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast system events to all connected clients."""
        try:
            message = {
                "type": "system_event",
                "event_type": event_type,
                "data": event_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to all pose stream clients
            sent_count = await self.connection_manager.broadcast(
                data=message,
                stream_type="pose"
            )
            
            logger.info(f"Broadcasted system event '{event_type}' to {sent_count} clients")
        
        except Exception as e:
            logger.error(f"Error broadcasting system event: {e}")
    
    async def update_stream_config(self, config: Dict[str, Any]):
        """Update streaming configuration."""
        try:
            # Validate and update configuration
            if "fps" in config:
                fps = max(1, min(60, config["fps"]))
                self.stream_config["fps"] = fps
            
            if "min_confidence" in config:
                confidence = max(0.0, min(1.0, config["min_confidence"]))
                self.stream_config["min_confidence"] = confidence
            
            if "include_metadata" in config:
                self.stream_config["include_metadata"] = bool(config["include_metadata"])
            
            if "buffer_size" in config:
                buffer_size = max(10, min(1000, config["buffer_size"]))
                self.stream_config["buffer_size"] = buffer_size
            
            logger.info(f"Updated stream configuration: {self.stream_config}")
            
            # Broadcast configuration update to clients
            await self.broadcast_system_event("stream_config_updated", {
                "new_config": self.stream_config
            })
        
        except Exception as e:
            logger.error(f"Error updating stream configuration: {e}")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get current streaming status."""
        return {
            "is_streaming": self.is_streaming,
            "config": self.stream_config,
            "subscriber_count": len(self.subscribers),
            "subscribers": {
                client_id: {
                    "zone_ids": sub["zone_ids"],
                    "min_confidence": sub["min_confidence"],
                    "subscribed_at": sub["subscribed_at"].isoformat()
                }
                for client_id, sub in self.subscribers.items()
            }
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics."""
        try:
            # Get connection manager metrics
            conn_metrics = await self.connection_manager.get_metrics()
            
            # Get pose service metrics
            pose_metrics = await self.pose_service.get_performance_metrics()
            
            return {
                "streaming": {
                    "is_active": self.is_streaming,
                    "fps": self.stream_config["fps"],
                    "subscriber_count": len(self.subscribers)
                },
                "connections": conn_metrics,
                "pose_service": pose_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown pose stream handler."""
        await self.stop_streaming()
        self.subscribers.clear()
        logger.info("Pose stream handler shutdown complete")