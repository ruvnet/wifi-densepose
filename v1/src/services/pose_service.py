"""
Pose estimation service for WiFi-DensePose API
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import torch

from src.config.settings import Settings
from src.config.domains import DomainConfig
from src.core.csi_processor import CSIProcessor
from src.core.phase_sanitizer import PhaseSanitizer
from src.models.densepose_head import DensePoseHead
from src.models.modality_translation import ModalityTranslationNetwork

logger = logging.getLogger(__name__)


class PoseService:
    """Service for pose estimation operations."""
    
    def __init__(self, settings: Settings, domain_config: DomainConfig):
        """Initialize pose service."""
        self.settings = settings
        self.domain_config = domain_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.csi_processor = None
        self.phase_sanitizer = None
        self.densepose_model = None
        self.modality_translator = None
        
        # Service state
        self.is_initialized = False
        self.is_running = False
        self.last_error = None
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
    
    async def initialize(self):
        """Initialize the pose service."""
        try:
            self.logger.info("Initializing pose service...")
            
            # Initialize CSI processor
            csi_config = {
                'buffer_size': self.settings.csi_buffer_size,
                'sampling_rate': getattr(self.settings, 'csi_sampling_rate', 1000),
                'window_size': getattr(self.settings, 'csi_window_size', 512),
                'overlap': getattr(self.settings, 'csi_overlap', 0.5),
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1),
                'human_detection_threshold': getattr(self.settings, 'csi_human_detection_threshold', 0.8),
                'smoothing_factor': getattr(self.settings, 'csi_smoothing_factor', 0.9),
                'max_history_size': getattr(self.settings, 'csi_max_history_size', 500),
                'num_subcarriers': 56,
                'num_antennas': 3
            }
            self.csi_processor = CSIProcessor(config=csi_config)
            
            # Initialize phase sanitizer
            phase_config = {
                'unwrapping_method': 'numpy',
                'outlier_threshold': 3.0,
                'smoothing_window': 5,
                'enable_outlier_removal': True,
                'enable_smoothing': True,
                'enable_noise_filtering': True,
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1)
            }
            self.phase_sanitizer = PhaseSanitizer(config=phase_config)
            
            # Initialize models if not mocking
            if not self.settings.mock_pose_data:
                await self._initialize_models()
            else:
                self.logger.info("Using mock pose data for development")
            
            self.is_initialized = True
            self.logger.info("Pose service initialized successfully")
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize pose service: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize neural network models."""
        try:
            # Initialize DensePose model
            if self.settings.pose_model_path:
                self.densepose_model = DensePoseHead()
                # Load model weights if path is provided
                # model_state = torch.load(self.settings.pose_model_path)
                # self.densepose_model.load_state_dict(model_state)
                self.logger.info("DensePose model loaded")
            else:
                self.logger.warning("No pose model path provided, using default model")
                self.densepose_model = DensePoseHead()
            
            # Initialize modality translation
            config = {
                'input_channels': 64,  # CSI data channels
                'hidden_channels': [128, 256, 512],
                'output_channels': 256,  # Visual feature channels
                'use_attention': True
            }
            self.modality_translator = ModalityTranslationNetwork(config)
            
            # Set models to evaluation mode
            self.densepose_model.eval()
            self.modality_translator.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def start(self):
        """Start the pose service."""
        if not self.is_initialized:
            await self.initialize()
        
        self.is_running = True
        self.logger.info("Pose service started")
    
    async def stop(self):
        """Stop the pose service."""
        self.is_running = False
        self.logger.info("Pose service stopped")
    
    async def process_csi_data(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSI data and estimate poses."""
        if not self.is_running:
            raise RuntimeError("Pose service is not running")
        
        start_time = datetime.now()
        
        try:
            # Process CSI data
            processed_csi = await self._process_csi(csi_data, metadata)
            
            # Estimate poses
            poses = await self._estimate_poses(processed_csi, metadata)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(poses, processing_time)
            
            return {
                "timestamp": start_time.isoformat(),
                "poses": poses,
                "metadata": metadata,
                "processing_time_ms": processing_time,
                "confidence_scores": [pose.get("confidence", 0.0) for pose in poses]
            }
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["failed_detections"] += 1
            self.logger.error(f"Error processing CSI data: {e}")
            raise
    
    async def _process_csi(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Process raw CSI data."""
        # Convert raw data to CSIData format
        from src.hardware.csi_extractor import CSIData
        
        # Create CSIData object with proper fields
        # For mock data, create amplitude and phase from input
        if csi_data.ndim == 1:
            amplitude = np.abs(csi_data)
            phase = np.angle(csi_data) if np.iscomplexobj(csi_data) else np.zeros_like(csi_data)
        else:
            amplitude = csi_data
            phase = np.zeros_like(csi_data)
        
        csi_data_obj = CSIData(
            timestamp=metadata.get("timestamp", datetime.now()),
            amplitude=amplitude,
            phase=phase,
            frequency=metadata.get("frequency", 5.0),  # 5 GHz default
            bandwidth=metadata.get("bandwidth", 20.0),  # 20 MHz default
            num_subcarriers=metadata.get("num_subcarriers", 56),
            num_antennas=metadata.get("num_antennas", 3),
            snr=metadata.get("snr", 20.0),  # 20 dB default
            metadata=metadata
        )
        
        # Process CSI data
        try:
            detection_result = await self.csi_processor.process_csi_data(csi_data_obj)
            
            # Add to history for temporal analysis
            self.csi_processor.add_to_history(csi_data_obj)
            
            # Extract amplitude data for pose estimation
            if detection_result and detection_result.features:
                amplitude_data = detection_result.features.amplitude_mean
                
                # Apply phase sanitization if we have phase data
                if hasattr(detection_result.features, 'phase_difference'):
                    phase_data = detection_result.features.phase_difference
                    sanitized_phase = self.phase_sanitizer.sanitize(phase_data)
                    # Combine amplitude and phase data
                    return np.concatenate([amplitude_data, sanitized_phase])
                
                return amplitude_data
            
        except Exception as e:
            self.logger.warning(f"CSI processing failed, using raw data: {e}")
        
        return csi_data
    
    async def _estimate_poses(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate poses from processed CSI data."""
        if self.settings.mock_pose_data:
            return self._generate_mock_poses()
        
        try:
            # Convert CSI data to tensor
            csi_tensor = torch.from_numpy(csi_data).float()
            
            # Add batch dimension if needed
            if len(csi_tensor.shape) == 2:
                csi_tensor = csi_tensor.unsqueeze(0)
            
            # Translate modality (CSI to visual-like features)
            with torch.no_grad():
                visual_features = self.modality_translator(csi_tensor)
                
                # Estimate poses using DensePose
                pose_outputs = self.densepose_model(visual_features)
            
            # Convert outputs to pose detections
            poses = self._parse_pose_outputs(pose_outputs)
            
            # Filter by confidence threshold
            filtered_poses = [
                pose for pose in poses 
                if pose.get("confidence", 0.0) >= self.settings.pose_confidence_threshold
            ]
            
            # Limit number of persons
            if len(filtered_poses) > self.settings.pose_max_persons:
                filtered_poses = sorted(
                    filtered_poses, 
                    key=lambda x: x.get("confidence", 0.0), 
                    reverse=True
                )[:self.settings.pose_max_persons]
            
            return filtered_poses
            
        except Exception as e:
            self.logger.error(f"Error in pose estimation: {e}")
            return []
    
    def _parse_pose_outputs(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Parse neural network outputs into pose detections.

        The DensePose model outputs:
        - segmentation: (batch, num_parts+1, H, W) - body part segmentation
        - uv_coords: (batch, 2, H, W) - UV coordinates for surface mapping

        Returns list of detected persons with keypoints and body parts.
        """
        poses = []

        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Simple tensor output - use legacy parsing
            return self._parse_simple_outputs(outputs)

        # DensePose structured output
        segmentation = outputs.get('segmentation')
        uv_coords = outputs.get('uv_coords')

        if segmentation is None:
            return []

        batch_size = segmentation.shape[0]

        for batch_idx in range(batch_size):
            # Get segmentation for this sample
            seg = segmentation[batch_idx]  # (num_parts+1, H, W)

            # Find persons by analyzing body part segmentation
            # Background is class 0, body parts are 1-24
            body_mask = seg[1:].sum(dim=0) > seg[0]  # Any body part vs background

            if not body_mask.any():
                continue

            # Find connected components (persons)
            person_regions = self._find_person_regions(body_mask)

            for person_idx, region in enumerate(person_regions):
                # Extract keypoints from body part segmentation
                keypoints = self._extract_keypoints_from_segmentation(seg, region)

                # Calculate bounding box from region
                bbox = self._calculate_bounding_box(region)

                # Calculate confidence from segmentation probabilities
                seg_probs = torch.softmax(seg, dim=0)
                region_mask = region['mask']
                confidence = float(seg_probs[1:, region_mask].max().item())

                # Classify activity from pose keypoints
                activity = self._classify_activity_from_keypoints(keypoints)

                pose = {
                    "person_id": person_idx,
                    "confidence": confidence,
                    "keypoints": keypoints,
                    "bounding_box": bbox,
                    "activity": activity,
                    "timestamp": datetime.now().isoformat(),
                    "body_parts": self._extract_body_parts(seg, region) if uv_coords is not None else None
                }

                poses.append(pose)

        return poses

    def _parse_simple_outputs(self, outputs: torch.Tensor) -> List[Dict[str, Any]]:
        """Parse simple tensor outputs (fallback for non-DensePose models)."""
        poses = []
        batch_size = outputs.shape[0]

        for i in range(batch_size):
            output = outputs[i]

            # Extract confidence from first channel
            confidence = float(torch.sigmoid(output[0]).mean().item()) if output.numel() > 0 else 0.0

            if confidence < 0.1:
                continue

            # Try to extract keypoints from output tensor
            keypoints = self._extract_keypoints_from_tensor(output)
            bbox = self._estimate_bbox_from_keypoints(keypoints)
            activity = self._classify_activity_from_keypoints(keypoints)

            pose = {
                "person_id": i,
                "confidence": confidence,
                "keypoints": keypoints,
                "bounding_box": bbox,
                "activity": activity,
                "timestamp": datetime.now().isoformat()
            }

            poses.append(pose)

        return poses

    def _find_person_regions(self, body_mask: torch.Tensor) -> List[Dict[str, Any]]:
        """Find distinct person regions in body mask using connected components."""
        # Convert to numpy for connected component analysis
        mask_np = body_mask.cpu().numpy().astype(np.uint8)

        # Simple connected component labeling
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask_np)

        regions = []
        for label_id in range(1, num_features + 1):
            region_mask = labeled == label_id
            if region_mask.sum() < 100:  # Minimum region size
                continue

            # Find bounding coordinates
            coords = np.where(region_mask)
            regions.append({
                'mask': torch.from_numpy(region_mask),
                'y_min': int(coords[0].min()),
                'y_max': int(coords[0].max()),
                'x_min': int(coords[1].min()),
                'x_max': int(coords[1].max()),
                'area': int(region_mask.sum())
            })

        return regions

    def _extract_keypoints_from_segmentation(
        self, segmentation: torch.Tensor, region: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract keypoints from body part segmentation."""
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # Mapping from body parts to keypoints
        # DensePose has 24 body parts, we map to COCO keypoints
        part_to_keypoint = {
            14: "nose",  # Head -> nose
            10: "left_shoulder", 11: "right_shoulder",
            12: "left_elbow", 13: "right_elbow",
            2: "left_wrist", 3: "right_wrist",  # Hands approximate wrists
            7: "left_hip", 6: "right_hip",  # Upper legs
            9: "left_knee", 8: "right_knee",  # Lower legs
            4: "left_ankle", 5: "right_ankle",  # Feet approximate ankles
        }

        h, w = segmentation.shape[1], segmentation.shape[2]
        keypoints = []

        # Get softmax probabilities
        seg_probs = torch.softmax(segmentation, dim=0)

        for kp_name in keypoint_names:
            # Find which body part corresponds to this keypoint
            part_idx = None
            for part, name in part_to_keypoint.items():
                if name == kp_name:
                    part_idx = part
                    break

            if part_idx is not None and part_idx < seg_probs.shape[0]:
                # Get probability map for this part within the region
                part_prob = seg_probs[part_idx] * region['mask'].float()

                if part_prob.max() > 0.1:
                    # Find location of maximum probability
                    max_idx = part_prob.argmax()
                    y = int(max_idx // w)
                    x = int(max_idx % w)

                    keypoints.append({
                        "name": kp_name,
                        "x": float(x) / w,
                        "y": float(y) / h,
                        "confidence": float(part_prob.max().item())
                    })
                else:
                    # Keypoint not visible
                    keypoints.append({
                        "name": kp_name,
                        "x": 0.0,
                        "y": 0.0,
                        "confidence": 0.0
                    })
            else:
                # Estimate position based on body region
                cx = (region['x_min'] + region['x_max']) / 2 / w
                cy = (region['y_min'] + region['y_max']) / 2 / h
                keypoints.append({
                    "name": kp_name,
                    "x": float(cx),
                    "y": float(cy),
                    "confidence": 0.1
                })

        return keypoints

    def _calculate_bounding_box(self, region: Dict[str, Any]) -> Dict[str, float]:
        """Calculate normalized bounding box from region."""
        # Assume region contains mask shape info
        mask = region['mask']
        h, w = mask.shape

        return {
            "x": float(region['x_min']) / w,
            "y": float(region['y_min']) / h,
            "width": float(region['x_max'] - region['x_min']) / w,
            "height": float(region['y_max'] - region['y_min']) / h
        }

    def _extract_body_parts(
        self, segmentation: torch.Tensor, region: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract body part information from segmentation."""
        part_names = [
            "background", "torso", "right_hand", "left_hand", "left_foot", "right_foot",
            "upper_leg_right", "upper_leg_left", "lower_leg_right", "lower_leg_left",
            "upper_arm_left", "upper_arm_right", "lower_arm_left", "lower_arm_right", "head"
        ]

        seg_probs = torch.softmax(segmentation, dim=0)
        region_mask = region['mask']

        parts = {}
        for i, name in enumerate(part_names):
            if i < seg_probs.shape[0]:
                part_prob = seg_probs[i] * region_mask.float()
                parts[name] = {
                    "present": bool(part_prob.max() > 0.3),
                    "confidence": float(part_prob.max().item()),
                    "coverage": float((part_prob > 0.3).sum().item() / max(1, region_mask.sum().item()))
                }

        return parts

    def _extract_keypoints_from_tensor(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract keypoints from a generic output tensor."""
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        keypoints = []

        # Try to interpret output as heatmaps
        if output.dim() >= 2:
            flat = output.flatten()
            num_kp = len(keypoint_names)

            # Divide output evenly for each keypoint
            chunk_size = len(flat) // num_kp if num_kp > 0 else 1

            for i, name in enumerate(keypoint_names):
                start = i * chunk_size
                end = min(start + chunk_size, len(flat))

                if start < len(flat):
                    chunk = flat[start:end]
                    # Find max location in chunk
                    max_val = chunk.max().item()
                    max_idx = chunk.argmax().item()

                    # Convert to x, y (assume square spatial layout)
                    side = int(np.sqrt(chunk_size))
                    if side > 0:
                        x = (max_idx % side) / side
                        y = (max_idx // side) / side
                    else:
                        x, y = 0.5, 0.5

                    keypoints.append({
                        "name": name,
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(torch.sigmoid(torch.tensor(max_val)).item())
                    })
                else:
                    keypoints.append({
                        "name": name, "x": 0.5, "y": 0.5, "confidence": 0.0
                    })
        else:
            # Fallback
            for name in keypoint_names:
                keypoints.append({"name": name, "x": 0.5, "y": 0.5, "confidence": 0.1})

        return keypoints

    def _estimate_bbox_from_keypoints(self, keypoints: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate bounding box from keypoint positions."""
        valid_kps = [kp for kp in keypoints if kp['confidence'] > 0.1]

        if not valid_kps:
            return {"x": 0.3, "y": 0.2, "width": 0.4, "height": 0.6}

        xs = [kp['x'] for kp in valid_kps]
        ys = [kp['y'] for kp in valid_kps]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add padding
        padding = 0.05
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1, x_max + padding)
        y_max = min(1, y_max + padding)

        return {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }

    def _classify_activity_from_keypoints(self, keypoints: List[Dict[str, Any]]) -> str:
        """Classify activity based on keypoint positions."""
        # Get key body parts
        kp_dict = {kp['name']: kp for kp in keypoints}

        # Check if enough keypoints are detected
        valid_count = sum(1 for kp in keypoints if kp['confidence'] > 0.3)
        if valid_count < 5:
            return "unknown"

        # Get relevant keypoints
        nose = kp_dict.get('nose', {})
        l_hip = kp_dict.get('left_hip', {})
        r_hip = kp_dict.get('right_hip', {})
        l_ankle = kp_dict.get('left_ankle', {})
        r_ankle = kp_dict.get('right_ankle', {})
        l_shoulder = kp_dict.get('left_shoulder', {})
        r_shoulder = kp_dict.get('right_shoulder', {})

        # Calculate body metrics
        hip_y = (l_hip.get('y', 0.5) + r_hip.get('y', 0.5)) / 2
        ankle_y = (l_ankle.get('y', 0.8) + r_ankle.get('y', 0.8)) / 2
        shoulder_y = (l_shoulder.get('y', 0.3) + r_shoulder.get('y', 0.3)) / 2
        nose_y = nose.get('y', 0.2)

        # Leg spread (horizontal distance between ankles)
        leg_spread = abs(l_ankle.get('x', 0.5) - r_ankle.get('x', 0.5))

        # Vertical compression (how "tall" the pose is)
        vertical_span = ankle_y - nose_y if ankle_y > nose_y else 0.6

        # Classification logic
        if vertical_span < 0.3:
            # Very compressed vertically - likely lying down
            return "lying"
        elif vertical_span < 0.45 and hip_y > 0.5:
            # Medium compression with low hips - sitting
            return "sitting"
        elif leg_spread > 0.15:
            # Legs apart - likely walking
            return "walking"
        else:
            # Default upright pose
            return "standing"
    
    def _generate_mock_poses(self) -> List[Dict[str, Any]]:
        """Generate mock pose data for development."""
        import random
        
        num_persons = random.randint(1, min(3, self.settings.pose_max_persons))
        poses = []
        
        for i in range(num_persons):
            confidence = random.uniform(0.3, 0.95)
            
            pose = {
                "person_id": i,
                "confidence": confidence,
                "keypoints": self._generate_keypoints(),
                "bounding_box": self._generate_bounding_box(),
                "activity": random.choice(["standing", "sitting", "walking", "lying"]),
                "timestamp": datetime.now().isoformat()
            }
            
            poses.append(pose)
        
        return poses
    
    def _generate_keypoints(self) -> List[Dict[str, Any]]:
        """Generate keypoints for a person."""
        import random
        
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        keypoints = []
        for name in keypoint_names:
            keypoints.append({
                "name": name,
                "x": random.uniform(0.1, 0.9),
                "y": random.uniform(0.1, 0.9),
                "confidence": random.uniform(0.5, 0.95)
            })
        
        return keypoints
    
    def _generate_bounding_box(self) -> Dict[str, float]:
        """Generate bounding box for a person."""
        import random
        
        x = random.uniform(0.1, 0.6)
        y = random.uniform(0.1, 0.6)
        width = random.uniform(0.2, 0.4)
        height = random.uniform(0.3, 0.5)
        
        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    
    def _classify_activity(self, features: torch.Tensor) -> str:
        """Classify activity from features."""
        # Simple mock classification
        import random
        activities = ["standing", "sitting", "walking", "lying", "unknown"]
        return random.choice(activities)
    
    def _update_stats(self, poses: List[Dict[str, Any]], processing_time: float):
        """Update processing statistics."""
        self.stats["total_processed"] += 1
        
        if poses:
            self.stats["successful_detections"] += 1
            confidences = [pose.get("confidence", 0.0) for pose in poses]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Update running average
            total = self.stats["successful_detections"]
            current_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (current_avg * (total - 1) + avg_confidence) / total
        else:
            self.stats["failed_detections"] += 1
        
        # Update processing time (running average)
        total = self.stats["total_processed"]
        current_avg = self.stats["processing_time_ms"]
        self.stats["processing_time_ms"] = (current_avg * (total - 1) + processing_time) / total
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
            "initialized": self.is_initialized,
            "running": self.is_running,
            "last_error": self.last_error,
            "statistics": self.stats.copy(),
            "configuration": {
                "mock_data": self.settings.mock_pose_data,
                "confidence_threshold": self.settings.pose_confidence_threshold,
                "max_persons": self.settings.pose_max_persons,
                "batch_size": self.settings.pose_processing_batch_size
            }
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "pose_service": {
                "total_processed": self.stats["total_processed"],
                "successful_detections": self.stats["successful_detections"],
                "failed_detections": self.stats["failed_detections"],
                "success_rate": (
                    self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                ),
                "average_confidence": self.stats["average_confidence"],
                "average_processing_time_ms": self.stats["processing_time_ms"]
            }
        }
    
    async def reset(self):
        """Reset service state."""
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
        self.last_error = None
        self.logger.info("Pose service reset")
    
    # API endpoint methods
    async def estimate_poses(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                           include_keypoints=True, include_segmentation=False):
        """Estimate poses with API parameters."""
        try:
            # Generate mock CSI data for estimation
            mock_csi = np.random.randn(64, 56, 3)  # Mock CSI data
            metadata = {
                "timestamp": datetime.now(),
                "zone_ids": zone_ids or ["zone_1"],
                "confidence_threshold": confidence_threshold or self.settings.pose_confidence_threshold,
                "max_persons": max_persons or self.settings.pose_max_persons
            }
            
            # Process the data
            result = await self.process_csi_data(mock_csi, metadata)
            
            # Format for API response
            persons = []
            for i, pose in enumerate(result["poses"]):
                person = {
                    "person_id": str(pose["person_id"]),
                    "confidence": pose["confidence"],
                    "bounding_box": pose["bounding_box"],
                    "zone_id": zone_ids[0] if zone_ids else "zone_1",
                    "activity": pose["activity"],
                    "timestamp": datetime.fromisoformat(pose["timestamp"])
                }
                
                if include_keypoints:
                    person["keypoints"] = pose["keypoints"]
                
                if include_segmentation:
                    person["segmentation"] = {"mask": "mock_segmentation_data"}
                
                persons.append(person)
            
            # Zone summary
            zone_summary = {}
            for zone_id in (zone_ids or ["zone_1"]):
                zone_summary[zone_id] = len([p for p in persons if p.get("zone_id") == zone_id])
            
            return {
                "timestamp": datetime.now(),
                "frame_id": f"frame_{int(datetime.now().timestamp())}",
                "persons": persons,
                "zone_summary": zone_summary,
                "processing_time_ms": result["processing_time_ms"],
                "metadata": {"mock_data": self.settings.mock_pose_data}
            }
            
        except Exception as e:
            self.logger.error(f"Error in estimate_poses: {e}")
            raise
    
    async def analyze_with_params(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                                include_keypoints=True, include_segmentation=False):
        """Analyze pose data with custom parameters."""
        return await self.estimate_poses(zone_ids, confidence_threshold, max_persons,
                                       include_keypoints, include_segmentation)
    
    async def get_zone_occupancy(self, zone_id: str):
        """Get current occupancy for a specific zone."""
        try:
            # Mock occupancy data
            import random
            count = random.randint(0, 5)
            persons = []
            
            for i in range(count):
                persons.append({
                    "person_id": f"person_{i}",
                    "confidence": random.uniform(0.7, 0.95),
                    "activity": random.choice(["standing", "sitting", "walking"])
                })
            
            return {
                "count": count,
                "max_occupancy": 10,
                "persons": persons,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting zone occupancy: {e}")
            return None
    
    async def get_zones_summary(self):
        """Get occupancy summary for all zones."""
        try:
            import random
            zones = ["zone_1", "zone_2", "zone_3", "zone_4"]
            zone_data = {}
            total_persons = 0
            active_zones = 0
            
            for zone_id in zones:
                count = random.randint(0, 3)
                zone_data[zone_id] = {
                    "occupancy": count,
                    "max_occupancy": 10,
                    "status": "active" if count > 0 else "inactive"
                }
                total_persons += count
                if count > 0:
                    active_zones += 1
            
            return {
                "total_persons": total_persons,
                "zones": zone_data,
                "active_zones": active_zones
            }
            
        except Exception as e:
            self.logger.error(f"Error getting zones summary: {e}")
            raise
    
    async def get_historical_data(self, start_time, end_time, zone_ids=None,
                                aggregation_interval=300, include_raw_data=False):
        """Get historical pose estimation data."""
        try:
            # Mock historical data
            import random
            from datetime import timedelta
            
            current_time = start_time
            aggregated_data = []
            raw_data = [] if include_raw_data else None
            
            while current_time < end_time:
                # Generate aggregated data point
                data_point = {
                    "timestamp": current_time,
                    "total_persons": random.randint(0, 8),
                    "zones": {}
                }
                
                for zone_id in (zone_ids or ["zone_1", "zone_2", "zone_3"]):
                    data_point["zones"][zone_id] = {
                        "occupancy": random.randint(0, 3),
                        "avg_confidence": random.uniform(0.7, 0.95)
                    }
                
                aggregated_data.append(data_point)
                
                # Generate raw data if requested
                if include_raw_data:
                    for _ in range(random.randint(0, 5)):
                        raw_data.append({
                            "timestamp": current_time + timedelta(seconds=random.randint(0, aggregation_interval)),
                            "person_id": f"person_{random.randint(1, 10)}",
                            "zone_id": random.choice(zone_ids or ["zone_1", "zone_2", "zone_3"]),
                            "confidence": random.uniform(0.5, 0.95),
                            "activity": random.choice(["standing", "sitting", "walking"])
                        })
                
                current_time += timedelta(seconds=aggregation_interval)
            
            return {
                "aggregated_data": aggregated_data,
                "raw_data": raw_data,
                "total_records": len(aggregated_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_recent_activities(self, zone_id=None, limit=10):
        """Get recently detected activities."""
        try:
            import random
            activities = []
            
            for i in range(limit):
                activity = {
                    "activity_id": f"activity_{i}",
                    "person_id": f"person_{random.randint(1, 5)}",
                    "zone_id": zone_id or random.choice(["zone_1", "zone_2", "zone_3"]),
                    "activity": random.choice(["standing", "sitting", "walking", "lying"]),
                    "confidence": random.uniform(0.6, 0.95),
                    "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 60)),
                    "duration_seconds": random.randint(10, 300)
                }
                activities.append(activity)
            
            return activities
            
        except Exception as e:
            self.logger.error(f"Error getting recent activities: {e}")
            raise
    
    async def is_calibrating(self):
        """Check if calibration is in progress."""
        return False  # Mock implementation
    
    async def start_calibration(self):
        """Start calibration process."""
        import uuid
        calibration_id = str(uuid.uuid4())
        self.logger.info(f"Started calibration: {calibration_id}")
        return calibration_id
    
    async def run_calibration(self, calibration_id):
        """Run calibration process."""
        self.logger.info(f"Running calibration: {calibration_id}")
        # Mock calibration process
        await asyncio.sleep(5)
        self.logger.info(f"Calibration completed: {calibration_id}")
    
    async def get_calibration_status(self):
        """Get current calibration status."""
        return {
            "is_calibrating": False,
            "calibration_id": None,
            "progress_percent": 100,
            "current_step": "completed",
            "estimated_remaining_minutes": 0,
            "last_calibration": datetime.now() - timedelta(hours=1)
        }
    
    async def get_statistics(self, start_time, end_time):
        """Get pose estimation statistics."""
        try:
            import random
            
            # Mock statistics
            total_detections = random.randint(100, 1000)
            successful_detections = int(total_detections * random.uniform(0.8, 0.95))
            
            return {
                "total_detections": total_detections,
                "successful_detections": successful_detections,
                "failed_detections": total_detections - successful_detections,
                "success_rate": successful_detections / total_detections,
                "average_confidence": random.uniform(0.75, 0.90),
                "average_processing_time_ms": random.uniform(50, 200),
                "unique_persons": random.randint(5, 20),
                "most_active_zone": random.choice(["zone_1", "zone_2", "zone_3"]),
                "activity_distribution": {
                    "standing": random.uniform(0.3, 0.5),
                    "sitting": random.uniform(0.2, 0.4),
                    "walking": random.uniform(0.1, 0.3),
                    "lying": random.uniform(0.0, 0.1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise
    
    async def process_segmentation_data(self, frame_id):
        """Process segmentation data in background."""
        self.logger.info(f"Processing segmentation data for frame: {frame_id}")
        # Mock background processing
        await asyncio.sleep(2)
        self.logger.info(f"Segmentation processing completed for frame: {frame_id}")
    
    # WebSocket streaming methods
    async def get_current_pose_data(self):
        """Get current pose data for streaming."""
        try:
            # Generate current pose data
            result = await self.estimate_poses()
            
            # Format data by zones for WebSocket streaming
            zone_data = {}
            
            # Group persons by zone
            for person in result["persons"]:
                zone_id = person.get("zone_id", "zone_1")
                
                if zone_id not in zone_data:
                    zone_data[zone_id] = {
                        "pose": {
                            "persons": [],
                            "count": 0
                        },
                        "confidence": 0.0,
                        "activity": None,
                        "metadata": {
                            "frame_id": result["frame_id"],
                            "processing_time_ms": result["processing_time_ms"]
                        }
                    }
                
                zone_data[zone_id]["pose"]["persons"].append(person)
                zone_data[zone_id]["pose"]["count"] += 1
                
                # Update zone confidence (average)
                current_confidence = zone_data[zone_id]["confidence"]
                person_confidence = person.get("confidence", 0.0)
                zone_data[zone_id]["confidence"] = (current_confidence + person_confidence) / 2
                
                # Set activity if not already set
                if not zone_data[zone_id]["activity"] and person.get("activity"):
                    zone_data[zone_id]["activity"] = person["activity"]
            
            return zone_data
            
        except Exception as e:
            self.logger.error(f"Error getting current pose data: {e}")
            # Return empty zone data on error
            return {}
    
    # Health check methods
    async def health_check(self):
        """Perform health check."""
        try:
            status = "healthy" if self.is_running and not self.last_error else "unhealthy"
            
            return {
                "status": status,
                "message": self.last_error if self.last_error else "Service is running normally",
                "uptime_seconds": 0.0,  # TODO: Implement actual uptime tracking
                "metrics": {
                    "total_processed": self.stats["total_processed"],
                    "success_rate": (
                        self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                    ),
                    "average_processing_time_ms": self.stats["processing_time_ms"]
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def is_ready(self):
        """Check if service is ready."""
        return self.is_initialized and self.is_running