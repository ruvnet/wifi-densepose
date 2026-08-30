import type { CameraPreviewFrame, LidarPointFrame, SpatialCalibrationArtifact } from '@/types/lidar';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';
import type { PoseCalibrationArtifact } from '@/types/poseCalibration';

export interface TransientNlosGate {
  status: 'blocked_raw_transients_unavailable' | 'track_stream_available';
  requiredSchema: 'ruview.nlos.transient.v1';
  requiredMeasurement: 'picosecond_photon_histograms';
  algorithmFamily: 'motion_induced_aperture_sampling';
  sensorModel: string | null;
  evidenceLevel: 'l0_synthetic' | 'l1_measured' | 'l2_calibrated' | 'l3_corroborated' | null;
}

export interface SensorFusionDisplayFrame {
  schema: 'ruview.sensor-fusion.display.v1';
  lidarFrame: LidarPointFrame | null;
  cameraPreview: CameraPreviewFrame | null;
  nlosTracks: NlosTrack[];
  nlosFreshness: NlosFreshness;
  calibration: SpatialCalibrationArtifact | null;
  poseCalibration: PoseCalibrationArtifact | null;
  alignment: 'overlay_only';
  transientNlos: TransientNlosGate;
  wallOverlay: {
    enabled: boolean;
    source: 'calibrated_rf_student' | 'live_rf' | 'simulated' | 'unavailable';
    evidenceLabel: 'validated_room_student' | 'rf_pose_hypothesis';
    confidence: number;
    sourceAgeMs: number | null;
    cameraAndLidarVisibleSurfacesOnly: true;
    throughWallClaim: false;
  };
}
