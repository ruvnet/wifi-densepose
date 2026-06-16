// Data Processor - WiFi DensePose 3D Visualization
// Transforms API data into Three.js geometry updates

export class DataProcessor {
  constructor() {
    // Smoothing buffers
    this._lastProcessedPersons = [];
    this._smoothingFactor = 0.3;
  }

  // Process incoming WebSocket message into visualization-ready data
  processMessage(message) {
    if (!message) return null;

    const result = {
      persons: [],
      zoneOccupancy: {},
      signalData: null,
      metadata: {
        isRealData: false,
        timestamp: null,
        processingTime: 0,
        frameId: null,
        sensingMode: 'Unavailable'
      }
    };

    // Handle different message types from the API
    if (message.type === 'pose_data') {
      const payload = message.data || message.payload;
      if (payload) {
        result.persons = this._extractPersons(payload);
        result.zoneOccupancy = this._extractZoneOccupancy(payload, message.zone_id);
        result.signalData = this._extractSignalData(payload);

        result.metadata.isRealData = payload.metadata?.mock_data === false;
        result.metadata.timestamp = message.timestamp;
        result.metadata.processingTime = payload.metadata?.processing_time_ms || 0;
        result.metadata.frameId = payload.metadata?.frame_id;

        // Determine sensing mode
        if (payload.metadata?.source === 'csi') {
          result.metadata.sensingMode = 'CSI';
        } else if (payload.metadata?.source === 'rssi') {
          result.metadata.sensingMode = 'RSSI';
        } else {
          result.metadata.sensingMode = payload.metadata?.mock_data === false ? 'CSI' : 'Unavailable';
        }
      }
    }

    return result;
  }

  // Extract person data with keypoints in COCO format
  _extractPersons(payload) {
    const persons = [];

    if (payload.pose && payload.pose.persons) {
      for (const person of payload.pose.persons) {
        const processed = {
          id: person.id || `person_${persons.length}`,
          confidence: person.confidence || 0,
          keypoints: this._normalizeKeypoints(person.keypoints),
          bbox: person.bbox || null,
          body_parts: person.densepose_parts || person.body_parts || null
        };
        persons.push(processed);
      }
    } else if (payload.persons) {
      // Alternative format: persons at top level
      for (const person of payload.persons) {
        persons.push({
          id: person.id || `person_${persons.length}`,
          confidence: person.confidence || 0,
          keypoints: this._normalizeKeypoints(person.keypoints),
          bbox: person.bbox || null,
          body_parts: person.densepose_parts || person.body_parts || null
        });
      }
    }

    return persons;
  }

  // Normalize keypoints to {x, y, confidence} format in [0,1] range
  _normalizeKeypoints(keypoints) {
    if (!keypoints || keypoints.length === 0) return [];

    return keypoints.map(kp => {
      // Handle various formats
      if (Array.isArray(kp)) {
        return { x: kp[0], y: kp[1], confidence: kp[2] || 0.5 };
      }
      return {
        x: kp.x !== undefined ? kp.x : 0,
        y: kp.y !== undefined ? kp.y : 0,
        confidence: kp.confidence !== undefined ? kp.confidence : (kp.score || 0.5)
      };
    });
  }

  // Extract zone occupancy data
  _extractZoneOccupancy(payload, zoneId) {
    const occupancy = {};

    if (payload.zone_summary) {
      Object.assign(occupancy, payload.zone_summary);
    }

    if (zoneId && payload.pose?.persons?.length > 0) {
      occupancy[zoneId] = payload.pose.persons.length;
    }

    return occupancy;
  }

  // Extract signal/CSI data if available
  _extractSignalData(payload) {
    if (payload.signal_data || payload.csi_data) {
      const sig = payload.signal_data || payload.csi_data;
      return {
        amplitude: sig.amplitude || null,
        phase: sig.phase || null,
        doppler: sig.doppler || sig.doppler_spectrum || null,
        motionEnergy: sig.motion_energy !== undefined ? sig.motion_energy : null
      };
    }
    return null;
  }

  // Visualization updates only from live messages.
  generateUnavailableData() {
    return null;
  }

  // Generate a confidence heatmap from person positions
  generateConfidenceHeatmap(persons, cols, rows, roomWidth, roomDepth) {
    const positions = (persons || []).map(p => {
      if (!p.keypoints || p.keypoints.length < 13) return null;
      const hipX = (p.keypoints[11].x + p.keypoints[12].x) / 2;
      const hipY = (p.keypoints[11].y + p.keypoints[12].y) / 2;
      return {
        x: (hipX - 0.5) * roomWidth,
        z: (hipY - 0.5) * roomDepth,
        confidence: p.confidence
      };
    }).filter(Boolean);

    const map = new Float32Array(cols * rows);
    const cellW = roomWidth / cols;
    const cellD = roomDepth / rows;

    for (const pos of positions) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cx = (c + 0.5) * cellW - roomWidth / 2;
          const cz = (r + 0.5) * cellD - roomDepth / 2;
          const dx = cx - pos.x;
          const dz = cz - pos.z;
          const dist = Math.sqrt(dx * dx + dz * dz);
          const conf = Math.exp(-dist * dist * 0.5) * pos.confidence;
          map[r * cols + c] = Math.max(map[r * cols + c], conf);
        }
      }
    }

    return map;
  }

  dispose() {
    this._lastProcessedPersons = [];
  }
}
