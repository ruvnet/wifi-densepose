// Training Service for WiFi-DensePose UI
// Manages training lifecycle, progress streaming, and CSI recordings.

import { apiService } from './api.service.js';

export class TrainingService {
  constructor() {
    this.progressSocket = null;
    this.progressPollTimer = null;
    this.listeners = {};
    this.logger = this.createLogger();
  }

  createLogger() {
    return {
      debug: (...args) => console.debug('[TRAIN-DEBUG]', new Date().toISOString(), ...args),
      info: (...args) => console.info('[TRAIN-INFO]', new Date().toISOString(), ...args),
      warn: (...args) => console.warn('[TRAIN-WARN]', new Date().toISOString(), ...args),
      error: (...args) => console.error('[TRAIN-ERROR]', new Date().toISOString(), ...args)
    };
  }

  // --- Event emitter helpers ---

  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
  }

  emit(event, data) {
    if (!this.listeners[event]) return;
    this.listeners[event].forEach(cb => {
      try { cb(data); } catch (err) { this.logger.error('Listener error', { event, err }); }
    });
  }

  normalizeTrainingRequest(payload = {}) {
    const config = payload.config || {};
    return {
      dataset_ids: payload.dataset_ids || [],
      config: {
        epochs: Number(config.epochs) || 100,
        batch_size: Number(config.batch_size) || 32,
        learning_rate: Number(config.learning_rate) || 3e-4,
        early_stopping_patience: Number(config.early_stopping_patience ?? config.patience) || 15,
        pretrained_rvf: config.pretrained_rvf || config.base_model || null,
        lora_profile: config.lora_profile || config.lora_profile_name || null
      }
    };
  }

  normalizePretrainRequest(payload = {}) {
    const config = payload.config || {};
    return {
      dataset_ids: payload.dataset_ids || [],
      epochs: Number(config.epochs) || 50,
      lr: Number(config.learning_rate) || 3e-4
    };
  }

  normalizeLoraRequest(payload = {}) {
    const config = payload.config || {};
    return {
      dataset_ids: payload.dataset_ids || [],
      base_model_id: config.base_model_id || config.base_model || config.pretrained_rvf || '',
      profile_name: config.profile_name || config.lora_profile_name || config.lora_profile || 'default',
      rank: Number(config.rank) || 8,
      epochs: Number(config.epochs) || 30
    };
  }

  // --- Training API methods ---

  async startTraining(config) {
    try {
      const request = this.normalizeTrainingRequest(config);
      this.logger.info('Starting training', { request });
      const data = await apiService.post('/api/v1/train/start', request);
      this.emit('training-started', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to start training', { error: error.message });
      throw error;
    }
  }

  async stopTraining() {
    try {
      this.logger.info('Stopping training');
      const data = await apiService.post('/api/v1/train/stop', {});
      this.emit('training-stopped', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to stop training', { error: error.message });
      throw error;
    }
  }

  async getTrainingStatus() {
    try {
      const data = await apiService.get('/api/v1/train/status');
      return data;
    } catch (error) {
      this.logger.error('Failed to get training status', { error: error.message });
      throw error;
    }
  }

  async getRvfReadiness() {
    try {
      const data = await apiService.get('/api/v1/train/rvf/readiness');
      return data;
    } catch (error) {
      this.logger.error('Failed to get RVF training readiness', { error: error.message });
      throw error;
    }
  }

  async startPretraining(config) {
    try {
      const request = this.normalizePretrainRequest(config);
      this.logger.info('Starting pretraining', { request });
      const data = await apiService.post('/api/v1/train/pretrain', request);
      this.emit('training-started', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to start pretraining', { error: error.message });
      throw error;
    }
  }

  async startLoraTraining(config) {
    try {
      const request = this.normalizeLoraRequest(config);
      this.logger.info('Starting LoRA training', { request });
      const data = await apiService.post('/api/v1/train/lora', request);
      this.emit('training-started', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to start LoRA training', { error: error.message });
      throw error;
    }
  }

  // --- Recording API methods ---

  async listRecordings() {
    try {
      const data = await apiService.get('/api/v1/recording/list');
      return data?.recordings ?? [];
    } catch (error) {
      this.logger.error('Failed to list recordings', { error: error.message });
      throw error;
    }
  }

  async startRecording(config) {
    try {
      this.logger.info('Starting recording', { config });
      const data = await apiService.post('/api/v1/recording/start', config);
      this.emit('recording-started', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to start recording', { error: error.message });
      throw error;
    }
  }

  async stopRecording() {
    try {
      this.logger.info('Stopping recording');
      const data = await apiService.post('/api/v1/recording/stop', {});
      this.emit('recording-stopped', data);
      return data;
    } catch (error) {
      this.logger.error('Failed to stop recording', { error: error.message });
      throw error;
    }
  }

  async deleteRecording(id) {
    try {
      this.logger.info('Deleting recording', { id });
      const data = await apiService.delete(
        `/api/v1/recording/${encodeURIComponent(id)}`
      );
      return data;
    } catch (error) {
      this.logger.error('Failed to delete recording', { id, error: error.message });
      throw error;
    }
  }

  // --- Progress stream ---

  connectProgressStream() {
    if (this.progressPollTimer) {
      this.logger.warn('Progress polling already connected');
      return this.progressPollTimer;
    }

    this.logger.info('Connecting progress stream over training status polling');
    this.emit('progress-connected', { transport: 'polling' });

    const poll = async () => {
      try {
        const data = await this.getTrainingStatus();
        if (data) {
          this.emit('progress', data);
          if (!data.active && ['completed', 'idle', 'stopped', 'error'].includes(data.status)) {
            this.disconnectProgressStream();
          }
        }
      } catch (err) {
        this.logger.warn('Progress polling failed', { error: err.message });
      }
    };

    poll();
    this.progressPollTimer = window.setInterval(poll, 750);
    return this.progressPollTimer;
  }

  disconnectProgressStream() {
    if (this.progressPollTimer) {
      window.clearInterval(this.progressPollTimer);
      this.progressPollTimer = null;
    }
    if (this.progressSocket) {
      this.progressSocket.close();
      this.progressSocket = null;
    }
  }

  dispose() {
    this.disconnectProgressStream();
    this.listeners = {};
    this.logger.info('TrainingService disposed');
  }
}

// Create singleton instance
export const trainingService = new TrainingService();
