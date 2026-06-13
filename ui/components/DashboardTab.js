// Dashboard Tab Component

import { healthService } from '../services/health.service.js';
import { poseService } from '../services/pose.service.js';
import { sensingService } from '../services/sensing.service.js';

export function parseFocusNodeIdFromSearch(search = '') {
  const params = new URLSearchParams(search);
  const raw = params.get('node') || params.get('node_id') || '';
  const normalized = raw.toLowerCase().replace(/^node-?/, '');
  const nodeId = Number.parseInt(normalized, 10);
  return Number.isFinite(nodeId) && nodeId > 0 ? nodeId : null;
}

export class DashboardTab {
  constructor(containerElement) {
    this.container = containerElement;
    this.statsElements = {};
    this.healthSubscription = null;
    this.statsInterval = null;
    this.nodeStatusInterval = null;
    this.focusNodeId = parseFocusNodeIdFromSearch(window.location.search);
  }

  // Initialize component
  async init() {
    this.cacheElements();
    await this.loadInitialData();
    this.startMonitoring();
  }

  // Cache DOM elements
  cacheElements() {
    // System stats
    const statsContainer = this.container.querySelector('.system-stats');
    if (statsContainer) {
      this.statsElements = {
        bodyRegions: statsContainer.querySelector('[data-stat="body-regions"] .stat-value'),
        samplingRate: statsContainer.querySelector('[data-stat="sampling-rate"] .stat-value'),
        accuracy: statsContainer.querySelector('[data-stat="accuracy"] .stat-value'),
        hardwareCost: statsContainer.querySelector('[data-stat="hardware-cost"] .stat-value')
      };
    }

    this.nodeSummary = this.container.querySelector('#node-summary');
    this.nodeStatusGrid = this.container.querySelector('#node-status-grid');
  }

  // Load initial data
  async loadInitialData() {
    try {
      // Get API info
      const info = await healthService.getApiInfo();
      this.updateApiInfo(info);

      // Get current stats
      const stats = await poseService.getStats(1);
      this.updateStats(stats);

    } catch (error) {
      // DensePose API may not be running (sensing-only mode) — fail silently
      console.log('Dashboard: DensePose API not available (sensing-only mode)');
    }
  }

  // Start monitoring
  startMonitoring() {
    // Subscribe to health updates
    this.healthSubscription = healthService.subscribeToHealth(health => {
      this.updateHealthStatus(health);
    });

    // Subscribe to sensing service state changes for data source indicator
    this._sensingUnsub = sensingService.onStateChange(() => {
      this.updateDataSourceIndicator();
    });
    // Also update on data — catches source changes mid-stream
    this._sensingDataUnsub = sensingService.onData(() => {
      this.updateDataSourceIndicator();
    });
    // Initial update
    this.updateDataSourceIndicator();
    this.updateNodeStatus();

    // Start periodic stats updates
    this.statsInterval = setInterval(() => {
      this.updateLiveStats();
    }, 5000);
    this.nodeStatusInterval = setInterval(() => {
      this.updateNodeStatus();
    }, 1000);

    // Start health monitoring
    healthService.startHealthMonitoring(30000);
  }

  // Update the data source indicator on the dashboard
  updateDataSourceIndicator() {
    const el = this.container.querySelector('#dashboard-datasource');
    if (!el) return;
    const ds = sensingService.dataSource;
    const statusText = el.querySelector('.status-text');
    const statusMsg  = el.querySelector('.status-message');
    const config = {
      'live':              { text: 'ESP32',     status: 'healthy', msg: 'Real hardware connected' },
      'stale':             { text: 'STALE',     status: 'degraded', msg: 'Waiting for fresh feature state' },
      'reconnecting':      { text: 'RECONNECTING', status: 'degraded', msg: 'Attempting to connect...' },
    };
    const cfg = config[ds] || config['reconnecting'];
    el.className = `component-status status-${cfg.status}`;
    if (statusText) statusText.textContent = cfg.text;
    if (statusMsg)  statusMsg.textContent = cfg.msg;
  }

  async updateNodeStatus() {
    if (!this.nodeStatusGrid || !this.nodeSummary) return;

    try {
      const response = await fetch('/api/v1/cardputer/status', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`status ${response.status}`);
      }
      const status = await response.json();
      this.renderNodeStatus(status);
    } catch (error) {
      this.nodeSummary.textContent = 'Status unavailable';
      this.nodeStatusGrid.replaceChildren(
        this.createNodeEmptyState(`Cardputer status endpoint unavailable: ${error.message}`)
      );
    }
  }

  renderNodeStatus(status) {
    const nodes = Array.isArray(status.nodes) ? status.nodes : [];
    const liveCount = Number.isFinite(status.live_node_count)
      ? status.live_node_count
      : nodes.filter(node => node.live).length;
    const totalCount = Number.isFinite(status.node_count) ? status.node_count : nodes.length;

    const focusNode = this.focusNodeId
      ? nodes.find(node => Number(node.node_id) === this.focusNodeId)
      : null;
    this.nodeSummary.textContent = totalCount > 0
      ? this.formatNodeSummary(liveCount, totalCount, focusNode)
      : 'Waiting for packets';

    if (nodes.length === 0) {
      this.nodeStatusGrid.replaceChildren(this.createNodeEmptyState('No node packets received'));
      return;
    }

    const cards = [...nodes]
      .sort((a, b) => this.compareNodes(a, b))
      .map(node => this.createNodeStatusCard(node));
    this.nodeStatusGrid.replaceChildren(...cards);
  }

  formatNodeSummary(liveCount, totalCount, focusNode) {
    if (!this.focusNodeId) return `${liveCount}/${totalCount} live`;
    if (!focusNode) return `Node ${this.focusNodeId} waiting - ${liveCount}/${totalCount} live`;
    return `Node ${this.focusNodeId} ${focusNode.live ? 'live' : 'stale'} - ${liveCount}/${totalCount} live`;
  }

  compareNodes(a, b) {
    const aId = Number(a.node_id);
    const bId = Number(b.node_id);
    if (this.focusNodeId) {
      if (aId === this.focusNodeId && bId !== this.focusNodeId) return -1;
      if (bId === this.focusNodeId && aId !== this.focusNodeId) return 1;
    }
    return aId - bId;
  }

  createNodeStatusCard(node) {
    const card = document.createElement('article');
    card.className = `node-status-card ${node.live ? 'node-live' : 'node-stale'}`;
    if (Number(node.node_id) === this.focusNodeId) {
      card.classList.add('node-focused');
    }

    const header = document.createElement('div');
    header.className = 'node-status-header';
    const title = document.createElement('h4');
    title.textContent = `Node ${node.node_id}`;
    const pill = document.createElement('span');
    pill.className = 'node-status-pill';
    pill.textContent = node.live ? 'LIVE' : 'STALE';
    header.append(title, pill);

    const fields = document.createElement('dl');
    fields.className = 'node-status-fields';
    [
      ['Source', node.last_source || '-'],
      ['Packets', this.formatNumber(node.packet_count || 0)],
      ['Age', this.formatAge(node.last_packet_age_s)],
      ['Type', node.last_packet_type || '-'],
      ['Presence', this.formatMetric(node.feature_state?.presence_score)],
      ['Motion', this.formatMetric(node.feature_state?.motion_score)],
      ['RSSI', this.formatRssi(node.rssi_dbm)],
      ['Battery', this.formatBattery(node.battery)],
      ['Seen', this.formatPacketTypes(node.packet_types)]
    ].forEach(([label, value]) => {
      const term = document.createElement('dt');
      term.textContent = label;
      const detail = document.createElement('dd');
      detail.textContent = value;
      fields.append(term, detail);
    });

    card.append(header, fields);
    return card;
  }

  createNodeEmptyState(message) {
    const empty = document.createElement('div');
    empty.className = 'node-status-empty';
    empty.textContent = message;
    return empty;
  }

  formatAge(value) {
    if (!Number.isFinite(value)) return '-';
    if (value < 1) return `${Math.round(value * 1000)}ms`;
    return `${value.toFixed(1)}s`;
  }

  formatMetric(value, digits = 2) {
    if (!Number.isFinite(value)) return '-';
    return Number(value).toFixed(digits);
  }

  formatRssi(value) {
    if (!Number.isFinite(value)) return 'not reported';
    return `${Math.round(value)} dBm`;
  }

  formatBattery(battery) {
    if (!battery?.valid) return 'not reported';
    const charge = battery.charging ? ' charging' : '';
    return `${battery.percent}%${charge}`;
  }

  formatPacketTypes(packetTypes) {
    if (!packetTypes || Object.keys(packetTypes).length === 0) return '-';
    return Object.entries(packetTypes)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([type, count]) => `${type}:${count}`)
      .join(' ');
  }

  // Update API info display
  updateApiInfo(info) {
    // Update version
    const versionElement = this.container.querySelector('.api-version');
    if (versionElement && info.version) {
      versionElement.textContent = `v${info.version}`;
    }

    // Update environment
    const envElement = this.container.querySelector('.api-environment');
    if (envElement && info.environment) {
      envElement.textContent = info.environment;
      envElement.className = `api-environment env-${info.environment}`;
    }

    // Update features status
    if (info.features) {
      this.updateFeatures(info.features);
    }
  }

  // Update features display
  updateFeatures(features) {
    const featuresContainer = this.container.querySelector('.features-status');
    if (!featuresContainer) return;

    featuresContainer.innerHTML = '';
    
    Object.entries(features).forEach(([feature, enabled]) => {
      const featureElement = document.createElement('div');
      featureElement.className = `feature-item ${enabled ? 'enabled' : 'disabled'}`;
      
      // Use textContent instead of innerHTML to prevent XSS
      const featureNameSpan = document.createElement('span');
      featureNameSpan.className = 'feature-name';
      featureNameSpan.textContent = this.formatFeatureName(feature);
      
      const featureStatusSpan = document.createElement('span');
      featureStatusSpan.className = 'feature-status';
      featureStatusSpan.textContent = enabled ? '✓' : '✗';
      
      featureElement.appendChild(featureNameSpan);
      featureElement.appendChild(featureStatusSpan);
      featuresContainer.appendChild(featureElement);
    });
  }

  // Update health status
  updateHealthStatus(health) {
    if (!health) return;

    // Update overall status
    const overallStatus = this.container.querySelector('.overall-health');
    if (overallStatus) {
      overallStatus.className = `overall-health status-${health.status}`;
      overallStatus.textContent = health.status.toUpperCase();
    }

    // Update component statuses
    if (health.components) {
      Object.entries(health.components).forEach(([component, status]) => {
        this.updateComponentStatus(component, status);
      });
    }

    // Update metrics
    if (health.metrics) {
      this.updateSystemMetrics(health.metrics);
    }
  }

  // Update component status
  updateComponentStatus(component, status) {
    // Map backend component names to UI component names
    const componentMap = {
      'pose': 'inference',
      'stream': 'streaming',
      'hardware': 'hardware'
    };
    
    const uiComponent = componentMap[component] || component;
    const element = this.container.querySelector(`[data-component="${uiComponent}"]`);
    
    if (element) {
      element.className = `component-status status-${status.status}`;
      const statusText = element.querySelector('.status-text');
      const statusMessage = element.querySelector('.status-message');
      
      if (statusText) {
        statusText.textContent = status.status.toUpperCase();
      }
      
      if (statusMessage && status.message) {
        statusMessage.textContent = status.message;
      }
    }
    
    // Also update API status based on overall health
    if (component === 'hardware') {
      const apiElement = this.container.querySelector(`[data-component="api"]`);
      if (apiElement) {
        apiElement.className = `component-status status-healthy`;
        const apiStatusText = apiElement.querySelector('.status-text');
        const apiStatusMessage = apiElement.querySelector('.status-message');
        
        if (apiStatusText) {
          apiStatusText.textContent = 'HEALTHY';
        }
        
        if (apiStatusMessage) {
          apiStatusMessage.textContent = 'API server is running normally';
        }
      }
    }
  }

  // Update system metrics
  updateSystemMetrics(metrics) {
    // Handle both flat and nested metric structures.
    const systemMetrics = metrics.system_metrics || metrics;
    const cpuPercent = systemMetrics.cpu?.percent || systemMetrics.cpu_percent;
    const memoryPercent = systemMetrics.memory?.percent || systemMetrics.memory_percent;
    const diskPercent = systemMetrics.disk?.percent || systemMetrics.disk_percent;

    // CPU usage
    const cpuElement = this.container.querySelector('.cpu-usage');
    if (cpuElement && cpuPercent !== undefined) {
      cpuElement.textContent = `${cpuPercent.toFixed(1)}%`;
      this.updateProgressBar('cpu', cpuPercent);
    }

    // Memory usage
    const memoryElement = this.container.querySelector('.memory-usage');
    if (memoryElement && memoryPercent !== undefined) {
      memoryElement.textContent = `${memoryPercent.toFixed(1)}%`;
      this.updateProgressBar('memory', memoryPercent);
    }

    // Disk usage
    const diskElement = this.container.querySelector('.disk-usage');
    if (diskElement && diskPercent !== undefined) {
      diskElement.textContent = `${diskPercent.toFixed(1)}%`;
      this.updateProgressBar('disk', diskPercent);
    }
  }

  // Update progress bar
  updateProgressBar(type, percent) {
    const progressBar = this.container.querySelector(`.progress-bar[data-type="${type}"]`);
    if (progressBar) {
      const fill = progressBar.querySelector('.progress-fill');
      if (fill) {
        fill.style.width = `${percent}%`;
        fill.className = `progress-fill ${this.getProgressClass(percent)}`;
      }
    }
  }

  // Get progress class based on percentage
  getProgressClass(percent) {
    if (percent >= 90) return 'critical';
    if (percent >= 75) return 'warning';
    return 'normal';
  }

  // Update live statistics
  async updateLiveStats() {
    try {
      // Get current pose data
      const currentPose = await poseService.getCurrentPose();
      this.updatePoseStats(currentPose);

      // Get zones summary
      const zonesSummary = await poseService.getZonesSummary();
      this.updateZonesDisplay(zonesSummary);

    } catch (error) {
      console.error('Failed to update live stats:', error);
    }
  }

  // Update pose statistics
  updatePoseStats(poseData) {
    if (!poseData) return;

    // Update person count
    const personCount = this.container.querySelector('.person-count');
    if (personCount) {
      const count = poseData.persons ? poseData.persons.length : (poseData.total_persons || 0);
      personCount.textContent = count;
    }

    // Update average confidence
    const avgConfidence = this.container.querySelector('.avg-confidence');
    if (avgConfidence && poseData.persons && poseData.persons.length > 0) {
      const confidences = poseData.persons.map(p => p.confidence);
      const avg = confidences.length > 0
        ? (confidences.reduce((a, b) => a + b, 0) / confidences.length * 100).toFixed(1)
        : 0;
      avgConfidence.textContent = `${avg}%`;
    } else if (avgConfidence) {
      avgConfidence.textContent = '0%';
    }

    // Update total detections from stats if available
    const detectionCount = this.container.querySelector('.detection-count');
    if (detectionCount && poseData.total_detections !== undefined) {
      detectionCount.textContent = this.formatNumber(poseData.total_detections);
    }
  }

  // Update zones display
  updateZonesDisplay(zonesSummary) {
    const zonesContainer = this.container.querySelector('.zones-summary');
    if (!zonesContainer) return;

    zonesContainer.innerHTML = '';
    
    // Handle different zone summary formats
    let zones = {};
    if (zonesSummary && zonesSummary.zones) {
      zones = zonesSummary.zones;
    } else if (zonesSummary && typeof zonesSummary === 'object') {
      zones = zonesSummary;
    }
    
    // If no zones data, show default zones
    if (Object.keys(zones).length === 0) {
      ['zone_1', 'zone_2', 'zone_3', 'zone_4'].forEach(zoneId => {
        const zoneElement = document.createElement('div');
        zoneElement.className = 'zone-item';
        
        // Use textContent instead of innerHTML to prevent XSS
        const zoneNameSpan = document.createElement('span');
        zoneNameSpan.className = 'zone-name';
        zoneNameSpan.textContent = zoneId;
        
        const zoneCountSpan = document.createElement('span');
        zoneCountSpan.className = 'zone-count';
        zoneCountSpan.textContent = 'undefined';
        
        zoneElement.appendChild(zoneNameSpan);
        zoneElement.appendChild(zoneCountSpan);
        zonesContainer.appendChild(zoneElement);
      });
      return;
    }
    
    Object.entries(zones).forEach(([zoneId, data]) => {
      const zoneElement = document.createElement('div');
      zoneElement.className = 'zone-item';
      const count = typeof data === 'object' ? (data.person_count || data.count || 0) : data;
      
      // Use textContent instead of innerHTML to prevent XSS
      const zoneNameSpan = document.createElement('span');
      zoneNameSpan.className = 'zone-name';
      zoneNameSpan.textContent = zoneId;
      
      const zoneCountSpan = document.createElement('span');
      zoneCountSpan.className = 'zone-count';
      zoneCountSpan.textContent = String(count);
      
      zoneElement.appendChild(zoneNameSpan);
      zoneElement.appendChild(zoneCountSpan);
      zonesContainer.appendChild(zoneElement);
    });
  }

  // Update statistics
  updateStats(stats) {
    if (!stats) return;

    // Update detection count
    const detectionCount = this.container.querySelector('.detection-count');
    if (detectionCount && stats.total_detections !== undefined) {
      detectionCount.textContent = this.formatNumber(stats.total_detections);
    }

    // Update accuracy if available
    if (this.statsElements.accuracy && stats.average_confidence !== undefined) {
      this.statsElements.accuracy.textContent = `${(stats.average_confidence * 100).toFixed(1)}%`;
    }
  }

  // Format feature name
  formatFeatureName(name) {
    return name.replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // Format large numbers
  formatNumber(num) {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  }

  // Show error message
  showError(message) {
    const errorContainer = this.container.querySelector('.error-container');
    if (errorContainer) {
      errorContainer.textContent = message;
      errorContainer.style.display = 'block';
      
      setTimeout(() => {
        errorContainer.style.display = 'none';
      }, 5000);
    }
  }

  // Clean up
  dispose() {
    if (this.healthSubscription) {
      this.healthSubscription();
    }
    if (this._sensingUnsub) this._sensingUnsub();
    if (this._sensingDataUnsub) this._sensingDataUnsub();

    if (this.statsInterval) {
      clearInterval(this.statsInterval);
    }
    if (this.nodeStatusInterval) {
      clearInterval(this.nodeStatusInterval);
    }

    healthService.stopHealthMonitoring();
  }
}
