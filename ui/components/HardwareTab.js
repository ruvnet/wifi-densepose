// Hardware Tab Component

export class HardwareTab {
  constructor(containerElement) {
    this.container = containerElement;
    this.antennas = [];
    this.csiUpdateInterval = null;
    this.isActive = false;
  }

  // Initialize component
  init() {
    this.setupAntennas();
    this.startCSIStatus();
  }

  // Set up antenna interactions
  setupAntennas() {
    this.antennas = Array.from(this.container.querySelectorAll('.antenna'));
    
    this.antennas.forEach(antenna => {
      antenna.addEventListener('click', () => {
        antenna.classList.toggle('active');
        this.updateCSIDisplay();
      });
    });
  }

  // Start CSI status display. Values remain empty until live hardware supplies data.
  startCSIStatus() {
    // Initial update
    this.updateCSIDisplay();
  }

  // Check if any antennas are active
  hasActiveAntennas() {
    return this.antennas.some(antenna => antenna.classList.contains('active'));
  }

  // Update CSI display
  updateCSIDisplay() {
    const activeAntennas = this.antennas.filter(a => a.classList.contains('active'));
    const isActive = activeAntennas.length > 0;
    
    // Get display elements
    const amplitudeFill = this.container.querySelector('.csi-fill.amplitude');
    const phaseFill = this.container.querySelector('.csi-fill.phase');
    const amplitudeValue = this.container.querySelector('.csi-row:first-child .csi-value');
    const phaseValue = this.container.querySelector('.csi-row:last-child .csi-value');
    
    if (!isActive) {
      // Set to zero when no antennas active
      if (amplitudeFill) amplitudeFill.style.width = '0%';
      if (phaseFill) phaseFill.style.width = '0%';
      if (amplitudeValue) amplitudeValue.textContent = '0.00';
      if (phaseValue) phaseValue.textContent = '0.0π';
      return;
    }
    
    if (amplitudeFill) amplitudeFill.style.width = '0%';
    if (phaseFill) phaseFill.style.width = '0%';
    if (amplitudeValue) amplitudeValue.textContent = '--';
    if (phaseValue) phaseValue.textContent = '--';
    
    // Update antenna array visualization
    this.updateAntennaArray(activeAntennas);
  }

  // Update antenna array visualization
  updateAntennaArray(activeAntennas) {
    const arrayStatus = this.container.querySelector('.array-status');
    if (!arrayStatus) return;
    
    const txActive = activeAntennas.filter(a => a.classList.contains('tx')).length;
    const rxActive = activeAntennas.filter(a => a.classList.contains('rx')).length;
    
    // Clear and rebuild using safe DOM methods to prevent XSS
    arrayStatus.innerHTML = '';
    
    const createInfoDiv = (label, value) => {
      const div = document.createElement('div');
      div.className = 'array-info';
      
      const labelSpan = document.createElement('span');
      labelSpan.className = 'info-label';
      labelSpan.textContent = label;
      
      const valueSpan = document.createElement('span');
      valueSpan.className = 'info-value';
      valueSpan.textContent = value;
      
      div.appendChild(labelSpan);
      div.appendChild(valueSpan);
      return div;
    };
    
    arrayStatus.appendChild(createInfoDiv('Active TX:', `${txActive}/3`));
    arrayStatus.appendChild(createInfoDiv('Active RX:', `${rxActive}/6`));
    arrayStatus.appendChild(createInfoDiv('Signal Quality:', 'Live data required'));
  }

  // Calculate signal quality from live CSI only.
  calculateSignalQuality(txCount, rxCount) {
    return 0;
  }

  // Toggle all antennas
  toggleAllAntennas(active) {
    this.antennas.forEach(antenna => {
      antenna.classList.toggle('active', active);
    });
    this.updateCSIDisplay();
  }

  // Reset antenna configuration
  resetAntennas() {
    // Set default configuration (all active)
    this.antennas.forEach(antenna => {
      antenna.classList.add('active');
    });
    this.updateCSIDisplay();
  }

  // Clean up
  dispose() {
    if (this.csiUpdateInterval) {
      clearInterval(this.csiUpdateInterval);
      this.csiUpdateInterval = null;
    }
    
    this.antennas.forEach(antenna => {
      antenna.removeEventListener('click', this.toggleAntenna);
    });
  }
}
