// WiFi DensePose Application - Main Entry Point

import { TabManager } from './components/TabManager.js';
import { DashboardTab } from './components/DashboardTab.js';
import { HardwareTab } from './components/HardwareTab.js';
import { LiveDemoTab } from './components/LiveDemoTab.js';
import { apiService } from './services/api.service.js';
import { wsService } from './services/websocket.service.js';
import { healthService } from './services/health.service.js';

class WiFiDensePoseApp {
  constructor() {
    this.components = {};
    this.isInitialized = false;
  }

  // Initialize application
  async init() {
    try {
      console.log('Initializing WiFi DensePose UI...');
      
      // Set up error handling
      this.setupErrorHandling();
      
      // Initialize services
      await this.initializeServices();
      
      // Initialize UI components
      this.initializeComponents();
      
      // Set up global event listeners
      this.setupEventListeners();
      
      this.isInitialized = true;
      console.log('WiFi DensePose UI initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize application:', error);
      this.showGlobalError('Failed to initialize application. Please refresh the page.');
    }
  }

  // Initialize services
  async initializeServices() {
    // Add request interceptor for error handling
    apiService.addResponseInterceptor(async (response, url) => {
      if (!response.ok && response.status === 401) {
        console.warn('Authentication required for:', url);
        // Handle authentication if needed
      }
      return response;
    });

    // Check API availability
    try {
      const health = await healthService.checkLiveness();
      console.log('API is available:', health);
    } catch (error) {
      console.error('API is not available:', error);
      throw new Error('API is not available. Please ensure the backend is running.');
    }
  }

  // Initialize UI components
  initializeComponents() {
    const container = document.querySelector('.container');
    if (!container) {
      throw new Error('Main container not found');
    }

    // Initialize tab manager
    this.components.tabManager = new TabManager(container);
    this.components.tabManager.init();

    // Initialize tab components
    this.initializeTabComponents();

    // Set up tab change handling
    this.components.tabManager.onTabChange((newTab, oldTab) => {
      this.handleTabChange(newTab, oldTab);
    });
  }

  // Initialize individual tab components
  initializeTabComponents() {
    // Dashboard tab
    const dashboardContainer = document.getElementById('dashboard');
    if (dashboardContainer) {
      this.components.dashboard = new DashboardTab(dashboardContainer);
      this.components.dashboard.init().catch(error => {
        console.error('Failed to initialize dashboard:', error);
      });
    }

    // Hardware tab
    const hardwareContainer = document.getElementById('hardware');
    if (hardwareContainer) {
      this.components.hardware = new HardwareTab(hardwareContainer);
      this.components.hardware.init();
    }

    // Live demo tab
    const demoContainer = document.getElementById('demo');
    if (demoContainer) {
      this.components.demo = new LiveDemoTab(demoContainer);
      this.components.demo.init();
    }

    // Architecture tab - static content, no component needed
    
    // Performance tab - static content, no component needed
    
    // Applications tab - static content, no component needed
  }

  // Handle tab changes
  handleTabChange(newTab, oldTab) {
    console.log(`Tab changed from ${oldTab} to ${newTab}`);
    
    // Stop demo if leaving demo tab
    if (oldTab === 'demo' && this.components.demo) {
      this.components.demo.stopDemo();
    }
    
    // Update components based on active tab
    switch (newTab) {
      case 'dashboard':
        // Dashboard auto-updates when visible
        break;
        
      case 'hardware':
        // Hardware visualization is always active
        break;
        
      case 'demo':
        // Demo starts manually
        break;
    }
  }

  // Set up global event listeners
  setupEventListeners() {
    // Handle window resize
    window.addEventListener('resize', () => {
      this.handleResize();
    });

    // Handle visibility change
    document.addEventListener('visibilitychange', () => {
      this.handleVisibilityChange();
    });

    // Handle before unload
    window.addEventListener('beforeunload', () => {
      this.cleanup();
    });
  }

  // Handle window resize
  handleResize() {
    // Update canvas sizes if needed
    const canvases = document.querySelectorAll('canvas');
    canvases.forEach(canvas => {
      const rect = canvas.parentElement.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    });
  }

  // Handle visibility change
  handleVisibilityChange() {
    if (document.hidden) {
      // Pause updates when page is hidden
      console.log('Page hidden, pausing updates');
      healthService.stopHealthMonitoring();
    } else {
      // Resume updates when page is visible
      console.log('Page visible, resuming updates');
      healthService.startHealthMonitoring();
    }
  }

  // Set up error handling
  setupErrorHandling() {
    window.addEventListener('error', (event) => {
      console.error('Global error:', event.error);
      this.showGlobalError('An unexpected error occurred');
    });

    window.addEventListener('unhandledrejection', (event) => {
      console.error('Unhandled promise rejection:', event.reason);
      this.showGlobalError('An unexpected error occurred');
    });
  }

  // Show global error message
  showGlobalError(message) {
    // Create error toast if it doesn't exist
    let errorToast = document.getElementById('globalErrorToast');
    if (!errorToast) {
      errorToast = document.createElement('div');
      errorToast.id = 'globalErrorToast';
      errorToast.className = 'error-toast';
      document.body.appendChild(errorToast);
    }

    errorToast.textContent = message;
    errorToast.classList.add('show');

    setTimeout(() => {
      errorToast.classList.remove('show');
    }, 5000);
  }

  // Clean up resources
  cleanup() {
    console.log('Cleaning up application resources...');
    
    // Dispose all components
    Object.values(this.components).forEach(component => {
      if (component && typeof component.dispose === 'function') {
        component.dispose();
      }
    });

    // Disconnect all WebSocket connections
    wsService.disconnectAll();
    
    // Stop health monitoring
    healthService.dispose();
  }

  // Public API
  getComponent(name) {
    return this.components[name];
  }

  isReady() {
    return this.isInitialized;
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.wifiDensePoseApp = new WiFiDensePoseApp();
  window.wifiDensePoseApp.init();
});

// Export for testing
export { WiFiDensePoseApp };