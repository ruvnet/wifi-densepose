export default {
  name: 'RuView NLOS Beta',
  slug: 'wifi-densepose',
  version: '1.0.0',
  description: 'Governed RuView NLOS beta viewer for synthetic replay and authenticated track evidence.',
  orientation: 'portrait',
  userInterfaceStyle: 'dark',
  icon: './assets/icon.png',
  backgroundColor: '#0A0E1A',
  primaryColor: '#32B8C6',
  ios: {
    bundleIdentifier: 'com.ruvnet.wifidensepose',
    supportsTablet: true,
  },
  android: {
    package: 'com.ruvnet.wifidensepose',
  },
  web: {
    favicon: './assets/favicon.png',
    name: 'RuView NLOS Beta',
    shortName: 'RuView NLOS',
    lang: 'en',
    themeColor: '#0A0E1A',
    backgroundColor: '#0A0E1A',
    display: 'standalone',
  },
};
