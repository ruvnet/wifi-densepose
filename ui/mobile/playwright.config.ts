import { defineConfig } from '@playwright/test';

const staticBundleMode = Boolean(process.env.RUVIEW_E2E_STATIC_DIR);

export default defineConfig({
  testDir: './e2e/web',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  timeout: 30_000,
  expect: { timeout: 8_000 },
  outputDir: 'test-results/playwright',
  use: {
    baseURL: staticBundleMode ? 'http://ruview.test' : 'http://127.0.0.1:4173',
    viewport: { width: 390, height: 844 },
    colorScheme: 'dark',
    locale: 'en-CA',
    contextOptions: { reducedMotion: 'reduce' },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? {
          executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--use-gl=angle',
            '--use-angle=swiftshader-webgl',
            '--enable-unsafe-swiftshader',
          ],
        }
      : undefined,
  },
  webServer: staticBundleMode
    ? undefined
    : {
        command: 'npm run build:web:e2e && npm run serve:web:e2e',
        url: 'http://127.0.0.1:4173',
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
