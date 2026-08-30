import { expect, test, type Page } from '@playwright/test';
import { mkdir, readFile, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';

const screenshotDirectory = resolve(
  process.cwd(),
  '../../docs/screenshots/consumer-nlos-mobile-ui',
);
const staticBundleDirectory = process.env.RUVIEW_E2E_STATIC_DIR
  ? resolve(process.cwd(), process.env.RUVIEW_E2E_STATIC_DIR)
  : null;

const contentTypes: Record<string, string> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
};

test.beforeEach(async ({ page }) => {
  if (!staticBundleDirectory) return;
  await page.route('http://ruview.test/**', async (route) => {
    const pathname = decodeURIComponent(new URL(route.request().url()).pathname);
    let localPath = resolve(staticBundleDirectory, `.${pathname}`);
    if (!localPath.startsWith(`${staticBundleDirectory}${sep}`) && localPath !== staticBundleDirectory) {
      await route.fulfill({ status: 404, body: 'Not found' });
      return;
    }
    try {
      if ((await stat(localPath)).isDirectory()) localPath = resolve(localPath, 'index.html');
    } catch {
      localPath = resolve(staticBundleDirectory, 'index.html');
    }
    await route.fulfill({
      status: 200,
      contentType: contentTypes[extname(localPath)] ?? 'application/octet-stream',
      body: await readFile(localPath),
    });
  });
});

const openNlos = async (page: Page) => {
  await page.goto('/');
  for (const tab of ['Live', 'Calibration', 'Vitals', 'Zones', 'MAT', 'Settings']) {
    await expect(page.getByText(tab, { exact: true }).last()).toBeVisible();
  }
  await page.getByText('Calibration', { exact: true }).last().click();
  await expect(page.getByText('RuView Calibration', { exact: true })).toBeVisible();
  await expect(page.getByTestId('calibration-guided-menu')).toBeVisible();
  await expect(page.getByText('Connect a verified source', { exact: true })).toBeVisible();
};

const capture = async (page: Page, name: string) => {
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({
    path: resolve(screenshotDirectory, name),
    animations: 'disabled',
    caret: 'hide',
  });
};

test.describe('RuView welcome navigation', () => {
  test('opens first and both header controls return to it', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.goto('/');
    await expect(page.getByTestId('welcome-screen')).toBeVisible();
    await expect(page.getByText('Understand the room.', { exact: false })).toBeVisible();
    const sweep = page.getByTestId('header-radar-sweep').last();
    await expect(sweep).toBeVisible();
    const firstTransform = await sweep.evaluate((element) => getComputedStyle(element).transform);
    await page.waitForTimeout(120);
    await expect.poll(() => sweep.evaluate((element) => getComputedStyle(element).transform)).not.toBe(firstTransform);

    await page.getByTestId('welcome-open-live').click();
    await expect(page.getByTestId('live-screen')).toBeVisible();
    await page.getByTestId('header-home-logo').last().click();
    await expect(page.getByTestId('welcome-screen')).toBeVisible();
    await expect.poll(() => page.getByTestId('welcome-screen').evaluate((element) => element.scrollTop)).toBe(0);

    await page.getByText('Calibration', { exact: true }).last().click();
    await expect(page.getByTestId('calibration-guided-menu')).toBeVisible();
    await page.getByTestId('header-home-nav').last().click();
    await expect(page.getByTestId('welcome-screen')).toBeVisible();
  });
});

test.describe('RuView Calibration mobile instrument UI', () => {
  test('guides the operator through one focused task at a time', async ({ page }) => {
    await openNlos(page);
    await expect(page.getByText('Connect a verified source', { exact: true })).toBeVisible();
    await expect(page.getByText('Measure and align the room', { exact: true })).toHaveCount(0);

    await page.getByRole('button', { name: 'CONTINUE TO ROOM' }).click();
    await expect(page.getByText('Measure and align the room', { exact: true })).toBeVisible();
    await expect(page.getByText('RETURN TO STEP 01 AND CONNECT LIVE RF', { exact: true })).toBeVisible();

    await page.getByTestId('calibration-step-pose').click();
    await expect(page.getByText('Teach coarse pose — optional', { exact: true })).toBeVisible();
    await page.getByTestId('calibration-step-review').click();
    await expect(page.getByText('Review evidence before promotion', { exact: true })).toBeVisible();
    await expect(page.getByTestId('nlos-provenance-panel')).toBeVisible();
  });

  test('captures the disconnected overview without horizontal overflow', async ({ page }) => {
    await openNlos(page);
    await page.evaluate(() => window.scrollTo(0, 0));

    const dimensions = await page.evaluate(() => ({
      viewport: window.innerWidth,
      content: document.documentElement.scrollWidth,
    }));
    const nestedDimensions = await page.getByTestId('nlos-scroll-view').evaluate((element) => ({
      viewport: element.clientWidth,
      content: element.scrollWidth,
    }));
    expect(dimensions.viewport).toBe(390);
    expect(dimensions.content).toBeLessThanOrEqual(390);
    expect(nestedDimensions.content).toBeLessThanOrEqual(nestedDimensions.viewport + 1);
    await expect(page.getByText('Connect a verified source', { exact: true })).toBeVisible();

    await capture(page, 'overview-390x844.png');
  });

  test('captures governed setup with fixed explainer and feedback controls', async ({ page }) => {
    await openNlos(page);
    await page.getByRole('button', { name: 'Show setup and safety help' }).click();
    const setup = page.getByTestId('nlos-beta-setup');
    await setup.scrollIntoViewIfNeeded();
    await expect(setup).toBeVisible();
    await expect(page.getByRole('link', { name: 'OPEN EXPLAINER' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'TEST STEPS AND FEEDBACK' })).toBeVisible();

    await capture(page, 'setup-390x844.png');
  });

  test('starts synthetic replay and preserves its visible watermark', async ({ page }) => {
    await openNlos(page);
    const replay = page.getByRole('button', { name: 'USE CALIBRATION REPLAY' });
    await replay.scrollIntoViewIfNeeded();
    await replay.click();
    await page.getByTestId('calibration-step-review').click();

    await expect(page.getByTestId('nlos-synthetic-watermark')).toBeVisible();
    await expect(page.getByTestId('nlos-provenance-badge')).toContainText('SYNTHETIC');
    await expect(page.getByTestId('nlos-evidence-state')).toContainText('SYNTHETIC');
    await page.getByTestId('nlos-provenance-panel').evaluate((element) => {
      element.scrollIntoView({ behavior: 'auto', block: 'start' });
    });

    await capture(page, 'synthetic-390x844.png');
  });

  test('renders a gated Three.js LiDAR point cloud without horizontal overflow', async ({ page }) => {
    await openNlos(page);
    const replay = page.getByRole('button', { name: 'USE CALIBRATION REPLAY' });
    await replay.scrollIntoViewIfNeeded();
    await replay.click();
    await page.getByTestId('calibration-step-review').click();

    await page.getByTestId('nlos-view-cloud').click();
    const cloud = page.getByTestId('nlos-lidar-point-cloud');
    await cloud.scrollIntoViewIfNeeded();
    await expect(cloud).toBeVisible();
    await expect(page.getByTestId('nlos-lidar-point-cloud-canvas')).toHaveAttribute('data-ready', 'true');
    await expect(page.getByTestId('nlos-cloud-target-count')).toHaveText('96');
    await expect(page.getByTestId('nlos-cloud-boundary-label')).toHaveText('RECONSTRUCTION / NOT RAW SCAN');
    await expect(page.getByText('THREE.JS / WEBGL')).toBeVisible();
    await expect(page.getByText('01 TRACK LOCK')).toBeVisible();
    await expect(page.getByText('72% CONFIDENCE')).toBeVisible();
    await expect(page.getByTestId('nlos-synthetic-watermark')).toBeVisible();

    const dimensions = await cloud.evaluate((element) => ({
      viewport: element.clientWidth,
      content: element.scrollWidth,
    }));
    expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport + 1);

    await capture(page, 'point-cloud-390x844.png');
  });
});

test.describe('RuView Vitals mobile instrument UI', () => {
  test('renders a fail-closed live dashboard and the real Apple Home boundary', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Vitals', { exact: true }).last().click();

    await expect(page.getByText('The room has a pulse.', { exact: true })).toBeVisible();
    await expect(page.getByText(/^(MEASURED \/ FRESH|NO FRESH EVIDENCE|SIMULATION HIDDEN)$/)).toBeVisible();
    const appleHomePanel = page.getByText('Apple Home / local HAP bridge', { exact: true });
    await appleHomePanel.scrollIntoViewIfNeeded();
    await expect(appleHomePanel).toBeVisible();
    await expect(page.getByText(/Breathing, heart-rate proxy, pose, raw CSI, and identity scores never cross this boundary/)).toBeVisible();
    await expect(page.getByText('Install the native iOS build to perform real Bonjour `_hap._tcp` discovery.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'DISCOVER HAP BRIDGE' })).toBeDisabled();

    const dimensions = await page.evaluate(() => ({
      viewport: window.innerWidth,
      content: document.documentElement.scrollWidth,
    }));
    expect(dimensions.viewport).toBe(390);
    expect(dimensions.content).toBeLessThanOrEqual(390);

    await capture(page, 'vitals-390x844.png');
  });
});

test.describe('RuView MAT governed incident UI', () => {
  test('renders fail-closed sources with locked navigation and no fabricated detections', async ({ page }) => {
    await page.goto('/');
    await page.getByText('MAT', { exact: true }).last().click();

    await expect(page.getByText('MISSION-AWARE TRIAGE / VERIFIED INPUTS', { exact: true })).toBeVisible();
    await expect(page.getByTestId('worldgraph-map')).toBeVisible();
    const survivorMetric = page.getByText('MAT SURVIVORS', { exact: true }).locator('..');
    await expect(survivorMetric.getByText('0', { exact: true })).toBeVisible();
    await expect(page.getByText('No incident events returned. Create an event through the MAT API before scanning.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: /START SCAN/ })).toBeDisabled();
    await expect(page.getByText('Training Scenario', { exact: true })).toHaveCount(0);
    await expect(page.getByText('SIMULATED DATA', { exact: true })).toHaveCount(0);

    await page.getByText('WORLDGRAPH SECURE LINK', { exact: true }).scrollIntoViewIfNeeded();
    await expect(page.getByTestId('header-home-logo').getByText('RuView', { exact: true }).first()).toBeVisible();
    await expect(page.getByRole('button', { name: 'MAT tab' })).toBeVisible();
    await expect(page.getByLabel('WorldGraph short-lived token')).toHaveValue('');

    const dimensions = await page.evaluate(() => ({
      viewport: window.innerWidth,
      content: document.documentElement.scrollWidth,
    }));
    expect(dimensions.viewport).toBe(390);
    expect(dimensions.content).toBeLessThanOrEqual(390);
  });
});

test.describe('RuView Zones spatial intelligence UI', () => {
  test('layers local evidence, topology, and OAuth-gated semantic context without fabricated state', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Zones', { exact: true }).last().click();
    await expect(page.getByTestId('zones-hero')).toBeVisible();
    await expect(page.getByText(/A room is more than/)).toBeVisible();
    await expect(page.getByTestId('zones-field-layer')).toBeVisible();
    await expect(page.getByText('LOCAL CSI · NO CAMERA · NO IDENTITY', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'TOPOLOGY' }).click();
    await expect(page.getByTestId('zones-topology-layer')).toBeVisible();
    await page.getByRole('button', { name: 'SPACES layer' }).click();
    await expect(page.getByTestId('zones-spaces-layer')).toBeVisible();
    await expect(page.getByText('SPACES IS PRIVATE BY DEFAULT', { exact: true })).toBeVisible();

    await expect(page.getByRole('switch', { name: 'Cloud spatial interpretation' })).not.toBeChecked();
    await expect(page.getByRole('button', { name: 'AUTHORIZE COGNITUM INFERENCE' })).toBeDisabled();
    await expect(page.getByText(/Raw CSI, pose frames, vital waveforms, recordings, and identity observations remain excluded/)).toBeVisible();
  });
});

test.describe('RuView product terminology', () => {
  test('does not expose the retired acronym on any primary screen', async ({ page }) => {
    await page.goto('/');
    for (const tab of ['Live', 'Calibration', 'Vitals', 'Zones', 'MAT', 'Settings']) {
      await page.getByText(tab, { exact: true }).last().click();
      expect(await page.locator('body').innerText()).not.toMatch(/\bNLOS\b/i);
    }
  });
});

test.describe('RuView tab navigation behavior', () => {
  test('resets every scrollable section to top on switch and repeated tab tap', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Vitals', { exact: true }).last().click();
    await expect(page.getByText('The room has a pulse.', { exact: true })).toBeVisible();
    const vitals = page.getByTestId('vitals-scroll-view');
    await page.waitForTimeout(100);
    await vitals.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await expect.poll(() => vitals.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);

    await page.getByText('Settings', { exact: true }).last().click();
    await page.getByText('Vitals', { exact: true }).last().click();
    await expect.poll(() => vitals.evaluate((element) => element.scrollTop)).toBe(0);

    await vitals.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await expect.poll(() => vitals.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
    await page.getByText('Vitals', { exact: true }).last().click();
    await expect.poll(() => vitals.evaluate((element) => element.scrollTop)).toBe(0);
  });

  test('shows real settings status, accepts a private calibration host, and hides stub controls', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Settings', { exact: true }).last().click();
    await expect(page.getByTestId('settings-hero')).toBeVisible();
    await expect(page.getByText('SENSING', { exact: true })).toBeVisible();
    await page.getByTestId('nlos-server-url-input').fill('http://192.168.1.166:3000');
    await expect(page.getByText('LOCAL HTTP', { exact: true })).toBeVisible();
    await expect(page.getByText(/requires HTTPS/)).toHaveCount(0);
    await expect(page.getByText('iOS: RSSI scanning uses stubbed telemetry in this build.', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('switch', { name: 'RSSI scanning' })).toHaveCount(0);
    await expect(page.getByRole('switch', { name: 'MAT alert sounds' })).toHaveCount(0);
  });
});
