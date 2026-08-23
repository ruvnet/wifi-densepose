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
  await page.getByText('NLOS', { exact: true }).last().click();
  await expect(page.getByText('RuView NLOS', { exact: true })).toBeVisible();
  await expect(page.getByTestId('nlos-evidence-state')).toBeVisible();
};

const capture = async (page: Page, name: string) => {
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({
    path: resolve(screenshotDirectory, name),
    animations: 'disabled',
    caret: 'hide',
  });
};

test.describe('RuView NLOS mobile instrument UI', () => {
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
    await expect(page.getByTestId('nlos-evidence-state')).toHaveText('DISCONNECTED');

    await capture(page, 'overview-390x844.png');
  });

  test('captures governed setup with fixed explainer and feedback controls', async ({ page }) => {
    await openNlos(page);
    const setup = page.getByTestId('nlos-beta-setup');
    await setup.scrollIntoViewIfNeeded();
    await expect(setup).toBeVisible();
    await expect(page.getByRole('link', { name: 'OPEN EXPLAINER' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'TEST STEPS AND FEEDBACK' })).toBeVisible();

    await capture(page, 'setup-390x844.png');
  });

  test('starts synthetic replay and preserves its visible watermark', async ({ page }) => {
    await openNlos(page);
    const replay = page.getByRole('button', { name: 'USE SYNTHETIC REPLAY' });
    await replay.scrollIntoViewIfNeeded();
    await replay.click();

    await expect(page.getByTestId('nlos-synthetic-watermark')).toBeVisible();
    await expect(page.getByTestId('nlos-provenance-badge')).toContainText('SYNTHETIC');
    await expect(page.getByTestId('nlos-evidence-state')).toContainText('SYNTHETIC');
    await page.getByTestId('nlos-provenance-panel').evaluate((element) => {
      element.scrollIntoView({ behavior: 'auto', block: 'start' });
    });

    await capture(page, 'synthetic-390x844.png');
  });
});
