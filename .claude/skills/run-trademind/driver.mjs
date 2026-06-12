#!/usr/bin/env node
/**
 * TradeMind driver — Playwright-based headless browser harness.
 * Usage: node driver.mjs [command] [args...]
 *
 * Commands:
 *   screenshot [path]         Take a screenshot (default: /tmp/trademind-shot.png)
 *   navigate <url>            Navigate to a URL
 *   login [user] [pass]       Log in with credentials (default: demo/demo)
 *   smoke                     Full smoke test: login → dashboard → signals → screenshot
 *
 * Env:
 *   BASE_URL   Frontend base URL (default: http://localhost:5173)
 */

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const SHOT_PATH = process.env.SHOT_PATH || '/tmp/trademind-shot.png';

const [, , cmd = 'smoke', ...args] = process.argv;

const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

page.on('console', msg => {
  if (msg.type() === 'error') process.stderr.write(`[console.error] ${msg.text()}\n`);
});

async function waitForApp() {
  await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
}

async function screenshot(path = SHOT_PATH) {
  await page.screenshot({ path, fullPage: false });
  console.log(`screenshot → ${path}`);
}

async function navigate(url) {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await waitForApp();
}

async function login(user = 'demo', pass = 'demo') {
  await navigate(BASE_URL);
  // If already on dashboard, skip
  if (page.url().includes('/dashboard')) return;
  // Fill login form
  const userInput = page.locator('input[type="text"], input[name="username"], input[placeholder*="sername"], input[placeholder*="Email"]').first();
  const passInput = page.locator('input[type="password"]').first();
  await userInput.waitFor({ timeout: 10000 });
  await userInput.fill(user);
  await passInput.fill(pass);
  await page.keyboard.press('Enter');
  await waitForApp();
}

async function smoke() {
  console.log('=== TradeMind smoke test ===');
  console.log(`BASE_URL: ${BASE_URL}`);

  // 1. Login
  console.log('[1] Logging in...');
  await login();
  await screenshot('/tmp/trademind-after-login.png');

  // 2. Navigate to dashboard
  console.log('[2] Dashboard...');
  await navigate(`${BASE_URL}/dashboard`);
  await screenshot('/tmp/trademind-dashboard.png');

  // 3. Navigate to AI Signals
  console.log('[3] AI Signals...');
  await navigate(`${BASE_URL}/signals`);
  await screenshot('/tmp/trademind-signals.png');

  // 4. Navigate to Market
  console.log('[4] Market...');
  await navigate(`${BASE_URL}/market`);
  await screenshot('/tmp/trademind-market.png');

  // 5. Check console errors
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });

  console.log('[5] Console errors:', errors.length === 0 ? 'none' : errors);
  console.log('=== smoke test complete ===');
  console.log('Screenshots:');
  ['/tmp/trademind-after-login.png', '/tmp/trademind-dashboard.png',
   '/tmp/trademind-signals.png', '/tmp/trademind-market.png'].forEach(p => console.log(' ', p));
}

try {
  switch (cmd) {
    case 'screenshot':
      await navigate(BASE_URL);
      await screenshot(args[0] || SHOT_PATH);
      break;
    case 'navigate':
      await navigate(args[0] || BASE_URL);
      await screenshot(SHOT_PATH);
      break;
    case 'login':
      await login(args[0], args[1]);
      await screenshot(SHOT_PATH);
      break;
    case 'smoke':
    default:
      await smoke();
  }
} finally {
  await browser.close();
}
