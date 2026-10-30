// Copies the generated results into the app. The explorer never computes a number itself.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, process.env.RESULTS ?? '../../reports/results.json');
const dst = resolve(here, '../public/results.json');
mkdirSync(dirname(dst), { recursive: true });
if (existsSync(src)) {
  copyFileSync(src, dst);
  console.log(`results: ${src}`);
} else {
  console.warn(`results: ${src} not found; the explorer will show its empty state`);
}
