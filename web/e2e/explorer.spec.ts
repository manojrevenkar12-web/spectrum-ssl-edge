import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('the explorer renders every section from the generated results', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('int8 model');
  for (const name of [
    /labels are scarce/,
    /not the bench/,
    /confidence still honest/,
    /never trained on/,
    /integer-only, in Rust/,
    /written blind/,
  ]) {
    await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
  }
  await expect(page.getByRole('img', { name: /emitters in a band of/ })).toBeVisible();
  expect(errors).toEqual([]);
});

test('switching the domain redraws the chart', async ({ page }) => {
  await page.goto('/');
  const chart = page.getByRole('img', { name: /Accuracy by labels per class/ });
  const before = await chart.getAttribute('aria-label');
  await page.locator('#q1').getByRole('button', { name: 'Bursts and dropouts' }).click();
  await expect(chart).not.toHaveAttribute('aria-label', before!);
});

test('no serious accessibility violations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  const a = await new AxeBuilder({ page } as unknown as ConstructorParameters<typeof AxeBuilder>[0])
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  expect(a.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')).toEqual([]);
});
