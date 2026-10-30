import type { CSSProperties, ReactNode } from 'react';
import t from './table.module.css';

/** Wide tables scroll sideways on a phone; the region is focusable so keyboard users can scroll it. */
export function ScrollRegion({ label, style, children }: { label: string; style?: CSSProperties; children: ReactNode }) {
  return (
    // biome-ignore lint/a11y/noNoninteractiveTabindex: a scrollable region must be reachable by keyboard (WCAG 2.1.1)
    <section className={t.scroll} tabIndex={0} aria-label={label} style={style}>
      {children}
    </section>
  );
}
