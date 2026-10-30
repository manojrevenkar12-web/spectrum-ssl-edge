import { motion } from 'motion/react';
import { useState } from 'react';
import styles from '../../shared/chart/chart.module.css';
import { Legend } from '../../shared/chart/Legend';
import { linear } from '../../shared/chart/scale';
import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { CHART, DOMAIN_LABEL, MOTION } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { Segmented } from '../../shared/ui/Segmented';
import t from '../../shared/ui/table.module.css';

const S = 300;

/** Reliability diagram per domain, and ECE for every calibration method on every domain. */
export function Calibration({ r }: { r: Results }) {
  const [domain, setDomain] = useState(r.domains.includes('hardware') ? 'hardware' : r.domains[0]!);
  const variants = Object.keys(r.reliability);
  const pad = CHART.pad.left;
  const s = linear(0, 1, pad, S - 8);
  const sy = linear(0, 1, S - pad, 8);
  const colors = ['var(--ink)', 'var(--novel)'];
  const names = Object.keys(r.calibration);
  const bestPerDomain = Object.fromEntries(r.domains.map((d) => [d, Math.min(...names.map((n) => r.calibration[n]![d]!.ece))]));
  return (
    <div style={{ display: 'grid', gap: 'var(--space-5)' }}>
      <figure className={styles.figure}>
        <Segmented
          label="Test domain"
          options={r.domains}
          value={domain}
          onChange={setDomain}
          format={(d) => DOMAIN_LABEL[d] ?? d}
        />
        <svg
          viewBox={`0 0 ${S} ${S}`}
          className={styles.svg}
          style={{ maxWidth: 420 }}
          role="img"
          aria-label={`Reliability diagram, ${DOMAIN_LABEL[domain] ?? domain}`}
        >
          <line x1={s(0)} y1={sy(0)} x2={s(1)} y2={sy(1)} stroke="var(--line)" strokeDasharray="4 4" />
          <line x1={s(0)} y1={sy(0)} x2={s(1)} y2={sy(0)} className={styles.axis} />
          <line x1={s(0)} y1={sy(0)} x2={s(0)} y2={sy(1)} className={styles.axis} />
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <text x={s(v)} y={sy(0) + 16} textAnchor="middle" className={styles.tick}>
                {pct(v, 0)}
              </text>
              <text x={s(0) - 6} y={sy(v) + 4} textAnchor="end" className={styles.tick}>
                {pct(v, 0)}
              </text>
            </g>
          ))}
          <text x={(s(0) + s(1)) / 2} y={S - 8} textAnchor="middle" className={styles.axisTitle}>
            confidence
          </text>
          <text
            x={10}
            y={(sy(0) + sy(1)) / 2}
            textAnchor="middle"
            transform={`rotate(-90 10 ${(sy(0) + sy(1)) / 2})`}
            className={styles.axisTitle}
          >
            accuracy
          </text>
          {variants.map((v, i) => {
            const pts = r.reliability[v]![domain]!.filter((b) => b.n > 0 && b.confidence !== null);
            const d = pts.map((b, j) => `${j ? 'L' : 'M'}${s(b.confidence!).toFixed(1)},${sy(b.accuracy!).toFixed(1)}`).join('');
            return (
              <g key={v}>
                <motion.path
                  d={d}
                  fill="none"
                  stroke={colors[i % colors.length]}
                  strokeWidth={CHART.stroke}
                  animate={{ d }}
                  transition={{ duration: MOTION.base, ease: MOTION.ease }}
                />
                {pts.map((b) => (
                  <circle
                    key={b.lo}
                    cx={s(b.confidence!)}
                    cy={sy(b.accuracy!)}
                    r={Math.min(6, 1.5 + Math.sqrt(b.n) / 6)}
                    fill={colors[i % colors.length]}
                    opacity={0.8}
                  />
                ))}
              </g>
            );
          })}
        </svg>
        <Legend items={variants.map((v, i) => ({ key: v, label: v, color: colors[i % colors.length]! }))} />
        <figcaption className={styles.caption}>
          On the diagonal, a model is right as often as it says it is. Below it, it is overconfident. Dot size grows with the
          number of windows in the bin. Temperatures were fitted on the bench only.
        </figcaption>
      </figure>
      <ScrollRegion label="Calibration error by method and domain">
        <table className={t.table}>
          <caption>Expected calibration error (lower is better); the best on each domain is bold.</caption>
          <thead>
            <tr>
              <th scope="col">Method</th>
              {r.domains.map((d) => (
                <th key={d} scope="col">
                  {DOMAIN_LABEL[d] ?? d}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((n) => (
              <tr key={n}>
                <th scope="row" style={{ fontWeight: 500 }}>
                  {n}
                </th>
                {r.domains.map((d) => {
                  const v = r.calibration[n]![d]!.ece;
                  return (
                    <td key={d} className={v === bestPerDomain[d] ? t.best : undefined}>
                      {pct(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
    </div>
  );
}
