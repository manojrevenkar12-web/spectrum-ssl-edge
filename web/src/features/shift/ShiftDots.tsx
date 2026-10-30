import { motion } from 'motion/react';
import { useState } from 'react';
import styles from '../../shared/chart/chart.module.css';
import { Legend } from '../../shared/chart/Legend';
import { linear, ticks } from '../../shared/chart/scale';
import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { CHART, DOMAIN_LABEL, METHOD_LABEL, METHOD_ORDER, MOTION, methodColor } from '../../shared/theme/tokens';
import { Segmented } from '../../shared/ui/Segmented';

const W = 640;
const ROW = 40;

/** One row per test domain, one dot per method: how far each falls off the bench. */
export function ShiftDots({ r }: { r: Results }) {
  const budgets = r.budgets.map(String);
  const [budget, setBudget] = useState(budgets[budgets.length - 1]!);
  const methods = METHOD_ORDER.filter((m) => r.label_efficiency[m]);
  const { right, left } = CHART.pad;
  const labelW = 150;
  const top = 8;
  const H = top + ROW * r.domains.length + 36;
  const all = methods.flatMap((m) => r.domains.map((d) => r.label_efficiency[m]![budget]![d]!.mean));
  const lo = Math.max(0, Math.floor(Math.min(...all) * 10) / 10);
  const x = linear(lo, 1, labelW + left / 2, W - right);
  const xt = ticks(lo, 1, CHART.tick);
  return (
    <figure className={styles.figure}>
      <Segmented
        label="Labels per class"
        options={budgets}
        value={budget}
        onChange={setBudget}
        format={(b) => `${b} labels per class`}
      />
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={styles.svg}
        role="img"
        aria-label={`Accuracy by test domain at ${budget} labels per class.`}
      >
        {xt.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top} y2={H - 30} className={styles.grid} />
            <text x={x(t)} y={H - 12} textAnchor="middle" className={styles.tick}>
              {pct(t, 0)}
            </text>
          </g>
        ))}
        {r.domains.map((d, i) => {
          const cy = top + ROW * i + ROW / 2;
          const xs = methods.map((m) => x(r.label_efficiency[m]![budget]![d]!.mean));
          return (
            <g key={d}>
              <text
                x={0}
                y={cy + 4}
                className={styles.tick}
                style={{ fontSize: 13, fill: 'var(--ink)', fontWeight: d === 'heldout' ? 700 : 500 }}
              >
                {DOMAIN_LABEL[d] ?? d}
              </text>
              <motion.line
                y1={cy}
                y2={cy}
                stroke="var(--line)"
                strokeWidth={6}
                strokeLinecap="round"
                animate={{ x1: Math.min(...xs), x2: Math.max(...xs) }}
                initial={false}
                transition={{ duration: MOTION.base, ease: MOTION.ease }}
              />
              {methods.map((m) => (
                <motion.circle
                  key={m}
                  cy={cy}
                  r={CHART.dot + 1.5}
                  fill={m.endsWith('probe') ? 'var(--surface)' : methodColor(m)}
                  stroke={methodColor(m)}
                  strokeWidth={2}
                  initial={false}
                  animate={{ cx: x(r.label_efficiency[m]![budget]![d]!.mean) }}
                  transition={{ duration: MOTION.base, ease: MOTION.ease }}
                >
                  <title>{`${METHOD_LABEL[m]}, ${DOMAIN_LABEL[d] ?? d}: ${pct(r.label_efficiency[m]![budget]![d]!.mean)}`}</title>
                </motion.circle>
              ))}
            </g>
          );
        })}
      </svg>
      <Legend
        items={methods.map((m) => ({ key: m, label: METHOD_LABEL[m] ?? m, color: methodColor(m), dashed: m.endsWith('probe') }))}
      />
      <figcaption className={styles.caption}>
        Field pretraining saw unlabelled multipath, cheaper-hardware and bursty recordings. No pretraining data contained the
        unseen symbol rates or the held-out domain.
      </figcaption>
    </figure>
  );
}
