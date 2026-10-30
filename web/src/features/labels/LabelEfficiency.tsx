import { motion } from 'motion/react';
import { useState } from 'react';
import styles from '../../shared/chart/chart.module.css';
import { Legend } from '../../shared/chart/Legend';
import { linear, log, ticks } from '../../shared/chart/scale';
import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { CHART, DOMAIN_LABEL, METHOD_LABEL, METHOD_ORDER, MOTION, methodColor } from '../../shared/theme/tokens';
import { Segmented } from '../../shared/ui/Segmented';

const W = 640;

/** Accuracy against labels per class, one line per method; the band spans the seeds. */
export function LabelEfficiency({ r }: { r: Results }) {
  const [domain, setDomain] = useState(r.domains[0] ?? 'lab');
  const { top, right, bottom, left } = CHART.pad;
  const H = CHART.height;
  const methods = METHOD_ORDER.filter((m) => r.label_efficiency[m]);
  const budgets = r.budgets;
  const x = log(budgets[0]!, budgets[budgets.length - 1]!, left, W - right);
  const vals = methods.flatMap((m) => budgets.map((b) => r.label_efficiency[m]![String(b)]![domain]!));
  const lo = Math.max(0, Math.floor(Math.min(...vals.map((v) => v.min)) * 10) / 10);
  const y = linear(lo, 1, H - bottom, top);
  const line = (m: string, key: 'mean' | 'min' | 'max') =>
    budgets
      .map((b, i) => `${i ? 'L' : 'M'}${x(b).toFixed(1)},${y(r.label_efficiency[m]![String(b)]![domain]![key]).toFixed(1)}`)
      .join('');
  const band = (m: string) => {
    const up = budgets.map((b) => `${x(b).toFixed(1)},${y(r.label_efficiency[m]![String(b)]![domain]!.max).toFixed(1)}`);
    const dn = [...budgets]
      .reverse()
      .map((b) => `${x(b).toFixed(1)},${y(r.label_efficiency[m]![String(b)]![domain]!.min).toFixed(1)}`);
    return `M${up.join('L')}L${dn.join('L')}Z`;
  };
  const yt = ticks(lo, 1, CHART.tick);
  const sentence = methods
    .map(
      (m) =>
        `${METHOD_LABEL[m]}: ${budgets.map((b) => `${pct(r.label_efficiency[m]![String(b)]![domain]!.mean)} at ${b}`).join(', ')}`,
    )
    .join('. ');

  return (
    <figure className={styles.figure}>
      <Segmented
        label="Test domain"
        options={r.domains}
        value={domain}
        onChange={setDomain}
        format={(d) => DOMAIN_LABEL[d] ?? d}
      />
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={styles.svg}
        role="img"
        aria-label={`Accuracy by labels per class, ${DOMAIN_LABEL[domain] ?? domain}. ${sentence}.`}
      >
        {yt.map((t) => (
          <g key={t}>
            <line x1={left} x2={W - right} y1={y(t)} y2={y(t)} className={styles.grid} />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className={styles.tick}>
              {pct(t, 0)}
            </text>
          </g>
        ))}
        {budgets.map((b) => (
          <text key={b} x={x(b)} y={H - bottom + 18} textAnchor="middle" className={styles.tick}>
            {b}
          </text>
        ))}
        <text x={(left + W - right) / 2} y={H - 4} textAnchor="middle" className={styles.axisTitle}>
          labelled windows per class (log scale)
        </text>
        {methods.map((m) => (
          <g key={m}>
            <motion.path
              d={band(m)}
              fill={methodColor(m)}
              opacity={0.12}
              animate={{ d: band(m) }}
              transition={{ duration: MOTION.base, ease: MOTION.ease }}
            />
            <motion.path
              d={line(m, 'mean')}
              fill="none"
              stroke={methodColor(m)}
              strokeWidth={CHART.stroke}
              strokeDasharray={m.endsWith('probe') ? '5 4' : undefined}
              animate={{ d: line(m, 'mean') }}
              transition={{ duration: MOTION.base, ease: MOTION.ease }}
            />
            {budgets.map((b) => (
              <motion.circle
                key={b}
                cx={x(b)}
                r={CHART.dot}
                fill={methodColor(m)}
                animate={{ cy: y(r.label_efficiency[m]![String(b)]![domain]!.mean) }}
                initial={false}
                transition={{ duration: MOTION.base, ease: MOTION.ease }}
              />
            ))}
          </g>
        ))}
      </svg>
      <Legend
        items={methods.map((m) => ({ key: m, label: METHOD_LABEL[m] ?? m, color: methodColor(m), dashed: m.endsWith('probe') }))}
      />
      <figcaption className={styles.caption}>
        Mean over {r.seeds.length} seeds; the shaded band spans the best and worst seed. Dashed lines keep the pretrained encoder
        frozen and train only a linear layer.
      </figcaption>
    </figure>
  );
}
