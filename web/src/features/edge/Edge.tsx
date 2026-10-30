import { motion } from 'motion/react';
import { int, pct, us } from '../../shared/lib/format';
import type { Edge as EdgeResults } from '../../shared/lib/results';
import { DOMAIN_LABEL, MOTION } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { Code, EmptyState } from '../../shared/ui/StateBlock';
import t from '../../shared/ui/table.module.css';
import styles from './edge.module.css';

function Verdict({ ok, children }: { ok: boolean; children: string }) {
  return <span className={`${styles.verdict} ${ok ? styles.ok : styles.no}`}>{children}</span>;
}

export function Edge({ e }: { e: EdgeResults | undefined }) {
  if (!e) {
    return (
      <EmptyState title="No edge results in this run">
        Run <Code>sse edge</Code> and <Code>sse report</Code> to quantise the deployed model and check the Rust runtime.
      </EmptyState>
    );
  }
  const rust = e.rust;
  const lab = e.accuracy.lab!;
  const lat = [
    { k: 'Rust int8 (this project)', v: rust.latency?.median_us ?? 0 },
    { k: 'ONNX Runtime float', v: e.onnx.latency_us.float },
    { k: 'ONNX Runtime int8 (QDQ)', v: e.onnx.latency_us.int8_qdq },
  ];
  const max = Math.max(...lat.map((l) => l.v));
  const steps = [
    {
      title: 'PyTorch → folded float graph',
      text: 'BatchNorm folded into each convolution, run in numpy float64.',
      ok: e.fold_max_abs_diff < 1e-3,
      v: `max |Δ| ${e.fold_max_abs_diff.toExponential(1)}`,
    },
    {
      title: 'Folded float → int8 reference',
      text: 'Per-channel int8 weights, per-tensor activations, integer requantisation only.',
      ok: lab.agreement >= 0.99,
      v: `${pct(lab.agreement)} same top-1`,
    },
    {
      title: 'int8 reference → Rust runtime',
      text: 'No dependencies, no allocation per window; every domain, masks and bursts included.',
      ok: !!rust.bit_exact,
      v: rust.available
        ? `${int((rust.windows ?? 0) - (rust.mismatched_windows ?? 0))} / ${int(rust.windows ?? 0)} bit-exact`
        : 'not run',
    },
    {
      title: 'PyTorch → ONNX Runtime',
      text: 'Exported graph with dynamic batch, CPU execution provider.',
      ok: e.onnx.max_abs_diff_vs_pytorch < 1e-3,
      v: `max |Δ| ${e.onnx.max_abs_diff_vs_pytorch.toExponential(1)}`,
    },
  ];
  return (
    <div style={{ display: 'grid', gap: 'var(--space-6)' }}>
      <div className={styles.facts}>
        <p className={styles.fact}>
          <b>{int(rust.macs ?? 0)}</b>
          <span>multiply-accumulates per window</span>
        </p>
        <p className={styles.fact}>
          <b>{int(e.weight_bytes)}</b>
          <span>bytes of weights</span>
        </p>
        <p className={styles.fact}>
          <b>{pct(lab.int8)}</b>
          <span>bench accuracy in int8 (float {pct(lab.float)})</span>
        </p>
      </div>
      <ol className={styles.chain} aria-label="Implementations checked against each other">
        {steps.map((s) => (
          <li key={s.title} className={styles.step}>
            <span className={styles.node} aria-hidden="true" />
            <span>
              <span className={styles.stepTitle}>{s.title}</span>
              <br />
              <span className={styles.stepText}>{s.text}</span>
            </span>
            <Verdict ok={s.ok}>{s.v}</Verdict>
          </li>
        ))}
      </ol>
      <ScrollRegion label="Accuracy by implementation">
        <table className={t.table}>
          <caption>Accuracy by implementation and test domain.</caption>
          <thead>
            <tr>
              <th scope="col">Domain</th>
              <th scope="col">Float</th>
              <th scope="col">int8 (ours)</th>
              <th scope="col">Same top-1</th>
              <th scope="col">ONNX Runtime int8</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(e.accuracy).map(([d, v]) => (
              <tr key={d}>
                <th scope="row" style={{ fontWeight: 500 }}>
                  {DOMAIN_LABEL[d] ?? d}
                </th>
                <td>{pct(v.float)}</td>
                <td>{pct(v.int8)}</td>
                <td>{pct(v.agreement)}</td>
                <td>{pct(e.onnx.int8_qdq_accuracy[d] ?? Number.NaN)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <div className={styles.latency} role="img" aria-label={lat.map((l) => `${l.k}: ${us(l.v)}`).join('; ')}>
        {lat.map((l) => (
          <div key={l.k} className={styles.latRow}>
            <span>{l.k}</span>
            <motion.span
              className={styles.latBar}
              initial={{ width: 0 }}
              whileInView={{ width: `${(100 * l.v) / max}%` }}
              viewport={{ once: true }}
              transition={{ duration: MOTION.slow, ease: MOTION.ease }}
            />
            <span className="num" style={{ textAlign: 'right' }}>
              {us(l.v)}
            </span>
          </div>
        ))}
        <p style={{ color: 'var(--ink-soft)' }}>
          Median latency for one window, one thread, on {e.cpu}. Not the target hardware: this compares implementations.
        </p>
      </div>
    </div>
  );
}
