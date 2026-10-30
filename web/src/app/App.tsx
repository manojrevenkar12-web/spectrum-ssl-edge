import { motion, useReducedMotion } from 'motion/react';
import { Calibration } from '../features/calibration/Calibration';
import { Edge } from '../features/edge/Edge';
import { HeldOut } from '../features/heldout/HeldOut';
import { LabelEfficiency } from '../features/labels/LabelEfficiency';
import { OpenSet } from '../features/openset/OpenSet';
import { SceneTotals } from '../features/scene/SceneTotals';
import { Waterfall } from '../features/scene/Waterfall';
import { ShiftDots } from '../features/shift/ShiftDots';
import { int, pct } from '../shared/lib/format';
import type { Results } from '../shared/lib/results';
import { useResults } from '../shared/lib/useResults';
import { MOTION } from '../shared/theme/tokens';
import { Section } from '../shared/ui/Section';
import { Code, EmptyState, ErrorState, Skeleton } from '../shared/ui/StateBlock';
import { answers } from './answers';
import styles from './app.module.css';

function Masthead({ r }: { r?: Results }) {
  return (
    <header className={styles.masthead}>
      <span className={styles.name}>spectrum-ssl-edge</span>
      <span>{r ? `Run ${r.generated_at.slice(0, 10)} · profile ${r.profile}` : 'Results explorer'}</span>
    </header>
  );
}

function Loaded({ r }: { r: Results }) {
  const reduce = useReducedMotion();
  const a = answers(r);
  const full = String(r.budgets[r.budgets.length - 1]);
  const deployedLab = r.label_efficiency[r.deployed.method]?.[full]?.lab?.mean ?? Number.NaN;
  const scene = r.edge?.scenes?.example;
  const rise = (i: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 12 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: MOTION.slow, ease: MOTION.ease, delay: 0.08 * i },
        };
  return (
    <>
      <section className={styles.hero} aria-labelledby="hero-h">
        <div className={styles.intro}>
          <motion.h1 id="hero-h" className={styles.headline} {...rise(0)}>
            A {int(r.model.parameters)}-parameter int8 model that names the signals in a band, measured where it holds up and
            where it does not.
          </motion.h1>
          <motion.p className={styles.introText} {...rise(1)}>
            Self-supervised pretraining on unlabelled I/Q, tested on the shifts a receiver meets in the field, on classes it never
            saw and on a domain written blind, then run integer-only in Rust, bit for bit equal to its reference. Every number
            here comes from <Code>reports/results.json</Code>.
          </motion.p>
          <motion.div className={styles.facts} {...rise(2)}>
            <p className={styles.fact}>
              <b>{pct(deployedLab)}</b>
              <span>bench accuracy, deployed model</span>
            </p>
            <p className={styles.fact}>
              <b>{int(r.edge?.rust.macs ?? r.model.macs_per_window)}</b>
              <span>multiply-accumulates per window</span>
            </p>
            <p className={styles.fact}>
              <b>{r.edge?.rust.bit_exact ? 'bit-exact' : 'not checked'}</b>
              <span>Rust runtime against the int8 reference</span>
            </p>
          </motion.div>
        </div>
        {scene ? (
          <Waterfall scene={scene} />
        ) : (
          <EmptyState title="No scene in this run">
            Run <Code>sse edge</Code> to replay a wideband scene.
          </EmptyState>
        )}
      </section>
      {r.edge?.scenes && <SceneTotals s={r.edge.scenes} />}

      <Section
        id="q1"
        kicker="Q1"
        title="Does self-supervised pretraining help when labels are scarce?"
        lede="SimCLR on unlabelled windows, then a linear probe or full fine-tuning, against training from scratch on the same labels."
        answer={a.q1}
      >
        <LabelEfficiency r={r} />
      </Section>
      <Section
        id="q2"
        kicker="Q2"
        title="Does it hold up when the receiver is not the bench?"
        lede="Each test domain is a distribution of conditions the labelled data never had."
        answer={a.q2}
      >
        <ShiftDots r={r} />
      </Section>
      <Section
        id="q3"
        kicker="Q3"
        title="Is its confidence still honest under shift?"
        lede="A replication of Ovadia et al. (NeurIPS 2019) on radio signals: temperature scaling against ensembles."
        answer={a.q3}
      >
        <Calibration r={r} />
      </Section>
      <Section
        id="q4"
        kicker="Q4"
        title="Does it notice a class it was never trained on?"
        lede="Four scores over the same models: max softmax, energy, Mahalanobis and nearest-neighbour distance in the embedding."
        answer={a.q4}
      >
        <OpenSet r={r} />
      </Section>
      <Section
        id="edge"
        kicker="Edge"
        title="The same model, integer-only, in Rust"
        lede="Each implementation is checked against the one before it."
      >
        <Edge e={r.edge} />
      </Section>
      <Section
        id="heldout"
        kicker="Held-out"
        title="A domain written blind, run once"
        lede="Written from a brief by someone who had not seen the code, the other domains or any result."
      >
        <HeldOut r={r} />
      </Section>
      <footer className={styles.footer}>
        <p>
          Simulated signals and channels; nothing here was recorded over the air. The scenes are civil spectrum monitoring:
          communications waveforms in a channel grid.
        </p>
        <p>
          Deployed model: <Code>{r.deployed.name}</Code>, chosen by a rule fixed before the run ({r.deployed.rule}).
        </p>
      </footer>
    </>
  );
}

export function App() {
  const res = useResults();
  return (
    <main className={styles.page}>
      <Masthead r={res.status === 'ready' ? res.data : undefined} />
      {res.status === 'loading' && (
        <div className={styles.hero}>
          <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
            <Skeleton height={96} label="Loading results" />
            <Skeleton height={64} />
            <Skeleton height={48} />
          </div>
          <Skeleton height={360} />
        </div>
      )}
      {res.status === 'error' && <ErrorState message={res.message} onRetry={res.retry} />}
      {res.status === 'empty' && (
        <EmptyState title="No results yet">
          Run <Code>sse all</Code> in the repository, then reload: the explorer reads <Code>reports/results.json</Code>.
        </EmptyState>
      )}
      {res.status === 'ready' && <Loaded r={res.data} />}
    </main>
  );
}
