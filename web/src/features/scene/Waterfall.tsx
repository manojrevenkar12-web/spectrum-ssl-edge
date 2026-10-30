import { motion, useReducedMotion } from 'motion/react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { SceneBox, SceneExample } from '../../shared/lib/results';
import { MOTION, WATERFALL } from '../../shared/theme/tokens';
import styles from './scene.module.css';

function lut(stops: readonly string[]): Uint8ClampedArray {
  const rgb = stops.map((h) => [1, 3, 5].map((i) => Number.parseInt(h.slice(i, i + 2), 16)));
  const out = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = (v / 255) * (rgb.length - 1);
    const i = Math.min(rgb.length - 2, Math.floor(t));
    const f = t - i;
    for (let c = 0; c < 3; c++) out[v * 3 + c] = rgb[i]![c]! + (rgb[i + 1]![c]! - rgb[i]![c]!) * f;
  }
  return out;
}

/** Frequency position (0..1 across the image) of a channel centre; the image is fft-shifted. */
export function channelX(channel: number, channels: number): number {
  let f = channel / channels;
  if (f >= 0.5) f -= 1;
  return 0.5 + f;
}

/**
 * The band over time, as the receiver saw it, replayed top to bottom. Dashed outlines are where
 * emitters really were; solid boxes are what the edge pipeline reported, labelled with its class or
 * flagged as never seen.
 */
export function Waterfall({ scene }: { scene: SceneExample }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();
  const [run, setRun] = useState(0);
  const [head, setHead] = useState(reduce ? scene.rows : 0);
  const W = scene.cols;
  const H = scene.rows;
  const chW = W / scene.channels;

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx) return;
    const img = ctx.createImageData(W, H);
    const map = lut(WATERFALL.stops);
    for (let i = 0; i < W * H; i++) {
      const v = scene.pixels[i] ?? 0;
      img.data[i * 4] = map[v * 3]!;
      img.data[i * 4 + 1] = map[v * 3 + 1]!;
      img.data[i * 4 + 2] = map[v * 3 + 2]!;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [scene, W, H]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `run` restarts the replay
  useEffect(() => {
    if (reduce) {
      setHead(H);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / (MOTION.replaySeconds * 1000));
      setHead(p * H);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    setHead(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [run, reduce, H]);

  const box = (b: SceneBox) => {
    const x = channelX(b.channel, scene.channels) * W - chW / 2 + 1;
    return { x, w: chW - 2, y: b.start, h: Math.max(1, b.end - b.start) };
  };
  const channelsShown = useMemo(() => [...new Set(scene.truth.map((t) => t.channel))].sort((a, b) => a - b), [scene.truth]);
  const shown = useMemo(() => scene.detections.filter((d) => d.start <= head), [scene.detections, head]);
  const novelCount = scene.detections.filter((d) => d.novel).length;
  const summary = `${scene.truth.length} emitters in a band of ${scene.channels} channels; the pipeline reported ${scene.detections.length} bursts, ${novelCount} flagged as an unseen class.`;

  return (
    <figure className={styles.wrap} style={{ margin: 0 }}>
      <div className={styles.stage}>
        <canvas ref={canvas} width={W} height={H} className={styles.canvas} />
        <svg className={styles.overlay} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={summary}>
          {scene.truth.map((t) => {
            const b = box(t);
            return (
              <rect
                key={`t${t.channel}-${t.start}`}
                x={b.x}
                y={b.y}
                width={b.w}
                height={b.h}
                fill="none"
                stroke="#e8ecf2"
                strokeWidth={0.6}
                strokeDasharray="2 2"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          {shown.map((d) => {
            const b = box(d);
            const h = Math.min(b.h, Math.max(0, head - b.y));
            const color = d.novel ? 'var(--novel)' : '#8fb0ff';
            return (
              <g key={`d${d.channel}-${d.start}`}>
                <motion.rect
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={h}
                  fill={d.novel ? 'rgb(184 61 11 / 0.18)' : 'rgb(143 176 255 / 0.16)'}
                  stroke={color}
                  strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: MOTION.fast }}
                />
              </g>
            );
          })}
          {!reduce && head < H && (
            <line x1={0} x2={W} y1={head} y2={head} className={styles.playhead} vectorEffect="non-scaling-stroke" />
          )}
        </svg>
        <div className={styles.labels} aria-hidden="true">
          {channelsShown.map((c) => (
            <span key={c} className={styles.channelLabel} style={{ left: `${100 * channelX(c, scene.channels)}%` }}>
              ch {c}
            </span>
          ))}
          {shown.map((d) => {
            const b = box(d);
            return (
              <span
                key={`l${d.channel}-${d.start}`}
                className={styles.tag}
                style={{
                  left: `${(100 * b.x) / W}%`,
                  top: `${(100 * b.y) / H}%`,
                  background: d.novel ? 'var(--novel)' : '#8fb0ff',
                  color: d.novel ? '#ffffff' : '#0b1020',
                }}
              >
                {d.novel ? 'unseen' : d.cls}
              </span>
            );
          })}
        </div>
      </div>
      <div className={styles.controls}>
        <button type="button" className={styles.replay} onClick={() => setRun((r) => r + 1)} disabled={!!reduce}>
          Replay
        </button>
        <span className={styles.key}>
          <span className={styles.box} style={{ border: '1.5px dashed var(--truth)' }} aria-hidden="true" /> where emitters were
        </span>
        <span className={styles.key}>
          <span className={styles.box} style={{ border: '2px solid var(--known)' }} aria-hidden="true" /> recognised
        </span>
        <span className={styles.key}>
          <span className={styles.box} style={{ border: '2px solid var(--novel)' }} aria-hidden="true" /> flagged as unseen
        </span>
      </div>
      <figcaption className={styles.counts}>{summary} Time runs down; frequency across.</figcaption>
      <ul className={styles.truthList} aria-label="What was transmitted and what was reported, by channel">
        {[...scene.truth]
          .sort((a, b) => a.channel - b.channel)
          .map((t) => {
            const d = scene.detections.find((x) => x.channel === t.channel && x.start < t.end && x.end > t.start);
            const said = d ? (d.novel ? 'flagged as unseen' : d.cls) : 'missed';
            const right = d ? (t.novel ? d.novel : !d.novel && d.cls === t.cls) : false;
            return (
              <li key={`${t.channel}-${t.start}`}>
                <span className={styles.ch}>ch {t.channel}</span> {t.cls}
                {t.novel && <span className={styles.unseen}> (never in training)</span>} →{' '}
                <b className={right ? styles.right : styles.wrong}>{said}</b>
              </li>
            );
          })}
      </ul>
    </figure>
  );
}
