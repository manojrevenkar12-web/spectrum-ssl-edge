import { motion } from 'motion/react';
import { useId } from 'react';
import { MOTION } from '../theme/tokens';
import styles from './ui.module.css';

/** A row of toggle buttons; the underline slides to the chosen one. */
export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  format = (v) => v,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  format?: (v: T) => string;
}) {
  const id = useId();
  return (
    <fieldset className={styles.segmented} style={{ border: 0, padding: 0, margin: 0, borderBottom: '1px solid var(--line)' }}>
      <legend className="visually-hidden">{label}</legend>
      {options.map((o) => (
        <button key={o} type="button" className={styles.segment} aria-pressed={o === value} onClick={() => onChange(o)}>
          {format(o)}
          {o === value && (
            <motion.span
              layoutId={`seg-${id}`}
              className={styles.underline}
              transition={{ duration: MOTION.base, ease: MOTION.ease }}
            />
          )}
        </button>
      ))}
    </fieldset>
  );
}
