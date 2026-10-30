import type { ReactNode } from 'react';
import styles from './ui.module.css';

export function Skeleton({
  height,
  width = '100%',
  label,
}: {
  height: number | string;
  width?: number | string;
  label?: string;
}) {
  return label ? (
    <div className="skeleton" style={{ height, width }} role="status" aria-label={label} />
  ) : (
    <div className="skeleton" style={{ height, width }} aria-hidden="true" />
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className={styles.state} role="alert">
      <p className={styles.stateTitle}>Results did not load</p>
      <p className={styles.stateText}>{message}</p>
      <button type="button" className={styles.button} onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className={styles.state}>
      <p className={styles.stateTitle}>{title}</p>
      <p className={styles.stateText}>{children}</p>
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className={styles.code}>{children}</code>;
}
