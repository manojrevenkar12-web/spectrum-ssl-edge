import type { ReactNode } from 'react';
import styles from './ui.module.css';

/** A research question: what was asked, the chart, and the answer in one sentence or two. */
export function Section({
  id,
  kicker,
  title,
  lede,
  answer,
  children,
}: {
  id: string;
  kicker: string;
  title: string;
  lede?: ReactNode;
  answer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={styles.section} aria-labelledby={`${id}-h`} id={id}>
      <div className={styles.sectionHead}>
        <p className={styles.kicker}>{kicker}</p>
        <h2 id={`${id}-h`} className={styles.sectionTitle}>
          {title}
        </h2>
        {lede && <p className={styles.lede}>{lede}</p>}
      </div>
      {children}
      {answer && <p className={styles.answer}>{answer}</p>}
    </section>
  );
}
