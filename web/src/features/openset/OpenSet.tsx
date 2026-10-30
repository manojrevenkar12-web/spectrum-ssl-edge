import { useState } from 'react';
import type { Results } from '../../shared/lib/results';
import { DOMAIN_LABEL, METHOD_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { Segmented } from '../../shared/ui/Segmented';
import t from '../../shared/ui/table.module.css';

function Bar({ v }: { v: number }) {
  // AUROC 0.5 is chance: the bar starts there
  const w = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
  return (
    <span className={t.barCell}>
      <span className={t.bar} aria-hidden="true">
        <span className={t.barFill} style={{ width: `${100 * w}%` }} />
      </span>
      <span>{v.toFixed(3)}</span>
    </span>
  );
}

/** AUROC of each novelty score, for near and far unseen classes and for each unseen class. */
export function OpenSet({ r }: { r: Results }) {
  const models = Object.keys(r.openset);
  const [model, setModel] = useState(models[models.length - 1]!);
  const domains = Object.keys(r.openset[model]!);
  const novel = r.classes.novel;
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <Segmented
        label="Model"
        options={models}
        value={model}
        onChange={setModel}
        format={(m) => METHOD_LABEL[m.split('/')[0]!] ?? m}
      />
      {domains.map((d) => {
        const rows = r.openset[model]![d]!;
        return (
          <ScrollRegion key={d} label={`Open-set results, ${DOMAIN_LABEL[d] ?? d}`}>
            <table className={t.table}>
              <caption>
                {DOMAIN_LABEL[d] ?? d}: AUROC for telling unseen classes from known ones (0.5 is chance, 1 is perfect).
              </caption>
              <thead>
                <tr>
                  <th scope="col">Score</th>
                  <th scope="col">Near ({r.classes.novel_groups.near?.join(', ')})</th>
                  <th scope="col">Far ({r.classes.novel_groups.far?.join(', ')})</th>
                  {novel.map((c) => (
                    <th key={c} scope="col">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(rows).map(([score, v]) => (
                  <tr key={score}>
                    <th scope="row" style={{ fontWeight: 500 }}>
                      {score}
                    </th>
                    <td>
                      <Bar v={v.near.auroc} />
                    </td>
                    <td>
                      <Bar v={v.far.auroc} />
                    </td>
                    {novel.map((c) => (
                      <td key={c}>{v[c]?.auroc.toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollRegion>
        );
      })}
    </div>
  );
}
