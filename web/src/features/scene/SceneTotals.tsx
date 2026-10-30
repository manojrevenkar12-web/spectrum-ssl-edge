import { int, pct } from '../../shared/lib/format';
import type { Edge } from '../../shared/lib/results';
import { DOMAIN_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import t from '../../shared/ui/table.module.css';

/** Scene results over every generated scene, per domain. */
export function SceneTotals({ s }: { s: NonNullable<Edge['scenes']> }) {
  return (
    <ScrollRegion label="Scene results by domain" style={{ paddingBottom: 'var(--space-6)' }}>
      <table className={t.table}>
        <caption>All scenes: bursts found by energy detection, then classified window by window with the int8 model.</caption>
        <thead>
          <tr>
            <th scope="col">Domain</th>
            <th scope="col">Emitters</th>
            <th scope="col">Found</th>
            <th scope="col">False detections</th>
            <th scope="col">Known, right class</th>
            <th scope="col">Unseen, flagged</th>
            <th scope="col">Known, wrongly flagged</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(s.domains).map(([d, v]) => (
            <tr key={d}>
              <th scope="row" style={{ fontWeight: 500 }}>
                {DOMAIN_LABEL[d] ?? d}
              </th>
              <td>{int(v.emitters)}</td>
              <td>{pct(v.found / Math.max(1, v.emitters))}</td>
              <td>{int(v.false_detections)}</td>
              <td>{pct(v.known_correct / Math.max(1, v.known_found))}</td>
              <td>{pct(v.novel_flagged / Math.max(1, v.novel_found))}</td>
              <td>{pct(v.known_flagged_novel / Math.max(1, v.known_found))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
