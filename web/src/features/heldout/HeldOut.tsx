import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { METHOD_LABEL, METHOD_ORDER } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { EmptyState } from '../../shared/ui/StateBlock';
import t from '../../shared/ui/table.module.css';

export function HeldOut({ r }: { r: Results }) {
  const f = r.heldout_first_run;
  if (!f) {
    return (
      <EmptyState title="No held-out first run yet">
        The held-out domain has not been run. Its first run is recorded once and never rewritten.
      </EmptyState>
    );
  }
  return (
    <ScrollRegion label="Held-out first run">
      <table className={t.table}>
        <caption>
          First run, recorded {f.recorded_at.slice(0, 10)}; domain file sha256 {f.domain_sha256.slice(0, 12)}…, never rewritten.
        </caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Accuracy, all labels</th>
          </tr>
        </thead>
        <tbody>
          {METHOD_ORDER.filter((m) => f.accuracy_full_labels[m] !== undefined).map((m) => (
            <tr key={m}>
              <th scope="row" style={{ fontWeight: m === r.deployed.method ? 700 : 500 }}>
                {METHOD_LABEL[m]}
                {m === r.deployed.method ? ' (deployed)' : ''}
              </th>
              <td>{pct(f.accuracy_full_labels[m]!)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
