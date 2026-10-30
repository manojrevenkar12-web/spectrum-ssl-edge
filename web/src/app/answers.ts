// One-sentence answers, computed from the results; they say which way a comparison went.
import { pct, pts } from '../shared/lib/format';
import type { Results } from '../shared/lib/results';

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

export function answers(r: Results) {
  const le = r.label_efficiency;
  const lo = String(r.budgets[0]);
  const hi = String(r.budgets[r.budgets.length - 1]);
  const acc = (m: string, b: string, d: string) => le[m]?.[b]?.[d]?.mean ?? Number.NaN;
  const avg = (m: string, b: string, ds: string[]) => mean(ds.map((d) => acc(m, b, d)));
  const shifts = r.domains.filter((d) => d !== 'lab');
  const seen = ['multipath', 'hardware', 'irregular'].filter((d) => r.domains.includes(d));
  const unseen = r.domains.filter((d) => d === 'oversampling' || d === 'heldout');
  const q1lo = acc('ssl-lab-finetune', lo, 'lab') - acc('supervised', lo, 'lab');
  const q1hi = acc('ssl-lab-finetune', hi, 'lab') - acc('supervised', hi, 'lab');
  const q2seen = avg('ssl-field-finetune', hi, seen) - avg('supervised', hi, seen);
  const q2unseen = avg('ssl-field-finetune', hi, unseen) - avg('supervised', hi, unseen);
  const cal = r.calibration;
  const ts = 'supervised + temperature';
  const ensTs = Object.keys(cal).find((k) => k.startsWith('ensemble') && k.endsWith('temperature')) ?? ts;
  const eceShift = (n: string) => mean(shifts.map((d) => cal[n]?.[d]?.ece ?? Number.NaN));
  const os = Object.values(r.openset).flatMap((per) => Object.values(per.lab ?? {}));
  const bestFar = Math.max(...os.map((v) => v.far.auroc));
  const bestNear = Math.max(...os.map((v) => v.near.auroc));
  return {
    q1: `With ${lo} labels per class, bench pretraining moves bench accuracy ${pts(q1lo)} against training from scratch; with ${hi} it moves it ${pts(q1hi)}.`,
    q2: `At ${hi} labels per class, field pretraining moves accuracy ${pts(q2seen)} on the shifts its unlabelled data contained, and ${pts(q2unseen)} on shifts it never saw.`,
    q3: `A temperature fitted on the bench gives ${pct(cal[ts]?.lab?.ece ?? Number.NaN)} ECE there and ${pct(eceShift(ts))} under shift; the ensemble with a temperature gives ${pct(eceShift(ensTs))} under shift.`,
    q4: `Best AUROC on the bench: ${bestFar.toFixed(3)} for far unseen classes, ${bestNear.toFixed(3)} for near ones that differ from a known class in one detail.`,
  };
}
