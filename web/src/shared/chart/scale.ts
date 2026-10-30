export type Scale = (v: number) => number;

export const linear =
  (d0: number, d1: number, r0: number, r1: number): Scale =>
  (v) =>
    r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

export const log = (d0: number, d1: number, r0: number, r1: number): Scale => {
  const a = Math.log(d0);
  const b = Math.log(d1);
  return (v) => r0 + ((Math.log(v) - a) / (b - a || 1)) * (r1 - r0);
};

/** Round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count: number): number[] {
  const step0 = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(step0 || 1));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}
