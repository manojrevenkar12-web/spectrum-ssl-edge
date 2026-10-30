// Theme constants used from TypeScript (charts, canvas, motion). Colours resolve CSS variables so
// light and dark mode stay in one place.
export const CHART = {
  height: 260,
  pad: { top: 16, right: 16, bottom: 40, left: 48 },
  tick: 5,
  stroke: 2,
  dot: 3.5,
} as const;

export const MOTION = {
  fast: 0.18,
  base: 0.32,
  slow: 0.6,
  ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
  replaySeconds: 9,
} as const;

export const WATERFALL = {
  /** cividis: perceptually uniform and readable with colour-vision deficiency */
  stops: ['#00204d', '#31446b', '#666970', '#958f78', '#cbba69', '#ffea46'],
} as const;

export const METHOD_ORDER = ['supervised', 'ssl-lab-probe', 'ssl-lab-finetune', 'ssl-field-probe', 'ssl-field-finetune'] as const;

export const METHOD_LABEL: Record<string, string> = {
  supervised: 'Supervised from scratch',
  'ssl-lab-probe': 'Bench pretraining, linear probe',
  'ssl-lab-finetune': 'Bench pretraining, fine-tuned',
  'ssl-field-probe': 'Field pretraining, linear probe',
  'ssl-field-finetune': 'Field pretraining, fine-tuned',
};

export const DOMAIN_LABEL: Record<string, string> = {
  lab: 'Bench',
  multipath: 'Multipath',
  hardware: 'Cheaper hardware',
  oversampling: 'Unseen symbol rates',
  irregular: 'Bursts and dropouts',
  heldout: 'Held-out (blind)',
};

export const methodColor = (m: string) => `var(--m-${m})`;
