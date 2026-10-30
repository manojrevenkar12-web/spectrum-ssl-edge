//! The integer forward pass. Buffers are allocated once in `Runtime::new`.

use crate::model::{Conv, Model, Shortcut};

pub const QMAX: i64 = 127;

/// Fixed-point rescale with round-half-up: (acc * m + 2^(s-1)) >> s, arithmetic shift.
#[inline(always)]
pub fn requant(acc: i64, m: i32, s: u32) -> i64 {
    (acc * m as i64 + (1i64 << (s - 1))) >> s
}

/// float32 -> int8 exactly as the Python reference: floor(x * inv_scale + 0.5), clamped.
#[inline(always)]
pub fn quantise_input(x: f32, inv_scale: f32) -> i8 {
    let v = (x * inv_scale + 0.5f32).floor();
    v.clamp(-(QMAX as f32), QMAX as f32) as i8
}

fn out_len(len: usize, c: &Conv) -> usize {
    (len + 2 * (c.k / 2) - c.k) / c.stride + 1
}

/// int8 (cin, len) -> accumulators (cout, out_len), "same" zero padding of k/2.
///
/// im2col then one dot product per (output channel, step): the patch for a step is the `cin * k`
/// input values under the kernel, laid out like a weight row, so the inner loop is a contiguous
/// i8 x i8 -> i32 dot product the compiler vectorises. Accumulation is in i32 (k * cin * 127 * 127
/// stays far below 2^31 for these layer sizes).
fn conv_acc(x: &[i8], len: usize, c: &Conv, out: &mut [i64], cols: &mut [i8]) -> usize {
    let ol = out_len(len, c);
    let pad = c.k / 2;
    let width = c.cin * c.k;
    let cols = &mut cols[..ol * width];
    for t in 0..ol {
        let patch = &mut cols[t * width..(t + 1) * width];
        for i in 0..c.cin {
            let xi = &x[i * len..(i + 1) * len];
            for k in 0..c.k {
                let pos = (t * c.stride + k) as isize - pad as isize;
                patch[i * c.k + k] = if pos >= 0 && (pos as usize) < len {
                    xi[pos as usize]
                } else {
                    0
                };
            }
        }
    }
    for o in 0..c.cout {
        let wrow = &c.w[o * width..(o + 1) * width];
        let bias = c.b[o];
        let dst = &mut out[o * ol..(o + 1) * ol];
        for (t, d) in dst.iter_mut().enumerate() {
            let patch = &cols[t * width..(t + 1) * width];
            let dot: i32 = wrow
                .iter()
                .zip(patch)
                .map(|(&w, &v)| w as i32 * v as i32)
                .sum();
            *d = (bias + dot) as i64;
        }
    }
    ol
}

fn requant_relu(acc: &[i64], c: &Conv, ol: usize, out: &mut [i8]) {
    for o in 0..c.cout {
        for t in 0..ol {
            let v = requant(acc[o * ol + t], c.m[o], c.s[o]).clamp(0, QMAX);
            out[o * ol + t] = v as i8;
        }
    }
}

pub struct Output<'a> {
    /// head accumulators, one per class
    pub acc: &'a [i32],
    /// int8 embedding
    pub embedding: &'a [i8],
}

impl Output<'_> {
    pub fn logits(&self, model: &Model) -> Vec<f64> {
        self.acc
            .iter()
            .zip(&model.head.out_scale)
            .map(|(&a, &s)| a as f64 * s)
            .collect()
    }
}

pub struct Runtime {
    pub model: Model,
    a: Vec<i8>,
    b: Vec<i8>,
    c: Vec<i8>,
    acc: Vec<i64>,
    acc2: Vec<i64>,
    cols: Vec<i8>,
    pooled: Vec<i64>,
    emb: Vec<i8>,
    head: Vec<i32>,
}

impl Runtime {
    pub fn new(model: Model) -> Runtime {
        // the largest activation is (channels x length) of any layer
        let mut biggest = 2 * model.length;
        let mut len = out_len(model.length, &model.stem);
        biggest = biggest.max(model.stem.cout * len);
        for b in &model.blocks {
            len = out_len(len, &b.c1);
            biggest = biggest.max(b.c1.cout * len).max(b.c2.cout * len);
        }
        let d = model.proj.n_out;
        let k = model.n_classes;
        let pooled = model.proj.n_in;
        // largest im2col buffer: out_len x (cin * k) over all convolutions
        let mut cols_len = 0;
        let mut l = model.length;
        let mut grow =
            |c: &crate::model::Conv, l: usize| cols_len = cols_len.max(out_len(l, c) * c.cin * c.k);
        grow(&model.stem, l);
        l = out_len(l, &model.stem);
        for b in &model.blocks {
            grow(&b.c1, l);
            if let Shortcut::Conv(sc) = &b.short {
                grow(sc, l);
            }
            l = out_len(l, &b.c1);
            grow(&b.c2, l);
        }
        Runtime {
            model,
            a: vec![0; biggest * 2],
            b: vec![0; biggest * 2],
            c: vec![0; biggest * 2],
            acc: vec![0; biggest * 2],
            acc2: vec![0; biggest * 2],
            cols: vec![0; cols_len],
            pooled: vec![0; pooled],
            emb: vec![0; d],
            head: vec![0; k],
        }
    }

    /// One window: `iq` is (2, length) float32 I then Q; `mask` marks recorded samples.
    pub fn run(&mut self, iq: &[f32], mask: &[u8]) -> Output<'_> {
        let m = &self.model;
        let len0 = m.length;
        assert_eq!(iq.len(), 2 * len0, "window must be 2 x length samples");
        assert_eq!(mask.len(), len0, "mask must have one entry per sample");
        for (q, &x) in self.a.iter_mut().zip(iq) {
            *q = quantise_input(x, m.in_inv_scale);
        }
        // stem
        let mut len = conv_acc(
            &self.a[..2 * len0],
            len0,
            &m.stem,
            &mut self.acc,
            &mut self.cols,
        );
        requant_relu(&self.acc, &m.stem, len, &mut self.b);
        let mut ch = m.stem.cout;
        // blocks: input in b
        for blk in &m.blocks {
            let l1 = conv_acc(
                &self.b[..ch * len],
                len,
                &blk.c1,
                &mut self.acc,
                &mut self.cols,
            );
            requant_relu(&self.acc, &blk.c1, l1, &mut self.c);
            let l2 = conv_acc(
                &self.c[..blk.c1.cout * l1],
                l1,
                &blk.c2,
                &mut self.acc,
                &mut self.cols,
            );
            debug_assert_eq!(l1, l2);
            let cout = blk.c2.cout;
            match &blk.short {
                Shortcut::Conv(sc) => {
                    conv_acc(&self.b[..ch * len], len, sc, &mut self.acc2, &mut self.cols);
                    for o in 0..cout {
                        for t in 0..l1 {
                            let i = o * l1 + t;
                            let y = requant(self.acc[i], blk.c2.m[o], blk.c2.s[o])
                                + requant(self.acc2[i], sc.m[o], sc.s[o]);
                            self.a[i] = y.clamp(0, QMAX) as i8;
                        }
                    }
                }
                Shortcut::Identity { m: im, s: is } => {
                    for o in 0..cout {
                        for t in 0..l1 {
                            let i = o * l1 + t;
                            let y = requant(self.acc[i], blk.c2.m[o], blk.c2.s[o])
                                + requant(self.b[i] as i64, *im, *is);
                            self.a[i] = y.clamp(0, QMAX) as i8;
                        }
                    }
                }
            }
            std::mem::swap(&mut self.a, &mut self.b);
            ch = cout;
            len = l1;
        }
        // masked mean and max over time; a step is valid if any sample in its stride block was
        let stride = m.total_stride;
        let mut count: i64 = 0;
        for t in 0..len {
            let lo = t * stride;
            let hi = ((t + 1) * stride).min(len0);
            if lo < len0 && mask[lo..hi].iter().any(|&v| v != 0) {
                count += 1;
            }
        }
        for c in 0..ch {
            let mut sum: i64 = 0;
            let mut mx: i64 = 0;
            for t in 0..len {
                let lo = t * stride;
                let hi = ((t + 1) * stride).min(len0);
                if lo < len0 && mask[lo..hi].iter().any(|&v| v != 0) {
                    let v = self.b[c * len + t] as i64;
                    sum += v;
                    mx = mx.max(v);
                }
            }
            self.pooled[c] = if count > 0 {
                (2 * sum + count).div_euclid(2 * count)
            } else {
                0
            };
            self.pooled[ch + c] = mx;
        }
        // projection -> int8 embedding
        let p = &m.proj;
        for o in 0..p.n_out {
            let row = &p.w[o * p.n_in..(o + 1) * p.n_in];
            let mut acc: i64 = p.b[o] as i64;
            for (w, x) in row.iter().zip(&self.pooled) {
                acc += *w as i64 * *x;
            }
            self.emb[o] = requant(acc, p.m[o], p.s[o]).clamp(-QMAX, QMAX) as i8;
        }
        // head -> int32 accumulators
        let h = &m.head;
        for o in 0..h.n_out {
            let row = &h.w[o * h.n_in..(o + 1) * h.n_in];
            let mut acc: i64 = h.b[o] as i64;
            for (w, x) in row.iter().zip(&self.emb) {
                acc += *w as i64 * *x as i64;
            }
            self.head[o] = acc as i32;
        }
        Output {
            acc: &self.head,
            embedding: &self.emb,
        }
    }

    /// Predicted class: argmax of the dequantised logits (per-class scales differ, so not of `acc`).
    pub fn classify(&mut self, iq: &[f32], mask: &[u8]) -> usize {
        self.run(iq, mask);
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (i, (&a, &s)) in self.head.iter().zip(&self.model.head.out_scale).enumerate() {
            let v = a as f64 * s;
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requant_rounds_half_up() {
        // m / 2^s = 0.5
        let (m, s) = (1 << 30, 31);
        assert_eq!(requant(3, m, s), 2);
        assert_eq!(requant(-3, m, s), -1);
        assert_eq!(requant(1, m, s), 1);
        assert_eq!(requant(-1, m, s), 0);
    }

    #[test]
    fn input_quantisation_matches_floor_half_up() {
        assert_eq!(quantise_input(0.5, 1.0), 1);
        assert_eq!(quantise_input(-0.5, 1.0), 0);
        assert_eq!(quantise_input(-0.51, 1.0), -1);
        assert_eq!(quantise_input(1e9, 1.0), 127);
        assert_eq!(quantise_input(-1e9, 1.0), -127);
    }

    #[test]
    fn conv_same_padding_and_stride() {
        // one input channel [1,2,3,4], kernel [1,1,1], stride 2 -> taps centred on 0 and 2
        let c = Conv {
            cin: 1,
            cout: 1,
            k: 3,
            stride: 2,
            w: vec![1, 1, 1],
            b: vec![10],
            m: vec![1 << 30],
            s: vec![30],
        };
        let mut out = vec![0i64; 2];
        let mut cols = vec![0i8; 8];
        let n = conv_acc(&[1, 2, 3, 4], 4, &c, &mut out, &mut cols);
        assert_eq!(n, 2);
        assert_eq!(out, vec![10 + 1 + 2, 10 + 2 + 3 + 4]);
    }
}
