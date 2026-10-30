//! Spectrum front end for the scene detector: a radix-2 FFT, a Hann-windowed power spectrogram and the
//! per-channel energy the detector thresholds. Mirrors `src/sse/detect/scene.py`; `tests/dsp.rs` checks
//! it against numpy on a committed scene.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    fn mul(self, o: C64) -> C64 {
        C64 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// In-place iterative radix-2 FFT (forward, no scaling). `x.len()` must be a power of two.
pub fn fft(x: &mut [C64]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = C64 {
                    re: (ang * k as f64).cos(),
                    im: (ang * k as f64).sin(),
                };
                let a = x[start + k];
                let b = x[start + k + len / 2].mul(w);
                x[start + k] = C64 {
                    re: a.re + b.re,
                    im: a.im + b.im,
                };
                x[start + k + len / 2] = C64 {
                    re: a.re - b.re,
                    im: a.im - b.im,
                };
            }
        }
        len <<= 1;
    }
}

/// Power spectrogram, frames of `nfft` samples with hop `hop`, Hann window, FFT-shifted so bin 0 is
/// the most negative frequency. Returns (frames, nfft) row-major.
pub fn spectrogram(iq: &[C64], nfft: usize, hop: usize) -> (usize, Vec<f64>) {
    let frames = if iq.len() >= nfft {
        (iq.len() - nfft) / hop + 1
    } else {
        0
    };
    let win: Vec<f64> = (0..nfft)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / nfft as f64).cos())
        .collect();
    let mut out = vec![0.0; frames * nfft];
    let mut buf = vec![C64::default(); nfft];
    for f in 0..frames {
        for i in 0..nfft {
            let s = iq[f * hop + i];
            buf[i] = C64 {
                re: s.re * win[i],
                im: s.im * win[i],
            };
        }
        fft(&mut buf);
        for i in 0..nfft {
            out[f * nfft + (i + nfft / 2) % nfft] = buf[i].norm_sqr();
        }
    }
    (frames, out)
}

/// Mean power per channel and frame in units of the band's noise floor, estimated as
/// median(power) / ln 2. Channel `c` is centred at frequency c / channels (wrapping), i.e. at bin
/// nfft/2 + f * nfft of the shifted spectrogram; `edge_bins` are dropped at each channel edge.
/// Mirrors `channel_energy` in `src/sse/detect/scene.py`. Returns (frames, channels) row-major.
pub fn channel_energy(
    power: &[f64],
    frames: usize,
    nfft: usize,
    channels: usize,
    edge_bins: usize,
) -> Vec<f64> {
    let mut sorted = power.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let floor = median / std::f64::consts::LN_2;
    let half = (nfft / channels / 2) as isize - edge_bins as isize;
    let mut out = vec![0.0; frames * channels];
    for c in 0..channels {
        let mut f = c as f64 / channels as f64;
        if f >= 0.5 {
            f -= 1.0;
        }
        let centre = (nfft as f64 / 2.0 + f * nfft as f64).round() as isize;
        let lo = (centre - half).max(0) as usize;
        let hi = ((centre + half + 1) as usize).min(nfft);
        for fr in 0..frames {
            let row = &power[fr * nfft..(fr + 1) * nfft];
            let s: f64 = row[lo..hi].iter().sum();
            out[fr * channels + c] = s / (hi - lo) as f64 / floor;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_of_a_tone_peaks_at_its_bin() {
        let n = 64;
        let mut x: Vec<C64> = (0..n)
            .map(|t| {
                let a = 2.0 * PI * 5.0 * t as f64 / n as f64;
                C64 {
                    re: a.cos(),
                    im: a.sin(),
                }
            })
            .collect();
        fft(&mut x);
        let peak = (0..n)
            .max_by(|&a, &b| x[a].norm_sqr().partial_cmp(&x[b].norm_sqr()).unwrap())
            .unwrap();
        assert_eq!(peak, 5);
        assert!((x[5].norm_sqr().sqrt() - n as f64).abs() < 1e-9);
    }

    #[test]
    fn fft_satisfies_parseval() {
        let n = 128;
        let mut x: Vec<C64> = (0..n)
            .map(|t| C64 {
                re: ((t * 7919) % 13) as f64 - 6.0,
                im: ((t * 104729) % 7) as f64 - 3.0,
            })
            .collect();
        let e_t: f64 = x.iter().map(|c| c.norm_sqr()).sum();
        fft(&mut x);
        let e_f: f64 = x.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;
        assert!((e_t - e_f).abs() < 1e-6 * e_t);
    }

    #[test]
    fn channel_energy_sees_a_tone_in_its_channel_only() {
        let (nfft, channels) = (256, 8);
        // a tone at the centre of channel 2 plus a tiny floor so the median is not zero
        let iq: Vec<C64> = (0..4096)
            .map(|t| {
                let a = 2.0 * PI * 2.0 / 8.0 * t as f64;
                C64 {
                    re: a.cos() + 1e-3 * ((t * 7) % 5) as f64,
                    im: a.sin() + 1e-3 * ((t * 3) % 7) as f64,
                }
            })
            .collect();
        let (frames, p) = spectrogram(&iq, nfft, 128);
        let e = channel_energy(&p, frames, nfft, channels, 2);
        let best = (0..channels)
            .max_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap())
            .unwrap();
        assert_eq!(best, 2);
    }
}
