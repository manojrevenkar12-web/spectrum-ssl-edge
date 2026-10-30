//! The spectrum front end against numpy on a committed scene (written by `sse edge`).

use sse_edge::dsp::{channel_energy, spectrogram, C64};
use std::fs;

fn read(name: &str) -> Vec<u8> {
    fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn channel_energy_matches_numpy() {
    let b = read("scene_iq.bin");
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let v: Vec<f64> = b[4..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let iq: Vec<C64> = (0..n)
        .map(|i| C64 {
            re: v[2 * i],
            im: v[2 * i + 1],
        })
        .collect();
    let e = read("scene_energy.bin");
    let frames = u32::from_le_bytes(e[0..4].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(e[4..8].try_into().unwrap()) as usize;
    let nfft = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
    let hop = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
    let edge = u32::from_le_bytes(e[16..20].try_into().unwrap()) as usize;
    let want: Vec<f64> = e[20..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (f, p) = spectrogram(&iq, nfft, hop);
    assert_eq!(f, frames);
    let got = channel_energy(&p, f, nfft, channels, edge);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "cell {i}: rust {g} numpy {w}"
        );
    }
}
