//! Bit-exact parity with the Python int8 reference on committed fixtures (written by `sse edge`).

use sse_edge::{Model, ParseError, Runtime};
use std::fs;

fn fixture(name: &str) -> Vec<u8> {
    fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

#[test]
fn rust_matches_the_python_reference_bit_for_bit() {
    let model = Model::parse(&fixture("model.bin")).expect("model parses");
    let v = fixture("vectors.bin");
    let e = fixture("expected.bin");
    let n = u32::from_le_bytes(v[0..4].try_into().unwrap()) as usize;
    let len = u32::from_le_bytes(v[4..8].try_into().unwrap()) as usize;
    let x: Vec<f32> = v[8..8 + 8 * n * len]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mask = &v[8 + 8 * n * len..];
    let k = u32::from_le_bytes(e[4..8].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
    let acc: Vec<i32> = e[12..12 + 4 * n * k]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let emb: Vec<i8> = e[12 + 4 * n * k..].iter().map(|&b| b as i8).collect();
    assert_eq!(emb.len(), n * d);

    let mut rt = Runtime::new(model);
    for i in 0..n {
        let o = rt.run(
            &x[i * 2 * len..(i + 1) * 2 * len],
            &mask[i * len..(i + 1) * len],
        );
        assert_eq!(
            o.acc,
            &acc[i * k..(i + 1) * k],
            "window {i}: head accumulators differ"
        );
        assert_eq!(
            o.embedding,
            &emb[i * d..(i + 1) * d],
            "window {i}: embedding differs"
        );
    }
}

#[test]
fn corrupted_files_are_refused_with_a_reason() {
    let good = fixture("model.bin");
    assert_eq!(Model::parse(b"NOPE").unwrap_err(), ParseError::BadMagic);
    assert!(matches!(
        Model::parse(&good[..good.len() - 3]).unwrap_err(),
        ParseError::Truncated { .. }
    ));
    let mut extra = good.clone();
    extra.push(0);
    assert_eq!(
        Model::parse(&extra).unwrap_err(),
        ParseError::TrailingBytes(1)
    );
    let mut v2 = good.clone();
    v2[4] = 2;
    assert_eq!(
        Model::parse(&v2).unwrap_err(),
        ParseError::UnsupportedVersion(2)
    );
}

#[test]
fn run_does_not_depend_on_the_previous_window() {
    let model = Model::parse(&fixture("model.bin")).unwrap();
    let v = fixture("vectors.bin");
    let len = u32::from_le_bytes(v[4..8].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(v[0..4].try_into().unwrap()) as usize;
    let x: Vec<f32> = v[8..8 + 8 * n * len]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mask = &v[8 + 8 * n * len..];
    let mut rt = Runtime::new(model);
    let first = rt.run(&x[..2 * len], &mask[..len]).acc.to_vec();
    rt.run(&x[2 * len..4 * len], &mask[len..2 * len]);
    assert_eq!(rt.run(&x[..2 * len], &mask[..len]).acc, first.as_slice());
}
