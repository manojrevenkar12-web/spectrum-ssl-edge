//! sse-edge: run the int8 classifier on recorded windows.
//!
//!   sse-edge info  <model.bin>
//!   sse-edge infer <model.bin> <vectors.bin> <out.bin>
//!   sse-edge bench <model.bin> <vectors.bin> [iterations]

use sse_edge::{Model, Runtime};
use std::{env, fs, process::ExitCode, time::Instant};

struct Vectors {
    n: usize,
    length: usize,
    x: Vec<f32>,
    mask: Vec<u8>,
}

fn read_vectors(path: &str) -> Result<Vectors, String> {
    let b = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 8 {
        return Err(format!("{path}: too short"));
    }
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let length = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let nx = n * 2 * length;
    let want = 8 + 4 * nx + n * length;
    if b.len() != want {
        return Err(format!("{path}: {} bytes, expected {want}", b.len()));
    }
    let x = b[8..8 + 4 * nx]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mask = b[8 + 4 * nx..].to_vec();
    Ok(Vectors { n, length, x, mask })
}

fn load(path: &str) -> Result<Runtime, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let model = Model::parse(&bytes).map_err(|e| format!("{path}: {e}"))?;
    Ok(Runtime::new(model))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let usage = "usage: sse-edge info <model> | infer <model> <vectors> <out> | bench <model> <vectors> [iterations]";
    match args.get(1).map(String::as_str) {
        Some("info") if args.len() == 3 => {
            let rt = load(&args[2])?;
            println!(
                "{{\"length\": {}, \"classes\": {}, \"macs\": {}, \"weight_bytes\": {}}}",
                rt.model.length,
                rt.model.n_classes,
                rt.model.macs(),
                rt.model.weight_bytes()
            );
            Ok(())
        }
        Some("infer") if args.len() == 5 => {
            let mut rt = load(&args[2])?;
            let v = read_vectors(&args[3])?;
            if v.length != rt.model.length {
                return Err(format!(
                    "vectors have length {}, model expects {}",
                    v.length, rt.model.length
                ));
            }
            let (k, d) = (rt.model.n_classes, rt.model.proj.n_out);
            let mut out = Vec::with_capacity(12 + v.n * (4 * k + d));
            out.extend_from_slice(&(v.n as u32).to_le_bytes());
            out.extend_from_slice(&(k as u32).to_le_bytes());
            out.extend_from_slice(&(d as u32).to_le_bytes());
            let mut embs = Vec::with_capacity(v.n * d);
            for i in 0..v.n {
                let o = rt.run(
                    &v.x[i * 2 * v.length..(i + 1) * 2 * v.length],
                    &v.mask[i * v.length..(i + 1) * v.length],
                );
                for a in o.acc {
                    out.extend_from_slice(&a.to_le_bytes());
                }
                embs.extend(o.embedding.iter().map(|&e| e as u8));
            }
            out.extend_from_slice(&embs);
            fs::write(&args[4], out).map_err(|e| format!("{}: {e}", args[4]))
        }
        Some("bench") if args.len() == 4 || args.len() == 5 => {
            let mut rt = load(&args[2])?;
            let v = read_vectors(&args[3])?;
            let iters: usize = args
                .get(4)
                .map(|s| {
                    s.parse()
                        .map_err(|_| "iterations must be a number".to_string())
                })
                .transpose()?
                .unwrap_or(2000);
            let mut times = Vec::with_capacity(iters);
            let mut sink = 0usize;
            for _ in 0..50.min(iters) {
                sink += rt.classify(&v.x[..2 * v.length], &v.mask[..v.length]);
            }
            for i in 0..iters {
                let j = i % v.n;
                let t = Instant::now();
                sink += rt.classify(
                    &v.x[j * 2 * v.length..(j + 1) * 2 * v.length],
                    &v.mask[j * v.length..(j + 1) * v.length],
                );
                times.push(t.elapsed().as_nanos() as f64 / 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
            println!(
                "{{\"iterations\": {iters}, \"median_us\": {:.2}, \"p99_us\": {:.2}, \"windows_per_s\": {:.0}, \"checksum\": {sink}}}",
                p(0.5),
                p(0.99),
                1e6 / p(0.5)
            );
            Ok(())
        }
        _ => Err(usage.to_string()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sse-edge: {e}");
            ExitCode::FAILURE
        }
    }
}
