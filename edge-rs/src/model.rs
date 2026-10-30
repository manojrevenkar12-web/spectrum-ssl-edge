//! Parser for the `SSEQ` model format (little-endian). See `export` in `src/sse/edge/int8.py`.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Conv {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    /// (cout, cin, k), row-major
    pub w: Vec<i8>,
    pub b: Vec<i32>,
    pub m: Vec<i32>,
    pub s: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Shortcut {
    Conv(Conv),
    Identity { m: i32, s: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub c1: Conv,
    pub c2: Conv,
    pub short: Shortcut,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub n_in: usize,
    pub n_out: usize,
    /// (n_out, n_in), row-major
    pub w: Vec<i8>,
    pub b: Vec<i32>,
    /// requantisation (projection) — empty for the head
    pub m: Vec<i32>,
    pub s: Vec<u32>,
    /// dequantisation scale per output (head only)
    pub out_scale: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub length: usize,
    pub n_classes: usize,
    pub total_stride: usize,
    pub in_inv_scale: f32,
    pub stem: Conv,
    pub blocks: Vec<Block>,
    pub proj: Linear,
    pub head: Linear,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated { at: usize, wanted: usize },
    Shape(&'static str),
    TrailingBytes(usize),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::BadMagic => write!(f, "not an SSEQ model file"),
            ParseError::UnsupportedVersion(v) => write!(f, "unsupported model version {v}"),
            ParseError::Truncated { at, wanted } => {
                write!(f, "file ends at byte {at}, {wanted} more bytes expected")
            }
            ParseError::Shape(what) => write!(f, "inconsistent shapes: {what}"),
            ParseError::TrailingBytes(n) => write!(f, "{n} unexpected bytes after the model"),
        }
    }
}

impl std::error::Error for ParseError {}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        if self.pos + n > self.buf.len() {
            return Err(ParseError::Truncated {
                at: self.buf.len(),
                wanted: self.pos + n - self.buf.len(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ParseError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, ParseError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ParseError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, ParseError> {
        Ok(self.u32()? as usize)
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, ParseError> {
        Ok(self.take(n)?.iter().map(|&b| b as i8).collect())
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, ParseError> {
        (0..n).map(|_| self.i32()).collect()
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, ParseError> {
        (0..n).map(|_| self.u32()).collect()
    }
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, ParseError> {
        (0..n)
            .map(|_| Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap())))
            .collect()
    }
    fn conv(&mut self) -> Result<Conv, ParseError> {
        let (cin, cout, k, stride) = (self.usize()?, self.usize()?, self.usize()?, self.usize()?);
        if cin == 0 || cout == 0 || k == 0 || k % 2 == 0 || stride == 0 {
            return Err(ParseError::Shape("convolution dimensions"));
        }
        let w = self.i8s(cout * cin * k)?;
        let b = self.i32s(cout)?;
        let m = self.i32s(cout)?;
        let s = self.u32s(cout)?;
        if s.iter().any(|&x| x == 0 || x > 62) {
            return Err(ParseError::Shape("requantisation shift"));
        }
        Ok(Conv {
            cin,
            cout,
            k,
            stride,
            w,
            b,
            m,
            s,
        })
    }
}

impl Model {
    pub fn parse(buf: &[u8]) -> Result<Model, ParseError> {
        let mut r = Reader { buf, pos: 0 };
        if r.take(4)? != b"SSEQ" {
            return Err(ParseError::BadMagic);
        }
        let version = r.u32()?;
        if version != 1 {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let length = r.usize()?;
        let n_classes = r.usize()?;
        let total_stride = r.usize()?;
        let n_blocks = r.usize()?;
        let in_inv_scale = r.f32()?;
        let stem = r.conv()?;
        if stem.cin != 2 {
            return Err(ParseError::Shape("stem must take I and Q"));
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut c = stem.cout;
        let mut stride = stem.stride;
        for _ in 0..n_blocks {
            let c1 = r.conv()?;
            let c2 = r.conv()?;
            let short = match r.u32()? {
                1 => Shortcut::Conv(r.conv()?),
                0 => Shortcut::Identity {
                    m: r.i32()?,
                    s: r.u32()?,
                },
                _ => return Err(ParseError::Shape("shortcut kind")),
            };
            if c1.cin != c || c2.cin != c1.cout || c2.stride != 1 {
                return Err(ParseError::Shape("block channels"));
            }
            if let Shortcut::Conv(sc) = &short {
                if sc.cin != c || sc.cout != c2.cout || sc.stride != c1.stride {
                    return Err(ParseError::Shape("shortcut channels"));
                }
            } else if c1.stride != 1 || c2.cout != c {
                return Err(ParseError::Shape("identity shortcut needs equal shapes"));
            }
            c = c2.cout;
            stride *= c1.stride;
            blocks.push(Block { c1, c2, short });
        }
        if stride != total_stride {
            return Err(ParseError::Shape("total stride"));
        }
        let (pi, po) = (r.usize()?, r.usize()?);
        if pi != 2 * c {
            return Err(ParseError::Shape(
                "projection input must be mean+max of the last block",
            ));
        }
        let proj = Linear {
            n_in: pi,
            n_out: po,
            w: r.i8s(pi * po)?,
            b: r.i32s(po)?,
            m: r.i32s(po)?,
            s: r.u32s(po)?,
            out_scale: vec![],
        };
        let (hi, ho) = (r.usize()?, r.usize()?);
        if hi != po || ho != n_classes {
            return Err(ParseError::Shape("head"));
        }
        let head = Linear {
            n_in: hi,
            n_out: ho,
            w: r.i8s(hi * ho)?,
            b: r.i32s(ho)?,
            m: vec![],
            s: vec![],
            out_scale: r.f64s(ho)?,
        };
        if r.pos != buf.len() {
            return Err(ParseError::TrailingBytes(buf.len() - r.pos));
        }
        Ok(Model {
            length,
            n_classes,
            total_stride,
            in_inv_scale,
            stem,
            blocks,
            proj,
            head,
        })
    }

    /// Multiply-accumulates per window.
    pub fn macs(&self) -> usize {
        let mut len = self.length;
        let conv = |c: &Conv, len: usize| -> (usize, usize) {
            let out = (len + 2 * (c.k / 2) - c.k) / c.stride + 1;
            (c.cin * c.cout * c.k * out, out)
        };
        let (mut total, l) = conv(&self.stem, len);
        len = l;
        for b in &self.blocks {
            let (a, l1) = conv(&b.c1, len);
            let (c, _) = conv(&b.c2, l1);
            total += a + c;
            if let Shortcut::Conv(s) = &b.short {
                total += conv(s, len).0;
            }
            len = l1;
        }
        total + self.proj.n_in * self.proj.n_out + self.head.n_in * self.head.n_out
    }

    /// Bytes of weights and biases.
    pub fn weight_bytes(&self) -> usize {
        let conv = |c: &Conv| c.w.len() + 4 * c.b.len();
        let mut n = conv(&self.stem);
        for b in &self.blocks {
            n += conv(&b.c1) + conv(&b.c2);
            if let Shortcut::Conv(s) = &b.short {
                n += conv(s);
            }
        }
        n + self.proj.w.len() + 4 * self.proj.b.len() + self.head.w.len() + 4 * self.head.b.len()
    }
}
