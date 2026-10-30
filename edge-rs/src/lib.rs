//! Integer-only inference for the spectrum-ssl-edge classifier.
//!
//! The model file is produced by `sse edge` (Python). Every operation after input quantisation is an
//! integer operation with the same rounding as the Python reference (`src/sse/edge/int8.py`), so the
//! two must agree bit for bit; `tests/parity.rs` checks that on committed fixtures.
//!
//! The crate has no dependencies. `Runtime::new` allocates every buffer once; `Runtime::run` does
//! not allocate.

pub mod dsp;
pub mod model;
pub mod runtime;

pub use model::{Model, ParseError};
pub use runtime::{Output, Runtime};
