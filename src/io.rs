use std::fs;
use std::path::Path;

use hdf5_metno as hdf5;

use crate::constants::{FR_ORDER, N_VARS, PTS_PER_ELEM};

// 1D chunk for chunked+shuffled+deflated datasets: 262144 elements (1 MB
// per chunk in f32). Chunking only exists to enable the filter pipeline;
// readers always fetch whole planes here, so chunk boundaries don't matter.
const CHUNK_ELEMS: usize = 1 << 18;

// Conservative-variable plane order of the state buffer (var-major layout):
// rho, rho*u, rho*v, rho*w, rho*e_total.
const VAR_NAMES: [&str; N_VARS] = ["rho", "rhou", "rhov", "rhow", "E"];

/// Compression spec for [`save_state_h5`]: None = contiguous unfiltered
/// dataset; the filters are constructed in the writer. All variants are
/// bitwise lossless and transparent to readers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SaveCompression {
    None,
    /// Deflate filter at the given level (+ shuffle before it).
    Gzip(u8),
    /// Blosc LZ4 (needs `hdf5plugin` on the Python reader side); clevel 9.
    BloscLz4(u8),
    /// Blosc ZStd; clevel configured by the value.
    BloscZstd(u8),
}

/// Parse the `[io] compression` setting. Supported: "none", "gzip1"..
/// "gzip9" (built-in, slowest on this CPU), "blosc-lz4" (~10x faster than
/// gzip, needs `hdf5plugin` on the Python side), "blosc-zstd" (gzip4-like
/// ratio at 2x gzip speed, same plugin caveat).
pub fn parse_compression(s: &str) -> Result<SaveCompression, Box<dyn std::error::Error>> {
    match s {
        "none" => Ok(SaveCompression::None),
        "blosc-lz4" => Ok(SaveCompression::BloscLz4(9)),
        "blosc-zstd" => Ok(SaveCompression::BloscZstd(5)),
        other => {
            for level in 1..=9u8 {
                if format!("gzip{level}") == other {
                    return Ok(SaveCompression::Gzip(level));
                }
            }
            Err(format!(
                "invalid compression '{other}': expected \"none\", \"gzip1\"..\"gzip9\", \"blosc-lz4\" or \"blosc-zstd\""
            )
            .into())
        }
    }
}

/// Solver-native dump/checkpoint: N_VARS raw planes exactly as the device
/// state buffer holds them (element-major point blocks, `pt = i + P1*j +
/// P1^2*k`), written as 1D datasets with grid metadata as attrs. Note: this
/// is NOT directly ParaView-renderable; run `scripts/h5_to_xdmf.py` on the
/// files for visualization.
pub fn save_state_h5<T: hdf5::H5Type>(
    step_label: usize,
    t: f64,
    u_real_kind: &str,
    out_dir: &str,
    base_prefix: &str,
    u_data: &[T],
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    compression: SaveCompression,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    let h5_path = Path::new(out_dir).join(format!("{}_{}_{:06}.h5", base_prefix, u_real_kind, step_label));

    let total_points = nx * ny * nz * PTS_PER_ELEM;
    if u_data.len() != N_VARS * total_points {
        return Err(format!(
            "state length {} != {} vars x {} points",
            u_data.len(),
            N_VARS,
            total_points
        )
        .into());
    }

    // Chunked + filter pipeline when compression is on: fully transparent to
    // readers (and the loader below), lossless — resume states stay exact.
    // deflate carries no byte shuffle of its own, so an explicit shuffle
    // filter pays for its ~28-30% gain on f32 noise; blosc filters shuffle
    // internally.
    let file = hdf5::File::create(&h5_path)?;
    let group = file.create_group("data")?;
    for (v, name) in VAR_NAMES.iter().enumerate() {
        let mut dataset = group.new_dataset::<T>().shape(total_points);
        match compression {
            SaveCompression::None => {}
            SaveCompression::Gzip(level) => {
                // Deflate has no byte shuffle of its own: an explicit shuffle
                // filter buys the f32 gain in compression ratio.
                dataset = dataset.chunk(CHUNK_ELEMS).shuffle().deflate(level);
            }
            SaveCompression::BloscLz4(clevel) | SaveCompression::BloscZstd(clevel) => {
                // Blosc filters byte-shuffle internally and parallelize via
                // blosc_set_nthreads (set by the solver at startup).
                use hdf5::filters::{Blosc, BloscShuffle};
                let filter = hdf5::filters::Filter::Blosc(
                    match compression {
                        SaveCompression::BloscLz4(_) => Blosc::LZ4,
                        _ => Blosc::ZStd,
                    },
                    clevel,
                    BloscShuffle::Byte,
                );
                dataset = dataset.chunk(CHUNK_ELEMS).set_filters(&[filter]);
            }
        }
        let dataset = dataset.create(*name)?;
        dataset.write_raw(&u_data[v * total_points..(v + 1) * total_points])?;
    }
    file.new_attr::<u64>().create("nx")?.write_scalar(&(nx as u64))?;
    file.new_attr::<u64>().create("ny")?.write_scalar(&(ny as u64))?;
    file.new_attr::<u64>().create("nz")?.write_scalar(&(nz as u64))?;
    file.new_attr::<u64>().create("fr_order")?.write_scalar(&(FR_ORDER as u64))?;
    file.new_attr::<u64>().create("step")?.write_scalar(&(step_label as u64))?;
    file.new_attr::<f64>().create("t")?.write_scalar(&t)?;
    file.new_attr::<f64>().create("lx")?.write_scalar(&lx)?;
    file.new_attr::<f64>().create("ly")?.write_scalar(&ly)?;
    file.new_attr::<f64>().create("lz")?.write_scalar(&lz)?;
    file.new_attr::<hdf5::types::FixedAscii<3>>()
        .create("u_real")?
        .write_scalar(&hdf5::types::FixedAscii::<3>::from_ascii(u_real_kind.as_bytes())?)?;
    Ok(())
}

/// A state loaded from a checkpoint file.
pub struct LoadedState {
    /// N_VARS planes of `total_points` f32 values (device layout). The
    /// planes always carry the validated dims of this run (`nx..nz` above).
    pub planes: Vec<Vec<f32>>,
    /// Validated grid dims (equal to the checkpoint's, after the load check).
    #[allow(dead_code)]
    pub nx: usize,
    #[allow(dead_code)]
    pub ny: usize,
    #[allow(dead_code)]
    pub nz: usize,
    /// Step label the state was saved at (continue numbering from here).
    pub step: usize,
    /// Physical time at which the state was saved.
    pub t: f64,
    /// Precision tag of the dump ("f32"/"f64"), validated by the caller.
    #[allow(dead_code)]
    pub u_real: String,
}

/// Validate and load a checkpoint written by [`save_state_h5`]. The expected
/// mesh and precision kind must match the current run exactly; var values are
/// converted to f32 host planes (exact when the file is f32, rounding when it
/// is f64 and the current pipeline is f32 — configs should normally match).
pub fn load_state_h5(
    path: &str,
    expect_nx: usize,
    expect_ny: usize,
    expect_nz: usize,
    expect_kind: &str,
) -> Result<LoadedState, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("initial_state file not found: {path}").into());
    }
    let file = hdf5::File::open(path)?;
    let nx = file.attr("nx")?.read_scalar::<u64>()? as usize;
    let ny = file.attr("ny")?.read_scalar::<u64>()? as usize;
    let nz = file.attr("nz")?.read_scalar::<u64>()? as usize;
    let fr = file.attr("fr_order")?.read_scalar::<u64>()? as usize;
    let step = file.attr("step")?.read_scalar::<u64>()? as usize;
    let t = file.attr("t")?.read_scalar::<f64>()?;
    let kind_bytes = file
        .attr("u_real")?
        .read_scalar::<hdf5::types::FixedAscii<3>>()?;
    let kind = std::str::from_utf8(kind_bytes.as_bytes())
        .map_err(|_| "bad u_real attr")?
        .to_string();

    if nx != expect_nx || ny != expect_ny || nz != expect_nz {
        return Err(format!(
            "initial_state mesh {nx}x{ny}x{nz} does not match config {expect_nx}x{expect_ny}x{expect_nz}; refusing to load"
        )
        .into());
    }
    if fr != FR_ORDER {
        return Err(format!(
            "initial_state fr_order {fr} != configured FR_ORDER {FR_ORDER}"
        )
        .into());
    }
    if kind != expect_kind {
        return Err(format!(
            "initial_state u_real '{kind}' does not match the configured pipeline ('{expect_kind}'); refusing to load"
        )
        .into());
    }

    let group = file.group("data")?;
    let mut planes = Vec::with_capacity(N_VARS);
    for name in VAR_NAMES.iter() {
        let dataset = group
            .dataset(name)
            .map_err(|_| format!("initial_state missing dataset /data/{name}"))?;
        let len: usize = dataset.shape().iter().sum();
        let expected = expect_nx * expect_ny * expect_nz * PTS_PER_ELEM;
        if len != expected {
            return Err(format!(
                "initial_state /data/{name} has {} values, expected {expected}",
                len
            )
            .into());
        }
        let plane: Vec<f32> = dataset.read_raw()?;
        planes.push(plane);
    }

    Ok(LoadedState {
        planes,
        nx,
        ny,
        nz,
        step,
        t,
        u_real: kind,
    })
}
