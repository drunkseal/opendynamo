use std::fs;
use std::path::Path;

use hdf5_metno as hdf5;

use crate::constants::{FR_ORDER, N_VARS, PTS_PER_ELEM};
use crate::solver::PrecisionType;

// State element -> f32 dump storage (follows the solver's PrecisionType).
impl<T: PrecisionType> ToF32Val for T {
    fn to_f32_val(self) -> f32 {
        self.to_f32()
    }
}

pub trait ToF32Val: Copy {
    fn to_f32_val(self) -> f32;
}

// Conservative-variable plane order of the state buffer (var-major layout):
// rho, rho*u, rho*v, rho*w, rho*e_total. Stored under the conserved names;
// derived velocity magnitude `vel` and pressure `p` are added for visuals.
const VAR_NAMES: [&str; N_VARS] = ["rho", "rhou", "rhov", "rhow", "E"];

/// Save one dump: per-variable HDF5 datasets on the full solution-point grid
/// (tensor-ordered, x fastest) plus an XDMF wrapping them into a structured
/// mesh ParaView can ingest directly. Also adds host-computed velocity
/// magnitude and pressure (ideal gas, gamma = 1.4) fields.
///
/// `u_data` is the var-major device state: N_VARS contiguous planes of
/// `total_points` values, each plane indexed `elem * PTS_PER_ELEM + pt` with
/// `pt = i + P1*j + P1^2*k` (element-major blocks).
pub fn save_hdf5_xdmf<T: hdf5::H5Type + ToF32Val>(
    step: usize,
    u_data: &[T],
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    out_dir: &str,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    let h5_filename = format!("{}_{:05}.h5", prefix, step);
    let h5_path = Path::new(out_dir).join(&h5_filename);
    let xdmf_path = Path::new(out_dir).join(format!("{}_{:05}.xmf", prefix, step));

    let p1 = FR_ORDER + 1;
    let total_elements = nx * ny * nz;
    let total_points = total_elements * PTS_PER_ELEM;
    let (npx, npy, npz) = (nx * p1, ny * p1, nz * p1);

    if u_data.len() != N_VARS * total_points {
        return Err(format!(
            "state length {} != {} vars x {} points",
            u_data.len(),
            N_VARS,
            total_points
        )
        .into());
    }

    // Host copy per variable, reordered from element-major blocks to the
    // global point tensor (x fastest, z slowest) so the datasets are plain
    // structured grids.
    let mut vars: Vec<Vec<f32>> = Vec::with_capacity(N_VARS);
    for v in 0..N_VARS {
        let plane = &u_data[v * total_points..(v + 1) * total_points];
        let mut out = vec![0.0f32; total_points];
        for ez in 0..nz {
            for ey in 0..ny {
                for ex in 0..nx {
                    let elem = ex + nx * (ey + ny * ez);
                    let src_base = elem * PTS_PER_ELEM;
                    let dst_base = (ez * p1) * (npx * npy) + (ey * p1) * npx + ex * p1;
                    for k in 0..p1 {
                        for j in 0..p1 {
                            for i in 0..p1 {
                                let dst = dst_base + k * (npx * npy) + j * npx + i;
                                out[dst] = plane[src_base + k * p1 * p1 + j * p1 + i].to_f32_val();
                            }
                        }
                    }
                }
            }
        }
        vars.push(out);
    }

    // Derived fields for direct visualization: velocity magnitude and
    // pressure (ideal gas, same conventions as the device kernels).
    let mut names: Vec<&str> = VAR_NAMES.to_vec();
    {
        let (rho, rhou, rhov, rhow, e) = (&vars[0], &vars[1], &vars[2], &vars[3], &vars[4]);
        let mut vel = vec![0.0f32; total_points];
        let mut p = vec![0.0f32; total_points];
        for n in 0..total_points {
            let u = rhou[n] / rho[n];
            let v = rhov[n] / rho[n];
            let w = rhow[n] / rho[n];
            vel[n] = (u * u + v * v + w * w).sqrt();
            p[n] = f32::max(
                1.0e-7,
                0.4 * (e[n] - 0.5 * (rhou[n] * u + rhov[n] * v + rhow[n] * w)),
            );
        }
        vars.push(vel);
        vars.push(p);
        names.push("vel");
        names.push("p");
    }

    // Create HDF5 file and write data
    let file = hdf5::File::create(&h5_path)?;
    let group = file.create_group("data")?;
    for (name, arr) in names.iter().zip(vars.iter()) {
        let dataset = group
            .new_dataset::<f32>()
            .shape((npz, npy, npx))
            .create(*name)?;
        dataset.write_raw(arr.as_slice())?;
    }

    // Rectilinear axis coordinates for the XDMF geometry (VXVYVZ: explicit
    // and unambiguous). Solution points sit at half-cell-ish spacing dx = L/N.
    let x_axis: Vec<f64> = (0..=npx).map(|i| i as f64 * (lx / npx as f64)).collect();
    let y_axis: Vec<f64> = (0..=npy).map(|i| i as f64 * (ly / npy as f64)).collect();
    let z_axis: Vec<f64> = (0..=npz).map(|i| i as f64 * (lz / npz as f64)).collect();
    for (name, axis) in [("xc", &x_axis), ("yc", &y_axis), ("zc", &z_axis)] {
        let dataset = group.new_dataset::<f64>().shape(axis.len()).create(name)?;
        dataset.write_raw(axis.as_slice())?;
    }

    let precision = 4u32; // f32

    // Write XDMF file pointing to the HDF5 datasets. ParaView expects node
    // counts for the rectilinear topology (cells = nodes - 1 on every axis).
    let mut attributes = String::new();
    for name in &names {
        attributes.push_str(&format!(
            r#"      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{npz} {npy} {npx}" NumberType="Float" Precision="{precision}" Format="HDF">
          {h5_filename}:/data/{name}
        </DataItem>
      </Attribute>
"#
        ));
    }

    let xdmf_content = format!(
        r#"<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="FR_Grid" GridType="Uniform">
      <Topology TopologyType="3DRectMesh" Dimensions="{tpz} {tpy} {tpx}"/>
      <Geometry GeometryType="VXVYVZ">
        <DataItem Dimensions="{tpx}" NumberType="Float" Precision="8" Format="HDF">
          {h5_filename}:/data/xc
        </DataItem>
        <DataItem Dimensions="{tpy}" NumberType="Float" Precision="8" Format="HDF">
          {h5_filename}:/data/yc
        </DataItem>
        <DataItem Dimensions="{tpz}" NumberType="Float" Precision="8" Format="HDF">
          {h5_filename}:/data/zc
        </DataItem>
      </Geometry>
{attributes}    </Grid>
  </Domain>
</Xdmf>
"#,
        tpz = npz + 1,
        tpy = npy + 1,
        tpx = npx + 1,
        attributes = attributes,
    );

    fs::write(xdmf_path, xdmf_content)?;
    Ok(())
}
