use std::fs;
use std::path::Path;

use hdf5_metno as hdf5;

use crate::constants::{N_VARS, PTS_PER_ELEM};

pub fn save_hdf5_xdmf<T: hdf5::H5Type>(
    step: usize,
    u_data: &[T],
    nx: usize,
    ny: usize,
    nz: usize,
    out_dir: &str,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    let h5_filename = format!("{}_{:05}.h5", prefix, step);
    let h5_path = Path::new(out_dir).join(&h5_filename);
    let xdmf_path = Path::new(out_dir).join(format!("{}_{:05}.xmf", prefix, step));

    // Create HDF5 file and write data
    let file = hdf5::File::create(&h5_path)?;
    let group = file.create_group("data")?;

    let total_elements = nx * ny * nz;
    let total_dofs = total_elements * PTS_PER_ELEM * N_VARS;

    // Write entire array as 1D
    let dataset = group.new_dataset::<T>().shape(total_dofs).create("rho")?;

    dataset.write(u_data)?;

    let precision = std::mem::size_of::<T>();

    // Write XDMF file pointing to the HDF5 dataset
    let xdmf_content = format!(
        r#"<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="FR_Grid" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{} {} {}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">
          0.0 0.0 0.0
        </DataItem>
        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">
          1.0 1.0 1.0
        </DataItem>
      </Geometry>
      <Attribute Name="rho" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{}" NumberType="Float" Precision="{}" Format="HDF">
          {}:/data/rho
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"#,
        nz + 1,
        ny + 1,
        nx + 1,
        total_elements,
        precision,
        h5_filename
    );

    fs::write(xdmf_path, xdmf_content)?;
    Ok(())
}
