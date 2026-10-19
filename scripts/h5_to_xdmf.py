#!/usr/bin/env python3
"""Convert solver-native H5 dumps/checkpoints to ParaView-ready XDMF.

The solver writes raw var-major planes (rho, rhou, rhov, rhow, E) with
element-major point blocks. This script reshapes them to the global
solution-point tensor (appended under /reshaped in the same file), derives
velocity magnitude and pressure (ideal gas, gamma = 1.4), and writes one
.xmf per file (rectilinear grid, cell-centered attributes). Physical box
extents are read from the file's lx/ly/lz attrs.

Dump files (flow_*_{kind}_*.h5) are converted; checkpoint files
(checkpoint_*_{kind}_*.h5) are skipped unless passed explicitly.

Usage:
    python3 h5_to_xdmf.py runs/base/flow_f32_000500.h5       # one file
    python3 h5_to_xdmf.py runs/base/flow_f32_*.h5            # dumps of a run
    python3 h5_to_xdmf.py runs/base/checkpoint_f32_010000.h5 # one checkpoint
"""
import sys
from pathlib import Path

import h5py
import numpy as np

GAMMA = 1.4
VAR_NAMES = ["rho", "rhou", "rhov", "rhow", "E"]


def to_tensor(flat: np.ndarray, nx: int, ny: int, nz: int, p1: int) -> np.ndarray:
    """Flat var plane (element-major blocks) -> global (px, py, pz) tensor.

    Flat index = elem * p1^3 + k*p1^2 + j*p1 + i, with
    elem = ex + nx*ey + nx*ny*ez (x fastest, element order matches the CUDA
    layout). The solution-point field's x fastest, y middle, z slowest.
    """
    plane = flat.reshape(nz, ny, nx, p1, p1, p1)  # (ez, ey, ex, k, j, i)
    # Permute blocks to (ex, i, ey, j, ez, k) then flatten the point cube.
    return plane.transpose(2, 5, 1, 4, 0, 3).reshape(nx * p1, ny * p1, nz * p1)


def process(src: Path) -> None:
    with h5py.File(src, "r") as f:
        g = f["data"]
        nx = int(f.attrs["nx"])
        ny = int(f.attrs["ny"])
        nz = int(f.attrs["nz"])
        step = int(f.attrs["step"])
        t = float(f.attrs["t"])
        lx, ly, lz = (float(f.attrs[k]) for k in ("lx", "ly", "lz"))
        raw = [np.asarray(g[n], dtype=np.float32).reshape(-1) for n in VAR_NAMES]

    total_points = nx * ny * nz
    p1 = round((raw[0].size / total_points) ** (1.0 / 3.0))
    if p1 ** 3 * total_points != raw[0].size:
        sys.exit(f"{src.name}: unexpected plane length {raw[0].size}")

    fields = {}
    rho = None
    for name, plane in zip(VAR_NAMES, raw):
        fields[name] = to_tensor(plane, nx, ny, nz, p1)
    rho, u, v, w, e = (fields[name] for name in VAR_NAMES)
    r = np.maximum(rho, 1e-30)
    fields["vel"] = np.sqrt(u * u + v * v + w * w) / r
    fields["p"] = np.maximum(
        1e-7, (GAMMA - 1.0) * (e - 0.5 * (u * u + v * v + w * w) / r)
    )

    npx, npy, npz = (n * p1 for n in (nx, ny, nz))
    axes = [
        np.linspace(0.0, L, n + 1)
        for L, n in zip((lx, ly, lz), (npx, npy, npz))
    ]

    h5_out = src.with_suffix(".pv.h5")
    attrs = "\n".join(
        f"""      <Attribute Name="{n}" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{npz} {npy} {npx}" NumberType="Float" Precision="4" Format="HDF">
          {h5_out.name}:/data/{n}
        </DataItem>
      </Attribute>"""
        for n in fields
    )
    xml = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="FR_Grid" GridType="Uniform" t_step="{step}" t_time="{t:.6g}">
      <Topology TopologyType="3DRectMesh" Dimensions="{npz + 1} {npy + 1} {npx + 1}"/>
      <Geometry GeometryType="VXVYVZ">
        <DataItem Dimensions="{npx + 1}" NumberType="Float" Precision="8" Format="XML">
          {" ".join(f"{x:.10g}" for x in axes[0])}
        </DataItem>
        <DataItem Dimensions="{npy + 1}" NumberType="Float" Precision="8" Format="XML">
          {" ".join(f"{x:.10g}" for x in axes[1])}
        </DataItem>
        <DataItem Dimensions="{npz + 1}" NumberType="Float" Precision="8" Format="XML">
          {" ".join(f"{x:.10g}" for x in axes[2])}
        </DataItem>
      </Geometry>
{attrs}
    </Grid>
  </Domain>
</Xdmf>
"""
    with h5py.File(h5_out, "w") as f:
        g2 = f.create_group("data")
        for n, arr in fields.items():
            g2.create_dataset(n, data=arr.astype(np.float32))
    xmf = src.with_suffix(".xmf")
    xmf.write_text(xml)
    print(f"{src.name} (step {step}, t={t:.5g}): fields {sorted(fields)} -> {h5_out.name} + {xmf.name}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    files: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.glob("*.h5")))
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"no such path: {a}")
    auto_skip_checkpoints = len(files) > 1
    for f in files:
        if auto_skip_checkpoints and f.name.startswith("checkpoint_"):
            print(f"skip checkpoint {f.name} (resume file; pass it explicitly to view)")
            continue
        process(f)


if __name__ == "__main__":
    main()
