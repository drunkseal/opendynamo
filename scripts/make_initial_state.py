#!/usr/bin/env python3
"""Generate a solver-native initial state file for OpenDynamo.

The solver has NO hardcoded initialization kernel: every run starts from a
state file given by [physics].initial_state. For a fresh run, generate that
file with this script:

    python3 scripts/make_initial_state.py [config.toml]

It reads the working-directory config (or the path argument), evaluates the
initial condition selected by [physics].initial_condition ("channel" or
"uniform") with the bulk Mach from mach_bulk, optionally superposes the
divergence-free wall-damped trip (perturbation_amplitude / _modes / _seed),
and writes

    <output_dir>/<prefix>_<f32|f64>_000000.h5

in exactly the format the solver's loader (and save writer) uses:
root attrs nx ny nz fr_order step=0 t=0.0 lx ly lz u_real, and the
variable planes under group "data" (var-major datasets
rho/rhou/rhov/rhow/E, index = elem*64 + pt,
elem = ex + nx*ey + nx*ny*ez, pt = i + 4*j + 16*k).

The numerics mirror the former CUDA init/trip kernels operation-by-
operation in float32 (COMPUTE_REAL), so a generated channel file is
bitwise identical to what the old kernels produced. The trip uses the same
hash/mode algorithm; IEEE arithmetic replicates exactly, but the GPU's
sinf/cos may differ from numpy's at the last ulp for some arguments -
statistically the same perturbation either way. Resuming from any
checkpoint stays bitwise exact since no IC evaluation happens then.

Output compression follows [io].compression (hdf5plugin needed for blosc
variants: pip install hdf5plugin).
"""

import math
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

F32 = np.float32
F64 = np.float64
N = 4  # FR_ORDER + 1 solution points per element axis
NPTS_ELEM = N * N * N
VAR_NAMES = ["rho", "rhou", "rhov", "rhow", "E"]
CHUNK_ELEMS = 262144

# Elementwise fused multiply-add emulating the GPU's f32 FMA contraction
# (NVRTC contracts mul+add by default). Inputs are f32 (exact in f64), and
# math.fma rounds the exact product+add once; the f64 result then rounds to
# f32 with double-rounding quirks only in cases of probability ~2^-29.
_fma_ufunc = np.frompyfunc(math.fma, 3, 1)


def fma32(a, b, c):
    """f32 fused multiply-add: round_f32(exact(a*b + c))."""
    out = _fma_ufunc(
        np.asarray(a, dtype=F64).astype(F64),
        np.asarray(b, dtype=F64),
        np.asarray(c, dtype=F64),
    )
    return np.asarray(out, dtype=F64).astype(F32)


def f32(x):
    """Round a python double to the nearest float32 exactly like a C cast."""
    return F32(x)


def gauss_legendre_points():
    """Replicates src/fr_math.rs gauss_legendre_points(P=3): Newton on the
    Legendre recurrence (tol 1e-14), so the collocation points match the
    solver's c_points constant after the usual f64 -> f32 cast."""
    n = 4

    def legendre(p, x):
        if p == 0:
            return 1.0
        if p == 1:
            return x
        prev2, prev = 1.0, x
        for k in range(2, p + 1):
            kf = float(k)
            prev2, prev = prev, ((2.0 * kf - 1.0) * x * prev - (kf - 1.0) * prev2) / kf
        return prev

    roots = []
    for i in range(n):
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            f = legendre(n, x)
            df = (
                n / (x * x - 1.0) * (x * f - legendre(n - 1, x))
            )
            step = f / df
            x -= step
            if abs(step) < 1e-14:
                break
        roots.append(x)

    # The Rust version seeds roots in i-ascending order and then sorts;
    # Newton converges to the same signed roots either way.
    return sorted(roots)


def flat_elem_decomp(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    elem = np.arange(nx * ny * nz, dtype=np.int64)
    return (
        (elem % nx).astype(F32),
        ((elem // nx) % ny).astype(F32),
        (elem // (nx * ny)).astype(F32),
    )


def initial_condition(cfg):
    """Return the 5 planes of the initial condition, replicating the former
    init_flow_kernel (+ optional perturb_flow_kernel) op order in float32."""
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    lx, ly, lz = cfg["lx"], cfg["ly"], cfg["lz"]
    mach = f32(cfg["mach_bulk"])

    total = nx * ny * nz * NPTS_ELEM
    cpt = np.array(gauss_legendre_points(), dtype=F32)
    pt = np.arange(NPTS_ELEM, dtype=np.int64)
    i, j, k = pt % N, (pt // N) % N, pt // (N * N)
    eax, eay, eaz = flat_elem_decomp(cfg)

    # Positions, kernel op order with the GPU's FMA contraction:
    # y = fma(dy, mid_y, -0.5*ly)   (the -0.5*ly addend rounds to f32 first)
    def coord(length, elems, decomp, c_idx):
        mid = decomp[:, None] + f32(0.5) + f32(0.5) * cpt[c_idx][None, :]
        d = f32(length) / f32(elems)
        addend = (f32(-0.5) * f32(length)).astype(F32)
        return fma32(d, mid, addend).ravel()

    x = coord(lx, nx, eax, i)
    y = coord(ly, ny, eay, j)
    z = coord(lz, nz, eaz, k)

    # Base flow
    if cfg["initial_condition"] == "uniform":
        u_vel = np.full(total, mach, dtype=F32)
    else:  # channel: parabolic Poiseuille profile
        ywall = f32(0.5) * f32(ly)
        eta = (y / ywall).astype(F32)
        # 1.0 - eta*eta contracts to fma(-eta, eta, 1.0) on the GPU.
        inner = fma32(-eta, eta, f32(1.0))
        u_vel = (f32(1.5) * mach * inner).astype(F32)

    rho = np.full(total, f32(1.0), dtype=F32)
    # GAMMA is (COMPUTE_REAL)1.4, so p and E are computed in FLOAT:
    # p = 1.0f / 1.4f;  E = p / (1.4f - 1.0f) + ke  (all f32, fmad-safe:
    # fdiv can't fuse with the trailing add).
    p = (F32(1.0) / F32(1.4)).astype(F32)
    p_den = (F32(1.4) - F32(1.0)).astype(F32)
    ke = (f32(0.5) * rho * (u_vel * u_vel)).astype(F32)
    E = (p / p_den + ke).astype(F32)

    planes = [rho, (rho * u_vel).astype(F32), np.zeros_like(rho),
              np.zeros_like(rho), E]

    amp = f32(cfg.get("perturbation_amplitude", 0.0))
    if amp != f32(0.0):
        planes = apply_trip(planes, cfg, x, y, z)

    return planes


def hash01(s):
    """hash01: f32( sinf(s) * 43758.5453f ) - floorf(...)."""
    x = (np.sin(s.astype(F32), dtype=F32) * f32(43758.5453)).astype(F32)
    return (x - np.floor(x)).astype(F32)


def apply_trip(planes, cfg, x, y, z):
    lx, ly, lz = cfg["lx"], cfg["ly"], cfg["lz"]
    rho, m1, m2, m3, E = planes
    amp = f32(cfg["perturbation_amplitude"])
    n_modes = int(max(1, cfg.get("perturbation_modes", 16)))
    seed = f32(float(cfg.get("perturbation_seed", 7)))

    hw = f32(0.5) * f32(ly)
    eta = (y / hw).astype(F32)
    tau = (f32(1.0) - eta * eta).astype(F32)
    tau_y = ((f32(-2.0) * eta) / hw).astype(F32)

    weight = (amp / np.sqrt(F32(n_modes), dtype=F32)).astype(F32)
    two_pi = f32(6.283185307179586)

    du = np.zeros_like(rho)
    dv = np.zeros_like(rho)
    dw = np.zeros_like(rho)
    for m in range(n_modes):
        mix = f32(f32(m) * f32(0.113))
        r1 = hash01((mix * f32(1.113) + seed) - f32(3.7183))
        r2 = hash01((mix * f32(2.229) + seed) + f32(1.7429))
        r3 = hash01((mix * f32(3.717) + seed) + f32(4.8527))
        r4 = hash01((mix * f32(5.381) + seed) - f32(2.9956))
        r5 = hash01((mix * f32(7.993) + seed) + f32(6.1141))
        r6 = hash01((mix * f32(9.677) + seed) + f32(8.2437))

        # These are per-mode scalars drawn from r[0] (the hashes are
        # constant arrays here since s does not vary per point).
        mx = 1 + int(F32(r1 * f32(24.0)))
        mz = 1 + int(F32(r2 * f32(12.0)))
        kx = ((two_pi * f32(mx)) / f32(lx)).astype(F32)
        kz = ((two_pi * f32(mz)) / f32(lz)).astype(F32)
        phi = F32(two_pi * F32(r3))
        av = (f32(2.0) * r4 - f32(1.0)).astype(F32)
        ac = (f32(2.0) * r5 - f32(1.0)).astype(F32)
        wj = (f32(0.5) + r6).astype(F32)

        theta = ((kx * x + kz * z) + phi).astype(F32)
        cosT = np.cos(theta, dtype=F32).astype(F32)
        sinT = np.sin(theta, dtype=F32).astype(F32)
        kh2 = (kx * kx + kz * kz).astype(F32)
        kh = np.sqrt(kh2, dtype=F32).astype(F32)

        V = (((weight * wj) * av) * tau).astype(F32)
        Vp = (((weight * wj) * av) * tau_y).astype(F32)
        C = (((weight * wj) * ac) * tau).astype(F32)

        du += ((f32(-1.0) * (kz * C / kh)) * cosT - (kx * Vp / kh2) * sinT).astype(F32)
        dv += (V * cosT).astype(F32)
        dw += (((kx * C / kh) * cosT) - (kz * Vp / kh2) * sinT).astype(F32)

    n1 = (m1 + rho * du).astype(F32)
    n2 = (m2 + rho * dv).astype(F32)
    n3 = (m3 + rho * dw).astype(F32)
    E_new = (
        E + (f32(0.5) * ((n1 * n1 + n2 * n2 + n3 * n3) - (m1 * m1 + m2 * m2 + m3 * m3))
             / rho)
    ).astype(F32)
    return [rho, n1, n2, n3, E_new]


def load_config(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def write_state(path, planes, cfg, dtype, compression):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    total = nx * ny * nz * NPTS_ELEM
    chunk = min(total, CHUNK_ELEMS)
    gz = None
    blosc_filter = None
    if compression == "none":
        pass
    elif compression.startswith("gzip"):
        gz = int(compression[4:] or 4)
    else:  # blosc-lz4 / blosc-zstd
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            sys.exit(
                "blosc output needs hdf5plugin: pip install hdf5plugin "
                '(or set [io].compression = "gzip4")'
            )
        blosc_filter = hdf5plugin.Blosc(
            "lz4" if "lz4" in compression else "zstd",
            5,
            1,  # shuffle byte pattern (BloscSHUFFLE)
        )
    with h5py.File(path, "w") as f:
        f.attrs["nx"] = np.uint64(nx)
        f.attrs["ny"] = np.uint64(ny)
        f.attrs["nz"] = np.uint64(nz)
        f.attrs["fr_order"] = np.uint64(3)
        f.attrs["step"] = np.uint64(0)
        f.attrs["t"] = F64(0.0)
        f.attrs["lx"] = F64(np.float32(cfg["lx"]))
        f.attrs["ly"] = F64(np.float32(cfg["ly"]))
        f.attrs["lz"] = F64(np.float32(cfg["lz"]))
        # FixedAscii<3>: fixed-length 3-byte ASCII, same as the solver writer
        f.attrs.create(
            "u_real",
            np.bytes_(b"f32" if dtype == np.float32 else b"f64"),
            dtype=h5py.string_dtype("ascii", 3),
        )
        grp = f.create_group("data")
        for name, plane in zip(VAR_NAMES, planes):
            data = plane.astype(dtype)
            if blosc_filter is not None:
                grp.create_dataset(
                    name, data=data, chunks=(chunk,), **blosc_filter
                )
            else:
                grp.create_dataset(
                    name,
                    data=data,
                    chunks=(chunk,) if compression != "none" else None,
                    compression="gzip" if gz is not None else None,
                    compression_opts=gz,
                    shuffle=gz is not None,
                )
    return path


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./config.toml"
    cfg = load_config(cfg_path)
    mesh, physics, io, precision = (
        cfg["mesh"], cfg["physics"], cfg["io"], cfg["precision"],
    )
    c = {
        "nx": mesh["nx"],
        "ny": mesh["ny"],
        "nz": mesh["nz"],
        "lx": mesh.get("lx", 4.0 * math.pi),
        "ly": mesh.get("ly", 2.0),
        "lz": mesh.get("lz", (4.0 / 3.0) * math.pi),
        "mach_bulk": physics.get("mach_bulk", 0.3),
        "initial_condition": physics.get("initial_condition", "channel"),
        "perturbation_amplitude": physics.get("perturbation_amplitude", 0.0),
        "perturbation_modes": physics.get("perturbation_modes", 16),
        "perturbation_seed": physics.get("perturbation_seed", 7),
    }
    dtype = np.float32 if precision["u_real"] == "f32" else np.float64
    kind = "f32" if dtype == np.float32 else "f64"
    compression = io.get("compression", "blosc-lz4")

    planes = initial_condition(c)
    os.makedirs(io["output_dir"], exist_ok=True)
    out = os.path.join(
        io["output_dir"], f"{io['prefix']}_{kind}_000000.h5"
    )
    write_state(out, planes, c, dtype, compression)

    u = planes[1] / planes[0]
    v = planes[2] / planes[0]
    w = planes[3] / planes[0]
    print(f"Initial state written: {out}")
    print(
        f"  mesh {c['nx']}x{c['ny']}x{c['nz']}  IC {c['initial_condition']}  "
        f"mach_bulk {c['mach_bulk']}"
    )
    print(
        f"  trip amp {c['perturbation_amplitude']} "
        f"({c['perturbation_modes']} modes, seed {c['perturbation_seed']})"
    )
    print(
        f"  u range [{u.min():.6f}, {u.max():.6f}]  "
        f"|v'| max {np.abs(v).max():.6f}  |w'| max {np.abs(w).max():.6f}"
    )


if __name__ == "__main__":
    main()
