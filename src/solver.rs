use cudarc::driver::PushKernelArg;

use crate::config::{FluxScheme, InitialCondition, ViscosityModel, YBoundary};
use crate::constants::{
    FACES_PER_BLOCK, FR_ORDER, N_VARS, PTS_PER_ELEM, PTS_PER_FACE,
};

pub trait PrecisionType:
    cudarc::driver::DeviceRepr
    + cudarc::driver::ValidAsZeroBits
    + Send
    + Sync
    + Copy
    + 'static
    + std::ops::Mul<Output = Self>
    + std::fmt::Debug
    + hdf5_metno::H5Type
{
    fn c_type() -> &'static str;
    fn from_f64(val: f64) -> Self;
    fn to_f32(self) -> f32;
}

impl PrecisionType for f32 {
    fn c_type() -> &'static str {
        "float"
    }
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl PrecisionType for f64 {
    fn c_type() -> &'static str {
        "double"
    }
    fn from_f64(val: f64) -> Self {
        val
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}

// Precision tag for dump/checkpoint attrs ("f32"/"f64"), independent of the
// NVRTC c_type names.
fn precision_tag<U>() -> &'static str {
    if std::mem::size_of::<U>() == 4 {
        "f32"
    } else {
        "f64"
    }
}

// ---------------------------------------------------------------------------
// Plain data structs (no methods): everything the setup and launch helpers
// exchange.
// ---------------------------------------------------------------------------

struct KernelHandles {
    init_flow: cudarc::driver::CudaFunction,
    perturb_flow: cudarc::driver::CudaFunction,
    rhs_volume: cudarc::driver::CudaFunction,
    face_x: cudarc::driver::CudaFunction,
    face_y: cudarc::driver::CudaFunction,
    face_y_wall: cudarc::driver::CudaFunction,
    face_z: cudarc::driver::CudaFunction,
    rk_stage: cudarc::driver::CudaFunction,
    rk_stage1: cudarc::driver::CudaFunction,
}

// Launch configs derived from the mesh: `volume` covers the point-wise kernels
// (init + fused RHS), `add` the element-point update of the RK stages.
struct LaunchConfigs {
    volume: cudarc::driver::LaunchConfig,
    add: cudarc::driver::LaunchConfig,
    face_x: cudarc::driver::LaunchConfig,
    face_y: cudarc::driver::LaunchConfig,
    face_y_wall: Option<cudarc::driver::LaunchConfig>,
    face_z: cudarc::driver::LaunchConfig,
}

// Grid sizes plus the physical box extents (already converted to COMP, since
// every kernel launch passes lx/ly/lz as COMPUTE_REAL arguments).
struct MeshDims<COMP> {
    nx: usize,
    ny: usize,
    nz: usize,
    lx: COMP,
    ly: COMP,
    lz: COMP,
    num_elements: usize,
    total_points: usize,
    state_size: usize,
}

// Face correction arrays: written by the face kernels, read by the RK kernels.
struct FaceCorrections<COMP> {
    j_x_left: cudarc::driver::CudaSlice<COMP>,
    j_x_right: cudarc::driver::CudaSlice<COMP>,
    j_y_left: cudarc::driver::CudaSlice<COMP>,
    j_y_right: cudarc::driver::CudaSlice<COMP>,
    j_z_left: cudarc::driver::CudaSlice<COMP>,
    j_z_right: cudarc::driver::CudaSlice<COMP>,
}

// Gradient boundary slices: pre-extrapolated Q values for the face kernels,
// written by compute_rhs_volume_kernel (the global Q array is gone).
struct GradSlices<COMP> {
    x_left: cudarc::driver::CudaSlice<COMP>,
    x_right: cudarc::driver::CudaSlice<COMP>,
    y_left: cudarc::driver::CudaSlice<COMP>,
    y_right: cudarc::driver::CudaSlice<COMP>,
    z_left: cudarc::driver::CudaSlice<COMP>,
    z_right: cudarc::driver::CudaSlice<COMP>,
}

struct DeviceBuffers<U, RHS, COMP> {
    u: cudarc::driver::CudaSlice<U>,
    u_prev: cudarc::driver::CudaSlice<U>,
    rhs: cudarc::driver::CudaSlice<RHS>,
    face_j: FaceCorrections<COMP>,
    grads: GradSlices<COMP>,
}

// One RK stage's coefficients; stage 1 has no b term (it stores the original
// state instead of blending with it).
struct RkStageCoeffs<COMP> {
    a: COMP,
    b: Option<COMP>,
    c: COMP,
}

// ---------------------------------------------------------------------------
// Top-level pipeline.
// ---------------------------------------------------------------------------

pub fn run_solver_generic<
    U: PrecisionType,
    Q: PrecisionType,
    RHS: PrecisionType,
    COMP: PrecisionType,
>(
    config: crate::config::OpenDynamoConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = compile_module::<U, Q, RHS, COMP>(&ctx, &config)?;
    let kernels = load_kernels(&module)?;
    setup_fr_operators::<COMP>(&module, &stream)?;

    let mesh = mesh_dims::<COMP>(&config.mesh);
    println!(
        "Allocating device memory for {} elements, total DOFs: {}",
        mesh.num_elements, mesh.total_points
    );
    let mut buffers = allocate_buffers::<U, RHS, COMP>(&stream, &mesh)?;

    let io_stream = ctx.new_stream()?;

    // Reusable pinned staging buffers for every device→host save (IC, loop
    // steps, final). Buffers are loaned to save tasks and returned via the
    // channel when the HDF5 write finishes; the final task's buffer is sent
    // to `final_tx` so the end-of-run state report can reuse it instead of
    // allocating and copying the whole state again.
    let (pool_tx, pool_rx) = std::sync::mpsc::channel::<cudarc::driver::PinnedHostSlice<U>>();
    for _ in 0..3 {
        let buf = unsafe { ctx.alloc_pinned_with_flags::<U>(mesh.state_size, 0) }?;
        pool_tx
            .send(buf)
            .map_err(|_| "pinned buffer pool send failed")?;
    }
    let (final_tx, final_rx) = std::sync::mpsc::channel::<cudarc::driver::PinnedHostSlice<U>>();

    let mach_bulk = COMP::from_f64(config.physics.mach_bulk as f64);
    let wall_y = config.physics.y_boundary == crate::config::YBoundary::Wall;
    let cfgs = launch_configs(&mesh, wall_y);

    // Resume from a checkpoint when configured; otherwise initialize the
    // analytic initial condition (plus the optional trip). Resume is exact:
    // for the explicit RK scheme u fully defines the state at a step
    // boundary, and the checkpoint carries its own trip perturbation.
    use crate::io::LoadedState;
    let resume = match config.physics.initial_state.as_deref() {
        Some(path) => {
            println!("Resuming from {path} ...");
            let st: LoadedState = crate::io::load_state_h5(
                path,
                mesh.nx,
                mesh.ny,
                mesh.nz,
                precision_tag::<U>(),
            )?;
            // Host planes (exact for f32 pipelines; f64 configs convert via
            // f32 values as stored in the checkpoint).
            let mut host = Vec::with_capacity(mesh.state_size);
            for plane in &st.planes {
                host.extend(plane.iter().map(|x| U::from_f64(*x as f64)));
            }
            stream.memcpy_htod(&host, &mut buffers.u)?;
            println!(
                "Resumed at step {} (t = {:.6}), skipping init and trip kernels.",
                st.step, st.t
            );
            Some((st.step, st.t))
        }
        None => {
            println!("Initializing flow...");
            launch_init_flow(
                &stream,
                &kernels.init_flow,
                cfgs.volume,
                &mut buffers.u,
                &mesh,
                mach_bulk,
            )?;

            let perturb_amp = config.physics.perturbation_amplitude;
            if perturb_amp != 0.0 {
                let n_modes = config.physics.perturbation_modes.max(1) as i32;
                let seed = COMP::from_f64(config.physics.perturbation_seed as f64);
                println!("Tripping with {} divergence-free modes, amplitude {}...", n_modes, perturb_amp);
                launch_perturb_flow(
                    &stream,
                    &kernels.perturb_flow,
                    cfgs.volume,
                    &mut buffers.u,
                    &mesh,
                    COMP::from_f64(perturb_amp as f64),
                    n_modes,
                    seed,
                )?;
            }
            None
        }
    };
    let base_t = resume.as_ref().map(|r| r.1).unwrap_or(0.0);
    // Labels count completed RK3 steps: fresh start = 0, resumed start = the
    // checkpoint's own label (= steps completed at the time it was saved).
    let base_step_count = resume.as_ref().map(|r| r.0).unwrap_or(0);
    let compression = crate::io::parse_compression(&config.io.compression)?;
    if matches!(
        compression,
        crate::io::SaveCompression::BloscLz4(_) | crate::io::SaveCompression::BloscZstd(_)
    ) {
        // Blosc compresses in parallel; leave the rest of the cores to the
        // solver and OS.
        hdf5_metno::filters::blosc_set_nthreads(4);
    }

    // SAVE INITIAL CONDITION AS STEP 99999
    let mut h_u = pool_rx.recv()?;
    stream.memcpy_dtoh(&buffers.u, &mut h_u)?;
    let out_dir = config.io.output_dir.clone();
    let prefix = config.io.prefix.clone();
    let slice = h_u.as_slice().expect("pinned host buffer");
    // Label semantics (see spawn_io_task): labels count completed RK3 steps,
    // so the startup state (fresh IC or the resumed checkpoint itself) is
    // label = completed-so-far (0 fresh / S resumed), at t = base_t.
    crate::io::save_state_h5(
        base_step_count,
        base_t,
        precision_tag::<U>(),
        &out_dir,
        &prefix,
        slice,
        mesh.nx,
        mesh.ny,
        mesh.nz,
        mesh.lx.to_f32() as f64,
        mesh.ly.to_f32() as f64,
        mesh.lz.to_f32() as f64,
        compression,
    )?;
    pool_tx
        .send(h_u)
        .map_err(|_| "pinned buffer pool send failed")?;

    let dt = COMP::from_f64(config.solver.dt as f64);
    let n_steps = config.solver.n_steps;
    let mut io_threads = vec![];

    println!("Starting RK3 explicit time integration...");
    let start_time = std::time::Instant::now();
    for step in 0..n_steps {
        // Stage 1: advance u into u_prev, then swap so u_prev holds the
        // original state that stages 2/3 blend against.
        launch_rhs_stage(&stream, &kernels, &mut buffers, &mesh, &cfgs)?;
        launch_rk_stage(
            &stream,
            &kernels.rk_stage1,
            cfgs.add,
            &mut buffers.u_prev,
            &buffers.u,
            &buffers.rhs,
            &buffers.face_j,
            &mesh,
            RkStageCoeffs {
                a: COMP::from_f64(1.0),
                b: None,
                c: dt,
            },
        )?;
        std::mem::swap(&mut buffers.u, &mut buffers.u_prev);

        // Stage 2
        launch_rhs_stage(&stream, &kernels, &mut buffers, &mesh, &cfgs)?;
        launch_rk_stage(
            &stream,
            &kernels.rk_stage,
            cfgs.add,
            &mut buffers.u,
            &buffers.u_prev,
            &buffers.rhs,
            &buffers.face_j,
            &mesh,
            RkStageCoeffs {
                a: COMP::from_f64(0.75),
                b: Some(COMP::from_f64(0.25)),
                c: COMP::from_f64(0.25) * dt,
            },
        )?;

        // Stage 3
        launch_rhs_stage(&stream, &kernels, &mut buffers, &mesh, &cfgs)?;
        launch_rk_stage(
            &stream,
            &kernels.rk_stage,
            cfgs.add,
            &mut buffers.u,
            &buffers.u_prev,
            &buffers.rhs,
            &buffers.face_j,
            &mesh,
            RkStageCoeffs {
                a: COMP::from_f64(1.0 / 3.0),
                b: Some(COMP::from_f64(2.0 / 3.0)),
                c: COMP::from_f64(2.0 / 3.0) * dt,
            },
        )?;

        if step.is_multiple_of(config.io.save_freq) || step == n_steps - 1 {
            let buf = pool_rx.recv()?;
            let done = if step == n_steps - 1 {
                final_tx.clone()
            } else {
                pool_tx.clone()
            };
            io_threads.push(spawn_io_task(
                &stream, &io_stream, &buffers.u, &mesh, step, resume, compression, &config, buf,
                done,
            )?);
        }
    }

    // Wait for all IO streams
    io_stream.synchronize()?;

    // Join all background IO threads
    for handle in io_threads {
        if let Err(panic) = handle.join() {
            eprintln!("IO thread panicked: {panic:?}");
        }
    }

    stream.synchronize()?;

    let elapsed = start_time.elapsed();
    println!(
        "Simulation finished in {:.3} seconds.",
        elapsed.as_secs_f64()
    );
    println!(
        "Time per step: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0 / (n_steps as f64)
    );

    // Reuse the final save's staging buffer (it already holds this state on
    // host); fall back to a fresh copy only if no final save was made.
    let h_u_final = match final_rx.try_recv() {
        Ok(buf) => buf,
        Err(_) => {
            let mut buf = unsafe { ctx.alloc_pinned_with_flags::<U>(mesh.state_size, 0) }?;
            stream.memcpy_dtoh(&buffers.u, &mut buf)?;
            buf
        }
    };
    stream.synchronize()?;

    let slice = h_u_final.as_slice().expect("pinned host buffer");
    physical_state_report(slice, mesh.total_points);

    // Release the pooled staging buffers (all save tasks have joined).
    for _ in pool_rx.try_iter() {}

    Ok(())
}

// ---------------------------------------------------------------------------
// Setup: NVRTC options, module compilation, kernel loading, FR operators.
// ---------------------------------------------------------------------------

fn nvrtc_options<U: PrecisionType, Q: PrecisionType, RHS: PrecisionType, COMP: PrecisionType>(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    config: &crate::config::OpenDynamoConfig,
) -> cudarc::nvrtc::CompileOptions {
    let mut opts = cudarc::nvrtc::CompileOptions::default();
    if config.solver.fast_math {
        opts.options.push("--use_fast_math".to_string());
    }
    // Target the actual device arch instead of NVRTC's default (compute_52).
    // Arch-specific PTX lets the JIT apply sm_XX scheduling (PyFR reports
    // runtime codegen targeting the current GPU as a major win).
    let (major, minor) = ctx
        .compute_capability()
        .expect("failed to query compute capability");
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    opts.arch = Some(arch);
    opts.options.push(format!("-DU_REAL={}", U::c_type()));
    opts.options.push(format!("-DQ_REAL={}", Q::c_type()));
    opts.options.push(format!("-DRHS_REAL={}", RHS::c_type()));
    opts.options
        .push(format!("-DCOMPUTE_REAL={}", COMP::c_type()));
    opts.options.push(format!("-DFR_ORDER={FR_ORDER}"));
    opts.options.push(format!(
        "-DINIT_CHANNEL={}",
        match config.physics.initial_condition {
            InitialCondition::Channel => 1,
            InitialCondition::Uniform => 0,
        }
    ));
    opts.options.push(format!(
        "-DBC_Y_WALL={}",
        match config.physics.y_boundary {
            YBoundary::Wall => 1,
            YBoundary::Periodic => 0,
        }
    ));

    if config.physics.viscosity_model == ViscosityModel::Sutherland {
        opts.options.push("-DVISCOSITY_SUTHERLAND".to_string());
    }
    opts.options
        .push(format!("-DMU_REF={}", config.physics.mu_ref));
    opts.options
        .push(format!("-DT_REF={}", config.physics.t_ref));
    opts.options
        .push(format!("-DS_SUTH={}", config.physics.s_suth));

    opts.options.push(
        match config.solver.flux_scheme {
            FluxScheme::Slau2 => "-DFLUX_SCHEME_SLAU2",
            FluxScheme::AusmPlus => "-DFLUX_SCHEME_AUSMPLUS",
            FluxScheme::Rusanov => "-DFLUX_SCHEME_RUSANOV",
        }
        .to_string(),
    );

    opts
}

fn compile_module<U: PrecisionType, Q: PrecisionType, RHS: PrecisionType, COMP: PrecisionType>(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    config: &crate::config::OpenDynamoConfig,
) -> Result<std::sync::Arc<cudarc::driver::CudaModule>, Box<dyn std::error::Error>> {
    let opts = nvrtc_options::<U, Q, RHS, COMP>(ctx, config);
    let ptx_source = format!(
        "{}\n{}\n{}\n{}\n{}",
        include_str!("cuda/constants.cuh"),
        include_str!("cuda/device_functions.cuh"),
        include_str!("cuda/init.cu"),
        include_str!("cuda/rhs.cu"),
        include_str!("cuda/rk.cu")
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(ptx_source, opts)
        .map_err(|e| format!("NVRTC compile error: {e:?}"))?;
    Ok(ctx.load_module(ptx)?)
}

fn load_kernels(
    module: &std::sync::Arc<cudarc::driver::CudaModule>,
) -> Result<KernelHandles, Box<dyn std::error::Error>> {
    Ok(KernelHandles {
        init_flow: module.load_function("init_flow_kernel")?,
        perturb_flow: module.load_function("perturb_flow_kernel")?,
        rhs_volume: module.load_function("compute_rhs_volume_kernel")?,
        face_x: module.load_function("compute_face_flux_x_kernel")?,
        face_y: module.load_function("compute_face_flux_y_kernel")?,
        face_y_wall: module.load_function("compute_face_flux_y_wall_kernel")?,
        face_z: module.load_function("compute_face_flux_z_kernel")?,
        rk_stage: module.load_function("rk_stage_kernel")?,
        rk_stage1: module.load_function("rk_stage1_kernel")?,
    })
}

// Generate the FR operators and upload them to device __constant__ memory.
fn setup_fr_operators<COMP: PrecisionType>(
    module: &std::sync::Arc<cudarc::driver::CudaModule>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating FR operators for P={FR_ORDER}...");
    let ops = crate::fr_math::generate_fr_operators(FR_ORDER);

    let points_comp: Vec<COMP> = ops.points.iter().map(|&x| COMP::from_f64(x)).collect();
    let extrap_left_comp: Vec<COMP> = ops.extrap_left.iter().map(|&x| COMP::from_f64(x)).collect();
    let extrap_right_comp: Vec<COMP> = ops
        .extrap_right
        .iter()
        .map(|&x| COMP::from_f64(x))
        .collect();
    let g_left_prime_comp: Vec<COMP> = ops
        .g_left_prime
        .iter()
        .map(|&x| COMP::from_f64(x))
        .collect();
    let g_right_prime_comp: Vec<COMP> = ops
        .g_right_prime
        .iter()
        .map(|&x| COMP::from_f64(x))
        .collect();
    let d_matrix_comp: Vec<COMP> = ops.d_matrix.iter().map(|&x| COMP::from_f64(x)).collect();

    let mut c_points = module.get_global("c_points", stream)?;
    let mut c_extrap_left = module.get_global("c_extrap_left", stream)?;
    let mut c_extrap_right = module.get_global("c_extrap_right", stream)?;
    let mut c_g_left_prime = module.get_global("c_g_left_prime", stream)?;
    let mut c_g_right_prime = module.get_global("c_g_right_prime", stream)?;
    let mut c_d_matrix = module.get_global("c_d_matrix", stream)?;

    unsafe {
        let u8_slice = |data: &[COMP]| {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        stream.memcpy_htod(u8_slice(&points_comp), &mut c_points)?;
        stream.memcpy_htod(u8_slice(&extrap_left_comp), &mut c_extrap_left)?;
        stream.memcpy_htod(u8_slice(&extrap_right_comp), &mut c_extrap_right)?;
        stream.memcpy_htod(u8_slice(&g_left_prime_comp), &mut c_g_left_prime)?;
        stream.memcpy_htod(u8_slice(&g_right_prime_comp), &mut c_g_right_prime)?;
        stream.memcpy_htod(u8_slice(&d_matrix_comp), &mut c_d_matrix)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Mesh, launch configs, device allocations.
// ---------------------------------------------------------------------------

fn mesh_dims<COMP: PrecisionType>(mesh_cfg: &crate::config::MeshConfig) -> MeshDims<COMP> {
    let num_elements = mesh_cfg.nx * mesh_cfg.ny * mesh_cfg.nz;
    let total_points = num_elements * PTS_PER_ELEM;
    MeshDims {
        nx: mesh_cfg.nx,
        ny: mesh_cfg.ny,
        nz: mesh_cfg.nz,
        lx: COMP::from_f64(mesh_cfg.lx as f64),
        ly: COMP::from_f64(mesh_cfg.ly as f64),
        lz: COMP::from_f64(mesh_cfg.lz as f64),
        num_elements,
        total_points,
        state_size: total_points * N_VARS,
    }
}

fn launch_configs<COMP>(mesh: &MeshDims<COMP>, wall_y: bool) -> LaunchConfigs {
    let block_dim = (PTS_PER_ELEM as u32, 1, 1);
    LaunchConfigs {
        volume: cudarc::driver::LaunchConfig {
            grid_dim: (mesh.num_elements as u32, 1, 1),
            block_dim,
            shared_mem_bytes: 0,
        },
        // RK kernels process 4 points per thread (vectorized loads).
        add: cudarc::driver::LaunchConfig::for_num_elems((mesh.state_size / 4) as u32),
        face_x: cudarc::driver::LaunchConfig {
            grid_dim: (mesh.num_elements.div_ceil(FACES_PER_BLOCK) as u32, 1, 1),
            block_dim,
            shared_mem_bytes: 0,
        },
        face_y: cudarc::driver::LaunchConfig {
            grid_dim: (
                (mesh.nx * mesh.ny.div_ceil(FACES_PER_BLOCK) * mesh.nz) as u32,
                1,
                1,
            ),
            block_dim,
            shared_mem_bytes: 0,
        },
        face_z: cudarc::driver::LaunchConfig {
            grid_dim: (
                (mesh.nx * mesh.ny * mesh.nz.div_ceil(FACES_PER_BLOCK)) as u32,
                1,
                1,
            ),
            block_dim,
            shared_mem_bytes: 0,
        },
        // Wall faces get a dedicated launch (only when wall BC is active).
        face_y_wall: (wall_y as usize != 0).then(|| cudarc::driver::LaunchConfig {
            grid_dim: ((2 * mesh.nx * mesh.nz) as u32, 1, 1),
            block_dim: (PTS_PER_FACE as u32, 1, 1),
            shared_mem_bytes: 0,
        }),
    }
}

fn allocate_buffers<U: PrecisionType, RHS: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    mesh: &MeshDims<COMP>,
) -> Result<DeviceBuffers<U, RHS, COMP>, Box<dyn std::error::Error>> {
    let face_array_size = mesh.num_elements * PTS_PER_FACE * N_VARS;
    let slice_array_size = mesh.num_elements * PTS_PER_FACE * 12;
    Ok(DeviceBuffers {
        u: stream.alloc_zeros::<U>(mesh.state_size)?,
        u_prev: stream.alloc_zeros::<U>(mesh.state_size)?,
        rhs: stream.alloc_zeros::<RHS>(mesh.state_size)?,
        face_j: FaceCorrections {
            j_x_left: stream.alloc_zeros::<COMP>(face_array_size)?,
            j_x_right: stream.alloc_zeros::<COMP>(face_array_size)?,
            j_y_left: stream.alloc_zeros::<COMP>(face_array_size)?,
            j_y_right: stream.alloc_zeros::<COMP>(face_array_size)?,
            j_z_left: stream.alloc_zeros::<COMP>(face_array_size)?,
            j_z_right: stream.alloc_zeros::<COMP>(face_array_size)?,
        },
        grads: GradSlices {
            x_left: stream.alloc_zeros::<COMP>(slice_array_size)?,
            x_right: stream.alloc_zeros::<COMP>(slice_array_size)?,
            y_left: stream.alloc_zeros::<COMP>(slice_array_size)?,
            y_right: stream.alloc_zeros::<COMP>(slice_array_size)?,
            z_left: stream.alloc_zeros::<COMP>(slice_array_size)?,
            z_right: stream.alloc_zeros::<COMP>(slice_array_size)?,
        },
    })
}

// ---------------------------------------------------------------------------
// Kernel launches.
// ---------------------------------------------------------------------------

fn launch_init_flow<U: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    cfg: cudarc::driver::LaunchConfig,
    d_u: &mut cudarc::driver::CudaSlice<U>,
    mesh: &MeshDims<COMP>,
    mach_bulk: COMP,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx_i32 = mesh.nx as i32;
    let ny_i32 = mesh.ny as i32;
    let nz_i32 = mesh.nz as i32;

    let mut args = stream.launch_builder(kernel);
    args.arg(d_u)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mach_bulk)
        .arg(&mesh.ly);
    unsafe { args.launch(cfg)? };
    Ok(())
}

// Div-free wall-damped trip: amplitude in velocity units, hashed mode mix
// seeded on config (same config + binary => identical perturbation).
#[allow(clippy::too_many_arguments)]
fn launch_perturb_flow<U: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    cfg: cudarc::driver::LaunchConfig,
    d_u: &mut cudarc::driver::CudaSlice<U>,
    mesh: &MeshDims<COMP>,
    amp: COMP,
    n_modes: i32,
    seed: COMP,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx_i32 = mesh.nx as i32;
    let ny_i32 = mesh.ny as i32;
    let nz_i32 = mesh.nz as i32;

    let mut args = stream.launch_builder(kernel);
    args.arg(d_u)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.lx)
        .arg(&mesh.ly)
        .arg(&mesh.lz)
        .arg(&amp)
        .arg(&n_modes)
        .arg(&seed);
    unsafe { args.launch(cfg)? };
    Ok(())
}

// One RK stage's PDE work: the fused gradient+volume kernel (writes the
// gradient boundary slices), then the three face directions (read slices,
// write the j arrays consumed by the RK kernels).
fn launch_rhs_stage<U: PrecisionType, RHS: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &KernelHandles,
    buffers: &mut DeviceBuffers<U, RHS, COMP>,
    mesh: &MeshDims<COMP>,
    cfgs: &LaunchConfigs,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx_i32 = mesh.nx as i32;
    let ny_i32 = mesh.ny as i32;
    let nz_i32 = mesh.nz as i32;

    let mut args_v = stream.launch_builder(&kernels.rhs_volume);
    args_v
        .arg(&buffers.u)
        .arg(&mut buffers.rhs)
        .arg(&buffers.grads.x_left)
        .arg(&buffers.grads.x_right)
        .arg(&buffers.grads.y_left)
        .arg(&buffers.grads.y_right)
        .arg(&buffers.grads.z_left)
        .arg(&buffers.grads.z_right)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.lx)
        .arg(&mesh.ly)
        .arg(&mesh.lz);
    unsafe { args_v.launch(cfgs.volume)? };

    let mut args_x = stream.launch_builder(&kernels.face_x);
    args_x
        .arg(&buffers.u)
        .arg(&buffers.face_j.j_x_left)
        .arg(&buffers.face_j.j_x_right)
        .arg(&buffers.grads.x_left)
        .arg(&buffers.grads.x_right)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.lx);
    unsafe { args_x.launch(cfgs.face_x)? };

    let mut args_y = stream.launch_builder(&kernels.face_y);
    args_y
        .arg(&buffers.u)
        .arg(&buffers.face_j.j_y_left)
        .arg(&buffers.face_j.j_y_right)
        .arg(&buffers.grads.y_left)
        .arg(&buffers.grads.y_right)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.ly);
    unsafe { args_y.launch(cfgs.face_y)? };

    if let Some(cfg_wall) = &cfgs.face_y_wall {
        let mut args_w = stream.launch_builder(&kernels.face_y_wall);
        args_w
            .arg(&buffers.u)
            .arg(&buffers.face_j.j_y_left)
            .arg(&buffers.face_j.j_y_right)
            .arg(&buffers.grads.y_left)
            .arg(&buffers.grads.y_right)
            .arg(&nx_i32)
            .arg(&ny_i32)
            .arg(&nz_i32)
            .arg(&mesh.ly);
        unsafe { args_w.launch(*cfg_wall)? };
    }

    let mut args_z = stream.launch_builder(&kernels.face_z);
    args_z
        .arg(&buffers.u)
        .arg(&buffers.face_j.j_z_left)
        .arg(&buffers.face_j.j_z_right)
        .arg(&buffers.grads.z_left)
        .arg(&buffers.grads.z_right)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.lz);
    unsafe { args_z.launch(cfgs.face_z)? };

    Ok(())
}

// One RK stage update: dst = a*dst + b*src + c*(RHS + face j corrections).
// Stage 1 has no b term (dst is the pristine copy of the state).
#[allow(clippy::too_many_arguments)]
fn launch_rk_stage<U: PrecisionType, RHS: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    cfg: cudarc::driver::LaunchConfig,
    dst: &mut cudarc::driver::CudaSlice<U>,
    src: &cudarc::driver::CudaSlice<U>,
    rhs: &cudarc::driver::CudaSlice<RHS>,
    face_j: &FaceCorrections<COMP>,
    mesh: &MeshDims<COMP>,
    coeffs: RkStageCoeffs<COMP>,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx_i32 = mesh.nx as i32;
    let ny_i32 = mesh.ny as i32;
    let nz_i32 = mesh.nz as i32;
    let state_size_i32 = mesh.state_size as i32;

    let mut args = stream.launch_builder(kernel);
    args.arg(dst)
        .arg(src)
        .arg(rhs)
        .arg(&face_j.j_x_left)
        .arg(&face_j.j_x_right)
        .arg(&face_j.j_y_left)
        .arg(&face_j.j_y_right)
        .arg(&face_j.j_z_left)
        .arg(&face_j.j_z_right)
        .arg(&nx_i32)
        .arg(&ny_i32)
        .arg(&nz_i32)
        .arg(&mesh.lx)
        .arg(&mesh.ly)
        .arg(&mesh.lz)
        .arg(&coeffs.a);
    if let Some(ref b) = coeffs.b {
        args.arg(b);
    }
    args.arg(&coeffs.c).arg(&state_size_i32);
    unsafe { args.launch(cfg)? };
    Ok(())
}

// ---------------------------------------------------------------------------
// Asynchronous HDF5 output and final state report.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn spawn_io_task<U: PrecisionType, COMP: PrecisionType>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    io_stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d_u: &cudarc::driver::CudaSlice<U>,
    mesh: &MeshDims<COMP>,
    step: usize,
    resume: Option<(usize, f64)>,
    compression: crate::io::SaveCompression,
    config: &crate::config::OpenDynamoConfig,
    mut buf: cudarc::driver::PinnedHostSlice<U>,
    done: std::sync::mpsc::Sender<cudarc::driver::PinnedHostSlice<U>>,
) -> Result<std::thread::JoinHandle<()>, Box<dyn std::error::Error>> {
    let event = stream.record_event(None)?;
    io_stream.wait(&event)?;
    io_stream.memcpy_dtoh(d_u, &mut buf)?;

    let out_dir = config.io.output_dir.clone();
    let prefix = config.io.prefix.clone();
    let (nx, ny, nz) = (mesh.nx, mesh.ny, mesh.nz);
    let (lx, ly, lz) = (
        mesh.lx.to_f32() as f64,
        mesh.ly.to_f32() as f64,
        mesh.lz.to_f32() as f64,
    );
    // Label semantics: labels count completed RK3 steps, so each dump at the
    // end of loop step `k` is label base + k + 1 (base 0 fresh, or the
    // resumed startup's own label). Physical time: t = base_t + (k+1)·dt.
    // Fresh runs therefore produce 000000 (IC, t = base_t), 000001 ... 00n;
    // resumed runs continue seamlessly from the checkpoint label.
    let (base_step, base_t) = match resume {
        Some((s, t)) => (Some(s), t),
        None => (None, 0.0),
    };
    let n_steps = config.solver.n_steps;
    let config_dt = config.solver.dt;
    let checkpoint_freq = config.io.checkpoint_freq;
    let kind = precision_tag::<U>();
    let io_stream_clone = io_stream.clone();

    Ok(std::thread::spawn(move || {
        // Block this background thread until the async copy finishes, then
        // write the HDF5 file and hand the staging buffer back to its pool.
        let result: Result<(), Box<dyn std::error::Error>> = (|| {
            io_stream_clone.synchronize()?;
            let slice = buf.as_slice()?;
            let label = base_step.map_or(step + 1, |s| s + step + 1);
            let t = base_t + (step as f64 + 1.0) * (config_dt as f64);
            crate::io::save_state_h5(label, t, kind, &out_dir, &prefix, slice, nx, ny, nz, lx, ly, lz, compression)?;
            // Checkpoint on the save cadence, and always at the final step
            // so every run ends with a resumable state.
            let want_checkpoint = step == n_steps - 1
                || (checkpoint_freq > 0 && step.is_multiple_of(checkpoint_freq));
            if want_checkpoint {
                crate::io::save_state_h5(
                    label,
                    t,
                    kind,
                    &out_dir,
                    "checkpoint",
                    slice,
                    nx,
                    ny,
                    nz,
                    lx,
                    ly,
                    lz,
                    compression,
                )?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("IO task failed: {e}");
        }
        let _ = done.send(buf);
    }))
}

fn physical_state_report<U: PrecisionType>(slice: &[U], total_points: usize) {
    let mut min_rho = f32::MAX;
    let mut max_rho = f32::MIN;
    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;

    let rho_var = &slice[..total_points];
    let rhou_var = &slice[total_points..2 * total_points];
    for (&rho_i, &rhou_i) in rho_var.iter().zip(rhou_var) {
        let rho = rho_i.to_f32();
        let u = rhou_i.to_f32() / rho;
        min_rho = min_rho.min(rho);
        max_rho = max_rho.max(rho);
        min_u = min_u.min(u);
        max_u = max_u.max(u);
    }
    println!("Physical State Check (Final):");
    println!("  Rho: [{min_rho:.5}, {max_rho:.5}]");
    println!("  U:   [{min_u:.5}, {max_u:.5}]");
}

macro_rules! dispatch_compute {
    ($cfg:expr, $U:ident, $Q:ident, $RHS:ident, $comp:expr) => {
        match $comp {
            $crate::config::RealType::F32 => {
                $crate::solver::run_solver_generic::<$U, $Q, $RHS, f32>($cfg)
            }
            $crate::config::RealType::F64 => {
                $crate::solver::run_solver_generic::<$U, $Q, $RHS, f64>($cfg)
            }
        }
    };
}
pub(crate) use dispatch_compute;

macro_rules! dispatch_rhs {
    ($cfg:expr, $U:ident, $Q:ident, $rhs:expr, $comp:expr) => {
        match $rhs {
            $crate::config::RealType::F32 => {
                $crate::solver::dispatch_compute!($cfg, $U, $Q, f32, $comp)
            }
            $crate::config::RealType::F64 => {
                $crate::solver::dispatch_compute!($cfg, $U, $Q, f64, $comp)
            }
        }
    };
}
pub(crate) use dispatch_rhs;

macro_rules! dispatch_q {
    ($cfg:expr, $U:ident, $q:expr, $rhs:expr, $comp:expr) => {
        match $q {
            $crate::config::RealType::F32 => {
                $crate::solver::dispatch_rhs!($cfg, $U, f32, $rhs, $comp)
            }
            $crate::config::RealType::F64 => {
                $crate::solver::dispatch_rhs!($cfg, $U, f64, $rhs, $comp)
            }
        }
    };
}
pub(crate) use dispatch_q;

macro_rules! dispatch_u {
    ($cfg:expr, $u:expr, $q:expr, $rhs:expr, $comp:expr) => {
        match $u {
            $crate::config::RealType::F32 => {
                $crate::solver::dispatch_q!($cfg, f32, $q, $rhs, $comp)
            }
            $crate::config::RealType::F64 => {
                $crate::solver::dispatch_q!($cfg, f64, $q, $rhs, $comp)
            }
        }
    };
}
pub(crate) use dispatch_u;
