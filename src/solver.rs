use cudarc::driver::PushKernelArg;
pub trait PrecisionType:
    cudarc::driver::DeviceRepr 
    + cudarc::driver::ValidAsZeroBits 
    + Send 
    + Sync 
    + Clone 
    + Copy 
    + 'static
    + std::ops::Mul<Output = Self> + std::fmt::Debug + hdf5_metno::H5Type
{
    fn c_type() -> &'static str;
    fn from_f64(val: f64) -> Self;
    fn to_f32(self) -> f32;
}

impl PrecisionType for f32 {
    fn c_type() -> &'static str { "float" }
    fn from_f64(val: f64) -> Self { val as f32 }
    fn to_f32(self) -> f32 { self }
}

impl PrecisionType for f64 {
    fn c_type() -> &'static str { "double" }
    fn from_f64(val: f64) -> Self { val }
    fn to_f32(self) -> f32 { self as f32 }
}

pub fn run_solver_generic<
    U: PrecisionType,
    Q: PrecisionType,
    RHS: PrecisionType,
    COMP: PrecisionType,
>(config: crate::config::OpenDynamoConfig) -> Result<(), Box<dyn std::error::Error>> {

        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let config = config;
        
        let mut opts = cudarc::nvrtc::CompileOptions::default();
        opts.options.push("--use_fast_math".to_string());
        opts.options.push(format!("-DU_REAL={}", U::c_type()));
        opts.options.push(format!("-DQ_REAL={}", Q::c_type()));
        opts.options.push(format!("-DRHS_REAL={}", RHS::c_type()));
        opts.options.push(format!("-DCOMPUTE_REAL={}", COMP::c_type()));
        
        if config.physics.viscosity_model == "sutherland" {
            opts.options.push("-DVISCOSITY_SUTHERLAND".to_string());
        }
        opts.options.push(format!("-DMU_REF={}", config.physics.mu_ref));
        opts.options.push(format!("-DT_REF={}", config.physics.t_ref));
        opts.options.push(format!("-DS_SUTH={}", config.physics.s_suth));
        
        if config.solver.flux_scheme == "slau2" {
            opts.options.push("-DFLUX_SCHEME_SLAU2".to_string());
        } else {
            opts.options.push("-DFLUX_SCHEME_RUSANOV".to_string());
        }
        
        let ptx_source = format!("{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("cuda/constants.cuh"),
            include_str!("cuda/device_functions.cuh"),
            include_str!("cuda/init.cu"),
            include_str!("cuda/gradient.cu"),
            include_str!("cuda/rhs.cu"),
            include_str!("cuda/rk.cu")
        );
        let ptx = match cudarc::nvrtc::compile_ptx_with_opts(ptx_source, opts) {
            Ok(p) => p,
            Err(e) => {
                println!("Compile Error: {:?}", e);
                panic!("Compilation failed");
            }
        };
        let module = ctx.load_module(ptx)?;
        
        let init_flow_kernel = module.load_function("init_flow_kernel")?;
        let compute_gradient_kernel = module.load_function("compute_gradient_kernel")?;
        let compute_rhs_volume_kernel = module.load_function("compute_rhs_volume_kernel")?;
        let compute_face_flux_x_kernel = module.load_function("compute_face_flux_x_kernel")?;
        let compute_face_flux_y_kernel = module.load_function("compute_face_flux_y_kernel")?;
        let compute_face_flux_z_kernel = module.load_function("compute_face_flux_z_kernel")?;
        let rk_stage_kernel = module.load_function("rk_stage_kernel")?;
        let rk_stage1_kernel = module.load_function("rk_stage1_kernel")?;
        
        println!("Generating FR operators for P=3...");
        let ops = crate::fr_math::generate_fr_operators(3);
        
        // Upload FR operators to device
        let extrap_left_comp: Vec<COMP> = ops.extrap_left.iter().map(|&x| COMP::from_f64(x)).collect();
        // Upload FR operators to device __constant__ memory
        let mut c_points = module.get_global("c_points", &stream)?;
        let mut c_extrap_left = module.get_global("c_extrap_left", &stream)?;
        let mut c_extrap_right = module.get_global("c_extrap_right", &stream)?;
        let mut c_g_left_prime = module.get_global("c_g_left_prime", &stream)?;
        let mut c_g_right_prime = module.get_global("c_g_right_prime", &stream)?;
        let mut c_d_matrix = module.get_global("c_d_matrix", &stream)?;
        
        let points_comp: Vec<COMP> = ops.points.iter().map(|&x| COMP::from_f64(x)).collect();
        
        let extrap_right_comp: Vec<COMP> = ops.extrap_right.iter().map(|&x| COMP::from_f64(x)).collect();
        let g_left_prime_comp: Vec<COMP> = ops.g_left_prime.iter().map(|&x| COMP::from_f64(x)).collect();
        let g_right_prime_comp: Vec<COMP> = ops.g_right_prime.iter().map(|&x| COMP::from_f64(x)).collect();
        let d_matrix_comp: Vec<COMP> = ops.d_matrix.iter().map(|&x| COMP::from_f64(x)).collect();
        
        unsafe {
            let u8_slice = |data: &Vec<COMP>| std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<COMP>());
            stream.memcpy_htod(u8_slice(&points_comp), &mut c_points)?;
            stream.memcpy_htod(u8_slice(&extrap_left_comp), &mut c_extrap_left)?;
            stream.memcpy_htod(u8_slice(&extrap_right_comp), &mut c_extrap_right)?;
            stream.memcpy_htod(u8_slice(&g_left_prime_comp), &mut c_g_left_prime)?;
            stream.memcpy_htod(u8_slice(&g_right_prime_comp), &mut c_g_right_prime)?;
            stream.memcpy_htod(u8_slice(&d_matrix_comp), &mut c_d_matrix)?;
        }
        
        let nx = config.mesh.nx;
        let ny = config.mesh.ny;
        let nz = config.mesh.nz;
        let num_elements = nx * ny * nz;
        let n_points_elem = 64;
        let n_vars = 5;
        let total_points = num_elements * n_points_elem;
        let state_size = total_points * n_vars;
        let q_size = total_points * 15; // 5 vars * 3 directions
        
        println!("Allocating device memory for {} elements, total DOFs: {}", num_elements, total_points);
        
        let mut d_u = stream.alloc_zeros::<U>(state_size as usize)?;
        let mut d_u0 = stream.alloc_zeros::<U>(state_size as usize)?;
        let mut d_rhs = stream.alloc_zeros::<RHS>(state_size as usize)?;
        let mut d_q = stream.alloc_zeros::<Q>(q_size as usize)?;
        let face_array_size = (num_elements * 16 * 5) as usize;
        let mut d_j_x_l = stream.alloc_zeros::<COMP>(face_array_size)?;
        let mut d_j_x_r = stream.alloc_zeros::<COMP>(face_array_size)?;
        let mut d_j_y_l = stream.alloc_zeros::<COMP>(face_array_size)?;
        let mut d_j_y_r = stream.alloc_zeros::<COMP>(face_array_size)?;
        let mut d_j_z_l = stream.alloc_zeros::<COMP>(face_array_size)?;
        let mut d_j_z_r = stream.alloc_zeros::<COMP>(face_array_size)?;

        
        let io_stream = ctx.new_stream()?;
        
        let mach_bulk = COMP::from_f64(config.physics.mach_bulk as f64);
        
        println!("Initializing flow...");
        let init_cfg = cudarc::driver::LaunchConfig { grid_dim: (num_elements as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let nx_i32 = nx as i32; let ny_i32 = ny as i32; let nz_i32 = nz as i32;
        
        let mut args_init = stream.launch_builder(&init_flow_kernel);
        args_init.arg(&mut d_u).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32).arg(&mach_bulk);
        unsafe { args_init.launch(init_cfg)? };
        
        // SAVE INITIAL CONDITION AS STEP 99999
        let mut h_u = unsafe { ctx.alloc_pinned_with_flags::<U>(state_size as usize, 0) }?;
        stream.memcpy_dtoh(&d_u, &mut h_u)?;
        let out_dir = config.io.output_dir.clone();
        let prefix = config.io.prefix.clone();
        let slice = h_u.as_slice().unwrap();
        crate::io::save_hdf5_xdmf(99999, 0.0, slice, nx, ny, nz, &out_dir, &prefix).unwrap();
        
        let dt = COMP::from_f64(config.solver.dt as f64);
        let n_steps = config.solver.n_steps;
        let rhs_cfg = cudarc::driver::LaunchConfig { grid_dim: (num_elements as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let add_cfg = cudarc::driver::LaunchConfig::for_num_elems(state_size as u32);
        let face_x_cfg = cudarc::driver::LaunchConfig { grid_dim: ((num_elements as u32 + 3) / 4, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let face_y_cfg = cudarc::driver::LaunchConfig { grid_dim: (nx as u32 * ((ny as u32 + 3) / 4) * nz as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let face_z_cfg = cudarc::driver::LaunchConfig { grid_dim: (nx as u32 * ny as u32 * ((nz as u32 + 3) / 4), 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        
        let ss_i32 = state_size as i32;
        
        let mut io_threads = vec![];
        
        println!("Starting RK3 explicit time integration...");
        let start_time = std::time::Instant::now();
        for step in 0..n_steps {
            // Stage 1
            let mut args_grad = stream.launch_builder(&compute_gradient_kernel);
            args_grad.arg(&d_u).arg(&mut d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_grad.launch(rhs_cfg)? };
            
            let mut args_x = stream.launch_builder(&compute_face_flux_x_kernel);
            args_x.arg(&d_u).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_x.launch(face_x_cfg)? };
            
            let mut args_y = stream.launch_builder(&compute_face_flux_y_kernel);
            args_y.arg(&d_u).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_y.launch(face_y_cfg)? };
            
            let mut args_z = stream.launch_builder(&compute_face_flux_z_kernel);
            args_z.arg(&d_u).arg(&d_j_z_l).arg(&d_j_z_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_z.launch(face_z_cfg)? };
            
            let mut args_v = stream.launch_builder(&compute_rhs_volume_kernel);
            args_v.arg(&d_u).arg(&mut d_rhs).arg(&d_q).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_j_z_l).arg(&d_j_z_r).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_v.launch(rhs_cfg)? };

let a1 = COMP::from_f64(1.0); let c1 = dt;
            let mut args_rk1 = stream.launch_builder(&rk_stage1_kernel);
            args_rk1.arg(&mut d_u0).arg(&d_u).arg(&d_rhs).arg(&a1).arg(&c1).arg(&ss_i32);
            unsafe { args_rk1.launch(add_cfg)? };
            std::mem::swap(&mut d_u, &mut d_u0);
            
            // Stage 2
            let mut args_grad2 = stream.launch_builder(&compute_gradient_kernel);
            args_grad2.arg(&d_u).arg(&mut d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_grad2.launch(rhs_cfg)? };
            
            let mut args_x = stream.launch_builder(&compute_face_flux_x_kernel);
            args_x.arg(&d_u).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_x.launch(face_x_cfg)? };
            
            let mut args_y = stream.launch_builder(&compute_face_flux_y_kernel);
            args_y.arg(&d_u).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_y.launch(face_y_cfg)? };
            
            let mut args_z = stream.launch_builder(&compute_face_flux_z_kernel);
            args_z.arg(&d_u).arg(&d_j_z_l).arg(&d_j_z_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_z.launch(face_z_cfg)? };
            
            let mut args_v = stream.launch_builder(&compute_rhs_volume_kernel);
            args_v.arg(&d_u).arg(&mut d_rhs).arg(&d_q).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_j_z_l).arg(&d_j_z_r).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_v.launch(rhs_cfg)? };

let a2 = COMP::from_f64(0.75); let b2 = COMP::from_f64(0.25); let c2 = COMP::from_f64(0.25) * dt;
            let mut args_rk2 = stream.launch_builder(&rk_stage_kernel);
            args_rk2.arg(&mut d_u).arg(&d_u0).arg(&d_rhs).arg(&a2).arg(&b2).arg(&c2).arg(&ss_i32);
            unsafe { args_rk2.launch(add_cfg)? };
            
            // Stage 3
            let mut args_grad3 = stream.launch_builder(&compute_gradient_kernel);
            args_grad3.arg(&d_u).arg(&mut d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_grad3.launch(rhs_cfg)? };
            
            let mut args_x = stream.launch_builder(&compute_face_flux_x_kernel);
            args_x.arg(&d_u).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_x.launch(face_x_cfg)? };
            
            let mut args_y = stream.launch_builder(&compute_face_flux_y_kernel);
            args_y.arg(&d_u).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_y.launch(face_y_cfg)? };
            
            let mut args_z = stream.launch_builder(&compute_face_flux_z_kernel);
            args_z.arg(&d_u).arg(&d_j_z_l).arg(&d_j_z_r).arg(&d_q).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_z.launch(face_z_cfg)? };
            
            let mut args_v = stream.launch_builder(&compute_rhs_volume_kernel);
            args_v.arg(&d_u).arg(&mut d_rhs).arg(&d_q).arg(&d_j_x_l).arg(&d_j_x_r).arg(&d_j_y_l).arg(&d_j_y_r).arg(&d_j_z_l).arg(&d_j_z_r).arg(&nx_i32).arg(&ny_i32).arg(&nz_i32);
            unsafe { args_v.launch(rhs_cfg)? };

let a3 = COMP::from_f64(1.0/3.0); let b3 = COMP::from_f64(2.0/3.0); let c3 = COMP::from_f64(2.0/3.0) * dt;
            let mut args_rk3 = stream.launch_builder(&rk_stage_kernel);
            args_rk3.arg(&mut d_u).arg(&d_u0).arg(&d_rhs).arg(&a3).arg(&b3).arg(&c3).arg(&ss_i32);
            unsafe { args_rk3.launch(add_cfg)? };
            
            if step % config.io.save_freq == 0 || step == n_steps - 1 {
                let event = stream.record_event(None)?;
                io_stream.wait(&event)?;
                
                // Allocate pinned memory and copy asynchronously
                let mut h_u = unsafe { ctx.alloc_pinned_with_flags::<U>(state_size as usize, 0) }?;
                io_stream.memcpy_dtoh(&d_u, &mut h_u)?;
                
                let out_dir = config.io.output_dir.clone();
                let prefix = config.io.prefix.clone();
                let nx_c = nx; let ny_c = ny; let nz_c = nz;
                let io_stream_clone = io_stream.clone();
                let step_c = step;
                
                let handle = std::thread::spawn(move || {
                    // Block this background thread until the async copy finishes
                    io_stream_clone.synchronize().unwrap();
                    let slice = h_u.as_slice().unwrap();
                    if let Err(e) = crate::io::save_hdf5_xdmf(step_c, 0.0, slice, nx_c, ny_c, nz_c, &out_dir, &prefix) {
                        eprintln!("Failed to save HDF5: {}", e);
                    }
                });
                io_threads.push(handle);
            }
        }
        
        // Wait for all IO streams
        io_stream.synchronize()?;
        
        // Join all background IO threads
        for handle in io_threads {
            let _ = handle.join();
        }
        
        stream.synchronize()?;
        let elapsed = start_time.elapsed();
        println!("Simulation finished in {:.3} seconds.", elapsed.as_secs_f64());
        println!("Time per step: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / (n_steps as f64));
        
        let mut h_u_final = unsafe { ctx.alloc_pinned_with_flags::<U>(state_size as usize, 0) }?;
        stream.memcpy_dtoh(&d_u, &mut h_u_final)?;
        stream.synchronize()?;
        
        let slice = h_u_final.as_slice().unwrap();
        let mut min_rho = f32::MAX;
        let mut max_rho = f32::MIN;
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        
        for i in 0..total_points {
            let rho = slice[0 * total_points + i].to_f32();
            let rhou = slice[1 * total_points + i].to_f32();
            let u = rhou / rho;
            min_rho = min_rho.min(rho);
            max_rho = max_rho.max(rho);
            min_u = min_u.min(u);
            max_u = max_u.max(u);
        }
        println!("Physical State Check (Final):");
        println!("  Rho: [{:.5}, {:.5}]", min_rho, max_rho);
        println!("  U:   [{:.5}, {:.5}]", min_u, max_u);
    
    Ok(())
}

#[macro_export]
macro_rules! dispatch_compute {
    ($U:ident, $Q:ident, $RHS:ident, $cfg:expr, $u_str:expr, $q_str:expr, $rhs_str:expr, $comp_str:expr) => {
        match $comp_str {
            "f32" => $crate::solver::run_solver_generic::<$U, $Q, $RHS, f32>($cfg).unwrap(),
            "f64" => $crate::solver::run_solver_generic::<$U, $Q, $RHS, f64>($cfg).unwrap(),
            _ => panic!("Invalid compute_real type. Must be f32 or f64."),
        }
    }
}

#[macro_export]
macro_rules! dispatch_rhs {
    ($U:ident, $Q:ident, $cfg:expr, $u_str:expr, $q_str:expr, $rhs_str:expr, $comp_str:expr) => {
        match $rhs_str {
            "f32" => dispatch_compute!($U, $Q, f32, $cfg, $u_str, $q_str, "f32", $comp_str),
            "f64" => dispatch_compute!($U, $Q, f64, $cfg, $u_str, $q_str, "f64", $comp_str),
            _ => panic!("Invalid rhs_real type. Must be f32 or f64."),
        }
    }
}

#[macro_export]
macro_rules! dispatch_q {
    ($U:ident, $cfg:expr, $u_str:expr, $q_str:expr, $rhs_str:expr, $comp_str:expr) => {
        match $q_str {
            "f32" => dispatch_rhs!($U, f32, $cfg, $u_str, "f32", $rhs_str, $comp_str),
            "f64" => dispatch_rhs!($U, f64, $cfg, $u_str, "f64", $rhs_str, $comp_str),
            _ => panic!("Invalid q_real type. Must be f32 or f64."),
        }
    }
}

#[macro_export]
macro_rules! dispatch_u {
    ($cfg:expr, $u_str:expr, $q_str:expr, $rhs_str:expr, $comp_str:expr) => {
        match $u_str {
            "f32" => dispatch_q!(f32, $cfg, "f32", $q_str, $rhs_str, $comp_str),
            "f64" => dispatch_q!(f64, $cfg, "f64", $q_str, $rhs_str, $comp_str),
            _ => panic!("Invalid u_real type. Must be f32 or f64."),
        }
    }
}
