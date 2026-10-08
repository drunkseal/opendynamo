use cudarc::driver::DriverError;

mod fr_math;
mod config;
mod io;
mod solver;

fn main() -> Result<(), DriverError> {
    let config = config::load_config("config.toml").expect("Failed to load config.toml");
    
    println!("Configuring pipeline with precisions:");
    println!("  U:       {}", config.precision.u_real);
    println!("  Q:       {}", config.precision.q_real);
    println!("  RHS:     {}", config.precision.rhs_real);
    println!("  COMPUTE: {}", config.precision.compute_real);
    println!("  SCHEME:  {}", config.solver.flux_scheme);

    dispatch_u!(
        config.clone(),
        config.precision.u_real.as_str(),
        config.precision.q_real.as_str(),
        config.precision.rhs_real.as_str(),
        config.precision.compute_real.as_str()
    );
    
    Ok(())
}
