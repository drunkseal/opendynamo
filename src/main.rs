mod config;
mod constants;
mod fr_math;
mod io;
mod solver;

use solver::dispatch_u;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = config::load_config("config.toml").expect("Failed to load config.toml");

    println!("Configuring pipeline with precisions:");
    println!("  U:       {:?}", config.precision.u_real);
    println!("  Q:       {:?}", config.precision.q_real);
    println!("  RHS:     {:?}", config.precision.rhs_real);
    println!("  COMPUTE: {:?}", config.precision.compute_real);
    println!("  SCHEME:  {:?}", config.solver.flux_scheme);

    let (u, q, rhs, comp) = (
        config.precision.u_real,
        config.precision.q_real,
        config.precision.rhs_real,
        config.precision.compute_real,
    );
    dispatch_u!(config, u, q, rhs, comp)
}
