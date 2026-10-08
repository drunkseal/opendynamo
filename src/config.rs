use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct OpenDynamoConfig {
    pub mesh: MeshConfig,
    pub physics: PhysicsConfig,
    pub solver: SolverConfig,
    pub io: IOConfig,
    pub precision: PrecisionConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct MeshConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ViscosityModel {
    Fixed,
    Sutherland,
}

fn default_viscosity_model() -> ViscosityModel {
    ViscosityModel::Fixed
}

fn default_mu_ref() -> f32 {
    0.00016
}

fn default_t_ref() -> f32 {
    273.15
}

fn default_s_suth() -> f32 {
    110.4
}

#[derive(Deserialize, Debug, Clone)]
pub struct PhysicsConfig {
    pub mach_bulk: f32,
    #[serde(default = "default_viscosity_model")]
    pub viscosity_model: ViscosityModel,
    #[serde(default = "default_mu_ref")]
    pub mu_ref: f32,
    #[serde(default = "default_t_ref")]
    pub t_ref: f32,
    #[serde(default = "default_s_suth")]
    pub s_suth: f32,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum FluxScheme {
    Slau2,
    Rusanov,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SolverConfig {
    pub n_steps: usize,
    pub dt: f32,
    pub flux_scheme: FluxScheme,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RealType {
    F32,
    F64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PrecisionConfig {
    pub u_real: RealType,
    pub q_real: RealType,
    pub rhs_real: RealType,
    pub compute_real: RealType,
}

#[derive(Deserialize, Debug, Clone)]
pub struct IOConfig {
    pub save_freq: usize,
    pub output_dir: String,
    pub prefix: String,
}

pub fn load_config(path: &str) -> Result<OpenDynamoConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let config: OpenDynamoConfig = toml::from_str(&content)?;
    Ok(config)
}
