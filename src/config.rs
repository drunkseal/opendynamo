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
    /// Domain size in x. Defaults reproduce the historical hardcoded box (4*pi).
    #[serde(default = "default_lx")]
    pub lx: f32,
    /// Domain size in y (walls at y = +/- ly/2 in channel mode).
    #[serde(default = "default_ly")]
    pub ly: f32,
    /// Domain size in z. Defaults reproduce the historical hardcoded box (4*pi/3).
    #[serde(default = "default_lz")]
    pub lz: f32,
}

// Defaults mirror the exact f32 evaluation order of the former hardcoded CUDA
// expressions so existing configs stay bit-identical.
fn default_lx() -> f32 {
    4.0 * std::f32::consts::PI
}
fn default_ly() -> f32 {
    2.0
}
fn default_lz() -> f32 {
    (4.0 / 3.0) * std::f32::consts::PI
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

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InitialCondition {
    /// Poiseuille channel profile: u = 1.5 * mach_bulk * (1 - (y/(ly/2))^2).
    Channel,
    /// Uniform freestream at the channel bulk velocity (same mass flow).
    Uniform,
}

fn default_initial_condition() -> InitialCondition {
    InitialCondition::Channel
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum YBoundary {
    /// Zero-flux slip walls at y = +/- ly/2 (historical behavior).
    Wall,
    /// Periodic wrap in y.
    Periodic,
}

fn default_y_boundary() -> YBoundary {
    YBoundary::Wall
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

fn default_fast_math() -> bool {
    true
}

fn default_perturbation_modes() -> u32 {
    16
}

fn default_perturbation_seed() -> u32 {
    7
}

#[derive(Deserialize, Debug, Clone)]
pub struct PhysicsConfig {
    pub mach_bulk: f32,
    #[serde(default = "default_initial_condition")]
    pub initial_condition: InitialCondition,
    #[serde(default = "default_y_boundary")]
    pub y_boundary: YBoundary,
    #[serde(default = "default_viscosity_model")]
    pub viscosity_model: ViscosityModel,
    #[serde(default = "default_mu_ref")]
    pub mu_ref: f32,
    #[serde(default = "default_t_ref")]
    pub t_ref: f32,
    #[serde(default = "default_s_suth")]
    pub s_suth: f32,
    /// Divergence-free wall-damped tripping perturbation: requested u'
    /// amplitude scale in physical velocity units (compared to c=1).
    /// 0.0 disables the perturbation kernel entirely (bit-identical to the
    /// old laminar start).
    #[serde(default)]
    pub perturbation_amplitude: f32,
    /// Number of superposed Fourier modes in the trip.
    #[serde(default = "default_perturbation_modes")]
    pub perturbation_modes: u32,
    /// Hash seed for the mode mix (same seed + same binary => same trip).
    #[serde(default = "default_perturbation_seed")]
    pub perturbation_seed: u32,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum FluxScheme {
    Slau2,
    #[serde(rename = "ausm+")]
    AusmPlus,
    Rusanov,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SolverConfig {
    pub n_steps: usize,
    pub dt: f32,
    pub flux_scheme: FluxScheme,
    #[serde(default = "default_fast_math")]
    pub fast_math: bool,
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
