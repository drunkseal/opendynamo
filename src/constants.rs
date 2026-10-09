//! Host-side constants shared by the solver and IO. FR_ORDER is pushed to the
//! NVRTC compiler as -DFR_ORDER so the CUDA sources derive identical layout
//! macros (NPTS1D, NPTS_ELEM, ...) from the same single source of truth.

/// FR polynomial order P (point count per direction is P+1).
pub const FR_ORDER: usize = 3;
/// Solution points per hex element: (P+1)^3.
pub const PTS_PER_ELEM: usize = (FR_ORDER + 1) * (FR_ORDER + 1) * (FR_ORDER + 1);
/// Face points per quad face: (P+1)^2.
pub const PTS_PER_FACE: usize = (FR_ORDER + 1) * (FR_ORDER + 1);
/// Conservative variables (rho, rho*u, rho*v, rho*w, rho*E).
pub const N_VARS: usize = 5;
/// Faces grouped per thread block for in-block cache reuse.
pub const FACES_PER_BLOCK: usize = PTS_PER_ELEM / PTS_PER_FACE;
/// Step label used for the initial-condition output file.
pub const INIT_OUTPUT_STEP: usize = 99999;
