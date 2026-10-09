// The j-corrections were fused out of compute_rhs_volume_kernel (which now runs
// before the face kernels): each stage's face fluxes are applied here, right
// where RHS is consumed, so the volume kernel no longer reads the j arrays.
// Correction math mirrors the old volume kernel exactly for bit-identical results.
__device__ __forceinline__ COMPUTE_REAL rhs_with_j_corrections(int idx,
    const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_L, const COMPUTE_REAL* j_x_R,
    const COMPUTE_REAL* j_y_L, const COMPUTE_REAL* j_y_R,
    const COMPUTE_REAL* j_z_L, const COMPUTE_REAL* j_z_R,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz) {
    int total_points = nx * ny * nz * NPTS_ELEM;
    int v = idx / total_points;
    int local = idx - v * total_points;
    int elem_idx = local / NPTS_ELEM;
    int pt = local - elem_idx * NPTS_ELEM;
    int i = pt % NPTS1D;
    int j = (pt / NPTS1D) % NPTS1D;
    int k = pt / NPTS1D2;

    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx;
    COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy;
    COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;

    int num_faces_total = nx * ny * nz * NPTS_FACE;
    int face_pt_x = j + k * NPTS1D;
    int face_pt_y = i + k * NPTS1D;
    int face_pt_z = i + j * NPTS1D;

    COMPUTE_REAL j_x_L_val = j_x_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
    COMPUTE_REAL j_x_R_val = j_x_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
    COMPUTE_REAL c_x = -(j_x_L_val * c_g_left_prime[i] * Jx) - (j_x_R_val * c_g_right_prime[i] * Jx);

    COMPUTE_REAL j_y_L_val = j_y_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
    COMPUTE_REAL j_y_R_val = j_y_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
    COMPUTE_REAL c_y = -(j_y_L_val * c_g_left_prime[j] * Jy) - (j_y_R_val * c_g_right_prime[j] * Jy);

    COMPUTE_REAL j_z_L_val = j_z_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
    COMPUTE_REAL j_z_R_val = j_z_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
    COMPUTE_REAL c_z = -(j_z_L_val * c_g_left_prime[k] * Jz) - (j_z_R_val * c_g_right_prime[k] * Jz);

    return (COMPUTE_REAL)rhs[idx] + (c_x + c_y + c_z);
}

extern "C" __global__ void rk_stage_kernel(U_REAL* u, const U_REAL* u0, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_L, const COMPUTE_REAL* j_x_R,
    const COMPUTE_REAL* j_y_L, const COMPUTE_REAL* j_y_R,
    const COMPUTE_REAL* j_z_L, const COMPUTE_REAL* j_z_R,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL a, COMPUTE_REAL b, COMPUTE_REAL c_dt, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        COMPUTE_REAL rhs_total = rhs_with_j_corrections(idx, rhs,
            j_x_L, j_x_R, j_y_L, j_y_R, j_z_L, j_z_R, nx, ny, nz, lx, ly, lz);
        u[idx] = a * u0[idx] + b * u[idx] + c_dt * rhs_total;
    }
}

extern "C" __global__ void rk_stage1_kernel(U_REAL* u, const U_REAL* u0, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_L, const COMPUTE_REAL* j_x_R,
    const COMPUTE_REAL* j_y_L, const COMPUTE_REAL* j_y_R,
    const COMPUTE_REAL* j_z_L, const COMPUTE_REAL* j_z_R,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL a, COMPUTE_REAL c_dt, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        COMPUTE_REAL rhs_total = rhs_with_j_corrections(idx, rhs,
            j_x_L, j_x_R, j_y_L, j_y_R, j_z_L, j_z_R, nx, ny, nz, lx, ly, lz);
        u[idx] = a * u0[idx] + c_dt * rhs_total;
    }
}
