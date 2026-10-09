// The j-corrections were fused out of compute_rhs_volume_kernel (which now runs
// before the face kernels): each stage's face fluxes are applied here, right
// where RHS is consumed, so the volume kernel no longer reads the j arrays.
// Correction math mirrors the old volume kernel exactly for bit-identical results.
__device__ __forceinline__ COMPUTE_REAL rhs_with_j_corrections(int idx,
    const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
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
    COMPUTE_REAL jac_x = ((COMPUTE_REAL)2.0) / dx;
    COMPUTE_REAL jac_y = ((COMPUTE_REAL)2.0) / dy;
    COMPUTE_REAL jac_z = ((COMPUTE_REAL)2.0) / dz;

    int num_faces_total = nx * ny * nz * NPTS_FACE;
    int face_pt_x = j + k * NPTS1D;
    int face_pt_y = i + k * NPTS1D;
    int face_pt_z = i + j * NPTS1D;

    COMPUTE_REAL j_x_left_val = j_x_left[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
    COMPUTE_REAL j_x_right_val = j_x_right[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
    COMPUTE_REAL correction_x = -(j_x_left_val * c_g_left_prime[i] * jac_x) - (j_x_right_val * c_g_right_prime[i] * jac_x);

    COMPUTE_REAL j_y_left_val = j_y_left[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
    COMPUTE_REAL j_y_right_val = j_y_right[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
    COMPUTE_REAL correction_y = -(j_y_left_val * c_g_left_prime[j] * jac_y) - (j_y_right_val * c_g_right_prime[j] * jac_y);

    COMPUTE_REAL j_z_left_val = j_z_left[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
    COMPUTE_REAL j_z_right_val = j_z_right[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
    COMPUTE_REAL correction_z = -(j_z_left_val * c_g_left_prime[k] * jac_z) - (j_z_right_val * c_g_right_prime[k] * jac_z);

    return (COMPUTE_REAL)rhs[idx] + (correction_x + correction_y + correction_z);
}

extern "C" __global__ void rk_stage_kernel(U_REAL* u, const U_REAL* u_prev, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL rk_a, COMPUTE_REAL rk_b, COMPUTE_REAL dt_scaled, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        COMPUTE_REAL rhs_total = rhs_with_j_corrections(idx, rhs,
            j_x_left, j_x_right, j_y_left, j_y_right, j_z_left, j_z_right, nx, ny, nz, lx, ly, lz);
        u[idx] = rk_a * u_prev[idx] + rk_b * u[idx] + dt_scaled * rhs_total;
    }
}

extern "C" __global__ void rk_stage1_kernel(U_REAL* u, const U_REAL* u_prev, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL rk_a, COMPUTE_REAL dt_scaled, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        COMPUTE_REAL rhs_total = rhs_with_j_corrections(idx, rhs,
            j_x_left, j_x_right, j_y_left, j_y_right, j_z_left, j_z_right, nx, ny, nz, lx, ly, lz);
        u[idx] = rk_a * u_prev[idx] + dt_scaled * rhs_total;
    }
}
