// The j-corrections were fused out of compute_rhs_volume_kernel (which now runs
// before the face kernels): each stage's face fluxes are applied here, right
// where RHS is consumed, so the volume kernel no longer reads the j arrays.
// Correction math mirrors the old volume kernel exactly for bit-identical
// results.
//
// Each thread handles 4 consecutive points (one i-row of an element face) so
// every global load can be a single 128-bit vector transaction:
//   u, u_prev, rhs  — 4 consecutive points of one variable plane
//   j_y, j_z        — 4 consecutive face points (face_pt = i + ...)
//   j_x             — one face point shared by the 4 points (broadcast)
__device__ __forceinline__ void rhs_with_j_corrections4(COMPUTE_REAL rhs0, COMPUTE_REAL rhs1, COMPUTE_REAL rhs2, COMPUTE_REAL rhs3,
    COMPUTE_REAL j_x_left_val, COMPUTE_REAL j_x_right_val,
    const COMPUTE_REAL* jyl, const COMPUTE_REAL* jyr,
    const COMPUTE_REAL* jzl, const COMPUTE_REAL* jzr,
    int i, int j, int k, COMPUTE_REAL jac_x, COMPUTE_REAL jac_y, COMPUTE_REAL jac_z,
    COMPUTE_REAL* out) {
    COMPUTE_REAL rhs_vals[4] = {rhs0, rhs1, rhs2, rhs3};
    for (int t = 0; t < 4; ++t) {
        COMPUTE_REAL correction_x = -(j_x_left_val * c_g_left_prime[i + t] * jac_x) - (j_x_right_val * c_g_right_prime[i + t] * jac_x);
        COMPUTE_REAL correction_y = -(jyl[t] * c_g_left_prime[j] * jac_y) - (jyr[t] * c_g_right_prime[j] * jac_y);
        COMPUTE_REAL correction_z = -(jzl[t] * c_g_left_prime[k] * jac_z) - (jzr[t] * c_g_right_prime[k] * jac_z);
        out[t] = rhs_vals[t] + (correction_x + correction_y + correction_z);
    }
}

template <bool STAGE1>
__device__ __forceinline__ void rk_update4(U_REAL* u, const U_REAL* u_prev, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
    int total_points, COMPUTE_REAL jac_x, COMPUTE_REAL jac_y, COMPUTE_REAL jac_z,
    COMPUTE_REAL rk_a, COMPUTE_REAL rk_b, COMPUTE_REAL dt_scaled, int idx4) {
    typedef typename V4Type<U_REAL>::type U4;
    typedef typename V4Type<RHS_REAL>::type R4;
    typedef typename V4Type<COMPUTE_REAL>::type C4;

    int v = idx4 / total_points;
    int local = idx4 - v * total_points;
    int elem = local / NPTS_ELEM;
    int pt = local - elem * NPTS_ELEM;
    int i = pt % NPTS1D;
    int j = (pt / NPTS1D) % NPTS1D;
    int k = pt / NPTS1D2;

    int num_faces_total = total_points / NPTS1D; // = elements * NPTS_FACE
    int face_base = elem * NPTS_FACE;
    int plane_base = v * num_faces_total + face_base;

    U4 u4 = *reinterpret_cast<const U4*>(u + idx4);
    U4 up4 = *reinterpret_cast<const U4*>(u_prev + idx4);
    R4 r4 = *reinterpret_cast<const R4*>(rhs + idx4);

    // j_x: all 4 points share one face point (face_pt_x depends on j, k only).
    COMPUTE_REAL jxl = j_x_left[plane_base + j + k * NPTS1D];
    COMPUTE_REAL jxr = j_x_right[plane_base + j + k * NPTS1D];
    // j_y / j_z: face_pt walks with i, so 4 consecutive points are contiguous.
    C4 jyl4 = *reinterpret_cast<const C4*>(j_y_left + plane_base + i + k * NPTS1D);
    C4 jyr4 = *reinterpret_cast<const C4*>(j_y_right + plane_base + i + k * NPTS1D);
    C4 jzl4 = *reinterpret_cast<const C4*>(j_z_left + plane_base + i + j * NPTS1D);
    C4 jzr4 = *reinterpret_cast<const C4*>(j_z_right + plane_base + i + j * NPTS1D);

    COMPUTE_REAL u_prev_vals[4] = {(COMPUTE_REAL)up4.x, (COMPUTE_REAL)up4.y, (COMPUTE_REAL)up4.z, (COMPUTE_REAL)up4.w};
    COMPUTE_REAL u_vals[4] = {(COMPUTE_REAL)u4.x, (COMPUTE_REAL)u4.y, (COMPUTE_REAL)u4.z, (COMPUTE_REAL)u4.w};
    COMPUTE_REAL rhs_total[4];
    rhs_with_j_corrections4((COMPUTE_REAL)r4.x, (COMPUTE_REAL)r4.y, (COMPUTE_REAL)r4.z, (COMPUTE_REAL)r4.w,
        jxl, jxr, &jyl4.x, &jyr4.x, &jzl4.x, &jzr4.x,
        i, j, k, jac_x, jac_y, jac_z, rhs_total);

    U4 out;
    if (STAGE1) {
        out.x = (U_REAL)(rk_a * u_prev_vals[0] + dt_scaled * rhs_total[0]);
        out.y = (U_REAL)(rk_a * u_prev_vals[1] + dt_scaled * rhs_total[1]);
        out.z = (U_REAL)(rk_a * u_prev_vals[2] + dt_scaled * rhs_total[2]);
        out.w = (U_REAL)(rk_a * u_prev_vals[3] + dt_scaled * rhs_total[3]);
    } else {
        out.x = (U_REAL)(rk_a * u_prev_vals[0] + rk_b * u_vals[0] + dt_scaled * rhs_total[0]);
        out.y = (U_REAL)(rk_a * u_prev_vals[1] + rk_b * u_vals[1] + dt_scaled * rhs_total[1]);
        out.z = (U_REAL)(rk_a * u_prev_vals[2] + rk_b * u_vals[2] + dt_scaled * rhs_total[2]);
        out.w = (U_REAL)(rk_a * u_prev_vals[3] + rk_b * u_vals[3] + dt_scaled * rhs_total[3]);
    }
    *reinterpret_cast<U4*>(u + idx4) = out;
}

extern "C" __global__ void rk_stage_kernel(U_REAL* u, const U_REAL* u_prev, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL rk_a, COMPUTE_REAL rk_b, COMPUTE_REAL dt_scaled, int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 < size) {
        COMPUTE_REAL jac_x = ((COMPUTE_REAL)2.0) / (lx / ((COMPUTE_REAL)nx));
        COMPUTE_REAL jac_y = ((COMPUTE_REAL)2.0) / (ly / ((COMPUTE_REAL)ny));
        COMPUTE_REAL jac_z = ((COMPUTE_REAL)2.0) / (lz / ((COMPUTE_REAL)nz));
        rk_update4<false>(u, u_prev, rhs, j_x_left, j_x_right, j_y_left, j_y_right, j_z_left, j_z_right,
            nx * ny * nz * NPTS_ELEM, jac_x, jac_y, jac_z, rk_a, rk_b, dt_scaled, idx4);
    }
}

extern "C" __global__ void rk_stage1_kernel(U_REAL* u, const U_REAL* u_prev, const RHS_REAL* rhs,
    const COMPUTE_REAL* j_x_left, const COMPUTE_REAL* j_x_right,
    const COMPUTE_REAL* j_y_left, const COMPUTE_REAL* j_y_right,
    const COMPUTE_REAL* j_z_left, const COMPUTE_REAL* j_z_right,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL rk_a, COMPUTE_REAL dt_scaled, int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 < size) {
        COMPUTE_REAL jac_x = ((COMPUTE_REAL)2.0) / (lx / ((COMPUTE_REAL)nx));
        COMPUTE_REAL jac_y = ((COMPUTE_REAL)2.0) / (ly / ((COMPUTE_REAL)ny));
        COMPUTE_REAL jac_z = ((COMPUTE_REAL)2.0) / (lz / ((COMPUTE_REAL)nz));
        rk_update4<true>(u, u_prev, rhs, j_x_left, j_x_right, j_y_left, j_y_right, j_z_left, j_z_right,
            nx * ny * nz * NPTS_ELEM, jac_x, jac_y, jac_z, rk_a, (COMPUTE_REAL)0, dt_scaled, idx4);
    }
}
