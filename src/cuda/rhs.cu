template <typename T> struct V4Type;
template <> struct V4Type<float>  { typedef float4  type; };
template <> struct V4Type<double> { typedef double4 type; };

template <typename T>
__device__ __forceinline__ typename V4Type<T>::type ld4(const T* p) {
    return *reinterpret_cast<const typename V4Type<T>::type*>(p);
}

// Volume Kernel + face corrections (fused: apply_corrections eliminated RHS read/write round-trip)
extern "C" __global__ void compute_rhs_volume_kernel(const U_REAL* U, RHS_REAL* RHS, const Q_REAL* Q,
    const COMPUTE_REAL* j_x_L, const COMPUTE_REAL* j_x_R,
    const COMPUTE_REAL* j_y_L, const COMPUTE_REAL* j_y_R,
    const COMPUTE_REAL* j_z_L, const COMPUTE_REAL* j_z_R,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz) {
    __shared__ COMPUTE_REAL s_U[5][NPTS_ELEM];
    __shared__ COMPUTE_REAL s_Q[15][NPTS_ELEM];

    int pt = threadIdx.x;
    int elem_idx = blockIdx.x;
    int total_points = nx * ny * nz * NPTS_ELEM;
    int g_idx = elem_idx * NPTS_ELEM + pt;

    for (int v = 0; v < 5; ++v) s_U[v][pt] = U[v * total_points + g_idx];
    for (int v = 0; v < 15; ++v) s_Q[v][pt] = Q[v * total_points + g_idx];
    __syncthreads();

    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx; COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy; COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;

    COMPUTE_REAL rhs_acc[5] = {0};

    int i = pt % NPTS1D;
    int j = (pt / NPTS1D) % NPTS1D;
    int k = pt / NPTS1D2;

    COMPUTE_REAL u_tmp[5]; for(int v=0; v<5; ++v) u_tmp[v] = s_U[v][pt];
    COMPUTE_REAL q_tmp[15]; for(int v=0; v<15; ++v) q_tmp[v] = s_Q[v][pt];
    
    // X-Flux
    {
        COMPUTE_REAL fi[5], fv[5];
        compute_euler_flux_x(u_tmp, fi);
        compute_viscous_flux_x(u_tmp, q_tmp, fv);
        __syncthreads();
        for (int v = 0; v < 5; ++v) s_U[v][pt] = fi[v] - fv[v];
        __syncthreads();
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL dF = 0;
            for (int m = 0; m < NPTS1D; ++m) dF += c_d_matrix[i * NPTS1D + m] * s_U[v][(m + j * NPTS1D + k * NPTS1D2)];
            rhs_acc[v] += dF * Jx;
        }
    }
    
    // Y-Flux
    {
        __syncthreads();
        // for(int v=0; v<5; ++v) u_tmp[v] = U[v * total_points + g_idx];
        COMPUTE_REAL fi[5], fv[5];
        compute_euler_flux_y(u_tmp, fi);
        compute_viscous_flux_y(u_tmp, q_tmp, fv);
        __syncthreads();
        for (int v = 0; v < 5; ++v) s_U[v][pt] = fi[v] - fv[v];
        __syncthreads();
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL dF = 0;
            for (int m = 0; m < NPTS1D; ++m) dF += c_d_matrix[j * NPTS1D + m] * s_U[v][(i + m * NPTS1D + k * NPTS1D2)];
            rhs_acc[v] += dF * Jy;
        }
    }
    
    // Z-Flux
    {
        __syncthreads();
        // for(int v=0; v<5; ++v) u_tmp[v] = U[v * total_points + g_idx];
        COMPUTE_REAL fi[5], fv[5];
        compute_euler_flux_z(u_tmp, fi);
        compute_viscous_flux_z(u_tmp, q_tmp, fv);
        __syncthreads();
        for (int v = 0; v < 5; ++v) s_U[v][pt] = fi[v] - fv[v];
        __syncthreads();
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL dF = 0;
            for (int m = 0; m < NPTS1D; ++m) dF += c_d_matrix[k * NPTS1D + m] * s_U[v][(i + j * NPTS1D + m * NPTS1D2)];
            rhs_acc[v] += dF * Jz;
        }
    }

    int num_faces_total = nx * ny * nz * NPTS_FACE;
    int face_pt_x = j + k * NPTS1D;
    int face_pt_y = i + k * NPTS1D;
    int face_pt_z = i + j * NPTS1D;
    
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL rhs_val = -rhs_acc[v];
        if (v == 1) rhs_val += ((COMPUTE_REAL)0.001);
        
        COMPUTE_REAL j_x_L_val = j_x_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
        COMPUTE_REAL j_x_R_val = j_x_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_x];
        COMPUTE_REAL c_x = -(j_x_L_val * c_g_left_prime[i] * Jx) - (j_x_R_val * c_g_right_prime[i] * Jx);
        
        COMPUTE_REAL j_y_L_val = j_y_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
        COMPUTE_REAL j_y_R_val = j_y_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_y];
        COMPUTE_REAL c_y = -(j_y_L_val * c_g_left_prime[j] * Jy) - (j_y_R_val * c_g_right_prime[j] * Jy);
        
        COMPUTE_REAL j_z_L_val = j_z_L[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
        COMPUTE_REAL j_z_R_val = j_z_R[v * num_faces_total + elem_idx * NPTS_FACE + face_pt_z];
        COMPUTE_REAL c_z = -(j_z_L_val * c_g_left_prime[k] * Jz) - (j_z_R_val * c_g_right_prime[k] * Jz);
        
        RHS[v * total_points + g_idx] = (RHS_REAL)rhs_val + (RHS_REAL)(c_x + c_y + c_z);
    }
}

extern "C" __global__ void compute_face_flux_x_kernel(const U_REAL* U, COMPUTE_REAL* j_x_L, COMPUTE_REAL* j_x_R, const Q_REAL* Q, int nx, int ny, int nz, COMPUTE_REAL lx) {
    int face_idx = blockIdx.x * FACES_PER_BLOCK + threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;
    if (face_idx >= nx * ny * nz) return;
    
    int total_points = nx * ny * nz * NPTS_ELEM;
    int ex = face_idx % nx;
    int ey = (face_idx / nx) % ny;
    int ez = face_idx / (nx * ny);
    
    int ex_L = (ex - 1 + nx) % nx;
    int ex_R = ex;
    
    int elem_L = ex_L + nx * (ey + ny * ez);
    int elem_R = ex_R + nx * (ey + ny * ez);
    
    COMPUTE_REAL u_L[5] = {0}, q_L[15] = {0};
    COMPUTE_REAL u_R[5] = {0}, q_R[15] = {0};
    
    // Thread pt owns m = 0..NPTS1D-1 (contiguous floats): idx = elem*NPTS_ELEM + NPTS1D*pt + m
    int base_L = elem_L * NPTS_ELEM + NPTS1D * pt;
    int base_R = elem_R * NPTS_ELEM + NPTS1D * pt;
    
    typename V4Type<U_REAL>::type U_r[5], U_l[5];
    for (int v = 0; v < 5; ++v) U_r[v] = ld4(U + v * total_points + base_R);
    for (int v = 0; v < 5; ++v) U_l[v] = ld4(U + v * total_points + base_L);
    for (int v = 0; v < 5; ++v) {
        u_L[v] = (COMPUTE_REAL)U_l[v].x * c_extrap_right[0] + (COMPUTE_REAL)U_l[v].y * c_extrap_right[1]
               + (COMPUTE_REAL)U_l[v].z * c_extrap_right[2] + (COMPUTE_REAL)U_l[v].w * c_extrap_right[3];
        u_R[v] = (COMPUTE_REAL)U_r[v].x * c_extrap_left[0] + (COMPUTE_REAL)U_r[v].y * c_extrap_left[1]
               + (COMPUTE_REAL)U_r[v].z * c_extrap_left[2] + (COMPUTE_REAL)U_r[v].w * c_extrap_left[3];
    }
    typename V4Type<Q_REAL>::type Q_r[11], Q_l[11];
    int qi = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 8 || v == 10 || v == 13 || v == 14) continue;
        Q_r[qi] = ld4(Q + v * total_points + base_R); ++qi;
    }
    qi = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 8 || v == 10 || v == 13 || v == 14) continue;
        Q_l[qi] = ld4(Q + v * total_points + base_L); ++qi;
    }
    qi = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 8 || v == 10 || v == 13 || v == 14) continue;
        q_L[v] = (COMPUTE_REAL)Q_l[qi].x * c_extrap_right[0] + (COMPUTE_REAL)Q_l[qi].y * c_extrap_right[1]
               + (COMPUTE_REAL)Q_l[qi].z * c_extrap_right[2] + (COMPUTE_REAL)Q_l[qi].w * c_extrap_right[3];
        q_R[v] = (COMPUTE_REAL)Q_r[qi].x * c_extrap_left[0] + (COMPUTE_REAL)Q_r[qi].y * c_extrap_left[1]
               + (COMPUTE_REAL)Q_r[qi].z * c_extrap_left[2] + (COMPUTE_REAL)Q_r[qi].w * c_extrap_left[3];
        ++qi;
    }
    
    COMPUTE_REAL F_inv[5], F_v_L[5], F_v_R[5], F_i_L[5], F_i_R[5];
    COMPUTE_REAL n[3] = {1.0, 0.0, 0.0};
    
    // Note: compute_interface_flux is slau2_flux, ausm_plus_flux or rusanov_flux mapped via macro
    compute_interface_flux(u_L, u_R, n, F_inv);
    compute_viscous_flux_x(u_L, q_L, F_v_L);
    compute_viscous_flux_x(u_R, q_R, F_v_R);
    compute_euler_flux_x(u_L, F_i_L);
    compute_euler_flux_x(u_R, F_i_R);
    
    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx;
    
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL F_common = F_inv[v] - ((COMPUTE_REAL)0.5) * (F_v_L[v] + F_v_R[v]);
        COMPUTE_REAL j_L = F_common - (F_i_L[v] - F_v_L[v]);
        COMPUTE_REAL j_R = F_common - (F_i_R[v] - F_v_R[v]);
        
        
        int num_faces_total = nx * ny * nz * NPTS_FACE;
        j_x_R[v * num_faces_total + elem_L * NPTS_FACE + pt] = j_L;
        j_x_L[v * num_faces_total + elem_R * NPTS_FACE + pt] = j_R;

    }
}

extern "C" __global__ void compute_face_flux_y_kernel(const U_REAL* U, COMPUTE_REAL* j_y_L, COMPUTE_REAL* j_y_R, const Q_REAL* Q, int nx, int ny, int nz, COMPUTE_REAL ly) {
    int sub = threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;

    // Block groups FACES_PER_BLOCK faces sharing (ex, ez) with consecutive ey -> in-block L1 reuse
    int ny4 = (ny + FACES_PER_BLOCK - 1) / FACES_PER_BLOCK;
    int bf = blockIdx.x;
    int by4 = bf % ny4;
    int ex = (bf / ny4) % nx;
    int ez = bf / (ny4 * nx);
    int ey = by4 * FACES_PER_BLOCK + sub;
    if (ey >= ny) return;
#if BC_Y_WALL
    if (ey == 0) return; // wall face: flux slots stay zero (zero-flux slip wall)
#endif

    int total_points = nx * ny * nz * NPTS_ELEM;

    // Shuffle mask must name only lanes that survived the early returns above:
    // partial face groups (ey >= ny) and the ey == 0 wall face diverge within a warp.
    const unsigned mask = __activemask();

#if BC_Y_WALL
    int ey_L = ey - 1;
#else
    int ey_L = (ey - 1 + ny) % ny;
#endif
    int ey_R = ey;
    
    int elem_L = ex + nx * (ey_L + ny * ez);
    int elem_R = ex + nx * (ey_R + ny * ez);
    
    COMPUTE_REAL u_L[5] = {0}, q_L[15] = {0};
    COMPUTE_REAL u_R[5] = {0}, q_R[15] = {0};
    
    // Thread role remap: pt = m + NPTS1D*k where m = element j (extrap var), k = face point k.
    // ld4 reads points i'=0..NPTS1D-1 (contiguous) at (m, k); quad (xor1, xor2) = m-set at fixed k.
    // After reduce, lane picks component i = pt % NPTS1D = output face point i.
    int m = pt % NPTS1D;
    int k = pt / NPTS1D;
    int base_L = elem_L * NPTS_ELEM + (m * NPTS1D + k * NPTS1D2);
    int base_R = elem_R * NPTS_ELEM + (m * NPTS1D + k * NPTS1D2);
    COMPUTE_REAL c_R = c_extrap_right[m];
    COMPUTE_REAL c_L = c_extrap_left[m];
    
    for (int v = 0; v < 5; ++v) {
        typename V4Type<U_REAL>::type fr = ld4(U + v * total_points + base_R);
        typename V4Type<U_REAL>::type fl = ld4(U + v * total_points + base_L);
        COMPUTE_REAL sL0 = (COMPUTE_REAL)fl.x * c_R, sL1 = (COMPUTE_REAL)fl.y * c_R, sL2 = (COMPUTE_REAL)fl.z * c_R, sL3 = (COMPUTE_REAL)fl.w * c_R;
        COMPUTE_REAL sR0 = (COMPUTE_REAL)fr.x * c_L, sR1 = (COMPUTE_REAL)fr.y * c_L, sR2 = (COMPUTE_REAL)fr.z * c_L, sR3 = (COMPUTE_REAL)fr.w * c_L;
        sL0 += __shfl_xor_sync(mask, sL0, 1); sR0 += __shfl_xor_sync(mask, sR0, 1);
        sL1 += __shfl_xor_sync(mask, sL1, 1); sR1 += __shfl_xor_sync(mask, sR1, 1);
        sL2 += __shfl_xor_sync(mask, sL2, 1); sR2 += __shfl_xor_sync(mask, sR2, 1);
        sL3 += __shfl_xor_sync(mask, sL3, 1); sR3 += __shfl_xor_sync(mask, sR3, 1);
        sL0 += __shfl_xor_sync(mask, sL0, 2); sR0 += __shfl_xor_sync(mask, sR0, 2);
        sL1 += __shfl_xor_sync(mask, sL1, 2); sR1 += __shfl_xor_sync(mask, sR1, 2);
        sL2 += __shfl_xor_sync(mask, sL2, 2); sR2 += __shfl_xor_sync(mask, sR2, 2);
        sL3 += __shfl_xor_sync(mask, sL3, 2); sR3 += __shfl_xor_sync(mask, sR3, 2);
        u_L[v] = (m == 0) ? sL0 : (m == 1) ? sL1 : (m == 2) ? sL2 : sL3;
        u_R[v] = (m == 0) ? sR0 : (m == 1) ? sR1 : (m == 2) ? sR2 : sR3;
    }
    for (int v = 0; v < 15; ++v) {
        if (v == 5 || v == 9 || v == 12 || v == 14) continue; // unused by viscous y-flux
        typename V4Type<Q_REAL>::type fr = ld4(Q + v * total_points + base_R);
        typename V4Type<Q_REAL>::type fl = ld4(Q + v * total_points + base_L);
        COMPUTE_REAL sL0 = (COMPUTE_REAL)fl.x * c_R, sL1 = (COMPUTE_REAL)fl.y * c_R, sL2 = (COMPUTE_REAL)fl.z * c_R, sL3 = (COMPUTE_REAL)fl.w * c_R;
        COMPUTE_REAL sR0 = (COMPUTE_REAL)fr.x * c_L, sR1 = (COMPUTE_REAL)fr.y * c_L, sR2 = (COMPUTE_REAL)fr.z * c_L, sR3 = (COMPUTE_REAL)fr.w * c_L;
        sL0 += __shfl_xor_sync(mask, sL0, 1); sR0 += __shfl_xor_sync(mask, sR0, 1);
        sL1 += __shfl_xor_sync(mask, sL1, 1); sR1 += __shfl_xor_sync(mask, sR1, 1);
        sL2 += __shfl_xor_sync(mask, sL2, 1); sR2 += __shfl_xor_sync(mask, sR2, 1);
        sL3 += __shfl_xor_sync(mask, sL3, 1); sR3 += __shfl_xor_sync(mask, sR3, 1);
        sL0 += __shfl_xor_sync(mask, sL0, 2); sR0 += __shfl_xor_sync(mask, sR0, 2);
        sL1 += __shfl_xor_sync(mask, sL1, 2); sR1 += __shfl_xor_sync(mask, sR1, 2);
        sL2 += __shfl_xor_sync(mask, sL2, 2); sR2 += __shfl_xor_sync(mask, sR2, 2);
        sL3 += __shfl_xor_sync(mask, sL3, 2); sR3 += __shfl_xor_sync(mask, sR3, 2);
        q_L[v] = (m == 0) ? sL0 : (m == 1) ? sL1 : (m == 2) ? sL2 : sL3;
        q_R[v] = (m == 0) ? sR0 : (m == 1) ? sR1 : (m == 2) ? sR2 : sR3;
    }
    
    COMPUTE_REAL F_inv[5], F_v_L[5], F_v_R[5], F_i_L[5], F_i_R[5];
    COMPUTE_REAL n[3] = {0.0, 1.0, 0.0};
    
    compute_interface_flux(u_L, u_R, n, F_inv);
    compute_viscous_flux_y(u_L, q_L, F_v_L);
    compute_viscous_flux_y(u_R, q_R, F_v_R);
    compute_euler_flux_y(u_L, F_i_L);
    compute_euler_flux_y(u_R, F_i_R);
    
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy;
    
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL F_common = F_inv[v] - ((COMPUTE_REAL)0.5) * (F_v_L[v] + F_v_R[v]);
        COMPUTE_REAL j_L = F_common - (F_i_L[v] - F_v_L[v]);
        COMPUTE_REAL j_R = F_common - (F_i_R[v] - F_v_R[v]);
        
        
        int num_faces_total = nx * ny * nz * NPTS_FACE;
        j_y_R[v * num_faces_total + elem_L * NPTS_FACE + pt] = j_L;
        j_y_L[v * num_faces_total + elem_R * NPTS_FACE + pt] = j_R;

    }
}

extern "C" __global__ void compute_face_flux_z_kernel(const U_REAL* U, COMPUTE_REAL* j_z_L, COMPUTE_REAL* j_z_R, const Q_REAL* Q, int nx, int ny, int nz, COMPUTE_REAL lz) {
    int sub = threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;

    // Block groups FACES_PER_BLOCK faces sharing (ex, ey) with consecutive ez -> in-block L1 reuse
    int nz4 = (nz + FACES_PER_BLOCK - 1) / FACES_PER_BLOCK;
    int bf = blockIdx.x;
    int bz4 = bf % nz4;
    int ex = (bf / nz4) % nx;
    int ey = (bf / nz4) / nx;
    int ez = bz4 * FACES_PER_BLOCK + sub;
    if (ez >= nz) return;
    
    int total_points = nx * ny * nz * NPTS_ELEM;
    
    int ez_L = (ez - 1 + nz) % nz;
    int ez_R = ez;
    
    int elem_L = ex + nx * (ey + ny * ez_L);
    int elem_R = ex + nx * (ey + ny * ez_R);
    
    COMPUTE_REAL u_L[5] = {0}, q_L[15] = {0};
    COMPUTE_REAL u_R[5] = {0}, q_R[15] = {0};
    
    int f1 = pt % NPTS1D; // i
    int f2 = pt / NPTS1D; // j

    for (int m = 0; m < NPTS1D; ++m) {
        int idx_L = elem_L * NPTS_ELEM + (f1 + f2 * NPTS1D + m * NPTS1D2);
        int idx_R = elem_R * NPTS_ELEM + (f1 + f2 * NPTS1D + m * NPTS1D2);
        
        COMPUTE_REAL c_R = c_extrap_right[m];
        COMPUTE_REAL c_L = c_extrap_left[m];
        
        for (int v = 0; v < 5; ++v) {
            u_R[v] += U[v * total_points + idx_R] * c_L;
            u_L[v] += U[v * total_points + idx_L] * c_R;
        }
        for (int v = 0; v < 15; ++v) {
            if (v == 4 || v == 6 || v == 12 || v == 13) continue; // unused by viscous z-flux
            q_R[v] += Q[v * total_points + idx_R] * c_L;
            q_L[v] += Q[v * total_points + idx_L] * c_R;
        }
    }
    
    COMPUTE_REAL F_inv[5], F_v_L[5], F_v_R[5], F_i_L[5], F_i_R[5];
    COMPUTE_REAL n[3] = {0.0, 0.0, 1.0};
    
    compute_interface_flux(u_L, u_R, n, F_inv);
    compute_viscous_flux_z(u_L, q_L, F_v_L);
    compute_viscous_flux_z(u_R, q_R, F_v_R);
    compute_euler_flux_z(u_L, F_i_L);
    compute_euler_flux_z(u_R, F_i_R);
    
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;
    
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL F_common = F_inv[v] - ((COMPUTE_REAL)0.5) * (F_v_L[v] + F_v_R[v]);
        COMPUTE_REAL j_L = F_common - (F_i_L[v] - F_v_L[v]);
        COMPUTE_REAL j_R = F_common - (F_i_R[v] - F_v_R[v]);
        
        
        int num_faces_total = nx * ny * nz * NPTS_FACE;
        j_z_R[v * num_faces_total + elem_L * NPTS_FACE + pt] = j_L;
        j_z_L[v * num_faces_total + elem_R * NPTS_FACE + pt] = j_R;

    }
}
