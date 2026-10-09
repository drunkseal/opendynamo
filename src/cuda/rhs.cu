template <typename T> struct V4Type;
template <> struct V4Type<float>  { typedef float4  type; };
template <> struct V4Type<double> { typedef double4 type; };

template <typename T>
__device__ __forceinline__ typename V4Type<T>::type ld4(const T* p) {
    return *reinterpret_cast<const typename V4Type<T>::type*>(p);
}

// Fused gradient + volume RHS. The gradient stays in registers (Q eliminated):
// boundary extrapolations are written to per-face slices for the face kernels,
// and the volume integrals consume the register copy directly.
extern "C" __global__ void compute_rhs_volume_kernel(const U_REAL* U, RHS_REAL* RHS,
    const COMPUTE_REAL* g_x_L, const COMPUTE_REAL* g_x_R,
    const COMPUTE_REAL* g_y_L, const COMPUTE_REAL* g_y_R,
    const COMPUTE_REAL* g_z_L, const COMPUTE_REAL* g_z_R,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz) {
    __shared__ COMPUTE_REAL s_U[5][NPTS_ELEM];
    __shared__ COMPUTE_REAL s_Q[15][NPTS_ELEM];

    int pt = threadIdx.x;
    int elem_idx = blockIdx.x;
    int total_points = nx * ny * nz * NPTS_ELEM;
    int g_idx = elem_idx * NPTS_ELEM + pt;
    int i = pt % NPTS1D;
    int j = (pt / NPTS1D) % NPTS1D;
    int k = pt / NPTS1D2;
    int ex = elem_idx % nx;
    int ey = (elem_idx / nx) % ny;
    int ez = elem_idx / (nx * ny);
    const unsigned full_mask = 0xffffffffu;

    for (int v = 0; v < 5; ++v) s_U[v][pt] = U[v * total_points + g_idx];
    __syncthreads();

    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx; COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy; COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;

    COMPUTE_REAL q_x[5] = {0}, q_y[5] = {0}, q_z[5] = {0};

    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL u = s_U[v][pt];
        // dU/dzeta: group is stride-NPTS1D2 (crosses warps) -> direct shared reads
        COMPUTE_REAL dU_dzeta = 0;
        for (int m = 0; m < NPTS1D; ++m) dU_dzeta += c_d_matrix[k * NPTS1D + m] * s_U[v][(i + j * NPTS1D + m * NPTS1D2)];
        // dU/dxi: quad over i -> broadcast via shfl 1,2,3; U_m at register i^m
        COMPUTE_REAL u1 = __shfl_xor_sync(full_mask, u, 1);
        COMPUTE_REAL u2 = __shfl_xor_sync(full_mask, u, 2);
        COMPUTE_REAL u3 = __shfl_xor_sync(full_mask, u, 3);
        COMPUTE_REAL dU_dxi = 0;
        for (int m = 0; m < NPTS1D; ++m) {
            COMPUTE_REAL um = (i == m) ? u : (i == (m ^ 1)) ? u1 : (i == (m ^ 2)) ? u2 : u3;
            dU_dxi += c_d_matrix[i * NPTS1D + m] * um;
        }
        // dU/deta: group is stride-NPTS1D -> broadcast via shfl 4,8,12; U_m at register j^m
        COMPUTE_REAL e1 = __shfl_xor_sync(full_mask, u, 4);
        COMPUTE_REAL e2 = __shfl_xor_sync(full_mask, u, 8);
        COMPUTE_REAL e3 = __shfl_xor_sync(full_mask, u, 12);
        COMPUTE_REAL dU_deta = 0;
        for (int m = 0; m < NPTS1D; ++m) {
            COMPUTE_REAL um = (j == m) ? u : (j == (m ^ 1)) ? e1 : (j == (m ^ 2)) ? e2 : e3;
            dU_deta += c_d_matrix[j * NPTS1D + m] * um;
        }
        q_x[v] = dU_dxi * Jx;
        q_y[v] = dU_deta * Jy;
        q_z[v] = dU_dzeta * Jz;
    }

    COMPUTE_REAL ul[5], ur[5];

    // --- X Faces ---
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = s_U[v][pt];
        COMPUTE_REAL a = t * c_extrap_left[i];
        a += __shfl_xor_sync(full_mask, a, 1);
        a += __shfl_xor_sync(full_mask, a, 2);
        ul[v] = a;
        COMPUTE_REAL b = t * c_extrap_right[i];
        b += __shfl_xor_sync(full_mask, b, 1);
        b += __shfl_xor_sync(full_mask, b, 2);
        ur[v] = b;
    }
    int nx_left = (ex - 1 + nx) % nx; int n_elem_xL = nx_left + nx * (ey + ny * ez);
    int nx_right = (ex + 1) % nx; int n_elem_xR = nx_right + nx * (ey + ny * ez);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_xL * NPTS_ELEM + pt] * c_extrap_right[i];
        t += __shfl_xor_sync(full_mask, t, 1);
        t += __shfl_xor_sync(full_mask, t, 2);
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v];
        q_x[v] += jump * c_g_left_prime[i] * Jx;
    }
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_xR * NPTS_ELEM + pt] * c_extrap_left[i];
        t += __shfl_xor_sync(full_mask, t, 1);
        t += __shfl_xor_sync(full_mask, t, 2);
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v];
        q_x[v] += jump * c_g_right_prime[i] * Jx;
    }

    // --- Y Faces ---
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = s_U[v][pt];
        COMPUTE_REAL a = t * c_extrap_left[j];
        a += __shfl_xor_sync(full_mask, a, 4);
        a += __shfl_xor_sync(full_mask, a, 8);
        ul[v] = a;
        COMPUTE_REAL b = t * c_extrap_right[j];
        b += __shfl_xor_sync(full_mask, b, 4);
        b += __shfl_xor_sync(full_mask, b, 8);
        ur[v] = b;
    }
#if BC_Y_WALL
    if (ey == 0) {
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL sign = (v > 0 && v < 4) ? ((COMPUTE_REAL)-1.0) : ((COMPUTE_REAL)1.0);
            COMPUTE_REAL U_nR = ul[v] * sign;
            COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (U_nR + ul[v]) - ul[v];
            q_y[v] += jump * c_g_left_prime[j] * Jy;
        }
    } else {
        int ny_left = ey - 1; int n_elem_yL = ex + nx * (ny_left + ny * ez);
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_yL * NPTS_ELEM + pt] * c_extrap_right[j];
            t += __shfl_xor_sync(full_mask, t, 4);
            t += __shfl_xor_sync(full_mask, t, 8);
            COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v];
            q_y[v] += jump * c_g_left_prime[j] * Jy;
        }
    }
    if (ey == ny - 1) {
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL sign = (v > 0 && v < 4) ? ((COMPUTE_REAL)-1.0) : ((COMPUTE_REAL)1.0);
            COMPUTE_REAL U_nL = ur[v] * sign;
            COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + U_nL) - ur[v];
            q_y[v] += jump * c_g_right_prime[j] * Jy;
        }
    } else {
        int ny_right = ey + 1; int n_elem_yR = ex + nx * (ny_right + ny * ez);
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_yR * NPTS_ELEM + pt] * c_extrap_left[j];
            t += __shfl_xor_sync(full_mask, t, 4);
            t += __shfl_xor_sync(full_mask, t, 8);
            COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v];
            q_y[v] += jump * c_g_right_prime[j] * Jy;
        }
    }
#else
    int ny_left = (ey - 1 + ny) % ny; int n_elem_yL = ex + nx * (ny_left + ny * ez);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_yL * NPTS_ELEM + pt] * c_extrap_right[j];
        t += __shfl_xor_sync(full_mask, t, 4);
        t += __shfl_xor_sync(full_mask, t, 8);
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v];
        q_y[v] += jump * c_g_left_prime[j] * Jy;
    }
    int ny_right = (ey + 1) % ny; int n_elem_yR = ex + nx * (ny_right + ny * ez);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_yR * NPTS_ELEM + pt] * c_extrap_left[j];
        t += __shfl_xor_sync(full_mask, t, 4);
        t += __shfl_xor_sync(full_mask, t, 8);
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v];
        q_y[v] += jump * c_g_right_prime[j] * Jy;
    }
#endif

    // --- Z Faces ---
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL a = 0, b = 0;
        for (int m = 0; m < NPTS1D; ++m) {
            COMPUTE_REAL uv = s_U[v][(i + j * NPTS1D + m * NPTS1D2)];
            a += uv * c_extrap_left[m];
            b += uv * c_extrap_right[m];
        }
        ul[v] = a; ur[v] = b;
    }
    int nz_left = (ez - 1 + nz) % nz; int n_elem_zL = ex + nx * (ey + ny * nz_left);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = 0;
        for (int m = 0; m < NPTS1D; ++m)
            t += (COMPUTE_REAL)U[v * total_points + n_elem_zL * NPTS_ELEM + i + j * NPTS1D + m * NPTS1D2] * c_extrap_right[m];
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v];
        q_z[v] += jump * c_g_left_prime[k] * Jz;
    }

    int nz_right = (ez + 1) % nz; int n_elem_zR = ex + nx * (ey + ny * nz_right);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = 0;
        for (int m = 0; m < NPTS1D; ++m)
            t += (COMPUTE_REAL)U[v * total_points + n_elem_zR * NPTS_ELEM + i + j * NPTS1D + m * NPTS1D2] * c_extrap_left[m];
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v];
        q_z[v] += jump * c_g_right_prime[k] * Jz;
    }

    // Q_REAL roundtrip: replicates the old global store/load of Q.
    for (int v = 0; v < 5; ++v) {
        q_x[v] = (COMPUTE_REAL)(Q_REAL)q_x[v];
        q_y[v] = (COMPUTE_REAL)(Q_REAL)q_y[v];
        q_z[v] = (COMPUTE_REAL)(Q_REAL)q_z[v];
        s_Q[v * 3 + 0][pt] = q_x[v];
        s_Q[v * 3 + 1][pt] = q_y[v];
        s_Q[v * 3 + 2][pt] = q_z[v];
    }
    __syncthreads();

    // --- Face slices: extrapolate the gradient to the 6 element boundaries.
    // Accumulation order replicates the face kernels' old in-kernel Q
    // extrapolation exactly, so downstream results stay bit-identical.
    // Row layout: 12 floats, components in ascending v skipping the set unused
    // by that direction's viscous flux (pad slot stays zero).
    if (i == 0) {
        int fp = j + k * NPTS1D;
        COMPUTE_REAL rL[12] = {0}, rR[12] = {0};
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 8 || v == 10 || v == 13 || v == 14) continue;
            int so = v; // s_Q rows are flattened components (var*3 + dir)
            // x slice: extrap along i at fixed (j, k)
            COMPUTE_REAL aL = s_Q[so][0 + j * NPTS1D + k * NPTS1D2] * c_extrap_left[0];
            aL += s_Q[so][1 + j * NPTS1D + k * NPTS1D2] * c_extrap_left[1];
            aL += s_Q[so][2 + j * NPTS1D + k * NPTS1D2] * c_extrap_left[2];
            aL += s_Q[so][3 + j * NPTS1D + k * NPTS1D2] * c_extrap_left[3];
            COMPUTE_REAL aR = s_Q[so][0 + j * NPTS1D + k * NPTS1D2] * c_extrap_right[0];
            aR += s_Q[so][1 + j * NPTS1D + k * NPTS1D2] * c_extrap_right[1];
            aR += s_Q[so][2 + j * NPTS1D + k * NPTS1D2] * c_extrap_right[2];
            aR += s_Q[so][3 + j * NPTS1D + k * NPTS1D2] * c_extrap_right[3];
            rL[qi] = aL; rR[qi] = aR;
            ++qi;
        }
        COMPUTE_REAL* pL = (COMPUTE_REAL*)(g_x_L + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        COMPUTE_REAL* pR = (COMPUTE_REAL*)(g_x_R + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[0] = ld4(rL);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[4] = ld4(rL + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[8] = ld4(rL + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[0] = ld4(rR);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[4] = ld4(rR + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[8] = ld4(rR + 8);
    }
    if (j == 0) {
        int fp = i + k * NPTS1D;
        COMPUTE_REAL rL[12] = {0}, rR[12] = {0};
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 5 || v == 9 || v == 12 || v == 14) continue;
            int so = v; // s_Q rows are flattened components (var*3 + dir)
            // y slice: extrap along j at fixed (i, k).
            // Face_y's shuffle-reduce cannot FMA-contract, so its q is a PLAIN
            // pair sum of plain products. Force the same here: volatile products
            // (no mul feeds the adds => no contraction).
            COMPUTE_REAL s0 = s_Q[so][i + 0 * NPTS1D + k * NPTS1D2];
            COMPUTE_REAL s1 = s_Q[so][i + 1 * NPTS1D + k * NPTS1D2];
            COMPUTE_REAL s2 = s_Q[so][i + 2 * NPTS1D + k * NPTS1D2];
            COMPUTE_REAL s3 = s_Q[so][i + 3 * NPTS1D + k * NPTS1D2];
            rL[qi] = (__fmul_rn(s0, c_extrap_left[0]) + __fmul_rn(s1, c_extrap_left[1]))
                   + (__fmul_rn(s2, c_extrap_left[2]) + __fmul_rn(s3, c_extrap_left[3]));
            rR[qi] = (__fmul_rn(s0, c_extrap_right[0]) + __fmul_rn(s1, c_extrap_right[1]))
                   + (__fmul_rn(s2, c_extrap_right[2]) + __fmul_rn(s3, c_extrap_right[3]));
            ++qi;
        }
        COMPUTE_REAL* pL = (COMPUTE_REAL*)(g_y_L + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        COMPUTE_REAL* pR = (COMPUTE_REAL*)(g_y_R + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[0] = ld4(rL);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[4] = ld4(rL + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[8] = ld4(rL + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[0] = ld4(rR);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[4] = ld4(rR + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[8] = ld4(rR + 8);
    }
    if (k == 0) {
        int fp = i + j * NPTS1D;
        COMPUTE_REAL rL[12] = {0}, rR[12] = {0};
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 4 || v == 6 || v == 12 || v == 13) continue;
            int so = v; // s_Q rows are flattened components (var*3 + dir)
            // z slice: extrap along k at fixed (i, j)
            COMPUTE_REAL aL = 0, aR = 0;
            for (int m = 0; m < NPTS1D; ++m) {
                COMPUTE_REAL uv = s_Q[so][i + j * NPTS1D + m * NPTS1D2];
                aL += uv * c_extrap_left[m];
                aR += uv * c_extrap_right[m];
            }
            rL[qi] = aL; rR[qi] = aR;
            ++qi;
        }
        COMPUTE_REAL* pL = (COMPUTE_REAL*)(g_z_L + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        COMPUTE_REAL* pR = (COMPUTE_REAL*)(g_z_R + ((size_t)elem_idx * NPTS_FACE + fp) * 12);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[0] = ld4(rL);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[4] = ld4(rL + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pL[8] = ld4(rL + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[0] = ld4(rR);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[4] = ld4(rR + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&pR[8] = ld4(rR + 8);
    }

    COMPUTE_REAL rhs_acc[5] = {0};
    COMPUTE_REAL u_tmp[5];
    for (int v = 0; v < 5; ++v) u_tmp[v] = s_U[v][pt];
    COMPUTE_REAL q_tmp[15];
    for (int v = 0; v < 15; ++v) q_tmp[v] = s_Q[v][pt];

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

    // Face corrections from the j arrays are applied in the RK kernels.
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL rhs_val = -rhs_acc[v];
        if (v == 1) rhs_val += ((COMPUTE_REAL)0.001);
        RHS[v * total_points + g_idx] = (RHS_REAL)rhs_val;
    }
}

extern "C" __global__ void compute_face_flux_x_kernel(const U_REAL* U, COMPUTE_REAL* j_x_L, COMPUTE_REAL* j_x_R, const COMPUTE_REAL* g_x_L, const COMPUTE_REAL* g_x_R, int nx, int ny, int nz, COMPUTE_REAL lx) {
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
    // Gradient boundary slices: q_L = elem_L's +side extrap, q_R = elem_R's -side extrap.
    {
        const COMPUTE_REAL* sLs = g_x_R + ((size_t)elem_L * NPTS_FACE + pt) * 12;
        const COMPUTE_REAL* sRs = g_x_L + ((size_t)elem_R * NPTS_FACE + pt) * 12;
        COMPUTE_REAL rowL[12], rowR[12];
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[0] = ld4(sLs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[4] = ld4(sLs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[8] = ld4(sLs + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[0] = ld4(sRs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[4] = ld4(sRs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[8] = ld4(sRs + 8);
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 8 || v == 10 || v == 13 || v == 14) continue;
            q_L[v] = rowL[qi];
            q_R[v] = rowR[qi];
            ++qi;
        }
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

extern "C" __global__ void compute_face_flux_y_kernel(const U_REAL* U, COMPUTE_REAL* j_y_L, COMPUTE_REAL* j_y_R, const COMPUTE_REAL* g_y_L, const COMPUTE_REAL* g_y_R, int nx, int ny, int nz, COMPUTE_REAL ly) {
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
    // Gradient boundary slices: q_L = elem_L's +side extrap, q_R = elem_R's -side extrap.
    {
        const COMPUTE_REAL* sLs = g_y_R + ((size_t)elem_L * NPTS_FACE + pt) * 12;
        const COMPUTE_REAL* sRs = g_y_L + ((size_t)elem_R * NPTS_FACE + pt) * 12;
        COMPUTE_REAL rowL[12], rowR[12];
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[0] = ld4(sLs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[4] = ld4(sLs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[8] = ld4(sLs + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[0] = ld4(sRs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[4] = ld4(sRs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[8] = ld4(sRs + 8);
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 5 || v == 9 || v == 12 || v == 14) continue;
            q_L[v] = rowL[qi];
            q_R[v] = rowR[qi];
            ++qi;
        }
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

extern "C" __global__ void compute_face_flux_z_kernel(const U_REAL* U, COMPUTE_REAL* j_z_L, COMPUTE_REAL* j_z_R, const COMPUTE_REAL* g_z_L, const COMPUTE_REAL* g_z_R, int nx, int ny, int nz, COMPUTE_REAL lz) {
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
    }
    // Gradient boundary slices: q_L = elem_L's +side extrap, q_R = elem_R's -side extrap.
    {
        const COMPUTE_REAL* sLs = g_z_R + ((size_t)elem_L * NPTS_FACE + pt) * 12;
        const COMPUTE_REAL* sRs = g_z_L + ((size_t)elem_R * NPTS_FACE + pt) * 12;
        COMPUTE_REAL rowL[12], rowR[12];
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[0] = ld4(sLs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[4] = ld4(sLs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowL[8] = ld4(sLs + 8);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[0] = ld4(sRs);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[4] = ld4(sRs + 4);
        *(typename V4Type<COMPUTE_REAL>::type*)&rowR[8] = ld4(sRs + 8);
        int qi = 0;
        for (int v = 0; v < 15; ++v) {
            if (v == 4 || v == 6 || v == 12 || v == 13) continue;
            q_L[v] = rowL[qi];
            q_R[v] = rowR[qi];
            ++qi;
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
