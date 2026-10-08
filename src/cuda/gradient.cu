
extern "C" __global__ void compute_gradient_kernel(const U_REAL* U, Q_REAL* Q, int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz) 
{
        
    __shared__ COMPUTE_REAL s_U[5][NPTS_ELEM];

    int pt = threadIdx.x; int elem_idx = blockIdx.x;
    int total_points = nx * ny * nz * NPTS_ELEM; int g_idx = elem_idx * NPTS_ELEM + pt;
    int i = pt % NPTS1D; int j = (pt / NPTS1D) % NPTS1D; int k = pt / NPTS1D2;
    int ex = elem_idx % nx; int ey = (elem_idx / nx) % ny; int ez = elem_idx / (nx * ny);
    const unsigned full_mask = 0xffffffffu;

    for (int v = 0; v < 5; ++v) s_U[v][pt] = U[v * total_points + g_idx];
    __syncthreads();
    

    COMPUTE_REAL q_x[5] = {0}, q_y[5] = {0}, q_z[5] = {0};
    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx; COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy; COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;

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
    // Own element extrapolation to both x-faces: quad reduce over i (lanes pt&~3).
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
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v]; // jump_L
        q_x[v] += jump * c_g_left_prime[i] * Jx;
    }
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = (COMPUTE_REAL)U[v * total_points + n_elem_xR * NPTS_ELEM + pt] * c_extrap_left[i];
        t += __shfl_xor_sync(full_mask, t, 1);
        t += __shfl_xor_sync(full_mask, t, 2);
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v]; // jump_R
        q_x[v] += jump * c_g_right_prime[i] * Jx;
    }

    // --- Y Faces ---
    // Own extrapolation to both y-faces: reduce over j (lanes stride 4, xor 4/8).
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
    // Reduce over k (crosses warps): each thread reads all points directly.
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
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (t + ul[v]) - ul[v]; // jump_L
        q_z[v] += jump * c_g_left_prime[k] * Jz;
    }

    int nz_right = (ez + 1) % nz; int n_elem_zR = ex + nx * (ey + ny * nz_right);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL t = 0;
        for (int m = 0; m < NPTS1D; ++m)
            t += (COMPUTE_REAL)U[v * total_points + n_elem_zR * NPTS_ELEM + i + j * NPTS1D + m * NPTS1D2] * c_extrap_left[m];
        COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (ur[v] + t) - ur[v]; // jump_R
        q_z[v] += jump * c_g_right_prime[k] * Jz;
    }

    for (int v = 0; v < 5; ++v) {
        Q[(v*3 + 0) * total_points + g_idx] = (Q_REAL)q_x[v];
        Q[(v*3 + 1) * total_points + g_idx] = (Q_REAL)q_y[v];
        Q[(v*3 + 2) * total_points + g_idx] = (Q_REAL)q_z[v];
    }
}
