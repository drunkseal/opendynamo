
extern "C" __global__ void compute_gradient_kernel(const U_REAL* U, Q_REAL* Q, int nx, int ny, int nz) 
{
        
    __shared__ COMPUTE_REAL s_U[5][64];
    __shared__ COMPUTE_REAL corr[5][64];
#define s_neigh s_U


    int pt = threadIdx.x; int elem_idx = blockIdx.x;
    int total_points = nx * ny * nz * 64; int g_idx = elem_idx * 64 + pt;
    int i = pt % 4; int j = (pt / 4) % 4; int k = pt / 16;
    int ex = elem_idx % nx; int ey = (elem_idx / nx) % ny; int ez = elem_idx / (nx * ny);

    for (int v = 0; v < 5; ++v) s_U[v][pt] = U[v * total_points + g_idx];
    __syncthreads();
    

    COMPUTE_REAL q_x[5] = {0}, q_y[5] = {0}, q_z[5] = {0};
    COMPUTE_REAL dx = ((COMPUTE_REAL)4.0) * ((COMPUTE_REAL)3.141592653589793) / nx;
    COMPUTE_REAL dy = ((COMPUTE_REAL)2.0) / ny;
    COMPUTE_REAL dz = ((COMPUTE_REAL)4.0) / ((COMPUTE_REAL)3.0) * ((COMPUTE_REAL)3.141592653589793) / nz;
    COMPUTE_REAL Jx = ((COMPUTE_REAL)2.0) / dx; COMPUTE_REAL Jy = ((COMPUTE_REAL)2.0) / dy; COMPUTE_REAL Jz = ((COMPUTE_REAL)2.0) / dz;

    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL dU_dxi = 0, dU_deta = 0, dU_dzeta = 0;
        for (int m = 0; m < 4; ++m) {
            dU_dxi += c_d_matrix[i * 4 + m] * s_U[v][(m + j * 4 + k * 16)];
            dU_deta += c_d_matrix[j * 4 + m] * s_U[v][(i + m * 4 + k * 16)];
            dU_dzeta += c_d_matrix[k * 4 + m] * s_U[v][(i + j * 4 + m * 16)];
        }
        q_x[v] = dU_dxi * Jx;
        q_y[v] = dU_deta * Jy;
        q_z[v] = dU_dzeta * Jz;
    }
    
    // --- X Faces ---
    __syncthreads();
    COMPUTE_REAL u_loc[5]; for(int v=0; v<5; ++v) u_loc[v] = s_U[v][pt];
    __syncthreads();

    COMPUTE_REAL ul_tmp[5] = {0}, ur_tmp[5] = {0};
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        for(int m=0; m<4; ++m) {
            int p_idx = (m + f1 * 4 + f2 * 16);
            for(int v=0; v<5; ++v) {
                ul_tmp[v] += s_U[v][p_idx] * c_extrap_left[m];
                ur_tmp[v] += s_U[v][p_idx] * c_extrap_right[m];
            }
        }
    }
    __syncthreads();
    
    int nx_left = (ex - 1 + nx) % nx; int n_elem_xL = nx_left + nx * (ey + ny * ez);
    for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_xL * 64 + pt];
    __syncthreads();
    
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        COMPUTE_REAL U_neigh_R[5] = {0};
        for(int m=0; m<4; ++m) {
            for(int v=0; v<5; ++v) U_neigh_R[v] += s_neigh[v][(m + f1 * 4 + f2 * 16)] * c_extrap_right[m];
        }
        for(int v=0; v<5; ++v) {
            corr[v][pt] = ((COMPUTE_REAL)0.5) * (U_neigh_R[v] + ul_tmp[v]) - ul_tmp[v]; // jump_L
        }
    }
    __syncthreads();

    int nx_right = (ex + 1) % nx; int n_elem_xR = nx_right + nx * (ey + ny * ez);
    for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_xR * 64 + pt];
    __syncthreads();
    
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        COMPUTE_REAL U_neigh_L[5] = {0};
        for(int m=0; m<4; ++m) {
            for(int v=0; v<5; ++v) U_neigh_L[v] += s_neigh[v][(m + f1 * 4 + f2 * 16)] * c_extrap_left[m];
        }
        for(int v=0; v<5; ++v) {
            corr[v][16 + pt] = ((COMPUTE_REAL)0.5) * (ur_tmp[v] + U_neigh_L[v]) - ur_tmp[v]; // jump_R
        }
    }
    __syncthreads();
    
    // Apply X corrections using all 64 threads
    int f_idx_x = j + k * 4;
    for(int v=0; v<5; ++v) {
        q_x[v] += corr[v][f_idx_x] * c_g_left_prime[i] * Jx;
        q_x[v] += corr[v][16 + f_idx_x] * c_g_right_prime[i] * Jx;
    }

    // --- Y Faces ---
    for(int v=0; v<5; ++v) s_U[v][pt] = u_loc[v]; // restore s_U
    __syncthreads();

    for(int v=0; v<5; ++v) { ul_tmp[v] = 0; ur_tmp[v] = 0; }
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        for(int m=0; m<4; ++m) {
            int p_idx = (f1 + m * 4 + f2 * 16);
            for(int v=0; v<5; ++v) {
                ul_tmp[v] += s_U[v][p_idx] * c_extrap_left[m];
                ur_tmp[v] += s_U[v][p_idx] * c_extrap_right[m];
            }
        }
    }
    __syncthreads();

    if (ey == 0) {
        if (pt < 16) {
            for(int v=0; v<5; ++v) {
                COMPUTE_REAL sign = (v > 0 && v < 4) ? ((COMPUTE_REAL)-1.0) : ((COMPUTE_REAL)1.0);
                COMPUTE_REAL U_nR = ul_tmp[v] * sign;
                corr[v][pt] = ((COMPUTE_REAL)0.5) * (U_nR + ul_tmp[v]) - ul_tmp[v];
            }
        }
    } else {
        int ny_left = ey - 1; int n_elem_yL = ex + nx * (ny_left + ny * ez);
        for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_yL * 64 + pt];
        __syncthreads();
        if (pt < 16) {
            int f1 = pt % 4; int f2 = pt / 4;
            COMPUTE_REAL U_neigh_R[5] = {0};
            for(int m=0; m<4; ++m) {
                for(int v=0; v<5; ++v) U_neigh_R[v] += s_neigh[v][(f1 + m * 4 + f2 * 16)] * c_extrap_right[m];
            }
            for(int v=0; v<5; ++v) {
                corr[v][pt] = ((COMPUTE_REAL)0.5) * (U_neigh_R[v] + ul_tmp[v]) - ul_tmp[v];
            }
        }
    }
    __syncthreads();

    if (ey == ny - 1) {
        if (pt < 16) {
            for(int v=0; v<5; ++v) {
                COMPUTE_REAL sign = (v > 0 && v < 4) ? ((COMPUTE_REAL)-1.0) : ((COMPUTE_REAL)1.0);
                COMPUTE_REAL U_nL = ur_tmp[v] * sign;
                corr[v][16 + pt] = ((COMPUTE_REAL)0.5) * (ur_tmp[v] + U_nL) - ur_tmp[v];
            }
        }
    } else {
        int ny_right = ey + 1; int n_elem_yR = ex + nx * (ny_right + ny * ez);
        for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_yR * 64 + pt];
        __syncthreads();
        if (pt < 16) {
            int f1 = pt % 4; int f2 = pt / 4;
            COMPUTE_REAL U_neigh_L[5] = {0};
            for(int m=0; m<4; ++m) {
                for(int v=0; v<5; ++v) U_neigh_L[v] += s_neigh[v][(f1 + m * 4 + f2 * 16)] * c_extrap_left[m];
            }
            for(int v=0; v<5; ++v) {
                corr[v][16 + pt] = ((COMPUTE_REAL)0.5) * (ur_tmp[v] + U_neigh_L[v]) - ur_tmp[v];
            }
        }
    }
    __syncthreads();
    
    int f_idx_y = i + k * 4;
    for(int v=0; v<5; ++v) {
        q_y[v] += corr[v][f_idx_y] * c_g_left_prime[j] * Jy;
        q_y[v] += corr[v][16 + f_idx_y] * c_g_right_prime[j] * Jy;
    }

    // --- Z Faces ---
    for(int v=0; v<5; ++v) { s_U[v][pt] = u_loc[v]; } // restore s_U
    __syncthreads();
    
    for(int v=0; v<5; ++v) { ul_tmp[v] = 0; ur_tmp[v] = 0; }
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        for(int m=0; m<4; ++m) {
            int p_idx = (f1 + f2 * 4 + m * 16);
            for(int v=0; v<5; ++v) {
                ul_tmp[v] += s_U[v][p_idx] * c_extrap_left[m];
                ur_tmp[v] += s_U[v][p_idx] * c_extrap_right[m];
            }
        }
    }
    __syncthreads();
    
    int nz_left = (ez - 1 + nz) % nz; int n_elem_zL = ex + nx * (ey + ny * nz_left);
    for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_zL * 64 + pt];
    __syncthreads();
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        COMPUTE_REAL U_neigh_R[5] = {0};
        for(int m=0; m<4; ++m) {
            for(int v=0; v<5; ++v) U_neigh_R[v] += s_neigh[v][(f1 + f2 * 4 + m * 16)] * c_extrap_right[m];
        }
        for(int v=0; v<5; ++v) {
            corr[v][pt] = ((COMPUTE_REAL)0.5) * (U_neigh_R[v] + ul_tmp[v]) - ul_tmp[v];
        }
    }
    __syncthreads();
    
    int nz_right = (ez + 1) % nz; int n_elem_zR = ex + nx * (ey + ny * nz_right);
    for (int v = 0; v < 5; ++v) s_neigh[v][pt] = U[v * total_points + n_elem_zR * 64 + pt];
    __syncthreads();
    if (pt < 16) {
        int f1 = pt % 4; int f2 = pt / 4;
        COMPUTE_REAL U_neigh_L[5] = {0};
        for(int m=0; m<4; ++m) {
            for(int v=0; v<5; ++v) U_neigh_L[v] += s_neigh[v][(f1 + f2 * 4 + m * 16)] * c_extrap_left[m];
        }
        for(int v=0; v<5; ++v) {
            corr[v][16 + pt] = ((COMPUTE_REAL)0.5) * (ur_tmp[v] + U_neigh_L[v]) - ur_tmp[v];
        }
    }
    __syncthreads();
    
    int f_idx_z = i + j * 4;
    for(int v=0; v<5; ++v) {
        q_z[v] += corr[v][f_idx_z] * c_g_left_prime[k] * Jz;
        q_z[v] += corr[v][16 + f_idx_z] * c_g_right_prime[k] * Jz;
        
        Q[(v*3 + 0) * total_points + g_idx] = (Q_REAL)q_x[v];
        Q[(v*3 + 1) * total_points + g_idx] = (Q_REAL)q_y[v];
        Q[(v*3 + 2) * total_points + g_idx] = (Q_REAL)q_z[v];
    }
}
