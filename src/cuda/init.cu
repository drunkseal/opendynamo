extern "C" __global__ void init_flow_kernel(U_REAL* U, int nx, int ny, int nz, COMPUTE_REAL mach_bulk, COMPUTE_REAL ly) {
    int elem_idx = blockIdx.x;
    int pt = threadIdx.x;
    int num_elements = nx * ny * nz;
    int total_points = num_elements * NPTS_ELEM;
    int idx = elem_idx * NPTS_ELEM + pt;

    if (idx < total_points) {
        COMPUTE_REAL u_vel;
#if INIT_CHANNEL
        int elem_y = (elem_idx / nx) % ny;
        int j = (pt / NPTS1D) % NPTS1D; // y-index of the point

        COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
        COMPUTE_REAL y_wall = ((COMPUTE_REAL)0.5) * ly;
        COMPUTE_REAL y = -y_wall + dy * (elem_y + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * c_points[j]);
        COMPUTE_REAL eta = y / y_wall;
        u_vel = ((COMPUTE_REAL)1.5) * mach_bulk * (((COMPUTE_REAL)1.0) - eta * eta);
#else
        u_vel = mach_bulk;
#endif
        COMPUTE_REAL rho = ((COMPUTE_REAL)1.0);
        COMPUTE_REAL p = ((COMPUTE_REAL)1.0) / GAMMA;

        U[0 * total_points + idx] = rho;
        U[1 * total_points + idx] = rho * u_vel;
        U[2 * total_points + idx] = ((COMPUTE_REAL)0.0);
        U[3 * total_points + idx] = ((COMPUTE_REAL)0.0);

        COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * rho * (u_vel * u_vel);
        U[4 * total_points + idx] = p / (GAMMA - ((COMPUTE_REAL)1.0)) + ke;
    }
}

// ---------------------------------------------------------------------------
// Divergence-free wall-damped tripping perturbation, applied right after
// init_flow_kernel. Modes of the form
//   v' = V(y) * cos(theta)
//   u' = -(kz*C(y)/k_h) * cos(theta) - (kx*V'(y)/k_h^2) * sin(theta)
//   w' = +(kx*C(y)/k_h) * cos(theta) - (kz*V'(y)/k_h^2) * sin(theta)
// with theta = kx*x + kz*z + phi satisfy div(u') = 0 identically for any
// wall profiles V, C (check: d(u')/dx + d(v')/dy + d(w')/dz = -V'cos + V'cos).
// Both profiles use tau(y) = 1 - eta^2 (eta = y/(ly/2)), so the perturbation
// and its wall-normal value vanish at the slip walls. Modes are hashed
// deterministically from the mode index and seed (reproducible runs).
// Total energy is updated as E += 0.5/rho * (|momentum_new|^2 - |momentum|^2);
// rho and pressure are untouched, so the trip is purely vortical.
// ---------------------------------------------------------------------------
__device__ __forceinline__ COMPUTE_REAL hash01(float s) {
    COMPUTE_REAL x = (COMPUTE_REAL)(sinf(s) * 43758.5453f);
    return x - floorf(x);
}

extern "C" __global__ void perturb_flow_kernel(U_REAL* U, int nx, int ny, int nz,
    COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz,
    COMPUTE_REAL amp, int n_modes, COMPUTE_REAL seed) {
    int elem_idx = blockIdx.x;
    int pt = threadIdx.x;
    int num_elements = nx * ny * nz;
    int total_points = num_elements * NPTS_ELEM;
    int idx = elem_idx * NPTS_ELEM + pt;

    if (idx < total_points) {
        int i = pt % NPTS1D;
        int j = (pt / NPTS1D) % NPTS1D;
        int k = pt / NPTS1D2;
        int elem_x = elem_idx % nx;
        int elem_y = (elem_idx / nx) % ny;
        int elem_z = elem_idx / (nx * ny);

        COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
        COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
        COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
        COMPUTE_REAL x = ((COMPUTE_REAL)-0.5) * lx + dx * (elem_x + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * c_points[i]);
        COMPUTE_REAL y = ((COMPUTE_REAL)-0.5) * ly + dy * (elem_y + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * c_points[j]);
        COMPUTE_REAL z = ((COMPUTE_REAL)-0.5) * lz + dz * (elem_z + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * c_points[k]);

        COMPUTE_REAL hw = ((COMPUTE_REAL)0.5) * ly;
        COMPUTE_REAL eta = y / hw;
        COMPUTE_REAL tau = ((COMPUTE_REAL)1.0) - eta * eta;
        COMPUTE_REAL tau_y = ((COMPUTE_REAL)-2.0) * eta / hw;

        COMPUTE_REAL du = ((COMPUTE_REAL)0.0), dv = ((COMPUTE_REAL)0.0), dw = ((COMPUTE_REAL)0.0);
        COMPUTE_REAL weight = amp / sqrtf((COMPUTE_REAL)n_modes);

        for (int m = 0; m < n_modes; ++m) {
            float mix = ((float)m) * 0.113f;
            COMPUTE_REAL r1 = hash01(mix * 1.113f + (float)seed - 3.7183f);
            COMPUTE_REAL r2 = hash01(mix * 2.229f + (float)seed + 1.7429f);
            COMPUTE_REAL r3 = hash01(mix * 3.717f + (float)seed + 4.8527f);
            COMPUTE_REAL r4 = hash01(mix * 5.381f + (float)seed - 2.9956f);
            COMPUTE_REAL r5 = hash01(mix * 7.993f + (float)seed + 6.1141f);
            COMPUTE_REAL r6 = hash01(mix * 9.677f + (float)seed + 8.2437f);

            int mx = 1 + (int)(r1 * ((COMPUTE_REAL)24.0));
            int mz = 1 + (int)(r2 * ((COMPUTE_REAL)12.0));
            COMPUTE_REAL kx = ((COMPUTE_REAL)6.283185307179586) * ((COMPUTE_REAL)mx) / lx;
            COMPUTE_REAL kz = ((COMPUTE_REAL)6.283185307179586) * ((COMPUTE_REAL)mz) / lz;
            COMPUTE_REAL phi = ((COMPUTE_REAL)6.283185307179586) * r3;
            COMPUTE_REAL av = ((COMPUTE_REAL)2.0) * r4 - ((COMPUTE_REAL)1.0);
            COMPUTE_REAL ac = ((COMPUTE_REAL)2.0) * r5 - ((COMPUTE_REAL)1.0);
            COMPUTE_REAL wj = ((COMPUTE_REAL)0.5) + r6;

            COMPUTE_REAL theta = kx * x + kz * z + phi;
            COMPUTE_REAL cosT = cos(theta);
            COMPUTE_REAL sinT = sin(theta);
            COMPUTE_REAL kh2 = kx * kx + kz * kz;
            COMPUTE_REAL kh = sqrtf(kh2);

            COMPUTE_REAL V = weight * wj * av * tau;
            COMPUTE_REAL Vp = weight * wj * av * tau_y;
            COMPUTE_REAL C = weight * wj * ac * tau;

            du += (COMPUTE_REAL)(((COMPUTE_REAL)(-1.0)) * (kz * C / kh)) * cosT - (kx * Vp / kh2) * sinT;
            dv += V * cosT;
            dw += (kx * C / kh) * cosT - (kz * Vp / kh2) * sinT;
        }

        COMPUTE_REAL rho = (COMPUTE_REAL)U[0 * total_points + idx];
        COMPUTE_REAL m1 = (COMPUTE_REAL)U[1 * total_points + idx];
        COMPUTE_REAL m2 = (COMPUTE_REAL)U[2 * total_points + idx];
        COMPUTE_REAL m3 = (COMPUTE_REAL)U[3 * total_points + idx];
        COMPUTE_REAL E = (COMPUTE_REAL)U[4 * total_points + idx];

        COMPUTE_REAL n1 = m1 + rho * du;
        COMPUTE_REAL n2 = m2 + rho * dv;
        COMPUTE_REAL n3 = m3 + rho * dw;
        E += ((COMPUTE_REAL)0.5) * ((n1 * n1 + n2 * n2 + n3 * n3) - (m1 * m1 + m2 * m2 + m3 * m3)) / rho;

        U[1 * total_points + idx] = (U_REAL)n1;
        U[2 * total_points + idx] = (U_REAL)n2;
        U[3 * total_points + idx] = (U_REAL)n3;
        U[4 * total_points + idx] = (U_REAL)E;
    }
}
