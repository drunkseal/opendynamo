extern "C" __global__ void init_flow_kernel(U_REAL* U, int nx, int ny, int nz, COMPUTE_REAL mach_bulk, COMPUTE_REAL ly) {
    int elem_idx = blockIdx.x;
    int pt = threadIdx.x;
    int num_elements = nx * ny * nz;
    int total_points = num_elements * NPTS_ELEM;
    int idx = elem_idx * NPTS_ELEM + pt;

    if (idx < total_points) {
        COMPUTE_REAL u_vel;
#if INIT_CHANNEL
        int ey = (elem_idx / nx) % ny;
        int j = (pt / NPTS1D) % NPTS1D; // y-index of the point

        COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
        COMPUTE_REAL y_wall = ((COMPUTE_REAL)0.5) * ly;
        COMPUTE_REAL y = -y_wall + dy * (ey + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * c_points[j]);
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
