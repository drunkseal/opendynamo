extern "C" __global__ void init_flow_kernel(U_REAL* U, int nx, int ny, int nz, COMPUTE_REAL mach_bulk) {
    int elem_idx = blockIdx.x;
    int pt = threadIdx.x;
    int num_elements = nx * ny * nz;
    int total_points = num_elements * N_POINTS_ELEM;
    int idx = elem_idx * 64 + pt;
    
    if (idx < total_points) {
        int ey = (elem_idx / nx) % ny;
        int j = (pt / 4) % 4; // y-index of the point
        
        COMPUTE_REAL dy = ((COMPUTE_REAL)2.0) / ny;
        COMPUTE_REAL xi = c_points[j]; // 1D point in [-1, 1]
        COMPUTE_REAL y = ((COMPUTE_REAL)-1.0) + dy * (ey + ((COMPUTE_REAL)0.5) + ((COMPUTE_REAL)0.5) * xi);
        
        COMPUTE_REAL u_vel = ((COMPUTE_REAL)1.5) * mach_bulk * (((COMPUTE_REAL)1.0) - y*y); 
        
        COMPUTE_REAL rho = ((COMPUTE_REAL)1.0);
        COMPUTE_REAL p = ((COMPUTE_REAL)1.0) / GAMMA;
        
        U[0 * total_points + idx] = rho;
        U[1 * total_points + idx] = rho * u_vel;
        U[2 * total_points + idx] = ((COMPUTE_REAL)0.0);
        U[3 * total_points + idx] = ((COMPUTE_REAL)0.0);
        
        COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * rho * (u_vel*u_vel);
        U[4 * total_points + idx] = p / (GAMMA - ((COMPUTE_REAL)1.0)) + ke;
    }
}
