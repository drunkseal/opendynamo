extern "C" __global__ void rk_stage_kernel(U_REAL* u, const U_REAL* u0, const RHS_REAL* rhs, COMPUTE_REAL a, COMPUTE_REAL b, COMPUTE_REAL c_dt, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) u[idx] = a * u0[idx] + b * u[idx] + c_dt * rhs[idx];
}

extern "C" __global__ void rk_stage1_kernel(U_REAL* u, const U_REAL* u0, const RHS_REAL* rhs, COMPUTE_REAL a, COMPUTE_REAL c_dt, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) u[idx] = a * u0[idx] + c_dt * rhs[idx];
}
