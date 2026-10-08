#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#ifndef U_REAL
#define U_REAL double
#endif

#ifndef Q_REAL
#define Q_REAL float
#endif

#ifndef RHS_REAL
#define RHS_REAL float
#endif

#ifndef COMPUTE_REAL
#define COMPUTE_REAL float
#endif

#define N_VARS 5
#define N_POINTS_ELEM 64
#define GAMMA (((COMPUTE_REAL)1.4))
#define PR (((COMPUTE_REAL)0.72))
#define MU (((COMPUTE_REAL)0.00016))

// FR Operators in constant memory
__constant__ COMPUTE_REAL c_points[4];
__constant__ COMPUTE_REAL c_extrap_left[4];
__constant__ COMPUTE_REAL c_extrap_right[4];
__constant__ COMPUTE_REAL c_g_left_prime[4];
__constant__ COMPUTE_REAL c_g_right_prime[4];
__constant__ COMPUTE_REAL c_d_matrix[16];

#endif
