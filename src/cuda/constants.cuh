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

// FR polynomial order: single source of truth is constants.rs, passed via -DFR_ORDER.
#ifndef FR_ORDER
#define FR_ORDER 3
#endif
#define NPTS1D (FR_ORDER + 1)
#define NPTS1D2 (NPTS1D * NPTS1D)
#define NPTS_FACE NPTS1D2
#define NPTS_ELEM (NPTS1D2 * NPTS1D)
#define FACES_PER_BLOCK (NPTS_ELEM / NPTS_FACE)

#if NPTS1D != 4
#error "FR_ORDER other than 3 unsupported: float4 loads and 4-lane butterfly shuffles assume 4 points per direction"
#endif

#ifndef INIT_CHANNEL
#error "INIT_CHANNEL must be defined via -D (1 = channel profile, 0 = uniform)"
#endif
#ifndef BC_Y_WALL
#error "BC_Y_WALL must be defined via -D (1 = slip walls, 0 = periodic)"
#endif

#define N_VARS 5
#define N_POINTS_ELEM NPTS_ELEM
#define GAMMA (((COMPUTE_REAL)1.4))
#define PR (((COMPUTE_REAL)0.72))
#define MU (((COMPUTE_REAL)0.00016))

// FR Operators in constant memory
__constant__ COMPUTE_REAL c_points[NPTS1D];
__constant__ COMPUTE_REAL c_extrap_left[NPTS1D];
__constant__ COMPUTE_REAL c_extrap_right[NPTS1D];
__constant__ COMPUTE_REAL c_g_left_prime[NPTS1D];
__constant__ COMPUTE_REAL c_g_right_prime[NPTS1D];
__constant__ COMPUTE_REAL c_d_matrix[NPTS1D2];

#endif
