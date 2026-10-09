template <typename T> struct V4Type;
template <> struct V4Type<float>  { typedef float4  type; };
template <> struct V4Type<double> { typedef double4 type; };

template <typename T>
__device__ __forceinline__ typename V4Type<T>::type ld4(const T* p) {
    return *reinterpret_cast<const typename V4Type<T>::type*>(p);
}

// ---------------------------------------------------------------------------
// Direction tags: each per-direction helper takes an empty tag so the caller
// spells the direction out (flux_dir_x) instead of passing a magic int.
// Ordinary overload dispatch, everything __forceinline__ -> same machine code
// as writing the direction-specific calls by hand.
// ---------------------------------------------------------------------------
struct flux_dir_x {};
struct flux_dir_y {};
struct flux_dir_z {};

__device__ __forceinline__ void dispatch_euler_flux(flux_dir_x, const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) { compute_euler_flux_x(cons_vars, flux); }
__device__ __forceinline__ void dispatch_euler_flux(flux_dir_y, const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) { compute_euler_flux_y(cons_vars, flux); }
__device__ __forceinline__ void dispatch_euler_flux(flux_dir_z, const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) { compute_euler_flux_z(cons_vars, flux); }

__device__ __forceinline__ void dispatch_viscous_flux(flux_dir_x, const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) { compute_viscous_flux_x(cons_vars, grad_cons, flux); }
__device__ __forceinline__ void dispatch_viscous_flux(flux_dir_y, const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) { compute_viscous_flux_y(cons_vars, grad_cons, flux); }
__device__ __forceinline__ void dispatch_viscous_flux(flux_dir_z, const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) { compute_viscous_flux_z(cons_vars, grad_cons, flux); }

// Volume-integral flux pair: Euler first, then viscous (original block order).
template <typename DIR>
__device__ __forceinline__ void compute_volume_fluxes(DIR dir, const COMPUTE_REAL* state, const COMPUTE_REAL* grad, COMPUTE_REAL* flux_euler, COMPUTE_REAL* flux_viscous) {
    dispatch_euler_flux(dir, state, flux_euler);
    dispatch_viscous_flux(dir, state, grad, flux_viscous);
}

// ---------------------------------------------------------------------------
// Fused-kernel helpers: analytical-derivative quadrature.
//
// The FR derivative along an axis is a sum over the 4-point quadrature nodes.
// Within one warp the 4 nodes of the (xi, eta) directions differ only in the
// lane bits the point index does not own, so the missing samples arrive via
// shfl_xor; the zeta group has stride NPTS1D2 (crosses warps) and reads shared
// memory directly.
// ---------------------------------------------------------------------------

// d/dzeta at point k. `row` = s_buf[v] + i + j*NPTS1D (start of this lane's k-column).
__device__ __forceinline__ COMPUTE_REAL derivative_zeta(int k, const COMPUTE_REAL* row) {
    COMPUTE_REAL d = 0;
    for (int m = 0; m < NPTS1D; ++m) d += c_d_matrix[k * NPTS1D + m] * row[m * NPTS1D2];
    return d;
}

// d/dxi (S* = 1,2,3 over i) or d/deta (S* = 4,8,12 over j) at lane `idx`.
template <int S1, int S2, int S3>
__device__ __forceinline__ COMPUTE_REAL derivative_shfl(int idx, COMPUTE_REAL point_value) {
    const unsigned mask = 0xffffffffu;
    COMPUTE_REAL u_1 = __shfl_xor_sync(mask, point_value, S1);
    COMPUTE_REAL u_2 = __shfl_xor_sync(mask, point_value, S2);
    COMPUTE_REAL u_3 = __shfl_xor_sync(mask, point_value, S3);
    COMPUTE_REAL d = 0;
    for (int m = 0; m < NPTS1D; ++m) {
        COMPUTE_REAL u_m = (idx == m) ? point_value : (idx == (m ^ 1)) ? u_1 : (idx == (m ^ 2)) ? u_2 : u_3;
        d += c_d_matrix[idx * NPTS1D + m] * u_m;
    }
    return d;
}

// Boundary extrapolation of this element's own samples along xi/eta (shfl reduce).
template <int S1, int S2>
__device__ __forceinline__ COMPUTE_REAL extrap_shfl(COMPUTE_REAL term, const COMPUTE_REAL* coeff, int idx) {
    const unsigned mask = 0xffffffffu;
    COMPUTE_REAL acc = term * coeff[idx];
    acc += __shfl_xor_sync(mask, acc, S1);
    acc += __shfl_xor_sync(mask, acc, S2);
    return acc;
}

// Same reduce over the neighbour element's point value loaded from global U.
template <int S1, int S2>
__device__ __forceinline__ COMPUTE_REAL neighbor_extrap_shfl(const U_REAL* U, int v, int total_points, int elem, int pt, int idx, const COMPUTE_REAL* coeff) {
    const unsigned mask = 0xffffffffu;
    COMPUTE_REAL term = (COMPUTE_REAL)U[v * total_points + elem * NPTS_ELEM + pt] * coeff[idx];
    term += __shfl_xor_sync(mask, term, S1);
    term += __shfl_xor_sync(mask, term, S2);
    return term;
}

// z-direction extrapolation: m-loop over the k-column in shared memory.
// `row` = s_buf[v] + i + j*NPTS1D.
__device__ __forceinline__ COMPUTE_REAL extrap_z(const COMPUTE_REAL* row, const COMPUTE_REAL* coeff) {
    COMPUTE_REAL acc = 0;
    for (int m = 0; m < NPTS1D; ++m) acc += row[m * NPTS1D2] * coeff[m];
    return acc;
}

// z-direction extrapolation of the neighbour element's k-column from global U.
__device__ __forceinline__ COMPUTE_REAL neighbor_extrap_z(const U_REAL* U, int v, int total_points, int elem, int i, int j, const COMPUTE_REAL* coeff) {
    COMPUTE_REAL term = 0;
    for (int m = 0; m < NPTS1D; ++m)
        term += (COMPUTE_REAL)U[v * total_points + elem * NPTS_ELEM + i + j * NPTS1D + m * NPTS1D2] * coeff[m];
    return term;
}

// Ghost-value jump: half the average of both sides minus the own-side value,
// folded into the stored gradient component.
__device__ __forceinline__ void accumulate_jump(COMPUTE_REAL& grad_comp, COMPUTE_REAL own, COMPUTE_REAL neighbor, COMPUTE_REAL g_prime, COMPUTE_REAL jac) {
    COMPUTE_REAL jump = ((COMPUTE_REAL)0.5) * (neighbor + own) - own;
    grad_comp += jump * g_prime * jac;
}

// Slip wall (BC_Y_WALL): mirror the wall-normal momentum across the boundary.
__device__ __forceinline__ COMPUTE_REAL reflect_wall(int v, COMPUTE_REAL value) {
    COMPUTE_REAL sign = (v > 0 && v < 4) ? ((COMPUTE_REAL)-1.0) : ((COMPUTE_REAL)1.0);
    return value * sign;
}

// ---------------------------------------------------------------------------
// Gradient boundary slices.
//
// The fused kernel writes 12-float rows per element face: gradient components
// in ascending order, skipping the set this direction's viscous flux does not
// read (slot 11 stays zero). The face kernels read them back via
// read_gradient_slice(). Row layout must stay in sync between the two.
// ---------------------------------------------------------------------------

// x slice: extrap along i at fixed (j, k).
__device__ __forceinline__ void build_slice_row_x(COMPUTE_REAL (*s_grad)[NPTS_ELEM], int j, int k, COMPUTE_REAL* row_left, COMPUTE_REAL* row_right) {
    int row_slot = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 8 || v == 10 || v == 13 || v == 14) continue;
        const COMPUTE_REAL* col = s_grad[v] + j * NPTS1D + k * NPTS1D2;
        COMPUTE_REAL extrap_left = col[0] * c_extrap_left[0];
        extrap_left += col[1] * c_extrap_left[1];
        extrap_left += col[2] * c_extrap_left[2];
        extrap_left += col[3] * c_extrap_left[3];
        COMPUTE_REAL extrap_right = col[0] * c_extrap_right[0];
        extrap_right += col[1] * c_extrap_right[1];
        extrap_right += col[2] * c_extrap_right[2];
        extrap_right += col[3] * c_extrap_right[3];
        row_left[row_slot] = extrap_left; row_right[row_slot] = extrap_right;
        ++row_slot;
    }
}

// y slice: extrap along j at fixed (i, k).
// Face_y's shuffle-reduce cannot FMA-contract, so its q is a PLAIN
// pair sum of plain products. Force the same here: __fmul_rn products
// (no mul feeds the adds => no contraction).
__device__ __forceinline__ void build_slice_row_y(COMPUTE_REAL (*s_grad)[NPTS_ELEM], int i, int k, COMPUTE_REAL* row_left, COMPUTE_REAL* row_right) {
    int row_slot = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 5 || v == 9 || v == 12 || v == 14) continue;
        const COMPUTE_REAL* col = s_grad[v] + i + k * NPTS1D2;
        COMPUTE_REAL sample_0 = col[0];
        COMPUTE_REAL sample_1 = col[NPTS1D];
        COMPUTE_REAL sample_2 = col[2 * NPTS1D];
        COMPUTE_REAL sample_3 = col[3 * NPTS1D];
        row_left[row_slot] = (__fmul_rn(sample_0, c_extrap_left[0]) + __fmul_rn(sample_1, c_extrap_left[1]))
               + (__fmul_rn(sample_2, c_extrap_left[2]) + __fmul_rn(sample_3, c_extrap_left[3]));
        row_right[row_slot] = (__fmul_rn(sample_0, c_extrap_right[0]) + __fmul_rn(sample_1, c_extrap_right[1]))
               + (__fmul_rn(sample_2, c_extrap_right[2]) + __fmul_rn(sample_3, c_extrap_right[3]));
        ++row_slot;
    }
}

// z slice: extrap along k at fixed (i, j).
__device__ __forceinline__ void build_slice_row_z(COMPUTE_REAL (*s_grad)[NPTS_ELEM], int i, int j, COMPUTE_REAL* row_left, COMPUTE_REAL* row_right) {
    int row_slot = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == 4 || v == 6 || v == 12 || v == 13) continue;
        const COMPUTE_REAL* col = s_grad[v] + i + j * NPTS1D;
        COMPUTE_REAL extrap_left = 0, extrap_right = 0;
        for (int m = 0; m < NPTS1D; ++m) {
            COMPUTE_REAL sample = col[m * NPTS1D2];
            extrap_left += sample * c_extrap_left[m];
            extrap_right += sample * c_extrap_right[m];
        }
        row_left[row_slot] = extrap_left; row_right[row_slot] = extrap_right;
        ++row_slot;
    }
}

// Row pair -> the element's two face slices (3x ld4/st4 each).
template <typename T>
__device__ __forceinline__ T* slice_ptr(const T* base, int elem_idx, int face_pt) {
    return (T*)(base + ((size_t)elem_idx * NPTS_FACE + face_pt) * 12);
}

template <typename T>
__device__ __forceinline__ void store_slice_rows(T* dst_left, T* dst_right, const T* row_left, const T* row_right) {
    *(typename V4Type<T>::type*)&dst_left[0] = ld4(row_left);
    *(typename V4Type<T>::type*)&dst_left[4] = ld4(row_left + 4);
    *(typename V4Type<T>::type*)&dst_left[8] = ld4(row_left + 8);
    *(typename V4Type<T>::type*)&dst_right[0] = ld4(row_right);
    *(typename V4Type<T>::type*)&dst_right[4] = ld4(row_right + 4);
    *(typename V4Type<T>::type*)&dst_right[8] = ld4(row_right + 8);
}

// Face kernels read back the 12-float boundary rows and scatter them into the
// 15-component gradient arrays, skipping components S0..S3 (the ones this
// direction's viscous flux ignores) in ascending order.
template <int S0, int S1, int S2, int S3>
__device__ __forceinline__ void read_gradient_slice(const COMPUTE_REAL* src_left, const COMPUTE_REAL* src_right, COMPUTE_REAL* grad_left, COMPUTE_REAL* grad_right) {
    COMPUTE_REAL row_left[12], row_right[12];
    *(typename V4Type<COMPUTE_REAL>::type*)&row_left[0] = ld4(src_left);
    *(typename V4Type<COMPUTE_REAL>::type*)&row_left[4] = ld4(src_left + 4);
    *(typename V4Type<COMPUTE_REAL>::type*)&row_left[8] = ld4(src_left + 8);
    *(typename V4Type<COMPUTE_REAL>::type*)&row_right[0] = ld4(src_right);
    *(typename V4Type<COMPUTE_REAL>::type*)&row_right[4] = ld4(src_right + 4);
    *(typename V4Type<COMPUTE_REAL>::type*)&row_right[8] = ld4(src_right + 8);
    int row_slot = 0;
    for (int v = 0; v < 15; ++v) {
        if (v == S0 || v == S1 || v == S2 || v == S3) continue;
        grad_left[v] = row_left[row_slot];
        grad_right[v] = row_right[row_slot];
        ++row_slot;
    }
}

// ---------------------------------------------------------------------------
// Volume integral.
//
// One direction: evaluate the fluxes, stage them in s_buf (the two barriers
// keep every thread's writes ahead of the next read), differentiate along the
// direction and accumulate into rhs_sum. `row` is the c_d_matrix row;
// the shared gather for sample m is index base + m*stride.
// ---------------------------------------------------------------------------
template <typename DIR>
__device__ __forceinline__ void accumulate_volume_rhs(
    COMPUTE_REAL (*s_buf)[NPTS_ELEM], const COMPUTE_REAL* state, const COMPUTE_REAL* grad,
    int row, int base, int stride, int pt, COMPUTE_REAL jac, COMPUTE_REAL* rhs_sum) {
    COMPUTE_REAL flux_euler[5], flux_viscous[5];
    compute_volume_fluxes(DIR{}, state, grad, flux_euler, flux_viscous);
    __syncthreads();
    for (int v = 0; v < 5; ++v) s_buf[v][pt] = flux_euler[v] - flux_viscous[v];
    __syncthreads();
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL d_flux = 0;
        for (int m = 0; m < NPTS1D; ++m) d_flux += c_d_matrix[row * NPTS1D + m] * s_buf[v][base + m * stride];
        rhs_sum[v] += d_flux * jac;
    }
}

// ---------------------------------------------------------------------------
// Face kernel tail: interface flux + the j corrections written back to both
// elements. `j_for_elem_left` is filled at elem_left, `j_for_elem_right` at
// elem_right (the *_right j-array owns the left element's slot and vice versa).
// ---------------------------------------------------------------------------
template <typename DIR>
__device__ __forceinline__ void compute_face_j(
    const COMPUTE_REAL* u_left, const COMPUTE_REAL* u_right,
    const COMPUTE_REAL* grad_left, const COMPUTE_REAL* grad_right,
    const COMPUTE_REAL* normal,
    COMPUTE_REAL* j_for_elem_left, COMPUTE_REAL* j_for_elem_right,
    int elem_left, int elem_right, int face_pt, int num_faces_total) {
    COMPUTE_REAL flux_interface[5], viscous_flux_left[5], viscous_flux_right[5], euler_flux_left[5], euler_flux_right[5];

    // Note: compute_interface_flux is slau2_flux, ausm_plus_flux or rusanov_flux mapped via macro
    compute_interface_flux(u_left, u_right, normal, flux_interface);
    dispatch_viscous_flux(DIR{}, u_left, grad_left, viscous_flux_left);
    dispatch_viscous_flux(DIR{}, u_right, grad_right, viscous_flux_right);
    dispatch_euler_flux(DIR{}, u_left, euler_flux_left);
    dispatch_euler_flux(DIR{}, u_right, euler_flux_right);

    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL flux_common = flux_interface[v] - ((COMPUTE_REAL)0.5) * (viscous_flux_left[v] + viscous_flux_right[v]);
        COMPUTE_REAL j_left = flux_common - (euler_flux_left[v] - viscous_flux_left[v]);
        COMPUTE_REAL j_right = flux_common - (euler_flux_right[v] - viscous_flux_right[v]);

        j_for_elem_left[v * num_faces_total + elem_left * NPTS_FACE + face_pt] = j_left;
        j_for_elem_right[v * num_faces_total + elem_right * NPTS_FACE + face_pt] = j_right;
    }
}

// Fused gradient + volume RHS. The gradient stays in registers (Q eliminated):
// boundary extrapolations are written to per-face slices for the face kernels,
// and the volume integrals consume the register copy directly.
extern "C" __global__ void compute_rhs_volume_kernel(const U_REAL* U, RHS_REAL* RHS,
    const COMPUTE_REAL* grad_x_left, const COMPUTE_REAL* grad_x_right,
    const COMPUTE_REAL* grad_y_left, const COMPUTE_REAL* grad_y_right,
    const COMPUTE_REAL* grad_z_left, const COMPUTE_REAL* grad_z_right,
    int nx, int ny, int nz, COMPUTE_REAL lx, COMPUTE_REAL ly, COMPUTE_REAL lz) {
    __shared__ COMPUTE_REAL s_buf[5][NPTS_ELEM];
    __shared__ COMPUTE_REAL s_grad[15][NPTS_ELEM];

    int pt = threadIdx.x;
    int elem_idx = blockIdx.x;
    int total_points = nx * ny * nz * NPTS_ELEM;
    int global_pt_idx = elem_idx * NPTS_ELEM + pt;
    int i = pt % NPTS1D;
    int j = (pt / NPTS1D) % NPTS1D;
    int k = pt / NPTS1D2;
    int elem_x = elem_idx % nx;
    int elem_y = (elem_idx / nx) % ny;
    int elem_z = elem_idx / (nx * ny);

    for (int v = 0; v < 5; ++v) s_buf[v][pt] = U[v * total_points + global_pt_idx];
    __syncthreads();

    COMPUTE_REAL dx = lx / ((COMPUTE_REAL)nx);
    COMPUTE_REAL dy = ly / ((COMPUTE_REAL)ny);
    COMPUTE_REAL dz = lz / ((COMPUTE_REAL)nz);
    COMPUTE_REAL jac_x = ((COMPUTE_REAL)2.0) / dx; COMPUTE_REAL jac_y = ((COMPUTE_REAL)2.0) / dy; COMPUTE_REAL jac_z = ((COMPUTE_REAL)2.0) / dz;

    COMPUTE_REAL grad_x[5] = {0}, grad_y[5] = {0}, grad_z[5] = {0};

    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL u_pt = s_buf[v][pt];
        COMPUTE_REAL d_u_dzeta = derivative_zeta(k, &s_buf[v][i + j * NPTS1D]);
        COMPUTE_REAL d_u_dxi = derivative_shfl<1, 2, 3>(i, u_pt);
        COMPUTE_REAL d_u_deta = derivative_shfl<4, 8, 12>(j, u_pt);
        grad_x[v] = d_u_dxi * jac_x;
        grad_y[v] = d_u_deta * jac_y;
        grad_z[v] = d_u_dzeta * jac_z;
    }

    COMPUTE_REAL u_left[5], u_right[5];

    // --- X Faces ---
    for (int v = 0; v < 5; ++v) {
        u_left[v] = extrap_shfl<1, 2>(s_buf[v][pt], c_extrap_left, i);
        u_right[v] = extrap_shfl<1, 2>(s_buf[v][pt], c_extrap_right, i);
    }
    int x_left = (elem_x - 1 + nx) % nx; int elem_x_left = x_left + nx * (elem_y + ny * elem_z);
    int x_right = (elem_x + 1) % nx; int elem_x_right = x_right + nx * (elem_y + ny * elem_z);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_shfl<1, 2>(U, v, total_points, elem_x_left, pt, i, c_extrap_right);
        accumulate_jump(grad_x[v], u_left[v], extrap_term, c_g_left_prime[i], jac_x);
    }
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_shfl<1, 2>(U, v, total_points, elem_x_right, pt, i, c_extrap_left);
        accumulate_jump(grad_x[v], u_right[v], extrap_term, c_g_right_prime[i], jac_x);
    }

    // --- Y Faces ---
    for (int v = 0; v < 5; ++v) {
        u_left[v] = extrap_shfl<4, 8>(s_buf[v][pt], c_extrap_left, j);
        u_right[v] = extrap_shfl<4, 8>(s_buf[v][pt], c_extrap_right, j);
    }
#if BC_Y_WALL
    if (elem_y == 0) {
        for (int v = 0; v < 5; ++v)
            accumulate_jump(grad_y[v], u_left[v], reflect_wall(v, u_left[v]), c_g_left_prime[j], jac_y);
    } else {
        int y_left = elem_y - 1; int elem_y_left = elem_x + nx * (y_left + ny * elem_z);
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL extrap_term = neighbor_extrap_shfl<4, 8>(U, v, total_points, elem_y_left, pt, j, c_extrap_right);
            accumulate_jump(grad_y[v], u_left[v], extrap_term, c_g_left_prime[j], jac_y);
        }
    }
    if (elem_y == ny - 1) {
        for (int v = 0; v < 5; ++v)
            accumulate_jump(grad_y[v], u_right[v], reflect_wall(v, u_right[v]), c_g_right_prime[j], jac_y);
    } else {
        int y_right = elem_y + 1; int elem_y_right = elem_x + nx * (y_right + ny * elem_z);
        for (int v = 0; v < 5; ++v) {
            COMPUTE_REAL extrap_term = neighbor_extrap_shfl<4, 8>(U, v, total_points, elem_y_right, pt, j, c_extrap_left);
            accumulate_jump(grad_y[v], u_right[v], extrap_term, c_g_right_prime[j], jac_y);
        }
    }
#else
    int y_left = (elem_y - 1 + ny) % ny; int elem_y_left = elem_x + nx * (y_left + ny * elem_z);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_shfl<4, 8>(U, v, total_points, elem_y_left, pt, j, c_extrap_right);
        accumulate_jump(grad_y[v], u_left[v], extrap_term, c_g_left_prime[j], jac_y);
    }
    int y_right = (elem_y + 1) % ny; int elem_y_right = elem_x + nx * (y_right + ny * elem_z);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_shfl<4, 8>(U, v, total_points, elem_y_right, pt, j, c_extrap_left);
        accumulate_jump(grad_y[v], u_right[v], extrap_term, c_g_right_prime[j], jac_y);
    }
#endif

    // --- Z Faces ---
    for (int v = 0; v < 5; ++v) {
        u_left[v] = extrap_z(&s_buf[v][i + j * NPTS1D], c_extrap_left);
        u_right[v] = extrap_z(&s_buf[v][i + j * NPTS1D], c_extrap_right);
    }
    int z_left = (elem_z - 1 + nz) % nz; int elem_z_left = elem_x + nx * (elem_y + ny * z_left);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_z(U, v, total_points, elem_z_left, i, j, c_extrap_right);
        accumulate_jump(grad_z[v], u_left[v], extrap_term, c_g_left_prime[k], jac_z);
    }

    int z_right = (elem_z + 1) % nz; int elem_z_right = elem_x + nx * (elem_y + ny * z_right);
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL extrap_term = neighbor_extrap_z(U, v, total_points, elem_z_right, i, j, c_extrap_left);
        accumulate_jump(grad_z[v], u_right[v], extrap_term, c_g_right_prime[k], jac_z);
    }

    // Q_REAL roundtrip: replicates the old global store/load of Q.
    for (int v = 0; v < 5; ++v) {
        grad_x[v] = (COMPUTE_REAL)(Q_REAL)grad_x[v];
        grad_y[v] = (COMPUTE_REAL)(Q_REAL)grad_y[v];
        grad_z[v] = (COMPUTE_REAL)(Q_REAL)grad_z[v];
        s_grad[v * 3 + 0][pt] = grad_x[v];
        s_grad[v * 3 + 1][pt] = grad_y[v];
        s_grad[v * 3 + 2][pt] = grad_z[v];
    }
    __syncthreads();

    // --- Face slices: extrapolate the gradient to the 6 element boundaries.
    // Accumulation order replicates the face kernels' old in-kernel Q
    // extrapolation exactly, so downstream results stay bit-identical.
    if (i == 0) {
        int face_pt = j + k * NPTS1D;
        COMPUTE_REAL row_left[12] = {0}, row_right[12] = {0};
        build_slice_row_x(s_grad, j, k, row_left, row_right);
        store_slice_rows(slice_ptr(grad_x_left, elem_idx, face_pt), slice_ptr(grad_x_right, elem_idx, face_pt), row_left, row_right);
    }
    if (j == 0) {
        int face_pt = i + k * NPTS1D;
        COMPUTE_REAL row_left[12] = {0}, row_right[12] = {0};
        build_slice_row_y(s_grad, i, k, row_left, row_right);
        store_slice_rows(slice_ptr(grad_y_left, elem_idx, face_pt), slice_ptr(grad_y_right, elem_idx, face_pt), row_left, row_right);
    }
    if (k == 0) {
        int face_pt = i + j * NPTS1D;
        COMPUTE_REAL row_left[12] = {0}, row_right[12] = {0};
        build_slice_row_z(s_grad, i, j, row_left, row_right);
        store_slice_rows(slice_ptr(grad_z_left, elem_idx, face_pt), slice_ptr(grad_z_right, elem_idx, face_pt), row_left, row_right);
    }

    COMPUTE_REAL rhs_sum[5] = {0};
    COMPUTE_REAL state_local[5];
    for (int v = 0; v < 5; ++v) state_local[v] = s_buf[v][pt];
    COMPUTE_REAL grad_local[15];
    for (int v = 0; v < 15; ++v) grad_local[v] = s_grad[v][pt];

    // X-Flux
    accumulate_volume_rhs<flux_dir_x>(s_buf, state_local, grad_local, i, j * NPTS1D + k * NPTS1D2, 1, pt, jac_x, rhs_sum);

    // Y-Flux (leading barrier separates Y's stage write from X's derivative reads)
    __syncthreads();
    accumulate_volume_rhs<flux_dir_y>(s_buf, state_local, grad_local, j, i + k * NPTS1D2, NPTS1D, pt, jac_y, rhs_sum);

    // Z-Flux (ditto for Z)
    __syncthreads();
    accumulate_volume_rhs<flux_dir_z>(s_buf, state_local, grad_local, k, i + j * NPTS1D, NPTS1D2, pt, jac_z, rhs_sum);

    // Face corrections from the j arrays are applied in the RK kernels.
    for (int v = 0; v < 5; ++v) {
        COMPUTE_REAL rhs_value = -rhs_sum[v];
        if (v == 1) rhs_value += ((COMPUTE_REAL)0.001);
        RHS[v * total_points + global_pt_idx] = (RHS_REAL)rhs_value;
    }
}

extern "C" __global__ void compute_face_flux_x_kernel(const U_REAL* U, COMPUTE_REAL* j_x_left, COMPUTE_REAL* j_x_right, const COMPUTE_REAL* grad_x_left, const COMPUTE_REAL* grad_x_right, int nx, int ny, int nz, COMPUTE_REAL lx) {
    int face_idx = blockIdx.x * FACES_PER_BLOCK + threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;
    if (face_idx >= nx * ny * nz) return;

    int total_points = nx * ny * nz * NPTS_ELEM;
    int elem_x = face_idx % nx;
    int elem_y = (face_idx / nx) % ny;
    int elem_z = face_idx / (nx * ny);

    int x_left = (elem_x - 1 + nx) % nx;
    int x_right = elem_x;

    int elem_left = x_left + nx * (elem_y + ny * elem_z);
    int elem_right = x_right + nx * (elem_y + ny * elem_z);

    COMPUTE_REAL u_left[5] = {0}, grad_left[15] = {0};
    COMPUTE_REAL u_right[5] = {0}, grad_right[15] = {0};

    // Thread pt owns m = 0..NPTS1D-1 (contiguous floats): idx = elem*NPTS_ELEM + NPTS1D*pt + m
    int base_left = elem_left * NPTS_ELEM + NPTS1D * pt;
    int base_right = elem_right * NPTS_ELEM + NPTS1D * pt;

    typename V4Type<U_REAL>::type u_vec_right[5], u_vec_left[5];
    for (int v = 0; v < 5; ++v) u_vec_right[v] = ld4(U + v * total_points + base_right);
    for (int v = 0; v < 5; ++v) u_vec_left[v] = ld4(U + v * total_points + base_left);
    for (int v = 0; v < 5; ++v) {
        u_left[v] = (COMPUTE_REAL)u_vec_left[v].x * c_extrap_right[0] + (COMPUTE_REAL)u_vec_left[v].y * c_extrap_right[1]
               + (COMPUTE_REAL)u_vec_left[v].z * c_extrap_right[2] + (COMPUTE_REAL)u_vec_left[v].w * c_extrap_right[3];
        u_right[v] = (COMPUTE_REAL)u_vec_right[v].x * c_extrap_left[0] + (COMPUTE_REAL)u_vec_right[v].y * c_extrap_left[1]
               + (COMPUTE_REAL)u_vec_right[v].z * c_extrap_left[2] + (COMPUTE_REAL)u_vec_right[v].w * c_extrap_left[3];
    }
    // Gradient boundary slices: grad_left = elem_left's +side extrap, grad_right = elem_right's -side extrap.
    // viscous_x does not read components 8, 10, 13, 14 -> they stay zero in the row.
    read_gradient_slice<8, 10, 13, 14>(
        grad_x_right + ((size_t)elem_left * NPTS_FACE + pt) * 12,
        grad_x_left + ((size_t)elem_right * NPTS_FACE + pt) * 12,
        grad_left, grad_right);

    COMPUTE_REAL normal[3] = {1.0, 0.0, 0.0};
    int num_faces_total = nx * ny * nz * NPTS_FACE;
    compute_face_j<flux_dir_x>(u_left, u_right, grad_left, grad_right, normal,
        j_x_right, j_x_left, elem_left, elem_right, pt, num_faces_total);
}

extern "C" __global__ void compute_face_flux_y_kernel(const U_REAL* U, COMPUTE_REAL* j_y_left, COMPUTE_REAL* j_y_right, const COMPUTE_REAL* grad_y_left, const COMPUTE_REAL* grad_y_right, int nx, int ny, int nz, COMPUTE_REAL ly) {
    int face_in_block = threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;

    // Block groups FACES_PER_BLOCK faces sharing (elem_x, elem_z) with consecutive elem_y -> in-block L1 reuse
    int y_groups = (ny + FACES_PER_BLOCK - 1) / FACES_PER_BLOCK;
    int block_id = blockIdx.x;
    int y_group = block_id % y_groups;
    int elem_x = (block_id / y_groups) % nx;
    int elem_z = block_id / (y_groups * nx);
    int elem_y = y_group * FACES_PER_BLOCK + face_in_block;
    if (elem_y >= ny) return;
#if BC_Y_WALL
    if (elem_y == 0) return; // wall face: flux slots stay zero (zero-flux slip wall)
#endif

    int total_points = nx * ny * nz * NPTS_ELEM;

    // Shuffle mask must name only lanes that survived the early returns above:
    // partial face groups (elem_y >= ny) and the elem_y == 0 wall face diverge within a warp.
    const unsigned mask = __activemask();

#if BC_Y_WALL
    int y_left = elem_y - 1;
#else
    int y_left = (elem_y - 1 + ny) % ny;
#endif
    int y_right = elem_y;

    int elem_left = elem_x + nx * (y_left + ny * elem_z);
    int elem_right = elem_x + nx * (y_right + ny * elem_z);

    COMPUTE_REAL u_left[5] = {0}, grad_left[15] = {0};
    COMPUTE_REAL u_right[5] = {0}, grad_right[15] = {0};

    // Thread role remap: pt = m + NPTS1D*k where m = element j (extrap var), k = face point k.
    // ld4 reads points i'=0..NPTS1D-1 (contiguous) at (m, k); quad (xor1, xor2) = m-set at fixed k.
    // After reduce, lane picks component i = pt % NPTS1D = output face point i.
    int m = pt % NPTS1D;
    int k = pt / NPTS1D;
    int base_left = elem_left * NPTS_ELEM + (m * NPTS1D + k * NPTS1D2);
    int base_right = elem_right * NPTS_ELEM + (m * NPTS1D + k * NPTS1D2);
    COMPUTE_REAL coeff_right = c_extrap_right[m];
    COMPUTE_REAL coeff_left = c_extrap_left[m];

    for (int v = 0; v < 5; ++v) {
        typename V4Type<U_REAL>::type u_vec_right = ld4(U + v * total_points + base_right);
        typename V4Type<U_REAL>::type u_vec_left = ld4(U + v * total_points + base_left);
        COMPUTE_REAL left_sum_0 = (COMPUTE_REAL)u_vec_left.x * coeff_right, left_sum_1 = (COMPUTE_REAL)u_vec_left.y * coeff_right, left_sum_2 = (COMPUTE_REAL)u_vec_left.z * coeff_right, left_sum_3 = (COMPUTE_REAL)u_vec_left.w * coeff_right;
        COMPUTE_REAL right_sum_0 = (COMPUTE_REAL)u_vec_right.x * coeff_left, right_sum_1 = (COMPUTE_REAL)u_vec_right.y * coeff_left, right_sum_2 = (COMPUTE_REAL)u_vec_right.z * coeff_left, right_sum_3 = (COMPUTE_REAL)u_vec_right.w * coeff_left;
        left_sum_0 += __shfl_xor_sync(mask, left_sum_0, 1); right_sum_0 += __shfl_xor_sync(mask, right_sum_0, 1);
        left_sum_1 += __shfl_xor_sync(mask, left_sum_1, 1); right_sum_1 += __shfl_xor_sync(mask, right_sum_1, 1);
        left_sum_2 += __shfl_xor_sync(mask, left_sum_2, 1); right_sum_2 += __shfl_xor_sync(mask, right_sum_2, 1);
        left_sum_3 += __shfl_xor_sync(mask, left_sum_3, 1); right_sum_3 += __shfl_xor_sync(mask, right_sum_3, 1);
        left_sum_0 += __shfl_xor_sync(mask, left_sum_0, 2); right_sum_0 += __shfl_xor_sync(mask, right_sum_0, 2);
        left_sum_1 += __shfl_xor_sync(mask, left_sum_1, 2); right_sum_1 += __shfl_xor_sync(mask, right_sum_1, 2);
        left_sum_2 += __shfl_xor_sync(mask, left_sum_2, 2); right_sum_2 += __shfl_xor_sync(mask, right_sum_2, 2);
        left_sum_3 += __shfl_xor_sync(mask, left_sum_3, 2); right_sum_3 += __shfl_xor_sync(mask, right_sum_3, 2);
        u_left[v] = (m == 0) ? left_sum_0 : (m == 1) ? left_sum_1 : (m == 2) ? left_sum_2 : left_sum_3;
        u_right[v] = (m == 0) ? right_sum_0 : (m == 1) ? right_sum_1 : (m == 2) ? right_sum_2 : right_sum_3;
    }
    // Gradient boundary slices: grad_left = elem_left's +side extrap, grad_right = elem_right's -side extrap.
    // viscous_y does not read components 5, 9, 12, 14 -> they stay zero in the row.
    read_gradient_slice<5, 9, 12, 14>(
        grad_y_right + ((size_t)elem_left * NPTS_FACE + pt) * 12,
        grad_y_left + ((size_t)elem_right * NPTS_FACE + pt) * 12,
        grad_left, grad_right);

    COMPUTE_REAL normal[3] = {0.0, 1.0, 0.0};
    int num_faces_total = nx * ny * nz * NPTS_FACE;
    compute_face_j<flux_dir_y>(u_left, u_right, grad_left, grad_right, normal,
        j_y_right, j_y_left, elem_left, elem_right, pt, num_faces_total);
}

extern "C" __global__ void compute_face_flux_z_kernel(const U_REAL* U, COMPUTE_REAL* j_z_left, COMPUTE_REAL* j_z_right, const COMPUTE_REAL* grad_z_left, const COMPUTE_REAL* grad_z_right, int nx, int ny, int nz, COMPUTE_REAL lz) {
    int face_in_block = threadIdx.x / NPTS_FACE;
    int pt = threadIdx.x % NPTS_FACE;

    // Block groups FACES_PER_BLOCK faces sharing (elem_x, elem_y) with consecutive elem_z -> in-block L1 reuse
    int z_groups = (nz + FACES_PER_BLOCK - 1) / FACES_PER_BLOCK;
    int block_id = blockIdx.x;
    int z_group = block_id % z_groups;
    int elem_x = (block_id / z_groups) % nx;
    int elem_y = (block_id / z_groups) / nx;
    int elem_z = z_group * FACES_PER_BLOCK + face_in_block;
    if (elem_z >= nz) return;

    int total_points = nx * ny * nz * NPTS_ELEM;

    int z_left = (elem_z - 1 + nz) % nz;
    int z_right = elem_z;

    int elem_left = elem_x + nx * (elem_y + ny * z_left);
    int elem_right = elem_x + nx * (elem_y + ny * z_right);

    COMPUTE_REAL u_left[5] = {0}, grad_left[15] = {0};
    COMPUTE_REAL u_right[5] = {0}, grad_right[15] = {0};

    int face_i = pt % NPTS1D; // i
    int face_j = pt / NPTS1D; // j

    for (int m = 0; m < NPTS1D; ++m) {
        int point_left = elem_left * NPTS_ELEM + (face_i + face_j * NPTS1D + m * NPTS1D2);
        int point_right = elem_right * NPTS_ELEM + (face_i + face_j * NPTS1D + m * NPTS1D2);

        COMPUTE_REAL coeff_right = c_extrap_right[m];
        COMPUTE_REAL coeff_left = c_extrap_left[m];

        for (int v = 0; v < 5; ++v) {
            u_right[v] += U[v * total_points + point_right] * coeff_left;
            u_left[v] += U[v * total_points + point_left] * coeff_right;
        }
    }
    // Gradient boundary slices: grad_left = elem_left's +side extrap, grad_right = elem_right's -side extrap.
    // viscous_z does not read components 4, 6, 12, 13 -> they stay zero in the row.
    read_gradient_slice<4, 6, 12, 13>(
        grad_z_right + ((size_t)elem_left * NPTS_FACE + pt) * 12,
        grad_z_left + ((size_t)elem_right * NPTS_FACE + pt) * 12,
        grad_left, grad_right);

    COMPUTE_REAL normal[3] = {0.0, 0.0, 1.0};
    int num_faces_total = nx * ny * nz * NPTS_FACE;
    compute_face_j<flux_dir_z>(u_left, u_right, grad_left, grad_right, normal,
        j_z_right, j_z_left, elem_left, elem_right, pt, num_faces_total);
}
