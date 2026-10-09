__device__ inline COMPUTE_REAL compute_pressure(COMPUTE_REAL rho, COMPUTE_REAL rhou, COMPUTE_REAL rhov, COMPUTE_REAL rhow, COMPUTE_REAL total_energy) {
    COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL p = (GAMMA - ((COMPUTE_REAL)1.0)) * (total_energy - ke);
    return fmax(((COMPUTE_REAL)1e-7), p);
}

__device__ inline void compute_euler_flux_x(const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho  = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL u = rhou / rho, p = compute_pressure(rho, rhou, rhov, rhow, total_energy);
    flux[0] = rhou; flux[1] = rhou*u + p; flux[2] = rhov*u; flux[3] = rhow*u; flux[4] = u*(total_energy + p);
}
__device__ inline void compute_euler_flux_y(const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho  = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL v = rhov / rho, p = compute_pressure(rho, rhou, rhov, rhow, total_energy);
    flux[0] = rhov; flux[1] = rhou*v; flux[2] = rhov*v + p; flux[3] = rhow*v; flux[4] = v*(total_energy + p);
}
__device__ inline void compute_euler_flux_z(const COMPUTE_REAL* cons_vars, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho  = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL w = rhow / rho, p = compute_pressure(rho, rhou, rhov, rhow, total_energy);
    flux[0] = rhow; flux[1] = rhou*w; flux[2] = rhov*w; flux[3] = rhow*w + p; flux[4] = w*(total_energy + p);
}

__device__ inline void compute_viscous_flux_x(const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (grad_cons[1*3 + 0] - u * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dx = (grad_cons[2*3 + 0] - v * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dx = (grad_cons[3*3 + 0] - w * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dy = (grad_cons[2*3 + 1] - v * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (grad_cons[3*3 + 2] - w * grad_cons[0*3 + 2]) / rho;
    COMPUTE_REAL du_dy = (grad_cons[1*3 + 1] - u * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL du_dz = (grad_cons[1*3 + 2] - u * grad_cons[0*3 + 2]) / rho;
    
    COMPUTE_REAL div_vel = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL tau_xx = MU * (((COMPUTE_REAL)2.0) * du_dx - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_vel);
    COMPUTE_REAL tau_xy = MU * (du_dy + dv_dx);
    COMPUTE_REAL tau_xz = MU * (du_dz + dw_dx);
    
    COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL p = (GAMMA - ((COMPUTE_REAL)1.0)) * (total_energy - ke);
    COMPUTE_REAL d_ke_dx = (rhou * grad_cons[1*3 + 0] + rhov * grad_cons[2*3 + 0] + rhow * grad_cons[3*3 + 0]) / rho - ke * grad_cons[0*3 + 0] / rho;
    COMPUTE_REAL d_p_dx = (GAMMA - ((COMPUTE_REAL)1.0)) * (grad_cons[4*3 + 0] - d_ke_dx);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL heat_flux_x = -coeff * (d_p_dx / rho - p * grad_cons[0*3 + 0] / (rho * rho));
    
    flux[0] = ((COMPUTE_REAL)0.0); flux[1] = tau_xx; flux[2] = tau_xy; flux[3] = tau_xz; flux[4] = u * tau_xx + v * tau_xy + w * tau_xz - heat_flux_x;
}

__device__ inline void compute_viscous_flux_y(const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (grad_cons[1*3 + 0] - u * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dx = (grad_cons[2*3 + 0] - v * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dy = (grad_cons[3*3 + 1] - w * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL dv_dy = (grad_cons[2*3 + 1] - v * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (grad_cons[3*3 + 2] - w * grad_cons[0*3 + 2]) / rho;
    COMPUTE_REAL du_dy = (grad_cons[1*3 + 1] - u * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL dv_dz = (grad_cons[2*3 + 2] - v * grad_cons[0*3 + 2]) / rho;
    
    COMPUTE_REAL div_vel = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL tau_yy = MU * (((COMPUTE_REAL)2.0) * dv_dy - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_vel);
    COMPUTE_REAL tau_xy = MU * (du_dy + dv_dx);
    COMPUTE_REAL tau_yz = MU * (dv_dz + dw_dy);
    
    COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL p = (GAMMA - ((COMPUTE_REAL)1.0)) * (total_energy - ke);
    COMPUTE_REAL d_ke_dy = (rhou * grad_cons[1*3 + 1] + rhov * grad_cons[2*3 + 1] + rhow * grad_cons[3*3 + 1]) / rho - ke * grad_cons[0*3 + 1] / rho;
    COMPUTE_REAL d_p_dy = (GAMMA - ((COMPUTE_REAL)1.0)) * (grad_cons[4*3 + 1] - d_ke_dy);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL heat_flux_y = -coeff * (d_p_dy / rho - p * grad_cons[0*3 + 1] / (rho * rho));
    
    flux[0] = ((COMPUTE_REAL)0.0); flux[1] = tau_xy; flux[2] = tau_yy; flux[3] = tau_yz; flux[4] = u * tau_xy + v * tau_yy + w * tau_yz - heat_flux_y;
}

__device__ inline void compute_viscous_flux_z(const COMPUTE_REAL* cons_vars, const COMPUTE_REAL* grad_cons, COMPUTE_REAL* flux) {
    COMPUTE_REAL rho = cons_vars[0], rhou = cons_vars[1], rhov = cons_vars[2], rhow = cons_vars[3], total_energy = cons_vars[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (grad_cons[1*3 + 0] - u * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dy = (grad_cons[2*3 + 1] - v * grad_cons[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (grad_cons[3*3 + 2] - w * grad_cons[0*3 + 2]) / rho;
    COMPUTE_REAL du_dz = (grad_cons[1*3 + 2] - u * grad_cons[0*3 + 2]) / rho;
    COMPUTE_REAL dv_dz = (grad_cons[2*3 + 2] - v * grad_cons[0*3 + 2]) / rho;
    COMPUTE_REAL dw_dx = (grad_cons[3*3 + 0] - w * grad_cons[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dy = (grad_cons[3*3 + 1] - w * grad_cons[0*3 + 1]) / rho;
    
    COMPUTE_REAL div_vel = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL tau_zz = MU * (((COMPUTE_REAL)2.0) * dw_dz - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_vel);
    COMPUTE_REAL tau_xz = MU * (du_dz + dw_dx);
    COMPUTE_REAL tau_yz = MU * (dv_dz + dw_dy);
    
    COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL p = (GAMMA - ((COMPUTE_REAL)1.0)) * (total_energy - ke);
    COMPUTE_REAL d_ke_dz = (rhou * grad_cons[1*3 + 2] + rhov * grad_cons[2*3 + 2] + rhow * grad_cons[3*3 + 2]) / rho - ke * grad_cons[0*3 + 2] / rho;
    COMPUTE_REAL d_p_dz = (GAMMA - ((COMPUTE_REAL)1.0)) * (grad_cons[4*3 + 2] - d_ke_dz);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL heat_flux_z = -coeff * (d_p_dz / rho - p * grad_cons[0*3 + 2] / (rho * rho));
    
    flux[0] = ((COMPUTE_REAL)0.0); flux[1] = tau_xz; flux[2] = tau_yz; flux[3] = tau_zz; flux[4] = u * tau_xz + v * tau_yz + w * tau_zz - heat_flux_z;
}

__device__ inline void rusanov_flux(const COMPUTE_REAL* state_left, const COMPUTE_REAL* state_right, const COMPUTE_REAL normal[3], COMPUTE_REAL* flux) {
    COMPUTE_REAL rho_left = state_left[0], rhou_left = state_left[1], rhov_left = state_left[2], rhow_left = state_left[3], energy_left = state_left[4];
    COMPUTE_REAL rho_right = state_right[0], rhou_right = state_right[1], rhov_right = state_right[2], rhow_right = state_right[3], energy_right = state_right[4];
    
    COMPUTE_REAL u_left = rhou_left/rho_left, v_left = rhov_left/rho_left, w_left = rhow_left/rho_left;
    COMPUTE_REAL u_right = rhou_right/rho_right, v_right = rhov_right/rho_right, w_right = rhow_right/rho_right;
    COMPUTE_REAL p_left = compute_pressure(rho_left, rhou_left, rhov_left, rhow_left, energy_left);
    COMPUTE_REAL p_right = compute_pressure(rho_right, rhou_right, rhov_right, rhow_right, energy_right);
    
    COMPUTE_REAL normal_vel_left = u_left*normal[0] + v_left*normal[1] + w_left*normal[2];
    COMPUTE_REAL normal_vel_right = u_right*normal[0] + v_right*normal[1] + w_right*normal[2];
    
    COMPUTE_REAL sound_speed_left = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_left / fmax(((COMPUTE_REAL)1e-7), rho_left)));
    COMPUTE_REAL sound_speed_right = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_right / fmax(((COMPUTE_REAL)1e-7), rho_right)));
    
    COMPUTE_REAL max_wave_speed = fmax(fabs(normal_vel_left) + sound_speed_left, fabs(normal_vel_right) + sound_speed_right);
    
    COMPUTE_REAL flux_left[5], flux_right[5];
    flux_left[0] = rho_left*normal_vel_left; flux_left[1] = rhou_left*normal_vel_left + p_left*normal[0]; flux_left[2] = rhov_left*normal_vel_left + p_left*normal[1]; flux_left[3] = rhow_left*normal_vel_left + p_left*normal[2]; flux_left[4] = normal_vel_left*(energy_left + p_left);
    flux_right[0] = rho_right*normal_vel_right; flux_right[1] = rhou_right*normal_vel_right + p_right*normal[0]; flux_right[2] = rhov_right*normal_vel_right + p_right*normal[1]; flux_right[3] = rhow_right*normal_vel_right + p_right*normal[2]; flux_right[4] = normal_vel_right*(energy_right + p_right);
    
    for(int v=0; v<5; v++) {
        flux[v] = ((COMPUTE_REAL)0.5) * (flux_left[v] + flux_right[v] - max_wave_speed * (state_right[v] - state_left[v]));
    }
}

__device__ inline void ausm_plus_flux(const COMPUTE_REAL* state_left, const COMPUTE_REAL* state_right, const COMPUTE_REAL normal[3], COMPUTE_REAL* flux) {
    COMPUTE_REAL rho_left = state_left[0], rhou_left = state_left[1], rhov_left = state_left[2], rhow_left = state_left[3], energy_left = state_left[4];
    COMPUTE_REAL rho_right = state_right[0], rhou_right = state_right[1], rhov_right = state_right[2], rhow_right = state_right[3], energy_right = state_right[4];
    
    COMPUTE_REAL u_left = rhou_left/rho_left, v_left = rhov_left/rho_left, w_left = rhow_left/rho_left;
    COMPUTE_REAL u_right = rhou_right/rho_right, v_right = rhov_right/rho_right, w_right = rhow_right/rho_right;
    COMPUTE_REAL p_left = compute_pressure(rho_left, rhou_left, rhov_left, rhow_left, energy_left);
    COMPUTE_REAL p_right = compute_pressure(rho_right, rhou_right, rhov_right, rhow_right, energy_right);
    
    COMPUTE_REAL normal_vel_left = u_left*normal[0] + v_left*normal[1] + w_left*normal[2];
    COMPUTE_REAL normal_vel_right = u_right*normal[0] + v_right*normal[1] + w_right*normal[2];
    COMPUTE_REAL enthalpy_left = (energy_left + p_left)/rho_left;
    COMPUTE_REAL enthalpy_right = (energy_right + p_right)/rho_right;
    
    COMPUTE_REAL sound_speed_left = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_left / fmax(((COMPUTE_REAL)1e-7), rho_left)));
    COMPUTE_REAL sound_speed_right = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_right / fmax(((COMPUTE_REAL)1e-7), rho_right)));
    
    COMPUTE_REAL sound_speed_max = fmax(sound_speed_left, sound_speed_right);
    COMPUTE_REAL mach_left = normal_vel_left / sound_speed_max;
    COMPUTE_REAL mach_right = normal_vel_right / sound_speed_max;
    
    COMPUTE_REAL mach_plus, mach_minus, p_plus, p_minus;
    COMPUTE_REAL alpha = ((COMPUTE_REAL)0.1875);
    COMPUTE_REAL beta = ((COMPUTE_REAL)0.125);
    
    if (fabs(mach_left) >= ((COMPUTE_REAL)1.0)) {
        mach_plus = ((COMPUTE_REAL)0.5) * (mach_left + fabs(mach_left));
        p_plus = (mach_left > ((COMPUTE_REAL)0.0)) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    } else {
        mach_plus = ((COMPUTE_REAL)0.25) * (mach_left + ((COMPUTE_REAL)1.0)) * (mach_left + ((COMPUTE_REAL)1.0)) + beta * (mach_left * mach_left - ((COMPUTE_REAL)1.0)) * (mach_left * mach_left - ((COMPUTE_REAL)1.0));
        p_plus = ((COMPUTE_REAL)0.25) * (mach_left + ((COMPUTE_REAL)1.0)) * (mach_left + ((COMPUTE_REAL)1.0)) * (((COMPUTE_REAL)2.0) - mach_left) + alpha * mach_left * (mach_left * mach_left - ((COMPUTE_REAL)1.0)) * (mach_left * mach_left - ((COMPUTE_REAL)1.0));
    }
    
    if (fabs(mach_right) >= ((COMPUTE_REAL)1.0)) {
        mach_minus = ((COMPUTE_REAL)0.5) * (mach_right - fabs(mach_right));
        p_minus = (mach_right < ((COMPUTE_REAL)0.0)) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    } else {
        mach_minus = -((COMPUTE_REAL)0.25) * (mach_right - ((COMPUTE_REAL)1.0)) * (mach_right - ((COMPUTE_REAL)1.0)) - beta * (mach_right * mach_right - ((COMPUTE_REAL)1.0)) * (mach_right * mach_right - ((COMPUTE_REAL)1.0));
        p_minus = ((COMPUTE_REAL)0.25) * (mach_right - ((COMPUTE_REAL)1.0)) * (mach_right - ((COMPUTE_REAL)1.0)) * (((COMPUTE_REAL)2.0) + mach_right) - alpha * mach_right * (mach_right * mach_right - ((COMPUTE_REAL)1.0)) * (mach_right * mach_right - ((COMPUTE_REAL)1.0));
    }
    
    COMPUTE_REAL mach_number = mach_plus + mach_minus;
    COMPUTE_REAL p = p_plus * p_left + p_minus * p_right;
    COMPUTE_REAL mass_flux = mach_number * sound_speed_max * (mach_number > ((COMPUTE_REAL)0.0) ? rho_left : rho_right);
    
    if (mach_number > ((COMPUTE_REAL)0.0)) {
        flux[0] = mass_flux;
        flux[1] = mass_flux * u_left + p * normal[0];
        flux[2] = mass_flux * v_left + p * normal[1];
        flux[3] = mass_flux * w_left + p * normal[2];
        flux[4] = mass_flux * enthalpy_left;
    } else {
        flux[0] = mass_flux;
        flux[1] = mass_flux * u_right + p * normal[0];
        flux[2] = mass_flux * v_right + p * normal[1];
        flux[3] = mass_flux * w_right + p * normal[2];
        flux[4] = mass_flux * enthalpy_right;
    }
}

// SLAU2 (Shima 2011), following the SU2 implementation (CUpwSLAU_Flow with slau2 = true)
__device__ inline void slau2_flux(const COMPUTE_REAL* state_left, const COMPUTE_REAL* state_right, const COMPUTE_REAL normal[3], COMPUTE_REAL* flux) {
    COMPUTE_REAL rho_left = state_left[0], rhou_left = state_left[1], rhov_left = state_left[2], rhow_left = state_left[3], energy_left = state_left[4];
    COMPUTE_REAL rho_right = state_right[0], rhou_right = state_right[1], rhov_right = state_right[2], rhow_right = state_right[3], energy_right = state_right[4];
    
    COMPUTE_REAL u_left = rhou_left/rho_left, v_left = rhov_left/rho_left, w_left = rhow_left/rho_left;
    COMPUTE_REAL u_right = rhou_right/rho_right, v_right = rhov_right/rho_right, w_right = rhow_right/rho_right;
    COMPUTE_REAL p_left = compute_pressure(rho_left, rhou_left, rhov_left, rhow_left, energy_left);
    COMPUTE_REAL p_right = compute_pressure(rho_right, rhou_right, rhov_right, rhow_right, energy_right);
    
    COMPUTE_REAL normal_vel_left = u_left*normal[0] + v_left*normal[1] + w_left*normal[2];
    COMPUTE_REAL normal_vel_right = u_right*normal[0] + v_right*normal[1] + w_right*normal[2];
    COMPUTE_REAL enthalpy_left = (energy_left + p_left)/rho_left;
    COMPUTE_REAL enthalpy_right = (energy_right + p_right)/rho_right;
    
    COMPUTE_REAL speed_sq_left = u_left*u_left + v_left*v_left + w_left*w_left;
    COMPUTE_REAL speed_sq_right = u_right*u_right + v_right*v_right + w_right*w_right;
    
    COMPUTE_REAL sound_speed_left = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_left / fmax(((COMPUTE_REAL)1e-7), rho_left)));
    COMPUTE_REAL sound_speed_right = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * p_right / fmax(((COMPUTE_REAL)1e-7), rho_right)));
    
    COMPUTE_REAL sound_speed_face = ((COMPUTE_REAL)0.5) * (sound_speed_left + sound_speed_right);
    COMPUTE_REAL normal_mach_left = normal_vel_left / sound_speed_face;
    COMPUTE_REAL normal_mach_right = normal_vel_right / sound_speed_face;
    
    // Smooth local Mach number and contact discontinuity switch
    COMPUTE_REAL mach_tilde = fmin(((COMPUTE_REAL)1.0), (((COMPUTE_REAL)1.0)/sound_speed_face) * sqrt(((COMPUTE_REAL)0.5) * (speed_sq_left + speed_sq_right)));
    COMPUTE_REAL chi = ((COMPUTE_REAL)1.0) - mach_tilde;
    chi = chi * chi;
    COMPUTE_REAL mass_flux_factor = -fmax(fmin(normal_mach_left, ((COMPUTE_REAL)0.0)), ((COMPUTE_REAL)-1.0)) * fmin(fmax(normal_mach_right, ((COMPUTE_REAL)0.0)), ((COMPUTE_REAL)1.0));
    
    // Density-weighted mean normal velocity magnitude
    COMPUTE_REAL vn_abs_avg = (rho_left * fabs(normal_vel_left) + rho_right * fabs(normal_vel_right)) / (rho_left + rho_right);
    COMPUTE_REAL vn_abs_left = (((COMPUTE_REAL)1.0) - mass_flux_factor) * vn_abs_avg + mass_flux_factor * fabs(normal_vel_left);
    COMPUTE_REAL vn_abs_right = (((COMPUTE_REAL)1.0) - mass_flux_factor) * vn_abs_avg + mass_flux_factor * fabs(normal_vel_right);
    
    COMPUTE_REAL mass_flux = ((COMPUTE_REAL)0.5) * (rho_left * (normal_vel_left + vn_abs_left) + rho_right * (normal_vel_right - vn_abs_right) - (chi / sound_speed_face) * (p_right - p_left));
    
    // Pressure split: AUSM+ polynomials with alpha = 0
    COMPUTE_REAL beta_left, beta_right;
    if (fabs(normal_mach_left) < ((COMPUTE_REAL)1.0)) {
        beta_left = ((COMPUTE_REAL)0.25) * (((COMPUTE_REAL)2.0) - normal_mach_left) * (normal_mach_left + ((COMPUTE_REAL)1.0)) * (normal_mach_left + ((COMPUTE_REAL)1.0));
    } else {
        beta_left = (normal_mach_left >= ((COMPUTE_REAL)0.0)) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    }
    if (fabs(normal_mach_right) < ((COMPUTE_REAL)1.0)) {
        beta_right = ((COMPUTE_REAL)0.25) * (((COMPUTE_REAL)2.0) + normal_mach_right) * (normal_mach_right - ((COMPUTE_REAL)1.0)) * (normal_mach_right - ((COMPUTE_REAL)1.0));
    } else {
        beta_right = (normal_mach_right >= ((COMPUTE_REAL)0.0)) ? ((COMPUTE_REAL)0.0) : ((COMPUTE_REAL)1.0);
    }
    
    COMPUTE_REAL p_face = ((COMPUTE_REAL)0.5) * (p_left + p_right) + ((COMPUTE_REAL)0.5) * (beta_left - beta_right) * (p_left - p_right);
    p_face += sqrt(((COMPUTE_REAL)0.5) * (speed_sq_left + speed_sq_right)) * (beta_left + beta_right - ((COMPUTE_REAL)1.0)) * sound_speed_face * ((COMPUTE_REAL)0.5) * (rho_left + rho_right);
    
    if (mass_flux > ((COMPUTE_REAL)0.0)) {
        flux[0] = mass_flux;
        flux[1] = mass_flux * u_left + p_face * normal[0];
        flux[2] = mass_flux * v_left + p_face * normal[1];
        flux[3] = mass_flux * w_left + p_face * normal[2];
        flux[4] = mass_flux * enthalpy_left;
    } else {
        flux[0] = mass_flux;
        flux[1] = mass_flux * u_right + p_face * normal[0];
        flux[2] = mass_flux * v_right + p_face * normal[1];
        flux[3] = mass_flux * w_right + p_face * normal[2];
        flux[4] = mass_flux * enthalpy_right;
    }
}

#if defined(FLUX_SCHEME_SLAU2)
#define compute_interface_flux slau2_flux
#elif defined(FLUX_SCHEME_AUSMPLUS)
#define compute_interface_flux ausm_plus_flux
#else
#define compute_interface_flux rusanov_flux
#endif
