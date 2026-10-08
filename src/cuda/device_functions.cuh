__device__ inline COMPUTE_REAL compute_pressure(COMPUTE_REAL rho, COMPUTE_REAL rhou, COMPUTE_REAL rhov, COMPUTE_REAL rhow, COMPUTE_REAL E) {
    COMPUTE_REAL ke = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL p = (GAMMA - ((COMPUTE_REAL)1.0)) * (E - ke);
    return fmax(((COMPUTE_REAL)1e-7), p);
}

__device__ inline void compute_euler_flux_x(const COMPUTE_REAL* U_loc, COMPUTE_REAL* Fx) {
    COMPUTE_REAL rho  = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL u = rhou / rho, p = compute_pressure(rho, rhou, rhov, rhow, E);
    Fx[0] = rhou; Fx[1] = rhou*u + p; Fx[2] = rhov*u; Fx[3] = rhow*u; Fx[4] = u*(E + p);
}
__device__ inline void compute_euler_flux_y(const COMPUTE_REAL* U_loc, COMPUTE_REAL* Fy) {
    COMPUTE_REAL rho  = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL v = rhov / rho, p = compute_pressure(rho, rhou, rhov, rhow, E);
    Fy[0] = rhov; Fy[1] = rhou*v; Fy[2] = rhov*v + p; Fy[3] = rhow*v; Fy[4] = v*(E + p);
}
__device__ inline void compute_euler_flux_z(const COMPUTE_REAL* U_loc, COMPUTE_REAL* Fz) {
    COMPUTE_REAL rho  = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL w = rhow / rho, p = compute_pressure(rho, rhou, rhov, rhow, E);
    Fz[0] = rhow; Fz[1] = rhou*w; Fz[2] = rhov*w; Fz[3] = rhow*w + p; Fz[4] = w*(E + p);
}

__device__ inline void compute_viscous_flux_x(const COMPUTE_REAL* U_loc, const COMPUTE_REAL* Q_loc, COMPUTE_REAL* Fvx) {
    COMPUTE_REAL rho = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (Q_loc[1*3 + 0] - u * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dx = (Q_loc[2*3 + 0] - v * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dx = (Q_loc[3*3 + 0] - w * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dy = (Q_loc[2*3 + 1] - v * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (Q_loc[3*3 + 2] - w * Q_loc[0*3 + 2]) / rho;
    COMPUTE_REAL du_dy = (Q_loc[1*3 + 1] - u * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL du_dz = (Q_loc[1*3 + 2] - u * Q_loc[0*3 + 2]) / rho;
    
    COMPUTE_REAL div_V = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL txx = MU * (((COMPUTE_REAL)2.0) * du_dx - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_V);
    COMPUTE_REAL txy = MU * (du_dy + dv_dx);
    COMPUTE_REAL txz = MU * (du_dz + dw_dx);
    
    COMPUTE_REAL K = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL P = (GAMMA - ((COMPUTE_REAL)1.0)) * (E - K);
    COMPUTE_REAL dK_dx = (rhou * Q_loc[1*3 + 0] + rhov * Q_loc[2*3 + 0] + rhow * Q_loc[3*3 + 0]) / rho - K * Q_loc[0*3 + 0] / rho;
    COMPUTE_REAL dP_dx = (GAMMA - ((COMPUTE_REAL)1.0)) * (Q_loc[4*3 + 0] - dK_dx);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL qx = -coeff * (dP_dx / rho - P * Q_loc[0*3 + 0] / (rho * rho));
    
    Fvx[0] = ((COMPUTE_REAL)0.0); Fvx[1] = txx; Fvx[2] = txy; Fvx[3] = txz; Fvx[4] = u * txx + v * txy + w * txz - qx;
}

__device__ inline void compute_viscous_flux_y(const COMPUTE_REAL* U_loc, const COMPUTE_REAL* Q_loc, COMPUTE_REAL* Fvy) {
    COMPUTE_REAL rho = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (Q_loc[1*3 + 0] - u * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dx = (Q_loc[2*3 + 0] - v * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dy = (Q_loc[3*3 + 1] - w * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL dv_dy = (Q_loc[2*3 + 1] - v * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (Q_loc[3*3 + 2] - w * Q_loc[0*3 + 2]) / rho;
    COMPUTE_REAL du_dy = (Q_loc[1*3 + 1] - u * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL dv_dz = (Q_loc[2*3 + 2] - v * Q_loc[0*3 + 2]) / rho;
    
    COMPUTE_REAL div_V = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL tyy = MU * (((COMPUTE_REAL)2.0) * dv_dy - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_V);
    COMPUTE_REAL txy = MU * (du_dy + dv_dx);
    COMPUTE_REAL tyz = MU * (dv_dz + dw_dy);
    
    COMPUTE_REAL K = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL P = (GAMMA - ((COMPUTE_REAL)1.0)) * (E - K);
    COMPUTE_REAL dK_dy = (rhou * Q_loc[1*3 + 1] + rhov * Q_loc[2*3 + 1] + rhow * Q_loc[3*3 + 1]) / rho - K * Q_loc[0*3 + 1] / rho;
    COMPUTE_REAL dP_dy = (GAMMA - ((COMPUTE_REAL)1.0)) * (Q_loc[4*3 + 1] - dK_dy);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL qy = -coeff * (dP_dy / rho - P * Q_loc[0*3 + 1] / (rho * rho));
    
    Fvy[0] = ((COMPUTE_REAL)0.0); Fvy[1] = txy; Fvy[2] = tyy; Fvy[3] = tyz; Fvy[4] = u * txy + v * tyy + w * tyz - qy;
}

__device__ inline void compute_viscous_flux_z(const COMPUTE_REAL* U_loc, const COMPUTE_REAL* Q_loc, COMPUTE_REAL* Fvz) {
    COMPUTE_REAL rho = U_loc[0], rhou = U_loc[1], rhov = U_loc[2], rhow = U_loc[3], E = U_loc[4];
    COMPUTE_REAL u = rhou / rho, v = rhov / rho, w = rhow / rho;
    
    COMPUTE_REAL du_dx = (Q_loc[1*3 + 0] - u * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dv_dy = (Q_loc[2*3 + 1] - v * Q_loc[0*3 + 1]) / rho;
    COMPUTE_REAL dw_dz = (Q_loc[3*3 + 2] - w * Q_loc[0*3 + 2]) / rho;
    COMPUTE_REAL du_dz = (Q_loc[1*3 + 2] - u * Q_loc[0*3 + 2]) / rho;
    COMPUTE_REAL dv_dz = (Q_loc[2*3 + 2] - v * Q_loc[0*3 + 2]) / rho;
    COMPUTE_REAL dw_dx = (Q_loc[3*3 + 0] - w * Q_loc[0*3 + 0]) / rho;
    COMPUTE_REAL dw_dy = (Q_loc[3*3 + 1] - w * Q_loc[0*3 + 1]) / rho;
    
    COMPUTE_REAL div_V = du_dx + dv_dy + dw_dz;
    COMPUTE_REAL tzz = MU * (((COMPUTE_REAL)2.0) * dw_dz - ((COMPUTE_REAL)2.0)/((COMPUTE_REAL)3.0) * div_V);
    COMPUTE_REAL txz = MU * (du_dz + dw_dx);
    COMPUTE_REAL tyz = MU * (dv_dz + dw_dy);
    
    COMPUTE_REAL K = ((COMPUTE_REAL)0.5) * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    COMPUTE_REAL P = (GAMMA - ((COMPUTE_REAL)1.0)) * (E - K);
    COMPUTE_REAL dK_dz = (rhou * Q_loc[1*3 + 2] + rhov * Q_loc[2*3 + 2] + rhow * Q_loc[3*3 + 2]) / rho - K * Q_loc[0*3 + 2] / rho;
    COMPUTE_REAL dP_dz = (GAMMA - ((COMPUTE_REAL)1.0)) * (Q_loc[4*3 + 2] - dK_dz);
    COMPUTE_REAL coeff = MU * GAMMA / (PR * (GAMMA - ((COMPUTE_REAL)1.0)));
    COMPUTE_REAL qz = -coeff * (dP_dz / rho - P * Q_loc[0*3 + 2] / (rho * rho));
    
    Fvz[0] = ((COMPUTE_REAL)0.0); Fvz[1] = txz; Fvz[2] = tyz; Fvz[3] = tzz; Fvz[4] = u * txz + v * tyz + w * tzz - qz;
}

__device__ inline void rusanov_flux(const COMPUTE_REAL* UL, const COMPUTE_REAL* UR, const COMPUTE_REAL n[3], COMPUTE_REAL* F_com) {
    COMPUTE_REAL rhoL = UL[0], rhouL = UL[1], rhovL = UL[2], rhowL = UL[3], EL = UL[4];
    COMPUTE_REAL rhoR = UR[0], rhouR = UR[1], rhovR = UR[2], rhowR = UR[3], ER = UR[4];
    
    COMPUTE_REAL uL = rhouL/rhoL, vL = rhovL/rhoL, wL = rhowL/rhoL;
    COMPUTE_REAL uR = rhouR/rhoR, vR = rhovR/rhoR, wR = rhowR/rhoR;
    COMPUTE_REAL pL = compute_pressure(rhoL, rhouL, rhovL, rhowL, EL);
    COMPUTE_REAL pR = compute_pressure(rhoR, rhouR, rhovR, rhowR, ER);
    
    COMPUTE_REAL vnL = uL*n[0] + vL*n[1] + wL*n[2];
    COMPUTE_REAL vnR = uR*n[0] + vR*n[1] + wR*n[2];
    
    COMPUTE_REAL cL = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pL / fmax(((COMPUTE_REAL)1e-7), rhoL)));
    COMPUTE_REAL cR = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pR / fmax(((COMPUTE_REAL)1e-7), rhoR)));
    
    COMPUTE_REAL smax = fmax(fabs(vnL) + cL, fabs(vnR) + cR);
    
    COMPUTE_REAL FL[5], FR[5];
    FL[0] = rhoL*vnL; FL[1] = rhouL*vnL + pL*n[0]; FL[2] = rhovL*vnL + pL*n[1]; FL[3] = rhowL*vnL + pL*n[2]; FL[4] = vnL*(EL + pL);
    FR[0] = rhoR*vnR; FR[1] = rhouR*vnR + pR*n[0]; FR[2] = rhovR*vnR + pR*n[1]; FR[3] = rhowR*vnR + pR*n[2]; FR[4] = vnR*(ER + pR);
    
    for(int v=0; v<5; v++) {
        F_com[v] = ((COMPUTE_REAL)0.5) * (FL[v] + FR[v] - smax * (UR[v] - UL[v]));
    }
}

__device__ inline void ausm_plus_flux(const COMPUTE_REAL* UL, const COMPUTE_REAL* UR, const COMPUTE_REAL n[3], COMPUTE_REAL* F_com) {
    COMPUTE_REAL rhoL = UL[0], rhouL = UL[1], rhovL = UL[2], rhowL = UL[3], EL = UL[4];
    COMPUTE_REAL rhoR = UR[0], rhouR = UR[1], rhovR = UR[2], rhowR = UR[3], ER = UR[4];
    
    COMPUTE_REAL uL = rhouL/rhoL, vL = rhovL/rhoL, wL = rhowL/rhoL;
    COMPUTE_REAL uR = rhouR/rhoR, vR = rhovR/rhoR, wR = rhowR/rhoR;
    COMPUTE_REAL pL = compute_pressure(rhoL, rhouL, rhovL, rhowL, EL);
    COMPUTE_REAL pR = compute_pressure(rhoR, rhouR, rhovR, rhowR, ER);
    
    COMPUTE_REAL vnL = uL*n[0] + vL*n[1] + wL*n[2];
    COMPUTE_REAL vnR = uR*n[0] + vR*n[1] + wR*n[2];
    COMPUTE_REAL HL = (EL + pL)/rhoL;
    COMPUTE_REAL HR = (ER + pR)/rhoR;
    
    COMPUTE_REAL cL = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pL / fmax(((COMPUTE_REAL)1e-7), rhoL)));
    COMPUTE_REAL cR = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pR / fmax(((COMPUTE_REAL)1e-7), rhoR)));
    
    COMPUTE_REAL c = fmax(cL, cR);
    COMPUTE_REAL ML = vnL / c;
    COMPUTE_REAL MR = vnR / c;
    
    COMPUTE_REAL Mp, Mm, Pp, Pm;
    COMPUTE_REAL alpha = ((COMPUTE_REAL)0.1875);
    COMPUTE_REAL beta = ((COMPUTE_REAL)0.125);
    
    if (fabs(ML) >= 1.0) {
        Mp = ((COMPUTE_REAL)0.5) * (ML + fabs(ML));
        Pp = (ML > 0.0) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    } else {
        Mp = ((COMPUTE_REAL)0.25) * (ML + 1.0) * (ML + 1.0) + beta * (ML * ML - 1.0) * (ML * ML - 1.0);
        Pp = ((COMPUTE_REAL)0.25) * (ML + 1.0) * (ML + 1.0) * (((COMPUTE_REAL)2.0) - ML) + alpha * ML * (ML * ML - 1.0) * (ML * ML - 1.0);
    }
    
    if (fabs(MR) >= 1.0) {
        Mm = ((COMPUTE_REAL)0.5) * (MR - fabs(MR));
        Pm = (MR < 0.0) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    } else {
        Mm = -((COMPUTE_REAL)0.25) * (MR - 1.0) * (MR - 1.0) - beta * (MR * MR - 1.0) * (MR * MR - 1.0);
        Pm = ((COMPUTE_REAL)0.25) * (MR - 1.0) * (MR - 1.0) * (((COMPUTE_REAL)2.0) + MR) - alpha * MR * (MR * MR - 1.0) * (MR * MR - 1.0);
    }
    
    COMPUTE_REAL M = Mp + Mm;
    COMPUTE_REAL p = Pp * pL + Pm * pR;
    COMPUTE_REAL mdot = M * c * (M > 0.0 ? rhoL : rhoR);
    
    if (M > 0.0) {
        F_com[0] = mdot;
        F_com[1] = mdot * uL + p * n[0];
        F_com[2] = mdot * vL + p * n[1];
        F_com[3] = mdot * wL + p * n[2];
        F_com[4] = mdot * HL;
    } else {
        F_com[0] = mdot;
        F_com[1] = mdot * uR + p * n[0];
        F_com[2] = mdot * vR + p * n[1];
        F_com[3] = mdot * wR + p * n[2];
        F_com[4] = mdot * HR;
    }
}

// SLAU2 (Shima 2011), following the SU2 implementation (CUpwSLAU_Flow with slau2 = true)
__device__ inline void slau2_flux(const COMPUTE_REAL* UL, const COMPUTE_REAL* UR, const COMPUTE_REAL n[3], COMPUTE_REAL* F_com) {
    COMPUTE_REAL rhoL = UL[0], rhouL = UL[1], rhovL = UL[2], rhowL = UL[3], EL = UL[4];
    COMPUTE_REAL rhoR = UR[0], rhouR = UR[1], rhovR = UR[2], rhowR = UR[3], ER = UR[4];
    
    COMPUTE_REAL uL = rhouL/rhoL, vL = rhovL/rhoL, wL = rhowL/rhoL;
    COMPUTE_REAL uR = rhouR/rhoR, vR = rhovR/rhoR, wR = rhowR/rhoR;
    COMPUTE_REAL pL = compute_pressure(rhoL, rhouL, rhovL, rhowL, EL);
    COMPUTE_REAL pR = compute_pressure(rhoR, rhouR, rhovR, rhowR, ER);
    
    COMPUTE_REAL vnL = uL*n[0] + vL*n[1] + wL*n[2];
    COMPUTE_REAL vnR = uR*n[0] + vR*n[1] + wR*n[2];
    COMPUTE_REAL HL = (EL + pL)/rhoL;
    COMPUTE_REAL HR = (ER + pR)/rhoR;
    
    COMPUTE_REAL sqL = uL*uL + vL*vL + wL*wL;
    COMPUTE_REAL sqR = uR*uR + vR*vR + wR*wR;
    
    COMPUTE_REAL cL = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pL / fmax(((COMPUTE_REAL)1e-7), rhoL)));
    COMPUTE_REAL cR = sqrt(fmax(((COMPUTE_REAL)0.0), GAMMA * pR / fmax(((COMPUTE_REAL)1e-7), rhoR)));
    
    COMPUTE_REAL aF = ((COMPUTE_REAL)0.5) * (cL + cR);
    COMPUTE_REAL mL = vnL / aF;
    COMPUTE_REAL mR = vnR / aF;
    
    // Smooth local Mach number and contact discontinuity switch
    COMPUTE_REAL mach_tilde = fmin(((COMPUTE_REAL)1.0), (((COMPUTE_REAL)1.0)/aF) * sqrt(((COMPUTE_REAL)0.5) * (sqL + sqR)));
    COMPUTE_REAL chi = ((COMPUTE_REAL)1.0) - mach_tilde;
    chi = chi * chi;
    COMPUTE_REAL f_rho = -fmax(fmin(mL, ((COMPUTE_REAL)0.0)), ((COMPUTE_REAL)-1.0)) * fmin(fmax(mR, ((COMPUTE_REAL)0.0)), ((COMPUTE_REAL)1.0));
    
    // Density-weighted mean normal velocity magnitude
    COMPUTE_REAL vn_mag = (rhoL * fabs(vnL) + rhoR * fabs(vnR)) / (rhoL + rhoR);
    COMPUTE_REAL vn_magL = (((COMPUTE_REAL)1.0) - f_rho) * vn_mag + f_rho * fabs(vnL);
    COMPUTE_REAL vn_magR = (((COMPUTE_REAL)1.0) - f_rho) * vn_mag + f_rho * fabs(vnR);
    
    COMPUTE_REAL mdot = ((COMPUTE_REAL)0.5) * (rhoL * (vnL + vn_magL) + rhoR * (vnR - vn_magR) - (chi / aF) * (pR - pL));
    
    // Pressure split: AUSM+ polynomials with alpha = 0
    COMPUTE_REAL betaL, betaR;
    if (fabs(mL) < 1.0) {
        betaL = ((COMPUTE_REAL)0.25) * (((COMPUTE_REAL)2.0) - mL) * (mL + 1.0) * (mL + 1.0);
    } else {
        betaL = (mL >= 0.0) ? ((COMPUTE_REAL)1.0) : ((COMPUTE_REAL)0.0);
    }
    if (fabs(mR) < 1.0) {
        betaR = ((COMPUTE_REAL)0.25) * (((COMPUTE_REAL)2.0) + mR) * (mR - 1.0) * (mR - 1.0);
    } else {
        betaR = (mR >= 0.0) ? ((COMPUTE_REAL)0.0) : ((COMPUTE_REAL)1.0);
    }
    
    COMPUTE_REAL p_face = ((COMPUTE_REAL)0.5) * (pL + pR) + ((COMPUTE_REAL)0.5) * (betaL - betaR) * (pL - pR);
    p_face += sqrt(((COMPUTE_REAL)0.5) * (sqL + sqR)) * (betaL + betaR - ((COMPUTE_REAL)1.0)) * aF * ((COMPUTE_REAL)0.5) * (rhoL + rhoR);
    
    if (mdot > 0.0) {
        F_com[0] = mdot;
        F_com[1] = mdot * uL + p_face * n[0];
        F_com[2] = mdot * vL + p_face * n[1];
        F_com[3] = mdot * wL + p_face * n[2];
        F_com[4] = mdot * HL;
    } else {
        F_com[0] = mdot;
        F_com[1] = mdot * uR + p_face * n[0];
        F_com[2] = mdot * vR + p_face * n[1];
        F_com[3] = mdot * wR + p_face * n[2];
        F_com[4] = mdot * HR;
    }
}

#if defined(FLUX_SCHEME_SLAU2)
#define compute_interface_flux slau2_flux
#elif defined(FLUX_SCHEME_AUSMPLUS)
#define compute_interface_flux ausm_plus_flux
#else
#define compute_interface_flux rusanov_flux
#endif
