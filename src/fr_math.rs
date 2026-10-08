pub fn legendre_poly(p: usize, x: f64) -> f64 {
    if p == 0 { return 1.0; }
    if p == 1 { return x; }
    let mut l_m2 = 1.0;
    let mut l_m1 = x;
    let mut l = 0.0;
    for k in 2..=p {
        let k_f = k as f64;
        l = ((2.0 * k_f - 1.0) * x * l_m1 - (k_f - 1.0) * l_m2) / k_f;
        l_m2 = l_m1;
        l_m1 = l;
    }
    l
}

pub fn legendre_poly_deriv(p: usize, x: f64) -> f64 {
    if p == 0 { return 0.0; }
    let p_f = p as f64;
    p_f / (x * x - 1.0) * (x * legendre_poly(p, x) - legendre_poly(p - 1, x))
}

pub fn gauss_legendre_points(p: usize) -> Vec<f64> {
    // For a polynomial of degree P, we need P+1 points.
    let n = p + 1;
    let mut roots = vec![0.0; n];
    let tol = 1e-14;
    
    for i in 0..n {
        // Initial guess
        let mut x = f64::cos(std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5));
        loop {
            let l_n = legendre_poly(n, x);
            let l_n_prime = legendre_poly_deriv(n, x);
            let dx = l_n / l_n_prime;
            x -= dx;
            if dx.abs() < tol {
                break;
            }
        }
        roots[i] = x;
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots
}

pub fn lagrange_poly(points: &[f64], i: usize, x: f64) -> f64 {
    let mut l = 1.0;
    for (j, &xj) in points.iter().enumerate() {
        if i != j {
            l *= (x - xj) / (points[i] - xj);
        }
    }
    l
}

pub fn lagrange_deriv_matrix(points: &[f64]) -> Vec<f64> {
    let n = points.len();
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut prod = 1.0;
                for k in 0..n {
                    if k != i && k != j {
                        prod *= points[i] - points[k];
                    }
                }
                let mut den = 1.0;
                for k in 0..n {
                    if k != j {
                        den *= points[j] - points[k];
                    }
                }
                d[i * n + j] = prod / den;
            } else {
                let mut sum = 0.0;
                for k in 0..n {
                    if k != i {
                        sum += 1.0 / (points[i] - points[k]);
                    }
                }
                d[i * n + i] = sum;
            }
        }
    }
    d
}

pub struct FrOperators {
    pub points: Vec<f64>,
    pub d_matrix: Vec<f64>,
    pub extrap_left: Vec<f64>,
    pub extrap_right: Vec<f64>,
    pub g_left_prime: Vec<f64>,
    pub g_right_prime: Vec<f64>,
}

pub fn generate_fr_operators(p: usize) -> FrOperators {
    let points = gauss_legendre_points(p);
    let d_matrix = lagrange_deriv_matrix(&points);
    let n = p + 1;
    
    let mut extrap_left = vec![0.0; n];
    let mut extrap_right = vec![0.0; n];
    for i in 0..n {
        extrap_left[i] = lagrange_poly(&points, i, -1.0);
        extrap_right[i] = lagrange_poly(&points, i, 1.0);
    }
    
    let mut g_left_prime = vec![0.0; n];
    let mut g_right_prime = vec![0.0; n];
    
    // g_R = 0.5 * (L_P + L_{P+1})
    // g_L = 0.5 * (-1)^P * (L_P - L_{P+1})
    let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
    
    for (i, &xi) in points.iter().enumerate() {
        let lp_prime = legendre_poly_deriv(p, xi);
        let lp1_prime = legendre_poly_deriv(p + 1, xi);
        
        g_right_prime[i] = 0.5 * (lp_prime + lp1_prime);
        g_left_prime[i] = 0.5 * sign * (lp_prime - lp1_prime);
    }
    
    FrOperators {
        points,
        d_matrix,
        extrap_left,
        extrap_right,
        g_left_prime,
        g_right_prime,
    }
}
