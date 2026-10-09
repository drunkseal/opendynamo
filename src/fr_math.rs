use std::f64::consts;

pub fn legendre_poly(p: usize, x: f64) -> f64 {
    if p == 0 {
        return 1.0;
    }
    if p == 1 {
        return x;
    }
    let mut legendre_prev2 = 1.0;
    let mut legendre_prev = x;
    let mut poly_value = 0.0;
    for k in 2..=p {
        let k_float = k as f64;
        poly_value = ((2.0 * k_float - 1.0) * x * legendre_prev - (k_float - 1.0) * legendre_prev2)
            / k_float;
        legendre_prev2 = legendre_prev;
        legendre_prev = poly_value;
    }
    poly_value
}

pub fn legendre_poly_deriv(p: usize, x: f64) -> f64 {
    if p == 0 {
        return 0.0;
    }
    let p_float = p as f64;
    p_float / (x * x - 1.0) * (x * legendre_poly(p, x) - legendre_poly(p - 1, x))
}

pub fn gauss_legendre_points(p: usize) -> Vec<f64> {
    // For a polynomial of degree P, we need P+1 points.
    let n = p + 1;
    let mut roots = vec![0.0; n];
    let tol = 1e-14;

    for (i, root) in roots.iter_mut().enumerate() {
        // Initial guess
        let mut x = (consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        loop {
            let legendre_value = legendre_poly(n, x);
            let legendre_derivative = legendre_poly_deriv(n, x);
            let newton_step = legendre_value / legendre_derivative;
            x -= newton_step;
            if newton_step.abs() < tol {
                break;
            }
        }
        *root = x;
    }
    roots.sort_by(f64::total_cmp);
    roots
}

pub fn lagrange_poly(points: &[f64], i: usize, x: f64) -> f64 {
    let mut poly_value = 1.0;
    for (j, &x_j) in points.iter().enumerate() {
        if i != j {
            poly_value *= (x - x_j) / (points[i] - x_j);
        }
    }
    poly_value
}

pub fn lagrange_deriv_matrix(points: &[f64]) -> Vec<f64> {
    let n = points.len();
    let mut deriv_matrix = vec![0.0; n * n];
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
                deriv_matrix[i * n + j] = prod / den;
            } else {
                let mut sum = 0.0;
                for k in 0..n {
                    if k != i {
                        sum += 1.0 / (points[i] - points[k]);
                    }
                }
                deriv_matrix[i * n + i] = sum;
            }
        }
    }
    deriv_matrix
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

    // g_right_prime = 0.5 * (L_P + L_{P+1})
    // g_left_prime = 0.5 * (-1)^P * (L_P - L_{P+1})
    let sign = if p.is_multiple_of(2) { 1.0 } else { -1.0 };

    for (i, &xi) in points.iter().enumerate() {
        let deriv_p = legendre_poly_deriv(p, xi);
        let deriv_p_plus1 = legendre_poly_deriv(p + 1, xi);

        g_right_prime[i] = 0.5 * (deriv_p + deriv_p_plus1);
        g_left_prime[i] = 0.5 * sign * (deriv_p - deriv_p_plus1);
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
