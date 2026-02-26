/// Ported exactly from `svd.ts` to ensure 1:1 bit parity in pose estimation logic.
/// Singular Value Decomposition computed via Householder reduction to bidiagonal form
/// and QR transformations.
pub struct Svd;

impl Svd {
    /// SVD wrapper specialized for 3x3 matrices to avoid Vec allocations overheads.
    #[allow(
        clippy::needless_range_loop,
        clippy::assign_op_pattern,
        unused_assignments
    )]
    pub fn svdcmp(a: &mut [[f64; 3]; 3], w: &mut [f64; 3], v: &mut [[f64; 3]; 3]) -> bool {
        let m = 3;
        let n = 3;
        let mut flag;
        let mut its;
        let mut jj;
        let mut l = 0;
        let mut nm = 0;
        let mut anorm: f64 = 0.0;
        let mut c;
        let mut f;
        let mut g = 0.0;
        let mut h;
        let mut s;
        let mut scale = 0.0;
        let mut x;
        let mut y;
        let mut z;
        let mut rv1 = [0.0; 3];

        // Householder reduction to bidiagonal form
        for i in 0..n {
            l = i + 1;
            rv1[i] = scale * g;
            g = 0.0;
            s = 0.0;
            scale = 0.0;
            if i < m {
                for k in i..m {
                    scale += a[k][i].abs();
                }
                if scale != 0.0 {
                    for k in i..m {
                        a[k][i] /= scale;
                        s += a[k][i] * a[k][i];
                    }
                    f = a[i][i];
                    g = -Self::sign(s.sqrt(), f);
                    h = f * g - s;
                    a[i][i] = f - g;
                    for j in l..n {
                        s = 0.0;
                        for k in i..m {
                            s += a[k][i] * a[k][j];
                        }
                        f = s / h;
                        for k in i..m {
                            a[k][j] += f * a[k][i];
                        }
                    }
                    for k in i..m {
                        a[k][i] *= scale;
                    }
                }
            }
            w[i] = scale * g;
            g = 0.0;
            s = 0.0;
            scale = 0.0;
            if i < m && i != n - 1 {
                for k in l..n {
                    scale += a[i][k].abs();
                }
                if scale != 0.0 {
                    for k in l..n {
                        a[i][k] /= scale;
                        s += a[i][k] * a[i][k];
                    }
                    f = a[i][l];
                    g = -Self::sign(s.sqrt(), f);
                    h = f * g - s;
                    a[i][l] = f - g;
                    for k in l..n {
                        rv1[k] = a[i][k] / h;
                    }
                    for j in l..m {
                        s = 0.0;
                        for k in l..n {
                            s += a[j][k] * a[i][k];
                        }
                        for k in l..n {
                            a[j][k] += s * rv1[k];
                        }
                    }
                    for k in l..n {
                        a[i][k] *= scale;
                    }
                }
            }
            anorm = anorm.max(w[i].abs() + rv1[i].abs());
        }

        // Accumulation of right-hand transformation
        for i in (0..n).rev() {
            if i < n - 1 {
                if g != 0.0 {
                    for j in l..n {
                        v[j][i] = a[i][j] / a[i][l] / g;
                    }
                    for j in l..n {
                        s = 0.0;
                        for k in l..n {
                            s += a[i][k] * v[k][j];
                        }
                        for k in l..n {
                            v[k][j] += s * v[k][i];
                        }
                    }
                }
                for j in l..n {
                    v[i][j] = 0.0;
                    v[j][i] = 0.0;
                }
            }
            v[i][i] = 1.0;
            g = rv1[i];
            l = i;
        }

        // Accumulation of left-hand transformation
        for i in (0..m.min(n)).rev() {
            l = i + 1;
            g = w[i];
            for j in l..n {
                a[i][j] = 0.0;
            }
            if g != 0.0 {
                g = 1.0 / g;
                for j in l..n {
                    s = 0.0;
                    for k in l..m {
                        s += a[k][i] * a[k][j];
                    }
                    f = (s / a[i][i]) * g;
                    for k in i..m {
                        a[k][j] += f * a[k][i];
                    }
                }
                for j in i..m {
                    a[j][i] *= g;
                }
            } else {
                for j in i..m {
                    a[j][i] = 0.0;
                }
            }
            a[i][i] += 1.0;
        }

        // Diagonalization of the bidiagonal form
        for k in (0..n).rev() {
            for its_loop in 1..=30 {
                its = its_loop;
                flag = true;
                for l_loop in (0..=k).rev() {
                    l = l_loop;
                    if l == 0 {
                        nm = 0;
                    } else {
                        nm = l - 1;
                    }
                    if rv1[l].abs() + anorm == anorm {
                        flag = false;
                        break;
                    }
                    if w[nm].abs() + anorm == anorm {
                        break;
                    }
                }
                if flag {
                    c = 0.0;
                    s = 1.0;
                    for i in l..=k {
                        f = s * rv1[i];
                        if f.abs() + anorm == anorm {
                            break;
                        }
                        g = w[i];
                        h = Self::pythag(f, g);
                        w[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        for j in 0..m {
                            y = a[j][nm];
                            z = a[j][i];
                            a[j][nm] = y * c + z * s;
                            a[j][i] = z * c - y * s;
                        }
                    }
                }

                // Convergence
                z = w[k];
                if l == k {
                    if z < 0.0 {
                        w[k] = -z;
                        for j in 0..n {
                            v[j][k] = -v[j][k];
                        }
                    }
                    break;
                }

                if its == 30 {
                    return false;
                }

                // Shift from bottom 2-by-2 minor
                x = w[l];
                nm = k - 1;
                y = w[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = Self::pythag(f, 1.0);
                f = ((x - z) * (x + z) + h * (y / (f + Self::sign(g, f)) - h)) / x;

                // Next QR transformation
                c = 1.0;
                s = 1.0;
                for j in l..=nm {
                    let i = j + 1;
                    g = rv1[i];
                    y = w[i];
                    h = s * g;
                    g = c * g;
                    z = Self::pythag(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    for jj_loop in 0..n {
                        jj = jj_loop;
                        x = v[jj][j];
                        z = v[jj][i];
                        v[jj][j] = x * c + z * s;
                        v[jj][i] = z * c - x * s;
                    }
                    z = Self::pythag(f, h);
                    w[j] = z;
                    if z != 0.0 {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    for jj_loop in 0..m {
                        jj = jj_loop;
                        y = a[jj][j];
                        z = a[jj][i];
                        a[jj][j] = y * c + z * s;
                        a[jj][i] = z * c - y * s;
                    }
                }
                rv1[l] = 0.0;
                rv1[k] = f;
                w[k] = x;
            }
        }

        true
    }

    fn pythag(a: f64, b: f64) -> f64 {
        let at = a.abs();
        let bt = b.abs();
        let ct: f64;

        if at > bt {
            ct = bt / at;
            return at * (1.0 + ct * ct).sqrt();
        }

        if bt == 0.0 {
            return 0.0;
        }

        ct = at / bt;
        bt * (1.0 + ct * ct).sqrt()
    }

    fn sign(a: f64, b: f64) -> f64 {
        if b >= 0.0 {
            a.abs()
        } else {
            -a.abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd() {
        let mut a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut w = [0.0; 3];
        let mut v = [[0.0; 3]; 3];

        let result = Svd::svdcmp(&mut a, &mut w, &mut v);
        assert!(result);

        // Typical singular values for a 3x3 matrix 1-9
        assert!((w[0] - 16.848).abs() < 0.001); // Approximation for standard tests
    }
}
