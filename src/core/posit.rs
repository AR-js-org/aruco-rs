use crate::core::svd::Svd;
use crate::Point2f;
use nalgebra::{Matrix3, Vector3};

/// A computed 3D pose of a recognized marker.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub best_error: f64,
    pub best_rotation: Matrix3<f64>,
    pub best_translation: Vector3<f64>,
    pub alternative_error: f64,
    pub alternative_rotation: Matrix3<f64>,
    pub alternative_translation: Vector3<f64>,
}

impl Pose {
    pub fn new(
        error1: f64,
        rotation1: Matrix3<f64>,
        translation1: Vector3<f64>,
        error2: f64,
        rotation2: Matrix3<f64>,
        translation2: Vector3<f64>,
    ) -> Self {
        Self {
            best_error: error1,
            best_rotation: rotation1,
            best_translation: translation1,
            alternative_error: error2,
            alternative_rotation: rotation2,
            alternative_translation: translation2,
        }
    }
}

/// The POSIT (Pose from Orthography and Scaling with Iterations) algorithms
/// ported from `posit.ts`.
pub struct Posit {
    model: [Vector3<f64>; 4],
    focal_length: f64,
    model_vectors: Matrix3<f64>,
    model_normal: Vector3<f64>,
    model_pseudo_inverse: Matrix3<f64>,
}

impl Posit {
    /// Creates a new POSIT estimator for a square marker of `model_size`.
    pub fn new(model_size: f64, focal_length: f64) -> Self {
        let half = model_size / 2.0;
        let model = [
            Vector3::new(-half, half, 0.0),
            Vector3::new(half, half, 0.0),
            Vector3::new(half, -half, 0.0),
            Vector3::new(-half, -half, 0.0),
        ];

        // JS: Mat3.fromRows(Vec3, Vec3, Vec3) -> 3 rows.
        let model_vectors = Matrix3::from_rows(&[
            (model[1] - model[0]).transpose(),
            (model[2] - model[0]).transpose(),
            (model[3] - model[0]).transpose(),
        ]);

        // Let's use standard array conversion for SVD
        let mut u_arr = [
            [
                model_vectors[(0, 0)],
                model_vectors[(0, 1)],
                model_vectors[(0, 2)],
            ],
            [
                model_vectors[(1, 0)],
                model_vectors[(1, 1)],
                model_vectors[(1, 2)],
            ],
            [
                model_vectors[(2, 0)],
                model_vectors[(2, 1)],
                model_vectors[(2, 2)],
            ],
        ];

        let mut d_arr = [0.0; 3];
        let mut v_arr = [[0.0; 3]; 3];

        Svd::svdcmp(&mut u_arr, &mut d_arr, &mut v_arr);

        // Convert back
        let d = Vector3::new(d_arr[0], d_arr[1], d_arr[2]);
        let v = Matrix3::new(
            v_arr[0][0],
            v_arr[0][1],
            v_arr[0][2],
            v_arr[1][0],
            v_arr[1][1],
            v_arr[1][2],
            v_arr[2][0],
            v_arr[2][1],
            v_arr[2][2],
        );

        let u_mat = Matrix3::new(
            u_arr[0][0],
            u_arr[0][1],
            u_arr[0][2],
            u_arr[1][0],
            u_arr[1][1],
            u_arr[1][2],
            u_arr[2][0],
            u_arr[2][1],
            u_arr[2][2],
        );

        // Inverse of d elements
        let d_inv = Vector3::new(
            if d.x != 0.0 { 1.0 / d.x } else { 0.0 },
            if d.y != 0.0 { 1.0 / d.y } else { 0.0 },
            if d.z != 0.0 { 1.0 / d.z } else { 0.0 },
        );
        let d_mat = Matrix3::from_diagonal(&d_inv);

        // this.modelPseudoInverse = Mat3.mult(Mat3.mult(v, Mat3.fromDiagonal(Vec3.inverse(d))), Mat3.transpose(u));
        let model_pseudo_inverse = v * d_mat * u_mat.transpose();

        // this.modelNormal = v.column(d.minIndex());
        // Find min index of d
        let mut min_idx = 0;
        let mut min_val = d.x;
        if d.y < min_val {
            min_idx = 1;
            min_val = d.y;
        }
        if d.z < min_val {
            min_idx = 2;
        }

        let model_normal = v.column(min_idx).into_owned();

        Self {
            model,
            focal_length,
            model_vectors,
            model_normal,
            model_pseudo_inverse,
        }
    }

    /// Computes the Pose iterating over orthography hypotheses.
    pub fn pose(&self, points: &[Point2f; 4]) -> Pose {
        let mut rotation1 = Matrix3::identity();
        let mut rotation2 = Matrix3::identity();
        let mut translation1 = Vector3::zeros();
        let mut translation2 = Vector3::zeros();
        let eps = Vector3::new(1.0, 1.0, 1.0);

        self.pos(
            points,
            &eps,
            &mut rotation1,
            &mut rotation2,
            &mut translation1,
            &mut translation2,
        );

        let mut rot1_clone = rotation1;
        let mut trans1_clone = translation1;
        let error1 = self.iterate(points, &mut rot1_clone, &mut trans1_clone);

        let mut rot2_clone = rotation2;
        let mut trans2_clone = translation2;
        let error2 = self.iterate(points, &mut rot2_clone, &mut trans2_clone);

        if error1 < error2 {
            Pose::new(
                error1,
                rot1_clone,
                trans1_clone,
                error2,
                rot2_clone,
                trans2_clone,
            )
        } else {
            Pose::new(
                error2,
                rot2_clone,
                trans2_clone,
                error1,
                rot1_clone,
                trans1_clone,
            )
        }
    }

    fn pos(
        &self,
        points: &[Point2f; 4],
        eps: &Vector3<f64>,
        rotation1: &mut Matrix3<f64>,
        rotation2: &mut Matrix3<f64>,
        translation1: &mut Vector3<f64>,
        translation2: &mut Vector3<f64>,
    ) {
        let xi = Vector3::new(points[1].x as f64, points[2].x as f64, points[3].x as f64);
        let yi = Vector3::new(points[1].y as f64, points[2].y as f64, points[3].y as f64);

        let xs = xi.component_mul(eps).add_scalar(-(points[0].x as f64));
        let ys = yi.component_mul(eps).add_scalar(-(points[0].y as f64));

        let i0 = self.model_pseudo_inverse * xs;
        let j0 = self.model_pseudo_inverse * ys;

        let s = j0.norm_squared() - i0.norm_squared();
        let ij = i0.dot(&j0);

        let r;
        let mut theta;

        if s == 0.0 {
            r = (2.0 * ij).abs().sqrt();
            theta = (-std::f64::consts::PI / 2.0)
                * if ij < 0.0 {
                    -1.0
                } else if ij > 0.0 {
                    1.0
                } else {
                    0.0
                };
        } else {
            r = (s * s + 4.0 * ij * ij).sqrt().sqrt();
            theta = (-2.0 * ij / s).atan();
            if s < 0.0 {
                theta += std::f64::consts::PI;
            }
            theta /= 2.0;
        }

        let lambda = r * theta.cos();
        let mu = r * theta.sin();

        // First possible rotation/translation
        let mut i = i0 + self.model_normal * lambda;
        let mut j = j0 + self.model_normal * mu;
        let inorm = i.normalize_mut();
        let jnorm = j.normalize_mut();
        let k = i.cross(&j);
        *rotation1 = Matrix3::from_columns(&[i, j, k]); // Equivalent to JS fromRows because Mat3 mult vector does row.dot(vec).

        let scale = (inorm + jnorm) / 2.0;
        let temp = *rotation1 * self.model[0];
        *translation1 = Vector3::new(
            (points[0].x as f64) / scale - temp.x,
            (points[0].y as f64) / scale - temp.y,
            self.focal_length / scale,
        );

        // Second possible rotation/translation
        let mut i2 = i0 - self.model_normal * lambda;
        let mut j2 = j0 - self.model_normal * mu;
        let inorm2 = i2.normalize_mut();
        let jnorm2 = j2.normalize_mut();
        let k2 = i2.cross(&j2);
        *rotation2 = Matrix3::from_columns(&[i2, j2, k2]);

        let scale2 = (inorm2 + jnorm2) / 2.0;
        let temp2 = *rotation2 * self.model[0];
        *translation2 = Vector3::new(
            (points[0].x as f64) / scale2 - temp2.x,
            (points[0].y as f64) / scale2 - temp2.y,
            self.focal_length / scale2,
        );
    }

    fn iterate(
        &self,
        points: &[Point2f; 4],
        rotation: &mut Matrix3<f64>,
        translation: &mut Vector3<f64>,
    ) -> f64 {
        let mut prev_error = f64::INFINITY;
        let mut rotation1 = Matrix3::identity();
        let mut rotation2 = Matrix3::identity();
        let mut translation1 = Vector3::zeros();
        let mut translation2 = Vector3::zeros();
        let mut error = 0.0;

        for _ in 0..100 {
            // JS: eps = Vec3.addScalar(Vec3.multScalar(Mat3.multVector(this.modelVectors, rotation.row(2)), 1.0 / translation.v[2]), 1.0);
            let row2 = rotation.row(2).transpose(); // 3x1 vector
            let vec_eps = (self.model_vectors * row2) * (1.0 / translation.z);
            let eps = vec_eps.add_scalar(1.0);

            self.pos(
                points,
                &eps,
                &mut rotation1,
                &mut rotation2,
                &mut translation1,
                &mut translation2,
            );

            let error1 = self.get_error(points, &rotation1, &translation1);
            let error2 = self.get_error(points, &rotation2, &translation2);

            if error1 < error2 {
                *rotation = rotation1;
                *translation = translation1;
                error = error1;
            } else {
                *rotation = rotation2;
                *translation = translation2;
                error = error2;
            }

            if error <= 2.0 || error > prev_error {
                break;
            }
            prev_error = error;
        }

        error
    }

    fn get_error(
        &self,
        points: &[Point2f; 4],
        rotation: &Matrix3<f64>,
        translation: &Vector3<f64>,
    ) -> f64 {
        let v1 = rotation * self.model[0] + translation;
        let v2 = rotation * self.model[1] + translation;
        let v3 = rotation * self.model[2] + translation;
        let v4 = rotation * self.model[3] + translation;

        let modeled = [
            Point2f::new(
                (v1.x * self.focal_length / v1.z) as f32,
                (v1.y * self.focal_length / v1.z) as f32,
            ),
            Point2f::new(
                (v2.x * self.focal_length / v2.z) as f32,
                (v2.y * self.focal_length / v2.z) as f32,
            ),
            Point2f::new(
                (v3.x * self.focal_length / v3.z) as f32,
                (v3.y * self.focal_length / v3.z) as f32,
            ),
            Point2f::new(
                (v4.x * self.focal_length / v4.z) as f32,
                (v4.y * self.focal_length / v4.z) as f32,
            ),
        ];

        let ia1 = Self::angle(&points[0], &points[1], &points[3]);
        let ia2 = Self::angle(&points[1], &points[2], &points[0]);
        let ia3 = Self::angle(&points[2], &points[3], &points[1]);
        let ia4 = Self::angle(&points[3], &points[0], &points[2]);

        let ma1 = Self::angle(&modeled[0], &modeled[1], &modeled[3]);
        let ma2 = Self::angle(&modeled[1], &modeled[2], &modeled[0]);
        let ma3 = Self::angle(&modeled[2], &modeled[3], &modeled[1]);
        let ma4 = Self::angle(&modeled[3], &modeled[0], &modeled[2]);

        ((ia1 - ma1).abs() + (ia2 - ma2).abs() + (ia3 - ma3).abs() + (ia4 - ma4).abs()) / 4.0
    }

    fn angle(a: &Point2f, b: &Point2f, c: &Point2f) -> f64 {
        let x1 = (b.x - a.x) as f64;
        let y1 = (b.y - a.y) as f64;
        let x2 = (c.x - a.x) as f64;
        let y2 = (c.y - a.y) as f64;

        let dot = x1 * x2 + y1 * y2;
        let mag1 = (x1 * x1 + y1 * y1).sqrt();
        let mag2 = (x2 * x2 + y2 * y2).sqrt();

        // Clamp domain to avoid NaN from float imprecision
        let cos_val = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        cos_val.acos() * 180.0 / std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posit_basic() {
        let focal_length = 500.0;
        let model_size = 35.0; // standard marker size

        let posit = Posit::new(model_size, focal_length);

        // Simulate marker projected directly in front of the camera, completely flat.
        // It occupies a 100x100 pixel space at distance X.
        let half_screen = 50.0;
        let points = [
            Point2f::new(-half_screen, half_screen),  // top-left
            Point2f::new(half_screen, half_screen),   // top-right
            Point2f::new(half_screen, -half_screen),  // bottom-right
            Point2f::new(-half_screen, -half_screen), // bottom-left
        ];

        let pose = posit.pose(&points);

        // Due to the symmetry of a frontal view, both poses are valid but roughly frontal.
        // The rotation should be near identity or PI rotated, and translation Z should positive.

        let trans = pose.best_translation;
        // Basic sanity bounds on estimated translation distance
        assert!(trans.z > 0.0);

        // Exact Z depth expected: focal_length * (real_size / screen_size)
        // 500 * (35 / 100) = 175
        assert!((trans.z - 175.0).abs() < 1.0);

        // Since it's centered at 0,0 XY should be close to 0
        assert!(trans.x.abs() < 1.0);
        assert!(trans.y.abs() < 1.0);
    }
}
