// src/cv/geometry.rs

use crate::Point2i;

/// Douglas-Peucker contour simplification algorithm.
/// Iteratively approximates a curve to a lower number of vertices based on epsilon.
/// Strict functional port of `CV.approxPolyDP` from ARuco-ts.
///
/// # Arguments
/// * `contour` - The input boundary points.
/// * `epsilon` - Epsilon defining simplification scale.
///
/// # Returns
/// A smaller list of representative `Point2i` vertices forming the approximated shape.
pub fn approx_poly_dp(contour: &[Point2i], epsilon: f64) -> Vec<Point2i> {
    let len = contour.len();
    if len == 0 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Slice {
        start_index: usize,
        end_index: usize,
    }

    let mut slice = Slice {
        start_index: 0,
        end_index: 0,
    };
    let mut right_slice = Slice {
        start_index: 0,
        end_index: 0,
    };
    let mut poly = Vec::new();
    let mut stack = Vec::new();

    let epsilon_sq = epsilon * epsilon;

    let mut k = 0;
    let mut start_pt = contour[0];
    let mut max_dist = 0.0;

    for _ in 0..3 {
        max_dist = 0.0;
        k = (k + right_slice.start_index) % len;
        start_pt = contour[k];
        k += 1;
        if k == len {
            k = 0;
        }

        for j in 1..len {
            let pt = contour[k];
            k += 1;
            if k == len {
                k = 0;
            }

            let dx = (pt.x - start_pt.x) as f64;
            let dy = (pt.y - start_pt.y) as f64;
            let dist = dx * dx + dy * dy;

            if dist > max_dist {
                max_dist = dist;
                right_slice.start_index = j;
            }
        }
    }

    if max_dist <= epsilon_sq {
        poly.push(Point2i::new(start_pt.x, start_pt.y));
    } else {
        slice.start_index = k;
        right_slice.start_index += slice.start_index;
        slice.end_index = right_slice.start_index;

        right_slice.start_index -= if right_slice.start_index >= len {
            len
        } else {
            0
        };
        right_slice.end_index = slice.start_index;
        if right_slice.end_index < right_slice.start_index {
            right_slice.end_index += len;
        }

        stack.push(Slice {
            start_index: right_slice.start_index,
            end_index: right_slice.end_index,
        });
        stack.push(Slice {
            start_index: slice.start_index,
            end_index: slice.end_index,
        });
    }

    while let Some(mut current_slice) = stack.pop() {
        let end_pt = contour[current_slice.end_index % len];
        k = current_slice.start_index % len;
        start_pt = contour[k];
        k += 1;
        if k == len {
            k = 0;
        }

        let le_eps;

        if current_slice.end_index <= current_slice.start_index + 1 {
            le_eps = true;
        } else {
            max_dist = 0.0;
            let dx = (end_pt.x - start_pt.x) as f64;
            let dy = (end_pt.y - start_pt.y) as f64;

            for i in (current_slice.start_index + 1)..current_slice.end_index {
                let pt = contour[k];
                k += 1;
                if k == len {
                    k = 0;
                }

                let ptx = pt.x as f64;
                let pty = pt.y as f64;
                let stx = start_pt.x as f64;
                let sty = start_pt.y as f64;

                let dist = ((pty - sty) * dx - (ptx - stx) * dy).abs();

                if dist > max_dist {
                    max_dist = dist;
                    right_slice.start_index = i;
                }
            }

            le_eps = max_dist * max_dist <= epsilon_sq * (dx * dx + dy * dy);
        }

        if le_eps {
            poly.push(Point2i::new(start_pt.x, start_pt.y));
        } else {
            right_slice.end_index = current_slice.end_index;
            current_slice.end_index = right_slice.start_index;

            stack.push(Slice {
                start_index: right_slice.start_index,
                end_index: right_slice.end_index,
            });
            stack.push(Slice {
                start_index: current_slice.start_index,
                end_index: current_slice.end_index,
            });
        }
    }

    poly
}

/// Calculates the spatial perimeter length of a contour.
/// Strict functional port of `CV.perimeter` from ARuco-ts.
pub fn perimeter(poly: &[Point2i]) -> f64 {
    let len = poly.len();
    if len == 0 {
        return 0.0;
    }

    let mut p = 0.0;
    let mut j = len - 1;
    for i in 0..len {
        let dx = (poly[i].x - poly[j].x) as f64;
        let dy = (poly[i].y - poly[j].y) as f64;
        p += (dx * dx + dy * dy).sqrt();
        j = i;
    }
    p
}

/// Calculates the minimum squared distance boundary length.
/// Strict functional port of `CV.minEdgeLength` from ARuco-ts.
pub fn min_edge_length(poly: &[Point2i]) -> f64 {
    let len = poly.len();
    if len <= 1 {
        return 0.0;
    }

    let mut min_d = f64::INFINITY;
    let mut j = len - 1;
    for i in 0..len {
        let dx = (poly[i].x - poly[j].x) as f64;
        let dy = (poly[i].y - poly[j].y) as f64;
        let d = dx * dx + dy * dy;
        if d < min_d {
            min_d = d;
        }
        j = i;
    }
    min_d.sqrt()
}

/// Tests if the contour shape is strictly convex.
/// Strict functional port of `CV.isContourConvex` from ARuco-ts.
pub fn is_contour_convex(contour: &[Point2i]) -> bool {
    let len = contour.len();
    if len == 0 {
        return false;
    }

    let mut orientation = 0;
    let mut convex = true;

    let mut prev_pt = contour[len - 1];
    let mut cur_pt = contour[0];

    let mut dx0 = cur_pt.x - prev_pt.x;
    let mut dy0 = cur_pt.y - prev_pt.y;

    let mut j = 0;
    for _ in 0..len {
        j += 1;
        if j == len {
            j = 0;
        }

        prev_pt = cur_pt;
        cur_pt = contour[j];

        let dx = cur_pt.x - prev_pt.x;
        let dy = cur_pt.y - prev_pt.y;

        // i64 prevents cross-product overflow for extremely large arrays
        let dxdy0 = (dx as i64) * (dy0 as i64);
        let dydx0 = (dy as i64) * (dx0 as i64);

        let branch = if dydx0 > dxdy0 {
            1
        } else if dydx0 < dxdy0 {
            2
        } else {
            3
        };

        orientation |= branch;

        if orientation == 3 {
            convex = false;
            break;
        }

        dx0 = dx;
        dy0 = dy;
    }

    convex
}

/// Solves the perspective transform mapping array.
/// Strict functional port of `CV.square2quad` from ARuco-ts.
pub fn square2quad(src: &[crate::Point2f; 4]) -> [f64; 9] {
    let mut sq = [0.0; 9];
    let px = (src[0].x - src[1].x + src[2].x - src[3].x) as f64;
    let py = (src[0].y - src[1].y + src[2].y - src[3].y) as f64;

    if px == 0.0 && py == 0.0 {
        sq[0] = (src[1].x - src[0].x) as f64;
        sq[1] = (src[2].x - src[1].x) as f64;
        sq[2] = src[0].x as f64;
        sq[3] = (src[1].y - src[0].y) as f64;
        sq[4] = (src[2].y - src[1].y) as f64;
        sq[5] = src[0].y as f64;
        sq[6] = 0.0;
        sq[7] = 0.0;
        sq[8] = 1.0;
    } else {
        let dx1 = (src[1].x - src[2].x) as f64;
        let dx2 = (src[3].x - src[2].x) as f64;
        let dy1 = (src[1].y - src[2].y) as f64;
        let dy2 = (src[3].y - src[2].y) as f64;
        let den = dx1 * dy2 - dx2 * dy1;

        sq[6] = (px * dy2 - dx2 * py) / den;
        sq[7] = (dx1 * py - px * dy1) / den;
        sq[8] = 1.0;
        sq[0] = (src[1].x - src[0].x) as f64 + sq[6] * (src[1].x as f64);
        sq[1] = (src[3].x - src[0].x) as f64 + sq[7] * (src[3].x as f64);
        sq[2] = src[0].x as f64;
        sq[3] = (src[1].y - src[0].y) as f64 + sq[6] * (src[1].y as f64);
        sq[4] = (src[3].y - src[0].y) as f64 + sq[7] * (src[3].y as f64);
        sq[5] = src[0].y as f64;
    }
    sq
}

/// Computes the 3x3 homography matrix for warping.
/// Strict functional port of `CV.getPerspectiveTransform` from ARuco-ts.
pub fn get_perspective_transform(src: &[crate::Point2f; 4], size: f64) -> [f64; 9] {
    let mut rq = square2quad(src);

    rq[0] /= size;
    rq[1] /= size;
    rq[3] /= size;
    rq[4] /= size;
    rq[6] /= size;
    rq[7] /= size;

    rq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point2i;

    #[test]
    fn test_is_contour_convex() {
        // Square (convex)
        let square = vec![
            Point2i::new(0, 0),
            Point2i::new(1, 0),
            Point2i::new(1, 1),
            Point2i::new(0, 1),
        ];
        assert!(is_contour_convex(&square));

        // Concave arrow
        let concave = vec![
            Point2i::new(0, 0),
            Point2i::new(2, 0),
            Point2i::new(2, 1),
            Point2i::new(1, 1), // inward dent
            Point2i::new(1, 2),
            Point2i::new(0, 2),
        ];
        assert!(!is_contour_convex(&concave));
    }

    #[test]
    fn test_perimeter_and_min_edge() {
        let square = vec![
            Point2i::new(0, 0),
            Point2i::new(1, 0),
            Point2i::new(1, 1),
            Point2i::new(0, 1),
        ];
        assert!((perimeter(&square) - 4.0).abs() < 1e-9);
        assert!((min_edge_length(&square) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_approx_poly_dp() {
        // Square with extra points
        let contour = vec![
            Point2i::new(0, 0),
            Point2i::new(1, 0),
            Point2i::new(10, 0),
            Point2i::new(10, 1),
            Point2i::new(10, 10),
            Point2i::new(9, 10),
            Point2i::new(0, 10),
            Point2i::new(0, 9),
        ];

        // Using epsilon of 2.0 (so epsilon_sq = 4.0)
        let poly = approx_poly_dp(&contour, 2.0);

        // Should simplify to approximately 4 corners
        assert!(poly.len() <= contour.len());
        assert!(poly.len() >= 3);
        // The algorithm should eliminate the intermediatory edge points (1,0), (10,1), (9,10), (0,9)
        assert_eq!(poly.len(), 4);
    }
}
