// src/core/detector.rs
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_if)]

use crate::core::dictionary::{Dictionary, DictionaryMatch};
use crate::cv::contours::{find_contours, Contour};
use crate::cv::geometry::{approx_poly_dp, is_contour_convex, min_edge_length};
use crate::cv::{ComputerVision, Square};
use crate::{ImageBuffer, Marker, Point2f};

/// Helper for f32 perimeter
fn perimeter_f(poly: &[Point2f]) -> f32 {
    let mut len = 0.0;
    let n = poly.len();
    for i in 0..n {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % n];
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        len += (dx * dx + dy * dy).sqrt();
    }
    len
}

/// ArUco Marker Detector.
pub struct Detector<'a, CV: ComputerVision> {
    pub dictionary: &'a Dictionary,
    pub cv: CV,
    // Configuration thresholds matching ARuco-ts
    pub adaptive_th_size: usize,
    pub adaptive_th_constant: f64,
    pub poly_epsilon: f32,
    pub min_length: f32,      // Minimum relative size to image width (0.01)
    pub min_edge_length: f32, // Abs minimum edge length for candidate (10)
    pub warp_size: usize,     // e.g. 49
}

impl<'a, CV: ComputerVision> Detector<'a, CV> {
    /// Create a new Detector instance initialized with default parameters matching `ARuco-ts`.
    pub fn new(dictionary: &'a Dictionary, cv: CV) -> Self {
        Detector {
            dictionary,
            cv,
            adaptive_th_size: 7, // equivalent to ARuco-ts 2: 2*2 + 3 = 7
            adaptive_th_constant: 7.0,
            poly_epsilon: 0.05,
            min_length: 0.01,
            min_edge_length: 10.0,
            warp_size: 49,
        }
    }

    /// Primary pipeline: Grayscale -> Adaptive Threshold -> Find Contours -> Filter Candidates -> Find Markers
    pub fn detect(&self, image: &ImageBuffer) -> Vec<Marker> {
        let width = image.width;
        let height = image.height;
        let len = (width * height) as usize;

        // 1. Grayscale
        let mut grey = vec![0u8; len];
        CV::grayscale(image, &mut grey);

        // 2. Adaptive Threshold
        let mut thres = vec![0u8; len];
        let grey_buf = ImageBuffer {
            data: &grey,
            width,
            height,
        };
        CV::adaptive_threshold(&grey_buf, &mut thres, 2, 7);

        // 3. Find Contours
        let mut binary = vec![0i32; ((width + 2) * (height + 2)) as usize];
        let thres_buf = ImageBuffer {
            data: &thres,
            width,
            height,
        };
        let contours = find_contours(&thres_buf, &mut binary);

        // 4. Find Candidates
        let mut candidates = self.find_candidates(&contours, width, height);

        // Ensure clockwise
        self.clockwise_corners(&mut candidates);

        // Filter near candidates
        let min_dist = f32::max(30.0, (width as f32) * 0.05);
        let candidates = self.not_too_near(candidates, min_dist);

        // 5. Find Markers
        self.find_markers(&grey_buf, candidates)
    }

    /// Filters raw contours into 4-vertex quadrilateral candidates.
    fn find_candidates(
        &self,
        contours: &[Contour],
        image_width: u32,
        _image_height: u32,
    ) -> Vec<[Point2f; 4]> {
        let min_size = ((image_width as f32) * self.min_length) as usize;
        let mut candidates = Vec::new();

        for contour in contours {
            if contour.points.len() >= min_size {
                let epsilon = (contour.points.len() as f64) * (self.poly_epsilon as f64);
                let poly = approx_poly_dp(&contour.points, epsilon);

                if poly.len() == 4 && is_contour_convex(&poly) {
                    if min_edge_length(&poly) >= (self.min_edge_length as f64) {
                        let corners = [
                            Point2f::new(poly[0].x as f32, poly[0].y as f32),
                            Point2f::new(poly[1].x as f32, poly[1].y as f32),
                            Point2f::new(poly[2].x as f32, poly[2].y as f32),
                            Point2f::new(poly[3].x as f32, poly[3].y as f32),
                        ];
                        candidates.push(corners);
                    }
                }
            }
        }
        candidates
    }

    /// Sorts candidates corners in clockwise order
    fn clockwise_corners(&self, candidates: &mut [[Point2f; 4]]) {
        for candidate in candidates.iter_mut() {
            let dx1 = candidate[1].x - candidate[0].x;
            let dy1 = candidate[1].y - candidate[0].y;
            let dx2 = candidate[2].x - candidate[0].x;
            let dy2 = candidate[2].y - candidate[0].y;

            if (dx1 * dy2 - dy1 * dx2) < 0.0 {
                candidate.swap(1, 3);
            }
        }
    }

    /// Filters candidates that are too near each other (nested or overlapping squares), keeping the larger one.
    fn not_too_near(&self, candidates: Vec<[Point2f; 4]>, min_dist: f32) -> Vec<[Point2f; 4]> {
        let len = candidates.len();
        let mut too_near_flags = vec![false; len];

        for i in 0..len {
            for j in (i + 1)..len {
                let mut dist = 0.0;
                for k in 0..4 {
                    let dx = candidates[i][k].x - candidates[j][k].x;
                    let dy = candidates[i][k].y - candidates[j][k].y;
                    dist += dx * dx + dy * dy;
                }

                if dist / 4.0 < min_dist * min_dist {
                    let perim_i = perimeter_f(&candidates[i]);
                    let perim_j = perimeter_f(&candidates[j]);
                    if perim_i < perim_j {
                        too_near_flags[i] = true;
                    } else {
                        too_near_flags[j] = true;
                    }
                }
            }
        }

        candidates
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !too_near_flags[*i])
            .map(|(_, c)| c)
            .collect()
    }

    /// Warps candidates to square patches and decodes the bits using the Dictionary.
    fn find_markers(&self, image: &ImageBuffer, candidates: Vec<[Point2f; 4]>) -> Vec<Marker> {
        let mut markers = Vec::new();
        let warp_size = self.warp_size;
        let mut warped = vec![0u8; warp_size * warp_size];

        for candidate in candidates {
            // Warp extraction
            CV::warp(image, &mut warped, &candidate, warp_size);

            let warped_buf = ImageBuffer {
                data: &warped,
                width: warp_size as u32,
                height: warp_size as u32,
            };

            // Otsu thresholding
            let otsu_thresh = CV::otsu(warped_buf.data);

            // Re-use warped buffer to store thresholded output (doing it inline here, careful of mutability)
            let mut thresholded = vec![0u8; warp_size * warp_size];
            CV::threshold(warped_buf.data, &mut thresholded, otsu_thresh);

            let thresholded_buf = ImageBuffer {
                data: &thresholded,
                width: warp_size as u32,
                height: warp_size as u32,
            };

            if let Some((match_info, corners)) = self.get_marker(&thresholded_buf, candidate) {
                markers.push(Marker {
                    id: match_info.id as i32,
                    corners,
                    hamming_distance: match_info.distance as i32,
                });
            }
        }
        markers
    }

    /// Inner method matching ARuco-ts getMarker. Samples grid cells to determine matrix bits.
    fn get_marker(
        &self,
        image: &ImageBuffer,
        mut candidate: [Point2f; 4],
    ) -> Option<(DictionaryMatch, [Point2f; 4])> {
        let mark_size = self.dictionary.mark_size;
        let width = (image.width as usize) / mark_size;
        let min_zero = (width * width) / 2;

        // 1. Check outer border (should be black, i.e., 0, meaning non-zero count is small)
        for i in 0..mark_size {
            let inc = if i == 0 || i == mark_size - 1 {
                1
            } else {
                mark_size - 1
            };
            let mut j = 0;
            while j < mark_size {
                let sq = Square {
                    x: (j * width) as u32,
                    y: (i * width) as u32,
                    width: width as u32,
                    height: width as u32,
                };
                if CV::count_non_zero(image, &sq) > min_zero {
                    return None;
                }
                j += inc;
            }
        }

        // 2. Extract inner grid
        let mut bits = vec![0u8; (mark_size - 2) * (mark_size - 2)];
        for i in 0..(mark_size - 2) {
            for j in 0..(mark_size - 2) {
                let sq = Square {
                    x: ((j + 1) * width) as u32,
                    y: ((i + 1) * width) as u32,
                    width: width as u32,
                    height: width as u32,
                };
                let bit = if CV::count_non_zero(image, &sq) > min_zero {
                    1
                } else {
                    0
                };
                // Using bits mapping similar to ARuco-ts where it creates an array of bit arrays
                // We'll just build a flat array representing the boolean grid
                bits[i * (mark_size - 2) + j] = bit;
            }
        }

        // 3. Try to match the bits and its 3 rotations
        let mut best_match: Option<(DictionaryMatch, [Point2f; 4])> = None;
        let mut current_bits = bits;

        for rotation in 0..4 {
            if let Some(m) = self.dictionary.find(&current_bits) {
                if let Some((best_m, _)) = best_match {
                    if m.distance < best_m.distance {
                        // Better match
                        best_match = Some((m, candidate));
                    }
                } else {
                    best_match = Some((m, candidate));
                }

                if m.distance == 0 {
                    // Exact match, break early!
                    break;
                }
            }

            if rotation < 3 {
                // Rotate bits counter-clockwise (to match corner rotation clockwise?)
                // Wait, ARuco-ts rotate method:
                // rotate(src): [len=src.length], dst[i][j] = src[j][len-1-i]
                current_bits = self.rotate_grid(&current_bits, mark_size - 2);

                // Also rotate corners
                candidate = [candidate[1], candidate[2], candidate[3], candidate[0]];
            }
        }

        best_match
    }

    /// Rotates a flat grid representation by 90-degrees (matching ARuco-ts `rotate`)
    /// src[i][j] -> dst[j][dim - 1 - i]
    fn rotate_grid(&self, src: &[u8], dim: usize) -> Vec<u8> {
        let mut dst = vec![0u8; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                dst[j * dim + (dim - 1 - i)] = src[i * dim + j];
            }
        }
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::DICTIONARY_ARUCO;

    // A mock CV implementation for testing the detector logic
    struct MockCV;
    impl ComputerVision for MockCV {
        fn grayscale(_src: &ImageBuffer, _dst: &mut [u8]) {}
        fn threshold(_src: &[u8], _dst: &mut [u8], _threshold: u8) {}
        fn otsu(_src: &[u8]) -> u8 {
            128
        }
        fn adaptive_threshold(
            _src: &ImageBuffer,
            _dst: &mut [u8],
            _kernel_size: usize,
            _threshold: u8,
        ) {
        }
        fn warp(_src: &ImageBuffer, _dst: &mut [u8], _contour: &[Point2f; 4], _warp_size: usize) {}
        fn stack_box_blur(_src: &ImageBuffer, _dst: &mut [u8], _kernel_size: usize) {}
        fn count_non_zero(src: &ImageBuffer, square: &Square) -> usize {
            let mut count = 0;
            for y in square.y..(square.y + square.height) {
                for x in square.x..(square.x + square.width) {
                    if src.data[(y * src.width + x) as usize] != 0 {
                        count += 1;
                    }
                }
            }
            count
        }
        fn gaussian_blur(_src: &ImageBuffer, _dst: &mut [u8], _kernel_size: usize) {}
    }

    #[test]
    fn test_rotate_grid() {
        let dict = Dictionary::new(&DICTIONARY_ARUCO);
        let detector = Detector::new(&dict, MockCV);
        let src = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let rotated = detector.rotate_grid(&src, 3);
        assert_eq!(rotated, vec![7, 4, 1, 8, 5, 2, 9, 6, 3]);
    }

    #[test]
    fn test_clockwise_corners() {
        let dict = Dictionary::new(&DICTIONARY_ARUCO);
        let detector = Detector::new(&dict, MockCV);

        let mut candidates = vec![[
            Point2f::new(0.0, 0.0),
            Point2f::new(0.0, 10.0), // Flipped
            Point2f::new(10.0, 10.0),
            Point2f::new(10.0, 0.0),
        ]];

        detector.clockwise_corners(&mut candidates);

        // Should have swapped index 1 and 3
        assert_eq!(candidates[0][1], Point2f::new(10.0, 0.0));
        assert_eq!(candidates[0][3], Point2f::new(0.0, 10.0));
    }

    #[test]
    fn test_not_too_near() {
        let dict = Dictionary::new(&DICTIONARY_ARUCO);
        let detector = Detector::new(&dict, MockCV);

        let c1 = [
            Point2f::new(0.0, 0.0),
            Point2f::new(10.0, 0.0),
            Point2f::new(10.0, 10.0),
            Point2f::new(0.0, 10.0),
        ];

        let c2 = [
            Point2f::new(1.0, 1.0),
            Point2f::new(9.0, 1.0),
            Point2f::new(9.0, 9.0),
            Point2f::new(1.0, 9.0),
        ];

        let candidates = vec![c1, c2];
        let filtered = detector.not_too_near(candidates, 5.0);

        assert_eq!(filtered.len(), 1);
        // Should keep c1 as it has a larger perimeter
        assert_eq!(filtered[0], c1);
    }

    #[test]
    fn test_get_marker_valid() {
        let dict = Dictionary::new(&DICTIONARY_ARUCO);
        let detector = Detector::new(&dict, MockCV);

        let width = 7;
        let mut data = vec![0u8; width * width];

        // Inner 5x5 bits for ID 0 (0x1084210)
        let bits = [
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        ];

        for i in 0..5 {
            for j in 0..5 {
                data[(i + 1) * width + (j + 1)] = bits[i * 5 + j] * 255;
            }
        }

        let image = ImageBuffer {
            data: &data,
            width: width as u32,
            height: width as u32,
        };

        let candidate = [
            Point2f::new(0.0, 0.0),
            Point2f::new(7.0, 0.0),
            Point2f::new(7.0, 7.0),
            Point2f::new(0.0, 7.0),
        ];

        let match_res = detector.get_marker(&image, candidate);
        assert!(match_res.is_some());
        let (match_info, _corners) = match_res.unwrap();
        assert_eq!(match_info.id, 0);
        assert_eq!(match_info.distance, 0);
    }
}
