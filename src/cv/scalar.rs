// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
// src/cv/scalar.rs
#![allow(clippy::needless_range_loop)]

use crate::cv::ComputerVision;
use crate::ImageBuffer;

/// Scalar (non-SIMD) implementation of Computer Vision operations.
pub struct ScalarCV;

impl ComputerVision for ScalarCV {
    /// Converts an RGBA image buffer to grayscale.
    /// Mirror of `CV.grayscale` in ARuco-ts.
    fn grayscale(src: &ImageBuffer, dst: &mut [u8]) {
        let src_data = src.data;
        let mut i = 0;
        let mut j = 0;
        let len = src_data.len();

        // Process RGBA to grayscale
        while i < len {
            let r = src_data[i] as f32;
            let g = src_data[i + 1] as f32;
            let b = src_data[i + 2] as f32;

            // Weighted average: 0.299R + 0.587G + 0.114B
            dst[j] = ((r * 0.299 + g * 0.587 + b * 0.114 + 0.5) as u32 & 0xff) as u8;
            j += 1;
            i += 4;
        }
    }

    /// Applies a simple threshold to a grayscale image.
    /// Mirror of `CV.threshold` in ARuco-ts.
    fn threshold(src: &[u8], dst: &mut [u8], threshold: u8) {
        let len = src.len();
        let mut tab = [0u8; 256];

        // Build lookup table
        for i in 0..256 {
            tab[i] = if (i as u8) <= threshold { 0 } else { 255 };
        }

        // Apply threshold using lookup
        for i in 0..len {
            dst[i] = tab[src[i] as usize];
        }
    }

    /// Computes the Otsu threshold for a grayscale image.
    /// Mirror of `CV.otsu` in ARuco-ts.
    fn otsu(src: &[u8]) -> u8 {
        let len = src.len();
        let mut hist = [0u32; 256];
        let mut threshold = 0;
        let mut sum = 0.0;
        let mut sum_b = 0.0;
        let mut w_b = 0.0;
        let mut max = 0.0;

        for &pixel in src.iter() {
            hist[pixel as usize] += 1;
        }

        for i in 0..256 {
            sum += (hist[i] as f64) * (i as f64);
        }

        for i in 0..256 {
            w_b += hist[i] as f64;
            if w_b != 0.0 {
                let w_f = (len as f64) - w_b;
                if w_f == 0.0 {
                    break;
                }

                sum_b += (hist[i] as f64) * (i as f64);

                let mu = sum_b / w_b - (sum - sum_b) / w_f;
                let between = w_b * w_f * mu * mu;

                if between > max {
                    max = between;
                    threshold = i as u8;
                }
            }
        }

        threshold
    }

    /// Computes a fast box blur using a stack algorithm.
    /// Mirror of `CV.stackBoxBlur` in ARuco-ts.
    fn stack_box_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize) {
        assert!(kernel_size < 16, "kernel_size must be < 16");

        const STACK_BOX_BLUR_MULT: [u32; 16] = [
            1, 171, 205, 293, 57, 373, 79, 137, 241, 27, 391, 357, 41, 19, 283, 265,
        ];
        const STACK_BOX_BLUR_SHIFT: [u32; 16] =
            [0, 9, 10, 11, 9, 12, 10, 11, 12, 9, 13, 13, 10, 9, 13, 13];

        let src_data = src.data;
        let height = src.height as usize;
        let width = src.width as usize;
        let height_minus_1 = height.saturating_sub(1);
        let width_minus_1 = width.saturating_sub(1);
        let size = kernel_size * 2 + 1;
        let radius = kernel_size + 1;
        let mult = STACK_BOX_BLUR_MULT[kernel_size];
        let shift = STACK_BOX_BLUR_SHIFT[kernel_size];

        let mut stack = [0u8; 31]; // Max size is 15 * 2 + 1 = 31

        // Horizontal pass
        let mut pos = 0;
        for _y in 0..height {
            let start = pos;

            let color = src_data[pos] as u32;
            let mut sum = (radius as u32) * color;

            let mut sp = 0;
            for _ in 0..radius {
                stack[sp] = color as u8;
                sp = (sp + 1) % size;
            }
            for i in 1..radius {
                let c = src_data[pos + i];
                stack[sp] = c;
                sum += c as u32;
                sp = (sp + 1) % size;
            }

            let mut stack_start = 0;
            for x in 0..width {
                dst[pos] = ((sum * mult) >> shift) as u8;
                pos += 1;

                let mut p = x + radius;
                p = start + if p < width_minus_1 { p } else { width_minus_1 };

                sum -= stack[stack_start] as u32;
                let c = src_data[p];
                sum += c as u32;

                stack[stack_start] = c;
                stack_start = (stack_start + 1) % size;
            }
        }

        // Vertical pass
        for x in 0..width {
            let mut pos = x;
            let mut start = pos + width;

            let color = dst[pos] as u32;
            let mut sum = (radius as u32) * color;

            let mut sp = 0;
            for _ in 0..radius {
                stack[sp] = color as u8;
                sp = (sp + 1) % size;
            }
            for _ in 1..radius {
                let c = dst[start];
                stack[sp] = c;
                sum += c as u32;
                sp = (sp + 1) % size;
                start += width;
            }

            let mut stack_start = 0;
            for y in 0..height {
                dst[pos] = ((sum * mult) >> shift) as u8;

                let mut p = y + radius;
                p = x
                    + (if p < height_minus_1 {
                        p
                    } else {
                        height_minus_1
                    }) * width;

                sum -= stack[stack_start] as u32;
                let c = dst[p];
                sum += c as u32;

                stack[stack_start] = c;
                stack_start = (stack_start + 1) % size;

                pos += width;
            }
        }
    }

    /// Computes an adaptive threshold using a dynamic box blur.
    /// Mirror of `CV.adaptiveThreshold` in ARuco-ts.
    fn adaptive_threshold(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize, threshold: u8) {
        let src_data = src.data;
        let len = src_data.len();
        let mut tab = [0u8; 768];

        // First: blur the image
        Self::stack_box_blur(src, dst, kernel_size);

        // Build lookup table (768 = 3 * 256 per gestire range -255 to 510)
        for i in 0..768 {
            let val = i as i32 - 255;
            tab[i] = if val <= -(threshold as i32) { 255 } else { 0 };
        }

        // Apply adaptive threshold WITH CRITICAL +255 OFFSET
        for i in 0..len {
            // CRITICAL: +255 offset to handle negative values
            let idx = (src_data[i] as i32) - (dst[i] as i32) + 255;
            dst[i] = tab[idx as usize];
        }
    }

    /// Extracts a patch using perspective transform and bilinear interpolation.
    /// Mirror of `CV.warp` in ARuco-ts.
    fn warp(src: &ImageBuffer, dst: &mut [u8], contour: &[crate::Point2f; 4], warp_size: usize) {
        let src_data = src.data;
        let width = src.width as usize;
        let height = src.height as usize;
        let mut pos = 0;

        let m = crate::cv::geometry::get_perspective_transform(contour, (warp_size - 1) as f64);

        let mut r = m[8];
        let mut s = m[2];
        let mut t = m[5];

        for _i in 0..warp_size {
            r += m[7];
            s += m[1];
            t += m[4];

            let mut u = r;
            let mut v = s;
            let mut w = t;

            for _j in 0..warp_size {
                u += m[6];
                v += m[0];
                w += m[3];

                let x = v / u;
                let y = w / u;

                let sx1 = x.max(0.0).min((width - 1) as f64) as usize;
                let sx2 = if sx1 == width - 1 { sx1 } else { sx1 + 1 };
                let dx1 = x - (sx1 as f64);
                let dx2 = 1.0 - dx1;

                let sy1 = y.max(0.0).min((height - 1) as f64) as usize;
                let sy2 = if sy1 == height - 1 { sy1 } else { sy1 + 1 };
                let dy1 = y - (sy1 as f64);
                let dy2 = 1.0 - dy1;

                let p1 = sy1 * width;
                let p2 = p1;
                let p3 = sy2 * width;
                let p4 = p3;

                let val = dy2
                    * (dx2 * (src_data[p1 + sx1] as f64) + dx1 * (src_data[p2 + sx2] as f64))
                    + dy1 * (dx2 * (src_data[p3 + sx1] as f64) + dx1 * (src_data[p4 + sx2] as f64));

                dst[pos] = (val as i64 & 0xff) as u8;
                pos += 1;
            }
        }
    }

    /// Counts non-zero pixels within a specified square area safely.
    /// Mirror of `CV.countNonZero` in ARuco-ts.
    fn count_non_zero(src: &ImageBuffer, square: &crate::cv::Square) -> usize {
        let src_data = src.data;
        let sheight = square.height as usize;
        let swidth = square.width as usize;
        let mut pos = (square.x + square.y * src.width) as usize;
        let span = (src.width - square.width) as usize;
        let mut nz = 0;

        for _ in 0..sheight {
            for _ in 0..swidth {
                if src_data[pos] != 0 {
                    nz += 1;
                }
                pos += 1;
            }
            pos += span;
        }

        nz
    }

    /// Applies a Gaussian blur to the image.
    /// Mirror of `CV.gaussianBlur` in ARuco-ts.
    fn gaussian_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize) {
        let kernel = Self::gaussian_kernel(kernel_size);
        let mut temp = vec![0u8; dst.len()];
        let width = src.width as usize;
        let height = src.height as usize;

        // In JS: CV.gaussianBlurFilter(imageSrc, imageMean, kernel, true);
        Self::gaussian_blur_filter(src.data, &mut temp, width, height, &kernel, true);
        // In JS: CV.gaussianBlurFilter(imageMean, imageDst, kernel, false);
        Self::gaussian_blur_filter(&temp, dst, width, height, &kernel, false);
    }
}

// Private helper methods for Gaussian Blur (mirroring static CV methods)
impl ScalarCV {
    /// Generates a 1D Gaussian kernel array.
    /// Mirror of `CV.gaussianKernel` in ARuco-ts.
    fn gaussian_kernel(kernel_size: usize) -> Vec<f64> {
        let tab = [
            vec![1.0],
            vec![0.25, 0.5, 0.25],
            vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
            vec![
                0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
            ],
        ];

        if kernel_size <= 7 && kernel_size % 2 == 1 {
            return tab[kernel_size >> 1].clone();
        }

        let mut kernel = vec![0.0; kernel_size];
        let center = (kernel_size as f64 - 1.0) * 0.5;
        let sigma = 0.8 + 0.3 * (center - 1.0);
        let scale2x = -0.5 / (sigma * sigma);
        let mut sum = 0.0;

        for (i, val) in kernel.iter_mut().enumerate() {
            let x = i as f64 - center;
            let v = (scale2x * x * x).exp();
            *val = v;
            sum += v;
        }

        let inv_sum = 1.0 / sum;
        for val in kernel.iter_mut() {
            *val *= inv_sum;
        }

        kernel
    }

    /// Applies a 1D Gaussian kernel over the horizontal or vertical axis.
    /// Mirror of `CV.gaussianBlurFilter` in ARuco-ts.
    fn gaussian_blur_filter(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        kernel: &[f64],
        horizontal: bool,
    ) {
        let mut pos = 0;
        let limit = (kernel.len() >> 1) as i32;
        let w = width as i32;
        let h = height as i32;

        for i in 0..h {
            for j in 0..w {
                let mut value = 0.0;

                for k in -limit..=limit {
                    let mut cur;
                    if horizontal {
                        cur = pos as i32 + k;
                        if j + k < 0 || j + k >= w {
                            cur = pos as i32;
                        }
                    } else {
                        cur = pos as i32 + k * w;
                        if i + k < 0 || i + k >= h {
                            cur = pos as i32;
                        }
                    }

                    value += kernel[(limit + k) as usize] * (src[cur as usize] as f64);
                }

                if horizontal {
                    // TS: writes to Uint8ClampedArray directly (rounds to even integer)
                    dst[pos] = value.round_ties_even().clamp(0.0, 255.0) as u8;
                } else {
                    // TS: `(value + 0.5) | 0` directly on primitive Number
                    dst[pos] = (value + 0.5).floor().clamp(0.0, 255.0) as u8;
                }
                pos += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageBuffer;

    include!("../../tests/data/sample_8x8.rs");

    #[test]
    fn test_grayscale_logic() {
        // Create an ImageBuffer from the sample RGBA array
        let src = ImageBuffer {
            data: &SAMPLE_8X8_RGBA,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        ScalarCV::grayscale(&src, &mut dst);

        // Verify that the grayscale conversion matches the weighted sum formula
        // (R * 0.299 + G * 0.587 + B * 0.114)
        // Pixel 0: 100 * 0.299 + 150 * 0.587 + 200 * 0.114 + 0.5 = 141.25 -> 141
        assert_eq!(dst[0], 141);

        // Pixel 1: 50 * 0.299 + 50 * 0.587 + 50 * 0.114 + 0.5 = 50.5 -> 50
        assert_eq!(dst[1], 50);

        // Check the entire array since it alternates
        for i in 0..64 {
            if i % 2 == 0 {
                assert_eq!(dst[i], 141);
            } else {
                assert_eq!(dst[i], 50);
            }
        }
    }

    #[test]
    fn test_otsu_threshold() {
        // We use the same grayscale output array
        let mut gray = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 141 } else { 50 };
        }

        // Use a known histogram to verify that the otsu function
        // returns the correct threshold value as per cv.ts logic.
        // With exactly half 50s and half 141s, Otsu should pick the lower bound
        // where between class variance is maximized.
        let otsu_thresh = ScalarCV::otsu(&gray);
        assert_eq!(otsu_thresh, 50);
    }

    #[test]
    fn test_threshold_bounds() {
        let mut gray = [0u8; 64];
        let mut dst = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 141 } else { 50 };
        }

        // Apply threshold at 50
        // Ensure that pixel values exactly at the threshold are handled consistently
        // with the TypeScript implementation (<= threshold -> 0).
        ScalarCV::threshold(&gray, &mut dst, 50);

        for i in 0..64 {
            if i % 2 == 0 {
                // 141 > 50 -> 255
                assert_eq!(dst[i], 255);
            } else {
                // 50 <= 50 -> 0
                assert_eq!(dst[i], 0);
            }
        }
    }

    #[test]
    fn test_stack_box_blur() {
        let mut gray = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 200 } else { 50 };
        }
        let src = ImageBuffer {
            data: &gray,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        // Kernel size 1
        ScalarCV::stack_box_blur(&src, &mut dst, 1);

        // Ensure blurring happens
        assert_ne!(dst, gray);
        // Ensure no out-of-bounds panics
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut gray = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 4 == 0 { 200 } else { 50 };
        }
        let src = ImageBuffer {
            data: &gray,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        // Kernel size 2, threshold 10
        ScalarCV::adaptive_threshold(&src, &mut dst, 2, 10);

        // Ensure the output is strictly binary (0 or 255)
        for item in &dst {
            assert!(*item == 0 || *item == 255);
        }
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = ScalarCV::gaussian_kernel(3);
        assert_eq!(kernel.len(), 3);
        assert!((kernel[0] - 0.25).abs() < 1e-6);
        assert!((kernel[1] - 0.50).abs() < 1e-6);
        assert!((kernel[2] - 0.25).abs() < 1e-6);

        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_blur() {
        let mut gray = [0u8; 64];
        for (i, val) in gray.iter_mut().enumerate() {
            *val = if i % 4 == 0 { 200 } else { 50 };
        }
        let src = ImageBuffer {
            data: &gray,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        ScalarCV::gaussian_blur(&src, &mut dst, 3);

        let sum: u32 = dst.iter().map(|&x| x as u32).sum();
        assert!(sum > 0 && sum != (gray.iter().map(|&x| x as u32).sum()));
    }
}
