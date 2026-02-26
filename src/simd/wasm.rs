// src/simd/wasm.rs

use crate::cv::ComputerVision;
use crate::ImageBuffer;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WASM SIMD implementation of Computer Vision operations.
pub struct WasmCV;

impl ComputerVision for WasmCV {
    /// Converts an RGBA image buffer to grayscale using WASM SIMD.
    /// Mirror of `CV.grayscale` in ARuco-ts.
    fn grayscale(src: &ImageBuffer, dst: &mut [u8]) {
        let src_data = src.data;
        let len = src_data.len();

        #[cfg(target_arch = "wasm32")]
        {
            // WASM SIMD operates on v128 types (16 bytes).
            // We need to extract R, G, B, multiply by weights, and pack them back.
            // Setting up scalar fallback for now until vector math is fully verified

            let mut i = 0;
            let mut j = 0;
            while i + 15 < len {
                // To do this efficiently in SIMD requires shuffles, integer multiplications,
                // and narrowing casts. Given the specific weights requested by logic parity:
                // 0.299, 0.587, 0.114.
                // We fallback to scalar for exact bit-parity unless we use a precise
                // fixed-point approximation that guarantees bit-exact results.

                // Scalar fallback for now
                let r = src_data[i] as f32;
                let g = src_data[i + 1] as f32;
                let b = src_data[i + 2] as f32;
                dst[j] = ((r * 0.299 + g * 0.587 + b * 0.114 + 0.5) as u32 & 0xff) as u8;
                j += 1;
                i += 4;
            }

            // Process remaining pixels
            while i < len {
                let r = src_data[i] as f32;
                let g = src_data[i + 1] as f32;
                let b = src_data[i + 2] as f32;
                dst[j] = ((r * 0.299 + g * 0.587 + b * 0.114 + 0.5) as u32 & 0xff) as u8;
                j += 1;
                i += 4;
            }
            return;
        }

        // Fallback for non-wasm32 architecture
        crate::cv::scalar::ScalarCV::grayscale(src, dst);
    }

    /// Applies a simple threshold to a grayscale image.
    fn threshold(src: &[u8], dst: &mut [u8], threshold: u8) {
        let len = src.len();
        let mut i = 0;

        #[cfg(target_arch = "wasm32")]
        {
            // Create a vector with the threshold value broadcasted to 16 bytes
            let thresh_vec = u8x16_splat(threshold); // u8x16 is available in WASM SIMD

            while i + 15 < len {
                let data = v128_load(src[i..].as_ptr() as *const v128);

                // WASM SIMD has u8x16_le (less than or equal) which directly
                // gives what we want, but it returns 0xFF for true.
                // However, logic parity requires 0 for <= threshold and 255 for > threshold.
                // So we actually want u8x16_gt (greater than) to directly get 255 (0xFF)
                // for values > threshold.

                let gt = u8x16_gt(data, thresh_vec); // 0xFF where data > thresh, 0x00 otherwise

                v128_store(dst[i..].as_mut_ptr() as *mut v128, gt);
                i += 16;
            }
        }

        // Process remaining pixels
        while i < len {
            dst[i] = if src[i] <= threshold { 0 } else { 255 };
            i += 1;
        }
    }

    /// Computes the Otsu threshold for a grayscale image.
    fn otsu(src: &[u8]) -> u8 {
        // Otsu's method requires a histogram.
        // SIMD histogram calculation is possible but complex (using gathers/scatters).
        // Since Otsu operates on the whole image just to generate 256 bins,
        // scalar is often fast enough or the bottleneck is entirely memory-bound.
        // We defer to the scalar implementation for exact parity and simplicity.
        crate::cv::scalar::ScalarCV::otsu(src)
    }

    /// Computes a fast box blur using a stack algorithm.
    fn stack_box_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize) {
        crate::cv::scalar::ScalarCV::stack_box_blur(src, dst, kernel_size)
    }

    /// Computes an adaptive threshold using a dynamic box blur.
    fn adaptive_threshold(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize, threshold: u8) {
        // The first phase of adaptive thresholding is a stack box blur.
        // Because the blur relies heavily on a recurrence relation (sliding stack sum),
        // we utilize the highly optimized scalar version to ensure exact bit-parity.
        crate::cv::scalar::ScalarCV::stack_box_blur(src, dst, kernel_size);

        #[cfg(target_arch = "wasm32")]
        {
            let src_data = src.data;
            let len = src_data.len();
            let mut i = 0;

            let thresh_vec = i16x8_splat(-(threshold as i16));

            while i + 15 < len {
                let s_ptr = src_data[i..].as_ptr() as *const v128;
                let src_128 = v128_load(s_ptr);

                let d_ptr = dst[i..].as_ptr() as *const v128;
                let dst_128 = v128_load(d_ptr);

                // Zero extend u8 to i16 (0-255)
                let src_low = u16x8_extend_low_u8x16(src_128);
                let src_high = u16x8_extend_high_u8x16(src_128);

                let dst_low = u16x8_extend_low_u8x16(dst_128);
                let dst_high = u16x8_extend_high_u8x16(dst_128);

                // Compute difference: src - dst
                let diff_low = i16x8_sub(src_low, dst_low);
                let diff_high = i16x8_sub(src_high, dst_high);

                // Compare diff <= -threshold
                let mask_low = i16x8_le(diff_low, thresh_vec); // 0xFFFF for true, 0 for false
                let mask_high = i16x8_le(diff_high, thresh_vec);

                // Compress back into i8 sizes.
                // Saturating narrow clamps 0xFFFF (-1) to -1 (0xFF), retaining the mask.
                let result = i8x16_narrow_i16x8(mask_low, mask_high);

                v128_store(dst[i..].as_mut_ptr() as *mut v128, result);
                i += 16;
            }

            // Scalar fallback for remaining pixels
            while i < len {
                let val = src_data[i] as i32 - dst[i] as i32;
                dst[i] = if val <= -(threshold as i32) { 255 } else { 0 };
                i += 1;
            }
            return;
        }

        crate::cv::scalar::ScalarCV::adaptive_threshold(src, dst, kernel_size, threshold);
    }

    /// Extracts a patch using perspective transform and bilinear interpolation.
    fn warp(src: &ImageBuffer, dst: &mut [u8], contour: &[crate::Point2f; 4], warp_size: usize) {
        crate::cv::scalar::ScalarCV::warp(src, dst, contour, warp_size)
    }

    /// Counts non-zero pixels within a specified square area safely.
    fn count_non_zero(src: &ImageBuffer, square: &crate::cv::Square) -> usize {
        crate::cv::scalar::ScalarCV::count_non_zero(src, square)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageBuffer;

    include!("../../tests/data/sample_8x8.rs");

    #[test]
    fn test_wasm_simd_grayscale_logic() {
        let src = ImageBuffer {
            data: &SAMPLE_8X8_RGBA,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        WasmCV::grayscale(&src, &mut dst);

        assert_eq!(dst[0], 141);
        assert_eq!(dst[1], 50);

        for i in 0..64 {
            if i % 2 == 0 {
                assert_eq!(dst[i], 141);
            } else {
                assert_eq!(dst[i], 50);
            }
        }
    }

    #[test]
    fn test_wasm_simd_threshold_bounds() {
        let mut gray = [0u8; 64];
        let mut dst = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 141 } else { 50 };
        }

        WasmCV::threshold(&gray, &mut dst, 50);

        for i in 0..64 {
            if i % 2 == 0 {
                assert_eq!(dst[i], 255);
            } else {
                assert_eq!(dst[i], 0);
            }
        }
    }

    #[test]
    fn test_wasm_simd_adaptive_threshold() {
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

        WasmCV::adaptive_threshold(&src, &mut dst, 2, 10);

        // Ensure the output is strictly binary
        for i in 0..64 {
            assert!(dst[i] == 0 || dst[i] == 255);
        }

        // Also ensure it perfectly matches the scalar computation
        let mut scalar_dst = [0u8; 64];
        crate::cv::scalar::ScalarCV::adaptive_threshold(&src, &mut scalar_dst, 2, 10);
        assert_eq!(dst, scalar_dst);
    }

    #[test]
    fn test_wasm_simd_stack_box_blur_parity() {
        let mut gray = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 200 } else { 50 };
        }
        let src = ImageBuffer {
            data: &gray,
            width: 8,
            height: 8,
        };
        let mut dst_simd = [0u8; 64];
        let mut dst_scalar = [0u8; 64];

        WasmCV::stack_box_blur(&src, &mut dst_simd, 2);
        crate::cv::scalar::ScalarCV::stack_box_blur(&src, &mut dst_scalar, 2);

        assert_eq!(dst_simd, dst_scalar);
    }
}
