// src/simd/native.rs

use crate::cv::ComputerVision;
use crate::ImageBuffer;

#[cfg(target_arch = "x86_64")]
use safe_arch::*;

/// Native SIMD implementation of Computer Vision operations.
pub struct NativeCV;

impl ComputerVision for NativeCV {
    /// Converts an RGBA image buffer to grayscale using SSE/AVX where available.
    /// Mirror of `CV.grayscale` in ARuco-ts.
    fn grayscale(src: &ImageBuffer, dst: &mut [u8]) {
        let src_data = src.data;
        let len = src_data.len();

        #[cfg(target_arch = "x86_64")]
        {
            // For x86_64, use `safe_arch` to process pixels in bulk.
            // A typical 128-bit SSE register can hold 16 bytes (4 RGBA pixels).
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
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::cv::scalar::ScalarCV::grayscale(src, dst);
        }
    }

    /// Applies a simple threshold to a grayscale image.
    fn threshold(src: &[u8], dst: &mut [u8], threshold: u8) {
        let len = src.len();
        let mut i = 0;

        #[cfg(target_arch = "x86_64")]
        {
            // Create a vector with the threshold value broadcasted to 16 bytes
            let thresh_vec = set_splat_i8_m128i(threshold as i8); // Careful with signed/unsigned comparison

            // This is complex because SSE mainly does signed 8-bit comparison (PCMPGTB).
            // For an unsigned threshold, we'd need to subtract 128 from both input and threshold.
            let offset = set_splat_i8_m128i(-128i8);
            let adjusted_thresh = add_i8_m128i(thresh_vec, offset);

            while i + 15 < len {
                let src_chunk = unsafe { &*(src[i..].as_ptr() as *const [u8; 16]) };
                let data = load_unaligned_m128i(src_chunk);
                let adjusted_data = add_i8_m128i(data, offset);
                // Compare: data > threshold ? 0xFF : 0x00
                // We want: data <= threshold ? 0 : 255.
                // So the exact opposite of GT.
                let gt = cmp_gt_mask_i8_m128i(adjusted_data, adjusted_thresh);

                // If it's GT, we want 255. The mask itself is 0xFF for true, 0x00 for false.
                let dst_chunk = unsafe { &mut *(dst[i..].as_mut_ptr() as *mut [u8; 16]) };
                store_unaligned_m128i(dst_chunk, gt);
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

        #[cfg(target_arch = "x86_64")]
        {
            let src_data = src.data;
            let len = src_data.len();
            let mut i = 0;

            // diff <= -threshold => diff < -threshold + 1 => (-threshold + 1) > diff
            let thresh_plus_one = set_splat_i16_m128i(-(threshold as i16) + 1);
            let zero = set_splat_i8_m128i(0);

            while i + 15 < len {
                let src_chunk = unsafe { &*(src_data[i..].as_ptr() as *const [u8; 16]) };
                let data_src = load_unaligned_m128i(src_chunk);

                let dst_chunk = unsafe { &*(dst[i..].as_ptr() as *const [u8; 16]) };
                let data_dst = load_unaligned_m128i(dst_chunk);

                // Zero-extend u8 -> u16
                let src_low = unpack_low_i8_m128i(data_src, zero);
                let src_high = unpack_high_i8_m128i(data_src, zero);

                let dst_low = unpack_low_i8_m128i(data_dst, zero);
                let dst_high = unpack_high_i8_m128i(data_dst, zero);

                // diff = src - dst (16-bit signed arithmetic)
                let diff_low = sub_i16_m128i(src_low, dst_low);
                let diff_high = sub_i16_m128i(src_high, dst_high);

                // (-threshold + 1) > diff
                let mask_low = cmp_gt_mask_i16_m128i(thresh_plus_one, diff_low);
                let mask_high = cmp_gt_mask_i16_m128i(thresh_plus_one, diff_high);

                // Pack i16 to i8 using saturated signed math (0xFFFF clamped to 0xFF, limits mask properly)
                let packed = pack_i16_to_i8_m128i(mask_low, mask_high);

                let dst_out = unsafe { &mut *(dst[i..].as_mut_ptr() as *mut [u8; 16]) };
                store_unaligned_m128i(dst_out, packed);
                i += 16;
            }

            // Scalar fallback for remaining pixels
            while i < len {
                let val = src_data[i] as i32 - dst[i] as i32;
                dst[i] = if val <= -(threshold as i32) { 255 } else { 0 };
                i += 1;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::cv::scalar::ScalarCV::adaptive_threshold(src, dst, kernel_size, threshold);
        }
    }

    fn warp(src: &ImageBuffer, dst: &mut [u8], contour: &[crate::Point2f; 4], warp_size: usize) {
        crate::cv::scalar::ScalarCV::warp(src, dst, contour, warp_size)
    }

    fn count_non_zero(src: &ImageBuffer, square: &crate::cv::Square) -> usize {
        crate::cv::scalar::ScalarCV::count_non_zero(src, square)
    }

    fn gaussian_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize) {
        crate::cv::scalar::ScalarCV::gaussian_blur(src, dst, kernel_size)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use crate::ImageBuffer;

    include!("../../tests/data/sample_8x8.rs");

    #[test]
    fn test_native_simd_grayscale_logic() {
        let src = ImageBuffer {
            data: &SAMPLE_8X8_RGBA,
            width: 8,
            height: 8,
        };
        let mut dst = [0u8; 64];

        NativeCV::grayscale(&src, &mut dst);

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
    fn test_native_simd_threshold_bounds() {
        let mut gray = [0u8; 64];
        let mut dst = [0u8; 64];
        for i in 0..64 {
            gray[i] = if i % 2 == 0 { 141 } else { 50 };
        }

        NativeCV::threshold(&gray, &mut dst, 50);

        for i in 0..64 {
            if i % 2 == 0 {
                assert_eq!(dst[i], 255);
            } else {
                assert_eq!(dst[i], 0);
            }
        }
    }

    #[test]
    fn test_native_simd_adaptive_threshold() {
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

        NativeCV::adaptive_threshold(&src, &mut dst, 2, 10);

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
    fn test_native_simd_stack_box_blur_parity() {
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

        NativeCV::stack_box_blur(&src, &mut dst_simd, 2);
        crate::cv::scalar::ScalarCV::stack_box_blur(&src, &mut dst_scalar, 2);

        assert_eq!(dst_simd, dst_scalar);
    }
}
