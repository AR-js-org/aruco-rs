// src/cv/mod.rs

use crate::ImageBuffer;

/// Common trait for computer vision operations.
/// This ensures 1:1 functional identity with ARuco-ts while allowing
/// for different implementations (Scalar, SIMD).
pub trait ComputerVision {
    /// Converts an RGBA image buffer to grayscale.
    /// Mirror of `CV.grayscale` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer` containing RGBA pixels.
    /// * `dst` - The destination buffer where grayscale pixels will be written. Must be pre-allocated.
    fn grayscale(src: &ImageBuffer, dst: &mut [u8]);

    /// Applies a simple threshold to a grayscale image.
    /// Mirror of `CV.threshold` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source slice of grayscale pixels.
    /// * `dst` - The destination slice where thresholded binary pixels will be written.
    /// * `threshold` - The cutoff threshold limit (0-255).
    fn threshold(src: &[u8], dst: &mut [u8], threshold: u8);

    /// Computes the Otsu threshold for a grayscale image.
    /// Mirror of `CV.otsu` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source slice of grayscale pixels.
    ///
    /// # Returns
    /// The calculated optimal threshold parameter (0-255).
    fn otsu(src: &[u8]) -> u8;

    /// Computes a fast box blur using a stack algorithm.
    /// Mirror of `CV.stackBoxBlur` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer` containing pixels to blur.
    /// * `dst` - The destination buffer where the blurred image is placed.
    /// * `kernel_size` - Size of the internal blur stack window.
    fn stack_box_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize);

    /// Computes an adaptive threshold using a dynamic box blur.
    /// Mirror of `CV.adaptiveThreshold` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer` used for computations.
    /// * `dst` - The destination buffer array.
    /// * `kernel_size` - Size of the internal stack box blur kernel.
    /// * `threshold` - The threshold subtracted during comparison limits.
    fn adaptive_threshold(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize, threshold: u8);

    /// Extracts a patch using perspective transform and bilinear interpolation.
    /// Mirror of `CV.warp` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer`.
    /// * `dst` - The destination buffer array.
    /// * `contour` - The 4 corners defining the quadrilateral to warp from.
    /// * `warp_size` - The dimensions of the output square.
    fn warp(src: &ImageBuffer, dst: &mut [u8], contour: &[crate::Point2f; 4], warp_size: usize);

    /// Counts non-zero pixels within a specified square area safely.
    /// Mirror of `CV.countNonZero` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer`.
    /// * `square` - The rectangular region to check.
    fn count_non_zero(src: &ImageBuffer, square: &Square) -> usize;

    /// Applies a Gaussian blur to the image.
    /// Mirror of `CV.gaussianBlur` in ARuco-ts.
    ///
    /// # Arguments
    /// * `src` - The source `ImageBuffer`.
    /// * `dst` - The destination buffer array.
    /// * `kernel_size` - Size of the Gaussian kernel.
    fn gaussian_blur(src: &ImageBuffer, dst: &mut [u8], kernel_size: usize);
}

/// Defines a rectangular region of interest
#[derive(Debug, Clone, Copy)]
pub struct Square {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

// Submodules for specific CV algorithms
pub mod contours;
pub mod geometry;
pub mod scalar;
