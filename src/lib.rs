// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
use nalgebra::Vector2;

/// 2D Point with floating point precision (f32 for SIMD/WASM compatibility)
pub type Point2f = Vector2<f32>;

/// 2D Point in pixel coordinates
pub type Point2i = Vector2<i32>;

/// The four corners of a detected marker
pub type MarkerCorners = [Point2f; 4];

/// Core structure for a detected ArUco marker.
///
/// # Fields
/// * `id` - The numeric dictionary ID of the identified marker.
/// * `corners` - The 4 corners bounding the marker in 2D image coordinates.
/// * `hamming_distance` - The Hamming distance calculated during error correction.
#[derive(Debug, Clone)]
pub struct Marker {
    pub id: i32,
    pub corners: MarkerCorners,
    pub hamming_distance: i32,
}

/// Zero-copy image buffer for JS/Native interop.
/// Designed to map WASM memory or native video buffers without copying.
///
/// # Fields
/// * `data` - A slice representing a 1D contiguous array of 8-bit pixels.
/// * `width` - The logical width of the frame in pixels.
/// * `height` - The logical height of the frame in pixels.
pub struct ImageBuffer<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
}

/// Possible errors during detection or calibration
#[derive(Debug)]
pub enum ArUcoError {
    InvalidBuffer,
    MarkerNotFound,
    PoseEstimationFailed,
    DictionaryMismatch,
}

pub type Result<T> = std::result::Result<T, ArUcoError>;

pub mod core;
pub mod cv;
pub mod pose;
pub mod simd;

#[cfg(feature = "wasm")]
pub mod wasm_bridge;
