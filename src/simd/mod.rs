// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
// src/simd/mod.rs

/*
 * This module provides the SIMD Dispatcher for computer vision operations.
 * It selects the implementation (WASM SIMD, Native, or Scalar) at compile-time
 * based on the target architecture and the "simd" feature flag.
 */

// Placeholder traits for CV operations that will have SIMD versions.
// These will be implemented in the respective submodules.

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
pub mod wasm;

#[cfg(all(not(target_arch = "wasm32"), feature = "simd"))]
pub mod native;

// Re-export the appropriate implementation as `dispatch`
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
pub use wasm as dispatch;

#[cfg(all(not(target_arch = "wasm32"), feature = "simd"))]
pub use native as dispatch;

#[cfg(not(feature = "simd"))]
pub use crate::cv::scalar as dispatch;
