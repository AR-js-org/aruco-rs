# aruco-rs

A high-performance, cross-platform ArUco marker detection library written in Rust and compiled to WebAssembly.

## What is an ArUco Marker?
An **ArUco marker** is a synthetic square black marker with a wide black border and an inner binary matrix that determines its unique ID. Originally developed as part of OpenCV, ArUco markers are heavily utilized in Augmented Reality (AR), robotics, and camera calibration due to their robustness, high detection speed, and resilience to occlusion and lighting changes.

Their clear planar geometry allows for extremely fast and accurate 3D pose estimation (calculating camera position and rotation relative to the marker) from a single 2D camera feed.

## Overview
This project is a native Rust port of [ARuco-ts](https://github.com/kalwalt/ARuco-ts), strictly maintaining 100% bit-for-bit parity while drastically improving performance. 

It is specifically optimized for **WebAssembly (WASM)** and **Native architectures** using manual **SIMD** kernels. `aruco-rs` is designed to be the foundational tracking engine powering the next-generation Entity Component System (ECS) architecture for [webarkit](https://github.com/webarkit) and [AR.js](https://github.com/AR-js-org). By unifying the CV engine in Rust, both WebXR and Native target platforms benefit from a unified, zero-copy, highly optimized detection pipeline.

## Project Structure
- `src/cv/`: Computer vision primitives (Grayscale, Threshold, Contours).
- `src/core/`: ArUco specific logic (Detector, Dictionary mapping).
- `src/simd/`: Hardware-specific SIMD implementations (WASM, x86_64, ARM).
- `src/pose/`: 3D math and PnP solvers using `nalgebra`.

## Development Principles
- **Logic Parity**: We maintain strict 1:1 functional identity with the original `ARuco-ts` algorithms.
- **Performance**: We utilize zero-copy memory management and explicit SIMD instructions to ensure 60fps tracking on web and mobile.
- **Cleanliness**: The core is `no_std` compatible where possible, with all documentation in English.

## Performance
The underlying algorithmic port (Scalar fallback without explicit hardware SIMD enablement) performs extremely efficiently:
* **~1,300 FPS** tracking pipeline for `320x240` buffers
* **~295 FPS** tracking pipeline for `640x480` standard webcam buffers
* **~168 FPS** downscaled tracking pipeline interpolating from a raw `2048x2080 (4.2MP)` real-world image. 

These speeds reflect the entire `Detector` end-to-end execution, spanning raw RGBA grayscale mapping, adaptive thresholding, polygon extraction, marker filtering, homography interpolation, and Dictionary ID hashing.

## Usage Examples

### WebAssembly (JavaScript/TypeScript)
`aruco-rs` uses `wasm-bindgen` and `serde` to safely serialize deeply technical math objects into flat, memory-safe Javascript arrays, entirely avoiding manual WASM pointer GC management.
```js
import { ARucoDetector } from '@ar-js-org/aruco-rs'; // Assuming generated pkg/ folder is exported

// 1. Initialize detector predicting standard ARUCO dictionary
const detector = new ARucoDetector("ARUCO", undefined);

// 2. Obtain raw pixel buffer (e.g. from a <canvas> getImageData or WebGL buffer)
const width = 640;
const height = 480;
const imageData = new Uint8Array(width * height * 4); 

// 3. Detect and receive serialized JSON objects containing ID, Distance, and Corners
const markers = detector.detect_image(width, height, imageData);

markers.forEach(marker => {
    console.log(`Detected Marker ID: ${marker.id} with ${marker.distance} bit errors.`);
    console.log(`Top-Left Corner: ${marker.corners[0].x}, ${marker.corners[0].y}`);
});
```

### Native Rust
```rust
use aruco_rs::core::detector::{Detector, DetectorOptions};
use aruco_rs::core::dictionary::{Dictionary, PredefinedDictionary};
use aruco_rs::cv::scalar::ScalarCV;
use aruco_rs::ImageBuffer;

// 1. Setup the Detector Pipeline
let dict = Dictionary::from_predefined(PredefinedDictionary::Aruco);
let mut detector = Detector::new(dict, ScalarCV, DetectorOptions::default());

// 2. Prepare the ImageBuffer (wrapping an image decode)
let raw_pixels: Vec<u8> = vec![0; 640 * 480 * 4];
let buffer = ImageBuffer { data: &raw_pixels, width: 640, height: 480 };

// 3. Execute Detection
let markers = detector.detect(&buffer).unwrap();

for marker in markers {
    println!("Found ArUco ID {}, Corners: {:?}", marker.id, marker.corners);
}
```

## Contributing
When creating new source files (`.rs`, `.ts`, `.js`), please ensure they include the standard MIT license header at the very top:
```rust
// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
```
You can use the provided `add_license.ps1` script to automatically apply this header to missing files.

## License
MIT