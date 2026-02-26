# aruco-rs

A high-performance, cross-platform ArUco marker detection library written in Rust.

## Overview
This project is a native Rust port of [ARuco-ts](https://github.com/kalwalt/ARuco-ts), specifically optimized for **WebAssembly (WASM)** and **Native architectures** using manual **SIMD** kernels. It is designed to be the core tracking engine for the next generation of [webarkit](https://github.com/webarkit) and [AR.js](https://github.com/AR-js-org).

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

## License
MIT OR Apache-2.0