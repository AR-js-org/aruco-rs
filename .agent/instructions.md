# Technical Instructions for aruco-rs

## 1. Project Overview
- **Objective**: Develop a high-performance ArUco marker detector library in Rust [cite: 2025-10-22].
- **Target**: Cross-platform support for WebAssembly (WASM) and Native architectures (x86_64/ARM) [cite: 2026-02-25].
- **Maintainer**: kalwalt (GitHub: kalwalt), creator of webarkit and maintainer of AR-js-org [cite: 2026-02-02].

## 2. Architecture & Modules
- **Modular Dispatch**: Use a directory-based dispatch in `src/simd/mod.rs` to select the implementation (WASM SIMD, Native, or Scalar) at compile-time via `#[cfg]` [cite: 2026-02-25].
- **Crate Structure**:
    - `src/cv/`: Core computer vision algorithms.
    - `src/core/`: ArUco detector and dictionary logic.
    - `src/pose/`: 3D math and PnP solvers using `nalgebra`.
- **Portability**: Keep the core logic `no_std` compatible where possible to ensure it runs in any environment.

## 3. Performance & Memory
- **SIMD**: Explicitly implement SIMD-optimized methods for WASM and native CPU architectures [cite: 2026-02-25].
- **Zero-Copy**: Utilize slices (`&[u8]`) to map WASM linear memory or native video buffers without unnecessary allocations.
- **Buffer Management**: Reuse intermediate buffers in the detection loop to minimize heap pressure.

## 4. Coding Standards
- **Language**: All code comments, documentation, and strings must be in **English** [cite: 2026-02-02].
- **Safety**: Every `unsafe` block used for SIMD intrinsics must include a `// SAFETY:` comment justifying the operation.
- **Traceability**: Reference specific ARuco-ts functions in comments when porting logic.
- **Tests**: Implement comprehensive unit tests for every function. For CV routines, include Parity Tests that compare the output of SIMD implementations against the Scalar baseline to ensure bit-perfect consistency.
- **Benchmarks**: Provide benchmarks for performance-critical functions using Criterion. Benchmarks should specifically measure the speedup of SIMD kernels over Scalar versions across different input sizes [cite: 2026-02-25].
- **Documentation**: Every public-facing function, struct, and trait must have Doc-comments (///) explaining its purpose, parameters, and return values, following standard Rust documentation practices.
