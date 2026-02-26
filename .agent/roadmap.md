# Roadmap: aruco-rs

## Phase 0: Project Scaffolding & Source Mapping
- [x] Initialize Cargo workspace.
- [x] **Map ARuco-ts Source Files**:
    - `cv.ts` -> `src/cv/`
    - `aruco.ts` -> `src/core/detector.rs`
    - `dictionary.ts` -> `src/core/dictionary.rs`
    - `pos.ts` -> `src/pose/`
- [x] Implement `src/lib.rs` core types.
- [x] Define `src/simd/` dispatch architecture (WASM vs Native).

## Phase 1: SIMD-Accelerated CV Kernels
- [x] Port `grayscale` from `cv.ts` to `src/cv/` (Scalar + WASM SIMD).
- [x] Port `threshold` and `otsu` methods from `cv.ts`.
- [x] Port `adaptiveThreshold`, `stackBoxBlur` from `cv.ts`.
- [x] Benchmarking: Compare Scalar vs SIMD performance in WASM.

## Phase 2: Contours & Geometry
- [x] Implement `findContours` (Suzuki algorithm) in `src/cv/contours.rs`.
- [x] Implement `isConvex` and `polySimplify` in `src/cv/geometry.rs`.
- [x] Implement `warp`, `getPerspectiveTransform`, and `square2quad` in `src/cv/geometry.rs`.
- [x] Implement `countNonZero` in `src/cv/scalar.rs`.

## Phase 3: Marker Decoding (Complete)
- [x] Implement bit-matrix sampling logic.
- [x] Port standard ArUco Dictionaries to `src/core/dictionary.rs`.
- [x] Implement Hamming distance error correction.

## 4. Phase 4: Pose Estimation & WebGL Integration (Completed)
- Port POSIT (`src/core/posit.rs`)
- Port SVD math (`src/core/svd.rs`)
- Implement `wasm-bindgen` API wrappers for high-performance WebAR usage.

## CI/CD Infrastructure
- [ ] Create `.github/workflows/rust.yml` to trigger tests & formatting checks on push.