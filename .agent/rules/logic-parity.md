---
trigger: always_on
---

# RULE: LOGIC_PARITY_TS_RUST

## Reference Source
- **Repository**: [https://github.com/kalwalt/ARuco-ts](https://github.com/kalwalt/ARuco-ts)
- **Constraint**: You must maintain 1:1 functional identity with the algorithms in this repository.

## Operating Procedures
1. **Source First**: Analyze the original `.ts` files in `ARuco-ts/src/` before proposing any Rust implementation.
2. **Algorithm Mirroring**: Always implement a "Safe Rust" scalar version that mirrors the TypeScript logic bit-for-bit before attempting SIMD optimizations.
3. **Parameter Sync**: Epsilon values, magic numbers, and binarization thresholds (e.g., for Otsu or Douglas-Peucker) must match the TS source exactly.
4. **Sampling Pattern**: The internal grid sampling for marker ID extraction must match the pattern defined in `ARuco-ts` to prevent detection mismatches.
5. **No Innovation**: Do not "improve" the base algorithm. The priority is to reach feature parity with the existing TypeScript library first.