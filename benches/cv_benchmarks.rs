// benches/cv_benchmarks.rs
#![allow(clippy::needless_range_loop)]
use aruco_rs::cv::scalar::ScalarCV;
use aruco_rs::cv::ComputerVision;
use aruco_rs::{ImageBuffer, Point2f};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use aruco_rs::simd::native::NativeCV;

const SIZES: [(usize, usize); 3] = [(320, 240), (640, 480), (1280, 720)];

fn bench_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grayscale");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let data: Vec<u8> = (0..(num_pixels * 4)).map(|i| (i % 256) as u8).collect();
        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };
        let mut out = vec![0u8; num_pixels];
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| ScalarCV::grayscale(black_box(&buffer), black_box(&mut out)))
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("simd_native", &size_str),
            &size_str,
            |b, _| b.iter(|| NativeCV::grayscale(black_box(&buffer), black_box(&mut out))),
        );
    }
    group.finish();
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("Threshold");
    for &(width, height) in SIZES.iter() {
        let size = width * height;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut out = vec![0u8; size];
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| ScalarCV::threshold(black_box(&data), black_box(&mut out), black_box(128)))
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("simd_native", &size_str),
            &size_str,
            |b, _| {
                b.iter(|| {
                    NativeCV::threshold(black_box(&data), black_box(&mut out), black_box(128))
                })
            },
        );
    }
    group.finish();
}

fn bench_otsu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Otsu");
    for &(width, height) in SIZES.iter() {
        let size = width * height;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| ScalarCV::otsu(black_box(&data)))
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("simd_native", &size_str),
            &size_str,
            |b, _| b.iter(|| NativeCV::otsu(black_box(&data))),
        );
    }
    group.finish();
}

fn bench_stack_box_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("StackBoxBlur");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let data: Vec<u8> = (0..num_pixels).map(|i| (i % 256) as u8).collect();
        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };
        let mut out = vec![0u8; num_pixels];
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| {
                ScalarCV::stack_box_blur(black_box(&buffer), black_box(&mut out), black_box(2))
            })
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("simd_native", &size_str),
            &size_str,
            |b, _| {
                b.iter(|| {
                    NativeCV::stack_box_blur(black_box(&buffer), black_box(&mut out), black_box(2))
                })
            },
        );
    }
    group.finish();
}

fn bench_adaptive_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdaptiveThreshold");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let data: Vec<u8> = (0..num_pixels).map(|i| (i % 256) as u8).collect();
        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };
        let mut out = vec![0u8; num_pixels];
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| {
                ScalarCV::adaptive_threshold(
                    black_box(&buffer),
                    black_box(&mut out),
                    black_box(2),
                    black_box(7),
                )
            })
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("simd_native", &size_str),
            &size_str,
            |b, _| {
                b.iter(|| {
                    NativeCV::adaptive_threshold(
                        black_box(&buffer),
                        black_box(&mut out),
                        black_box(2),
                        black_box(7),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_find_contours(c: &mut Criterion) {
    let mut group = c.benchmark_group("FindContours");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        // Create an alternating sparse grid of boxes so there are many contours to find
        let mut data = vec![0u8; num_pixels];
        for y in 0..height {
            for x in 0..width {
                // Creates 10x10 squares of 255s spaced out
                if (x / 10) % 2 == 0 && (y / 10) % 2 == 0 {
                    data[y * width + x] = 255;
                }
            }
        }
        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };
        let mut binary_scratch = vec![0i32; (width + 2) * (height + 2)];
        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| {
                aruco_rs::cv::contours::find_contours(
                    black_box(&buffer),
                    black_box(&mut binary_scratch),
                )
            })
        });
    }
    group.finish();
}

fn bench_warp(c: &mut Criterion) {
    let mut group = c.benchmark_group("PerspectiveWarp");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let mut data = vec![0u8; num_pixels];
        for i in 0..num_pixels {
            data[i] = (i % 256) as u8;
        }

        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };

        // Extract a 49x49 patch from the center of the image
        let w = width as f32;
        let h = height as f32;
        let contour = [
            Point2f::new(w * 0.25, h * 0.25),
            Point2f::new(w * 0.75, h * 0.25),
            Point2f::new(w * 0.75, h * 0.75),
            Point2f::new(w * 0.25, h * 0.75),
        ];

        let mut out = vec![0u8; 49 * 49];

        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| {
                ScalarCV::warp(
                    black_box(&buffer),
                    black_box(&mut out),
                    black_box(&contour),
                    black_box(49),
                )
            })
        });
    }
    group.finish();
}

fn bench_count_non_zero(c: &mut Criterion) {
    let mut group = c.benchmark_group("CountNonZero");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let mut data = vec![0u8; num_pixels];
        for i in 0..num_pixels {
            data[i] = if i % 2 == 0 { 255 } else { 0 };
        }

        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };

        // Test counting a 49x49 square in the middle
        let square = aruco_rs::cv::Square {
            x: (width / 2 - 24) as u32,
            y: (height / 2 - 24) as u32,
            width: 49,
            height: 49,
        };

        let size_str = format!("{}x{}", width, height);

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| ScalarCV::count_non_zero(black_box(&buffer), black_box(&square)))
        });
    }
    group.finish();
}

fn bench_detector_fake(c: &mut Criterion) {
    use aruco_rs::core::detector::Detector;
    use aruco_rs::core::dictionary::{Dictionary, DICTIONARY_ARUCO};

    let mut group = c.benchmark_group("Detector_Detect");
    let dict = Dictionary::new(&DICTIONARY_ARUCO);

    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let mut data = vec![255u8; num_pixels * 4];

        // Draw a simulated black marker 50x50 in the middle
        let start_x = width / 2 - 25;
        let start_y = height / 2 - 25;
        let end_x = start_x + 50;
        let end_y = start_y + 50;

        for y in start_y..end_y {
            for x in start_x..end_x {
                let idx = (y * width + x) * 4;
                data[idx] = 0; // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 0; // B
                data[idx + 3] = 255; // A
            }
        }

        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };

        let size_str = format!("{}x{}", width, height);

        let detector_scalar = Detector::new(&dict, ScalarCV);
        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| detector_scalar.detect(black_box(&buffer)))
        });

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            let detector_simd = Detector::new(&dict, NativeCV);
            group.bench_with_input(
                BenchmarkId::new("simd_native", &size_str),
                &size_str,
                |b, _| b.iter(|| detector_simd.detect(black_box(&buffer))),
            );
        }
    }
    group.finish();
}

fn bench_detector_real(c: &mut Criterion) {
    use aruco_rs::core::detector::Detector;
    use aruco_rs::core::dictionary::{Dictionary, DICTIONARY_ARUCO};
    use std::path::Path;

    let mut group = c.benchmark_group("Detector_RealImage");
    let dict = Dictionary::new(&DICTIONARY_ARUCO);

    let image_path = Path::new("tests/data/real_marker.png");

    // Fallback if the user hasn't provided the image yet
    if !image_path.exists() {
        println!("⚠️ Skipping bench_detector_real: tests/data/real_marker.png not found.");
        return;
    }

    let img = image::open(image_path).unwrap();

    // 1. Benchmark Native Resolution
    {
        let img_native = img.to_rgba8();
        let width = img_native.width();
        let height = img_native.height();
        let data = img_native.into_raw();

        let buffer = ImageBuffer {
            data: &data,
            width,
            height,
        };

        let size_str = format!("Native ({}x{})", width, height);

        let detector_scalar = Detector::new(&dict, ScalarCV);
        group.bench_with_input(
            BenchmarkId::new("scalar_real", &size_str),
            &size_str,
            |b, _| b.iter(|| detector_scalar.detect(black_box(&buffer))),
        );
    }

    // 2. Benchmark Downscaled AR Webcam Resolution (640x480)
    {
        let img_640 = img
            .resize_exact(640, 480, image::imageops::FilterType::Nearest)
            .to_rgba8();
        let width = img_640.width();
        let height = img_640.height();
        let data = img_640.into_raw();

        let buffer = ImageBuffer {
            data: &data,
            width,
            height,
        };

        let size_str = format!("Webcam ({}x{})", width, height);

        let detector_scalar = Detector::new(&dict, ScalarCV);
        group.bench_with_input(
            BenchmarkId::new("scalar_real", &size_str),
            &size_str,
            |b, _| b.iter(|| detector_scalar.detect(black_box(&buffer))),
        );
    }

    group.finish();
}

fn bench_gaussian_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianBlur");
    for &(width, height) in SIZES.iter() {
        let num_pixels = width * height;
        let mut data = vec![0u8; num_pixels];
        for (i, val) in data.iter_mut().enumerate().take(num_pixels) {
            *val = (i % 256) as u8;
        }

        let buffer = ImageBuffer {
            data: &data,
            width: width as u32,
            height: height as u32,
        };

        // Output buffer
        let mut out = vec![0u8; num_pixels];
        let size_str = format!("{}x{}", width, height);

        let kernel_size = 5;

        group.bench_with_input(BenchmarkId::new("scalar", &size_str), &size_str, |b, _| {
            b.iter(|| {
                ScalarCV::gaussian_blur(
                    black_box(&buffer),
                    black_box(&mut out),
                    black_box(kernel_size),
                )
            })
        });

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            group.bench_with_input(
                BenchmarkId::new("simd_wasm", &size_str),
                &size_str,
                |b, _| {
                    b.iter(|| {
                        aruco_rs::simd::wasm::WasmCV::gaussian_blur(
                            black_box(&buffer),
                            black_box(&mut out),
                            black_box(kernel_size),
                        )
                    })
                },
            );
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            group.bench_with_input(
                BenchmarkId::new("simd_native", &size_str),
                &size_str,
                |b, _| {
                    b.iter(|| {
                        aruco_rs::simd::native::NativeCV::gaussian_blur(
                            black_box(&buffer),
                            black_box(&mut out),
                            black_box(kernel_size),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_grayscale,
    bench_threshold,
    bench_otsu,
    bench_stack_box_blur,
    bench_gaussian_blur,
    bench_adaptive_threshold,
    bench_find_contours,
    bench_warp,
    bench_count_non_zero,
    bench_detector_fake,
    bench_detector_real
);
criterion_main!(benches);
