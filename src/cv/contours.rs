use crate::{ImageBuffer, Point2i};

/// Fills the destination array with a 1-pixel padded zero-border around the source image,
/// and converts source pixels (0 -> 0, anything else -> 1).
/// Exact mirror of ARuco-ts `CV.binaryBorder`.
///
/// # Arguments
/// * `src` - The source `ImageBuffer`.
/// * `dst` - The destination byte slice, which MUST be dynamically allocated to accommodate
///   the `(width + 2) * (height + 2)` zero-padded array size.
///
/// # Returns
/// A reference to the modified destination slice.
pub fn binary_border<'a>(src: &ImageBuffer, dst: &'a mut [i32]) -> &'a [i32] {
    let src_data = src.data;
    let height = src.height as usize;
    let width = src.width as usize;
    let mut pos_src = 0;
    let mut pos_dst = 0;

    // Top border padding (-2 to width)
    for _ in -2..(width as isize) {
        dst[pos_dst] = 0;
        pos_dst += 1;
    }

    for _ in 0..height {
        // Left border
        dst[pos_dst] = 0;
        pos_dst += 1;

        // Copy row with 0/1 compression
        for _ in 0..width {
            dst[pos_dst] = if src_data[pos_src] == 0 { 0 } else { 1 };
            pos_dst += 1;
            pos_src += 1;
        }

        // Right border
        dst[pos_dst] = 0;
        pos_dst += 1;
    }

    // Bottom border padding
    for _ in -2..(width as isize) {
        dst[pos_dst] = 0;
        pos_dst += 1;
    }

    dst
}

/// Constant offsets for 8-directional sweeping (x, y).
pub const NEIGHBORHOOD: [[i32; 2]; 8] = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
];

/// Calculates flattened sequence offsets for 8-directional sweeps given a known row width.
/// Returns an array of exactly 16 offsets (the 8 offsets duplicated twice sequentially)
/// matching ARuco-ts `CV.neighborhoodDeltas(width)`.
pub fn neighborhood_deltas(width: i32) -> [i32; 16] {
    let mut deltas = [0i32; 16];
    for i in 0..8 {
        let delta = NEIGHBORHOOD[i][0] + NEIGHBORHOOD[i][1] * width;
        deltas[i] = delta;
        deltas[i + 8] = delta;
    }
    deltas
}

/// Represents a single extracted boundary contour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contour {
    /// Coordinates of each vertex in the contour.
    pub points: Vec<Point2i>,
    /// Whether this contour bounds a hole inside another contour.
    pub hole: bool,
}

/// Suzuki contour tracing algorithm (single border trace).
/// A strict functional port of `CV.borderFollowing` from ARuco-ts.
///
/// # Arguments
/// * `src` - Flat array representing the bordered binary image. Overwritten during execution.
/// * `pos` - Current index position in the flat array.
/// * `nbd` - The "contour number" tracking sequence label.
/// * `mut point` - The base (x, y) coordinates of the starting `pos`.
/// * `hole` - True if tracing the border of a hole.
/// * `deltas` - Generated offsets to locate 8-connected neighbors radially.
///
/// # Returns
/// A single `Contour` object encapsulating the points forming the extracted border.
pub fn border_following(
    src: &mut [i32],
    pos: usize,
    nbd: i32,
    mut point: Point2i,
    hole: bool,
    deltas: &[i32; 16],
) -> Contour {
    let mut contour = Contour {
        points: Vec::new(),
        hole,
    };

    let mut s: usize = if hole { 0 } else { 4 };
    let mut s_end = s;
    let mut pos1;
    let mut pos3;
    let mut pos4;

    loop {
        s = s.wrapping_sub(1) & 7;
        pos1 = (pos as isize + deltas[s] as isize) as usize;
        if src[pos1] != 0 {
            break;
        }
        if s == s_end {
            break;
        }
    }

    if s == s_end {
        src[pos] = -nbd;
        contour.points.push(Point2i::new(point.x, point.y));
    } else {
        pos3 = pos;

        loop {
            s_end = s;

            loop {
                // Post-increment conceptually mirrors TS `deltas[++s]` effectively
                s = (s + 1) & 15;
                pos4 = (pos3 as isize + deltas[s] as isize) as usize;
                if src[pos4] != 0 {
                    break;
                }
            }

            s &= 7;

            // Mirror JS unsigned 32-bit shift subtraction comparison: `(s - 1) >>> 0 < s_end >>> 0`
            // Detect outer vs inner border transition checks.
            let s_minus_1 = s.wrapping_sub(1) as u32;
            let s_end_u32 = s_end as u32;
            if s_minus_1 < s_end_u32 {
                src[pos3] = -nbd;
            } else if src[pos3] == 1 {
                src[pos3] = nbd;
            }

            contour.points.push(Point2i::new(point.x, point.y));

            point.x += NEIGHBORHOOD[s][0];
            point.y += NEIGHBORHOOD[s][1];

            if pos4 == pos && pos3 == pos1 {
                break;
            }

            pos3 = pos4;
            s = (s + 4) & 7;
        }
    }

    contour
}

/// Applies Suzuki's find contours algorithm on a binary thresholded `ImageBuffer`.
/// A strict functional port of `CV.findContours` from ARuco-ts.
///
/// # Arguments
/// * `src_img` - The parsed `ImageBuffer` of the base threshold output.
/// * `binary` - A large dynamically allocated scratchpad array of `i32`
///   equal to size `(width + 2) * (height + 2)`.
///
/// # Returns
/// A comprehensive `Vec<Contour>` encapsulating all geometric boundary chains.
pub fn find_contours(src_img: &ImageBuffer, binary: &mut [i32]) -> Vec<Contour> {
    let width = src_img.width as usize;
    let height = src_img.height as usize;
    let mut contours = Vec::new();

    // Fill buffer with 0/1 compression surrounded by an empty border 0
    binary_border(src_img, binary);

    // Flat distance offsets to fetch neighborhood jumps
    let deltas = neighborhood_deltas((width + 2) as i32);

    let mut pos = width + 3; // Skips initial padding corner pixel
    let mut nbd = 1;

    for i in 0..height {
        for j in 0..width {
            let pix = binary[pos];

            if pix != 0 {
                let mut outer = false;
                let mut hole = false;

                if pix == 1 && binary[pos - 1] == 0 {
                    outer = true;
                } else if pix >= 1 && binary[pos + 1] == 0 {
                    hole = true;
                }

                if outer || hole {
                    nbd += 1;
                    let point = Point2i::new(j as i32, i as i32);
                    let contour = border_following(binary, pos, nbd, point, hole, &deltas);
                    contours.push(contour);
                }
            }

            pos += 1; // Slide to next inner pixel
        }
        pos += 2; // Jump across right border, newline, left border
    }

    contours
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_border() {
        let src_data = [1, 0, 1, 0, 1, 0, 0, 0, 1];
        let src = ImageBuffer {
            data: &src_data,
            width: 3,
            height: 3,
        };

        let bw = 5;
        let mut dst = vec![0i32; bw * bw];
        let bordered = binary_border(&src, &mut dst);

        // Assert padded length is 25
        assert_eq!(bordered.len(), 25);

        // Assert top and bottom padding rows
        for i in 0..5 {
            assert_eq!(bordered[i], 0);
            assert_eq!(bordered[20 + i], 0);
        }

        // Check interior compression matches strictly 0 or 1
        let interior_expected = [1, 0, 1, 0, 1, 0, 0, 0, 1];
        let mut idx = 0;
        for y in 0..3 {
            let row_start = (y + 1) * bw + 1;
            for x in 0..3 {
                assert_eq!(bordered[row_start + x], interior_expected[idx]);
                idx += 1;
            }
        }
    }

    #[test]
    fn test_find_contours() {
        // Simple 5x5 image with a square boundary
        let src_data = [
            0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0,
            0,
        ];
        let img = ImageBuffer {
            data: &src_data,
            width: 5,
            height: 5,
        };

        let mut binary = vec![0i32; 7 * 7]; // (5+2) * (5+2)
        let contours = find_contours(&img, &mut binary);

        assert!(!contours.is_empty());

        // An outer bounding box shape and an inner hole boundary shape are expected.
        assert_eq!(contours.len(), 2);

        // Check that holes are properly detected
        assert!(!contours[0].hole); // Outer trace
        assert!(contours[1].hole); // Inner hole trace
    }
}
