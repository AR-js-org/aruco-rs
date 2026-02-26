#![cfg(target_arch = "wasm32")]
#![cfg(feature = "wasm")]

use crate::core::detector::Detector;
use crate::cv::scalar::ScalarCV;
use crate::{ImageBuffer, Marker};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
pub struct WasmPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Serialize)]
pub struct WasmMarker {
    pub id: i32,
    pub corners: Vec<WasmPoint>,
    pub distance: i32,
}

impl From<&Marker> for WasmMarker {
    fn from(marker: &Marker) -> Self {
        WasmMarker {
            id: marker.id,
            corners: marker
                .corners
                .iter()
                .map(|p| WasmPoint { x: p.x, y: p.y })
                .collect(),
            distance: marker.hamming_distance,
        }
    }
}

/// JS-facing Detector
#[wasm_bindgen]
pub struct ARucoDetector {
    detector: Detector<'static, ScalarCV>,
}

#[wasm_bindgen]
impl ARucoDetector {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dict_name: &str,
        max_hamming_distance: Option<i32>,
    ) -> Result<ARucoDetector, JsValue> {
        let mut dict = if dict_name == "ARUCO_MIP_36h12" {
            crate::core::dictionary::Dictionary::new(&crate::core::dictionary::DICTIONARY_ARUCO_MIP_36H12)
        } else if dict_name == "ARUCO" {
            crate::core::dictionary::Dictionary::new(&crate::core::dictionary::DICTIONARY_ARUCO)
        } else {
            return Err(JsValue::from_str(&format!(
                "Dictionary {} is not compiled-in.",
                dict_name
            )));
        };

        if let Some(dist) = max_hamming_distance {
            dict.tau = dist as usize;
        }

        let dict_ref: &'static crate::core::dictionary::Dictionary = Box::leak(Box::new(dict));
        let cv = ScalarCV;
        let detector = Detector::new(dict_ref, cv);

        Ok(ARucoDetector { detector })
    }

    /// Primary detection endpoint avoiding memory copies.
    /// `image_data` should be a flatten continuous byte array.
    pub fn detect_image(
        &self,
        width: u32,
        height: u32,
        image_data: &[u8],
    ) -> Result<JsValue, JsValue> {
        let buffer = ImageBuffer {
            data: image_data,
            width,
            height,
        };

        let markers = self.detector.detect(&buffer);
        let js_markers: Vec<WasmMarker> = markers.iter().map(WasmMarker::from).collect();
        
        Ok(serde_wasm_bindgen::to_value(&js_markers).unwrap())
    }
}
