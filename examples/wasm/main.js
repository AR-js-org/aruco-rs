// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
import init, { ARucoDetector } from '@ar-js-org/aruco-rs';

async function main() {
    // 1. Initialize the WebAssembly module (Required for Vite)
    await init();

    const output = document.getElementById('output');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('source-img');

    img.onload = () => {
        // 2. Draw image to canvas to extract raw pixel data
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const rawPixels = new Uint8Array(imageData.data.buffer);

        output.innerText = "Initializing Detector...";

        try {
            // 3. Initialize the Detector 
            // ("ARUCO" dictionary, undefined tau gives the default max distance)
            const detector = new ARucoDetector("ARUCO");

            output.innerText = "Running Detection...\n";
            const start = performance.now();

            // 4. Run the WASM tracking pipeline!
            const markers = detector.detect_image(canvas.width, canvas.height, rawPixels);

            const timeMs = performance.now() - start;

            // 5. Parse the returned standard Javascript Objects
            output.innerText += `\nFound ${markers.length} markers in ${timeMs.toFixed(2)}ms.\n`;

            markers.forEach((marker, index) => {
                output.innerText += `\n[Marker ${index}] ID: ${marker.id}, Errors: ${marker.distance}\n`;
                output.innerText += `  Corners:\n`;
                marker.corners.forEach((c, i) => {
                    output.innerText += `    pt${i}: (${c.x.toFixed(1)}, ${c.y.toFixed(1)})\n`;
                });

                // Draw a box around the detection
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(marker.corners[0].x, marker.corners[0].y);
                for (let i = 1; i < 4; i++) {
                    ctx.lineTo(marker.corners[i].x, marker.corners[i].y);
                }
                ctx.closePath();
                ctx.stroke();
            });

        } catch (e) {
            output.innerText = `Error: ${e}`;
            console.error(e);
        }
    };

    // Ensure onload triggers if image is cached
    if (img.complete) {
        img.onload();
    }
}

main();
