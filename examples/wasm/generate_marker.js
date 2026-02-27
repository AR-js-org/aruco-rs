// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
// Small script to create a minimal 50x50 PNG testing image
const fs = require('fs');
const { createCanvas } = require('canvas');

const width = 200;
const height = 200;
const canvas = createCanvas(width, height);
const ctx = canvas.getContext('2d');

// White background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, width, height);

// Draw a simple 5x5 grid (simulating an ArUco marker)
// 1 = White, 0 = Black

// ArUco dictionary grid ID 0 (0x1084210)
const grid = [
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
];

const pixelSize = 20;
const offsetX = 50;
const offsetY = 50;

// Draw thick black border
ctx.fillStyle = 'black';
ctx.fillRect(offsetX - pixelSize, offsetY - pixelSize, pixelSize * 7, pixelSize * 7);

// Draw inner grid 
for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
        ctx.fillStyle = grid[y][x] ? 'white' : 'black';
        ctx.fillRect(offsetX + (x * pixelSize), offsetY + (y * pixelSize), pixelSize, pixelSize);
    }
}

// Save to file
const buffer = canvas.toBuffer('image/png');
fs.writeFileSync('marker_23.png', buffer);
console.log('Created marker_23.png');
