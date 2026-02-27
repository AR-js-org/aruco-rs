// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait()
    ],
    server: {
        fs: {
            allow: ['../..']
        }
    }
});
