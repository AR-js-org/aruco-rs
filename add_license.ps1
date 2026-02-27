$header = @"
// Copyright (c) 2026 kalwalt and AR.js-org contributors
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// See https://github.com/AR-js-org/aruco-rs/blob/main/LICENSE

"@

Get-ChildItem -Path "src", "tests", "benches", "examples" -Include *.rs,*.js,*.ts -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "node_modules|pkg" } | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if (-not $content.StartsWith("// Copyright")) {
        $newContent = $header + $content
        Set-Content -Path $_.FullName -Value $newContent -NoNewline
        Write-Host "Added header to $($_.Name)"
    }
}
