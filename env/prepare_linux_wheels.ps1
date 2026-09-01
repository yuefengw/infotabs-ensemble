param(
    [string]$Python = "python",
    [string]$WheelDir = "$PSScriptRoot\wheels"
)

$ErrorActionPreference = "Stop"
$requirements = Join-Path (Split-Path $PSScriptRoot -Parent) "requirements-top-level.txt"

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

& $Python -m pip download `
    --dest $WheelDir `
    --requirement $requirements `
    --only-binary=:all: `
    --implementation cp `
    --python-version 3.10 `
    --abi cp310 `
    --abi abi3 `
    --abi none `
    --platform manylinux_2_28_x86_64 `
    --platform manylinux2014_x86_64 `
    --platform manylinux_2_17_x86_64 `
    --platform manylinux1_x86_64 `
    --platform linux_x86_64

if ($LASTEXITCODE -ne 0) {
    throw "Linux wheel download failed with exit code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $WheelDir -File |
    Get-FileHash -Algorithm SHA256 |
    Sort-Object Path |
    ForEach-Object { "{0}  {1}" -f $_.Hash.ToLowerInvariant(), (Split-Path $_.Path -Leaf) } |
    Set-Content -LiteralPath (Join-Path $WheelDir "SHA256SUMS") -Encoding ascii

Write-Host "Linux wheel bundle ready at $WheelDir"
