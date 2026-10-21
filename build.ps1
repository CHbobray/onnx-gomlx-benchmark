# build.ps1 - build, test, and run the Go application on Windows.
#
# Usage (PowerShell, from the repository root):
#   .\build.ps1            # run tests, then build bin\fashion-go.exe
#   .\build.ps1 -Run       # ...and run it with the default settings
#   .\build.ps1 -SkipTests # build only
#
# If scripts are blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [switch]$Run,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "Go is not installed or not on PATH. Install Go 1.27 or newer from https://go.dev/dl/ and reopen PowerShell."
}

$goVersion = (go env GOVERSION)
Write-Host "Using $goVersion"

if (-not $SkipTests) {
    Write-Host "`nRunning unit tests..."
    go test ./...
    if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed; not building." }
}

Write-Host "`nBuilding bin\fashion-go.exe..."
New-Item -ItemType Directory -Force -Path bin | Out-Null
# CGO_ENABLED=0: the pure-Go GoMLX backend needs no C compiler, so the .exe is self-contained.
# GOEXPERIMENT=simd: compiles GoMLX's AVX2/AVX-512 kernels (used automatically when the CPU has them).
$env:CGO_ENABLED = "0"
$env:GOEXPERIMENT = "simd"
go build -trimpath -ldflags "-s -w" -o bin\fashion-go.exe .\cmd\fashion-go
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed." }

$size = [math]::Round((Get-Item bin\fashion-go.exe).Length / 1MB, 1)
Write-Host "Built bin\fashion-go.exe ($size MB)"

if ($Run) {
    Write-Host "`nRunning..."
    .\bin\fashion-go.exe -runs 30 -latency 500
}
