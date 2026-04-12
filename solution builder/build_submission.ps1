# ============================================================
# build_submission.ps1
# Compila el binario mlsys con Docker y empaqueta el .zip
# de entrega para MLSys 2026 Track A
# ============================================================

param(
    [string]$TeamName = "MiEquipo",
    [int]$SubmissionNumber = 1
)

$ErrorActionPreference = "Stop"
$SolutionDir = $PSScriptRoot
$ProjectRoot = Split-Path $SolutionDir -Parent
$SubmissionDir = Join-Path $SolutionDir "submission"
$ZipName = "${TeamName}_TrackA_${SubmissionNumber}.zip"
$ZipPath = Join-Path $ProjectRoot $ZipName

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MLSys 2026 Track A - Build & Package" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Compilar con Docker
Write-Host "`n[1/5] Building mlsys binary with Docker..." -ForegroundColor Yellow
docker build -t mlsys-builder "$SolutionDir"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK - Docker build succeeded" -ForegroundColor Green

# 2. Extraer el binario del contenedor
Write-Host "`n[2/5] Extracting mlsys binary from container..." -ForegroundColor Yellow
$ContainerId = docker create mlsys-builder
docker cp "${ContainerId}:/app/mlsys" "$SolutionDir\mlsys"
docker rm $ContainerId | Out-Null
if (-not (Test-Path "$SolutionDir\mlsys")) {
    Write-Host "ERROR: Failed to extract binary!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK - Binary extracted" -ForegroundColor Green

# 3. Crear estructura de submission
Write-Host "`n[3/5] Creating submission directory structure..." -ForegroundColor Yellow
if (Test-Path $SubmissionDir) { Remove-Item $SubmissionDir -Recurse -Force }
New-Item -ItemType Directory -Path $SubmissionDir | Out-Null
New-Item -ItemType Directory -Path "$SubmissionDir\source" | Out-Null
New-Item -ItemType Directory -Path "$SubmissionDir\source\nlohmann" | Out-Null

# Copiar binario
Copy-Item "$SolutionDir\mlsys" "$SubmissionDir\mlsys"
# Copiar fuentes
Copy-Item "$SolutionDir\main.cpp" "$SubmissionDir\source\"
Copy-Item "$SolutionDir\scheduler.cpp" "$SubmissionDir\source\"
Copy-Item "$SolutionDir\scheduler.h" "$SubmissionDir\source\"
Copy-Item "$SolutionDir\nlohmann\json.hpp" "$SubmissionDir\source\nlohmann\"
Write-Host "  OK - Directory structure created" -ForegroundColor Green

# 4. Verificar writeup.pdf
Write-Host "`n[4/5] Checking for writeup.pdf..." -ForegroundColor Yellow
$WriteupSrc = Join-Path $ProjectRoot "pdf\writeup.pdf"
if (Test-Path $WriteupSrc) {
    Copy-Item $WriteupSrc "$SubmissionDir\writeup.pdf"
    Write-Host "  OK - writeup.pdf found and copied" -ForegroundColor Green
} else {
    Write-Host "  WARNING: writeup.pdf not found in $SolutionDir" -ForegroundColor DarkYellow
    Write-Host "  You must add writeup.pdf to the submission folder before uploading!" -ForegroundColor DarkYellow
}

# 5. Crear ZIP
Write-Host "`n[5/5] Creating $ZipName..." -ForegroundColor Yellow
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path "$SubmissionDir\*" -DestinationPath $ZipPath
Write-Host "  OK - ZIP created at: $ZipPath" -ForegroundColor Green

# Resumen
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " Submission package ready!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Contents of $ZipName :"
Get-ChildItem $SubmissionDir -Recurse | ForEach-Object {
    $indent = "  " * ($_.FullName.Replace($SubmissionDir, "").Split([IO.Path]::DirectorySeparatorChar).Length - 1)
    if ($_.PSIsContainer) {
        Write-Host "  $indent$($_.Name)/" -ForegroundColor Blue
    } else {
        $size = "{0:N0} KB" -f ($_.Length / 1024)
        Write-Host "  $indent$($_.Name) ($size)"
    }
}

Write-Host ""
if (-not (Test-Path "$SubmissionDir\writeup.pdf")) {
    Write-Host "  REMINDER: Add writeup.pdf before uploading!" -ForegroundColor Red
}
Write-Host ""
Write-Host "To test the binary against benchmarks inside Docker:" -ForegroundColor Gray
Write-Host "  docker run --rm -v `"$($ProjectRoot):/work`" mlsys-builder /work/benchmarks/mlsys-2026-1.json /work/output.json" -ForegroundColor Gray
