param()

$ErrorActionPreference = "Continue"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "Windows version"
try {
    $cv = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    Write-Host "ProductName: $($cv.ProductName)"
    Write-Host "DisplayVersion: $($cv.DisplayVersion)"
    Write-Host "CurrentBuild: $($cv.CurrentBuild)"
    Write-Host "UBR: $($cv.UBR)"
} catch {
    Write-Warning "Unable to read Windows version details."
}

Write-Step "WSL status"
try {
    & wsl.exe --status
} catch {
    Write-Warning "Unable to read WSL status."
}

Write-Step "Installed distros"
try {
    & wsl.exe -l -v
} catch {
    Write-Warning "Unable to list WSL distros."
}

Write-Step "WSL GPU device path in Linux"
try {
    & wsl.exe -d Ubuntu-22.04 -- bash -lc 'ls -l /dev/dxg 2>/dev/null || echo "/dev/dxg missing"'
} catch {
    Write-Warning "Unable to query /dev/dxg inside Ubuntu-22.04."
}

Write-Step "NVIDIA driver on Windows"
try {
    & nvidia-smi.exe
} catch {
    Write-Warning "Unable to run nvidia-smi.exe from Windows."
}

Write-Step "Per-user .wslconfig"
$wslConfig = Join-Path $HOME ".wslconfig"
if (Test-Path $wslConfig) {
    Write-Host "Found: $wslConfig"
    Get-Content $wslConfig
} else {
    Write-Host "Not found: $wslConfig"
}

Write-Step "Next actions if /dev/dxg is missing"
Write-Host "1. In an elevated PowerShell window, run: wsl --shutdown"
Write-Host "2. Reboot Windows"
Write-Host "3. Open Ubuntu-22.04 again and run: bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/wsl-doctor.sh"
Write-Host "4. If /dev/dxg is still missing, share this script's full output."
