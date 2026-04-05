param(
    [switch]$Install,
    [string]$Distro = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking current WSL state"

try {
    & wsl.exe --status
} catch {
    Write-Warning "Unable to read WSL status. WSL may not be installed yet."
}

try {
    & wsl.exe -l -v
} catch {
    Write-Warning "No WSL distributions are currently available."
}

if (-not $Install) {
    Write-Step "No changes made"
    Write-Host "Re-run this script as Administrator with -Install to install WSL2 and $Distro."
    Write-Host "Example:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup\install-wsl-ubuntu.ps1 -Install -Distro $Distro"
    exit 0
}

if (-not (Test-IsAdmin)) {
    throw "This script must be run in an elevated PowerShell window when -Install is used."
}

Write-Step "Installing WSL2"
& wsl.exe --install -d $Distro

Write-Step "Setting WSL2 as the default version"
& wsl.exe --set-default-version 2

Write-Step "Post-install"
Write-Host "Target distro for the LiDAR detector stack: $Distro"
Write-Host "If Windows asks for a reboot, reboot first."
Write-Host "After reboot, check installed distros with:"
Write-Host "  wsl.exe -l -v"
Write-Host "If $Distro is not listed yet, run:"
Write-Host "  wsl.exe --install -d $Distro"
Write-Host "Then launch $Distro once, create your Linux user, and run:"
Write-Host "  bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-lidar.sh"
