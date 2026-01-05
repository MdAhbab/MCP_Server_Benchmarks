# MCP Benchmarking Environment Setup Script
# Run as Administrator for best results

Write-Host "=== MCP Benchmarking Environment Setup ===" -ForegroundColor Cyan

# Create benchmark directory structure
$benchmarkDir = "C:\MCP_Benchmarks"
$subdirs = @("servers", "results", "logs", "figures")

Write-Host "`nCreating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $benchmarkDir | Out-Null
foreach ($dir in $subdirs) {
    New-Item -ItemType Directory -Force -Path "$benchmarkDir\$dir" | Out-Null
}
Write-Host "Created: $benchmarkDir" -ForegroundColor Green

# Get system information
Write-Host "`n=== System Information ===" -ForegroundColor Cyan

$sysinfo = @{
    "OS" = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    "OS Version" = (Get-WmiObject -Class Win32_OperatingSystem).Version
    "CPU" = (Get-WmiObject -Class Win32_Processor).Name
    "CPU Cores" = (Get-WmiObject -Class Win32_Processor).NumberOfCores
    "CPU Threads" = (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors
    "RAM (GB)" = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
}

$sysinfo.GetEnumerator() | ForEach-Object {
    Write-Host "$($_.Key): $($_.Value)"
}

# Save system info
$sysinfo | ConvertTo-Json | Out-File "$benchmarkDir\system_info.json"
Write-Host "`nSystem info saved to $benchmarkDir\system_info.json" -ForegroundColor Green

# Check Python installation
Write-Host "`n=== Checking Python ===" -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
}

# Check Node.js installation
Write-Host "`n=== Checking Node.js ===" -ForegroundColor Cyan
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js not found. Optional for TypeScript MCP servers." -ForegroundColor Yellow
}

# Install Python dependencies
Write-Host "`n=== Installing Python Dependencies ===" -ForegroundColor Cyan
$packages = @("mcp", "tiktoken", "aiohttp", "matplotlib", "numpy", "fastapi", "uvicorn", "msgpack", "psutil")

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg..." -ForegroundColor Yellow
    pip install $pkg --quiet 2>$null
}
Write-Host "Python dependencies installed." -ForegroundColor Green

# Create requirements.txt
$packages -join "`n" | Out-File "$benchmarkDir\requirements.txt"

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Benchmark directory: $benchmarkDir"
Write-Host "Next steps:"
Write-Host "  1. Copy benchmark scripts to $benchmarkDir"
Write-Host "  2. Run individual benchmarks: python <benchmark_name>.py"
Write-Host "  3. Analyze results: python analyze_results.py"
