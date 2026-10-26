<# Stops every service started by run-all.ps1 (process trees, since a venv launcher spawns a child). #>
$root = Split-Path -Parent $PSScriptRoot
$file = "$root\data\pids.json"
if (Test-Path $file) {
    (Get-Content $file -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { taskkill /T /F /PID $_.Value 2>$null | Out-Null }
    Remove-Item $file
}
# Also free the ports in case a service was started by hand.
foreach ($port in 9100..9105) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { taskkill /T /F /PID $_.OwningProcess 2>$null | Out-Null }
}
