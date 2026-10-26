<#
  Starts the LLM gateway and the five agent services, each in its own venv, and waits
  until every /health answers. PIDs go to data/pids.json for stop-all.ps1.
  Usage:  scripts\run-all.ps1            keep existing data/ (memory persists across runs)
          scripts\run-all.ps1 -Fresh     wipe data/ and logs/ first
#>
param([switch]$Fresh)

$root = Split-Path -Parent $PSScriptRoot
& "$PSScriptRoot\stop-all.ps1"
if ($Fresh) { Remove-Item -Recurse -Force "$root\data", "$root\logs" -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force "$root\data", "$root\logs" | Out-Null

# name, port, venv, entry
$services = @(
    @{ n = "llm";    p = 9100; v = "policy_langgraph"; a = @("-m", "testbed_common.mockllm") },
    @{ n = "policy"; p = 9102; v = "policy_langgraph"; a = @("agents\policy_langgraph\app.py") },
    @{ n = "risk";   p = 9103; v = "risk_crewai";      a = @("agents\risk_crewai\app.py") },
    @{ n = "refund"; p = 9104; v = "refund_adk";       a = @("agents\refund_adk\app.py") },
    @{ n = "notify"; p = 9105; v = "notify_autogen";   a = @("agents\notify_autogen\app.py") },
    @{ n = "router"; p = 9101; v = "router_openai";    a = @("agents\router_openai\app.py") }
)

$pids = @{}
foreach ($s in $services) {
    $py = "$root\.venvs\$($s.v)\Scripts\python.exe"
    if (-not (Test-Path $py)) { Write-Warning "skip $($s.n): $($s.v) venv missing (run scripts\setup.ps1)"; continue }
    $proc = Start-Process -FilePath $py -ArgumentList $s.a -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardError "$root\logs\$($s.n).stderr.txt" -RedirectStandardOutput "$root\logs\$($s.n).stdout.txt"
    $pids[$s.n] = $proc.Id
}
$pids | ConvertTo-Json | Set-Content "$root\data\pids.json"

foreach ($s in $services) {
    if (-not $pids.ContainsKey($s.n)) { continue }
    $ok = $false
    for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
        try { Invoke-RestMethod "http://127.0.0.1:$($s.p)/health" -TimeoutSec 2 | Out-Null; $ok = $true } catch { Start-Sleep 1 }
    }
    Write-Host ("{0,-7} :{1}  {2}" -f $s.n, $s.p, $(if ($ok) { "up" } else { "NOT UP - see logs\$($s.n).stderr.txt" }))
}
