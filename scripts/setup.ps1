<#
  Creates one virtualenv per agent under .venvs/ and installs that agent's own
  framework stack (agents/<name>/requirements.txt) plus the shared testbed_common.
  Separate venvs are deliberate: the stacks pin conflicting versions, and in the
  real world each agent team ships its own runtime.
  Usage:  scripts\setup.ps1            (all agents)
          scripts\setup.ps1 -Only risk_crewai
#>
param([string[]]$Only)

$root = Split-Path -Parent $PSScriptRoot
$agents = Get-ChildItem "$root\agents" -Directory | Where-Object { $_.Name -notlike '_*' }
if ($Only) { $agents = $agents | Where-Object { $Only -contains $_.Name } }

foreach ($a in $agents) {
    $venv = "$root\.venvs\$($a.Name)"
    # An agent can pin its interpreter (e.g. CrewAI needs 3.13: tiktoken/regex have no 3.14 wheels).
    $pin = "$($a.FullName)\python.txt"
    $py = if (Test-Path $pin) { (Get-Content $pin -Raw).Trim() } else { "python" }
    if (-not (Test-Path "$venv\Scripts\python.exe")) { & $py -m venv $venv }
    & "$venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -e "$root\testbed_common"
    & "$venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r "$($a.FullName)\requirements.txt"
    if ($LASTEXITCODE -ne 0) { Write-Warning "install failed for $($a.Name)"; continue }
    & "$venv\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 "$($a.FullName)\requirements.lock.txt"
    Write-Host "ok  $($a.Name)"
}
