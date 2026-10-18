# Starts Riya: the local model server (minimised), then the core. PowerShell 5.1 compatible.
param(
    [string]$Model = "C:/Users/sreev/models/riya/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Health = "http://127.0.0.1:8090/health"

function Test-ModelServer {
    try {
        $r = Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

Set-Location $Root

if (-not (Test-ModelServer)) {
    Start-Process -FilePath $Python -ArgumentList @("-m", "riya.server", "--model", "`"$Model`"") `
        -WorkingDirectory $Root -WindowStyle Minimized
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-ModelServer)) {
        if ((Get-Date) -gt $deadline) {
            Write-Error "The model server did not become healthy at $Health within 120 s."
            exit 1
        }
        Start-Sleep -Seconds 2
    }
}

& $Python -m riya.core
