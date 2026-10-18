# Starts Riya: the local model server (minimised), the core, and the desktop body once the core accepts connections.
# PowerShell 5.1 compatible.
param(
    [string]$Model = "~/models/riya/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Health = "http://127.0.0.1:8090/health"
$CorePort = 8765
$BodyDir = Join-Path $Root "body"

function Test-ModelServer {
    try {
        $r = Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Test-CorePort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect("127.0.0.1", $CorePort, $null, $null)
        return ($pending.AsyncWaitHandle.WaitOne(500) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Close()
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

# The core runs in this console (its log stays visible); the body starts once its port accepts connections.
$core = Start-Process -FilePath $Python -ArgumentList @("-m", "riya.core") -WorkingDirectory $Root `
    -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(60)
while (-not $core.HasExited -and -not (Test-CorePort)) {
    if ((Get-Date) -gt $deadline) { break }
    Start-Sleep -Milliseconds 500
}
if (-not $core.HasExited -and (Test-CorePort)) {
    try {
        Start-Process -FilePath "npm.cmd" -ArgumentList @("--prefix", "`"$BodyDir`"", "start") `
            -WorkingDirectory $Root -WindowStyle Hidden
    } catch {
        Write-Warning "Could not start the body: $_"
    }
} elseif (-not $core.HasExited) {
    Write-Warning "The core's port $CorePort did not open within 60 s; not starting the body."
}
$core.WaitForExit()
exit $core.ExitCode
