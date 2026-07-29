<#
.SYNOPSIS
    Starts both the backend and frontend dev servers in the same terminal.
.DESCRIPTION
    Runs `pnpm dev` in Backend_UserModuleTemplate (port 5001) and
    Frontend_UserModuleTemplate (port 5173) concurrently. Press Ctrl+C to stop both.
.PARAMETER SkipBackend
    Skip starting the backend server.
.PARAMETER SkipFrontend
    Skip starting the frontend server.
.EXAMPLE
    .\start.ps1
.EXAMPLE
    .\start.ps1 -SkipBackend
#>

param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BackendDir = Resolve-Path "$ScriptDir\Backend_UserModuleTemplate" -ErrorAction SilentlyContinue

$processes = @()

function Stop-All {
    Write-Host "`nShutting down all services..." -ForegroundColor Red
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "All services stopped." -ForegroundColor Red
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Nobel - Dev Startup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($BackendDir) {
    Write-Host "[pgbouncer] Starting PgBouncer via Docker..." -ForegroundColor Green
    $pgProc = Start-Process -FilePath "cmd" -ArgumentList "/c docker compose --env-file .env.dev -f docker-compose.pgbouncer.yml up -d" -WorkingDirectory $BackendDir -NoNewWindow -PassThru
    if ($pgProc) {
        $pgProc.WaitForExit()
        Write-Host "[pgbouncer] Docker Compose exited with code $($pgProc.ExitCode)" -ForegroundColor DarkGreen
    } else {
        Write-Host "[pgbouncer] ERROR: Failed to start Docker Compose" -ForegroundColor Red
    }
    Write-Host ""
}

if (-not $SkipBackend) {
    if (-not $BackendDir) {
        Write-Host "[backend] ERROR: Backend_UserModuleTemplate directory not found at $ScriptDir\Backend_UserModuleTemplate" -ForegroundColor Red
    } else {
        Write-Host "[backend] Starting in $BackendDir" -ForegroundColor Green
        $proc = Start-Process -FilePath "cmd" -ArgumentList "/c pnpm dev" -WorkingDirectory $BackendDir -NoNewWindow -PassThru
        if ($proc) {
            $processes += $proc
            Write-Host "[backend] PID: $($proc.Id)" -ForegroundColor DarkGreen
        } else {
            Write-Host "[backend] ERROR: Failed to start. Is pnpm installed globally?" -ForegroundColor Red
        }
    }
}

if (-not $SkipFrontend) {
    $FrontendDir = Resolve-Path "$ScriptDir\Frontend_UserModuleTemplate" -ErrorAction SilentlyContinue
    if (-not $FrontendDir) {
        Write-Host "[frontend] ERROR: Frontend_UserModuleTemplate directory not found at $ScriptDir\Frontend_UserModuleTemplate" -ForegroundColor Red
    } else {
        Write-Host "[frontend] Starting in $FrontendDir" -ForegroundColor Green
        $proc = Start-Process -FilePath "cmd" -ArgumentList "/c pnpm dev" -WorkingDirectory $FrontendDir -NoNewWindow -PassThru
        if ($proc) {
            $processes += $proc
            Write-Host "[frontend] PID: $($proc.Id)" -ForegroundColor DarkGreen
        } else {
            Write-Host "[frontend] ERROR: Failed to start. Is pnpm installed globally?" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "--- All services started ---" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:5001" -ForegroundColor Cyan
Write-Host "  Frontend: http://championsforgood.localhost:5173" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop both" -ForegroundColor Yellow
Write-Host ""

if ($processes.Count -eq 0) {
    Write-Host "No processes were started. Exiting." -ForegroundColor Red
    exit 1
}

try {
    while ($true) {
        $allExited = $true
        foreach ($proc in $processes) {
            if ($proc -and -not $proc.HasExited) {
                $allExited = $false
            }
        }
        if ($allExited) {
            Write-Host "`nAll processes exited on their own." -ForegroundColor Yellow
            break
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Stop-All
}
