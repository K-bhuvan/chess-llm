# Unattended chess SFT/GRPO watchdog. Keep this process alive; it keeps Windows awake
# and restarts WSL training from the last complete checkpoint if the job dies or hangs.
$ErrorActionPreference = "Continue"
$Repo = Split-Path $PSScriptRoot -Parent
$drive = $Repo.Substring(0, 1).ToLower()
$rest = $Repo.Substring(2).Replace('\', '/')
$WslRepo = "/mnt/$drive$rest"
$Tick = "$WslRepo/scripts/watchdog_tick.sh"
$Train = "$WslRepo/scripts/train_full.sh"
$StatusLog = Join-Path $Repo "outputs\watchdog.log"
$StallSeconds = 720
$PollSeconds = 90

New-Item -ItemType Directory -Force -Path (Join-Path $Repo "outputs") | Out-Null

function Write-Status([string]$msg) {
  $line = "{0:u} {1}" -f (Get-Date).ToUniversalTime(), $msg
  Add-Content -Path $StatusLog -Value $line
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ChessExecState {
  [DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$ES_AWAYMODE = [uint32]64
[void][ChessExecState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE)

powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 2>$null | Out-Null
powercfg /SETACTIVE SCHEME_CURRENT | Out-Null

function Get-Tick {
  $raw = wsl -d Ubuntu -- bash $Tick 2>$null
  $map = @{}
  foreach ($line in $raw) {
    if ($line -match '^([A-Z_]+)=(.*)$') { $map[$Matches[1]] = $Matches[2].Trim() }
  }
  return $map
}

function Start-Train {
  Write-Status "starting train_full.sh"
  wsl -d Ubuntu -- bash -c "sed -i 's/\r$//' '$WslRepo/scripts/'*.sh; bash '$WslRepo/scripts/cleanup_incomplete_ckpts.sh'"
  Start-Process -FilePath "wsl.exe" -ArgumentList @("-d","Ubuntu","--","bash",$Train) -WindowStyle Hidden | Out-Null
}

function Recover-Hang {
  Write-Status "hang detected; wsl --shutdown and resume"
  wsl --shutdown
  Start-Sleep -Seconds 10
  Start-Train
}

Write-Status "watchdog started; stall=${StallSeconds}s"
$lastStep = ""
$lastStepAt = Get-Date

while ($true) {
  [void][ChessExecState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE)
  $t = Get-Tick
  $running = $t["RUNNING"]
  $step = $t["STEP"]
  $gpu = $t["GPU"]
  $power = $t["POWER"]
  $age = 0
  [void][int]::TryParse($t["LOG_AGE_SEC"], [ref]$age)
  $complete = $t["COMPLETE"]

  if ($complete -and [int]$complete -gt 0) {
    Write-Status "pipeline complete; watchdog exiting"
    break
  }

  if ($step -and $step -ne $lastStep) {
    $lastStep = $step
    $lastStepAt = Get-Date
    Write-Status "ok step=$step gpu=$gpu power=$power running=$running"
  }

  $stalled = ((Get-Date) - $lastStepAt).TotalSeconds -ge $StallSeconds
  $idleGpu = $false
  $gpuN = 100
  $pwrN = 200
  [void][double]::TryParse($gpu, [ref]$gpuN)
  [void][double]::TryParse($power, [ref]$pwrN)
  if ($gpuN -lt 20 -and $pwrN -lt 40) { $idleGpu = $true }

  if ($running -ne "1") {
    Write-Status "trainer not running; restart"
    Start-Train
    $lastStepAt = Get-Date
  } elseif ($stalled -and $idleGpu) {
    Recover-Hang
    $lastStepAt = Get-Date
  }

  Start-Sleep -Seconds $PollSeconds
}
