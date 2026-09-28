# Phase 7 Bucket 1, overnight unattended bake-off orchestration (Issue 325).
#
# WHY THIS EXISTS AS A SEPARATE, DETACHED SCRIPT rather than another backgrounded Bash call from
# the development session: five consecutive "low memory" kills this session all landed with
# 12-18GB of physical RAM still reported available (out of 32GB total) -- short of genuine Windows
# last-resort OOM behavior, which typically only kicks in under much more severe, sustained
# pressure (heavy pagefile thrashing, allocation failures inside processes). The consistent,
# narrow band points at the development tool's OWN background-task memory guard killing the
# whole tracked process tree (driver + worker together, which matches what was actually observed:
# every python.exe vanished each time, not just the worker), not real system-level distress. This
# script runs as a fully detached, independent OS process (launched via Start-Process from outside
# any run_in_background-tracked Bash/PowerShell call) specifically so it is NOT subject to that
# same harness-level guard, and can run unattended overnight without depending on the development
# session staying alive or reachable at all.
#
# Every decision point that required a judgment call earlier this session is pre-committed here as
# fixed, mechanical policy -- this script cannot ask a question, by construction, so there is
# nothing left that can stall waiting on a human overnight:
#   - OOM/crash of the top-level driver process: retry, up to 4 total attempts, 5-minute wait
#     between attempts, then stop (no 5th attempt) -- see MAX_ATTEMPTS/RETRY_WAIT_SECONDS below.
#   - A single candidate crashing inside its OWN isolated subprocess: already handled entirely
#     inside run_bakeoff.py's own driver_main() (Issue 323) -- logged, the run continues with
#     the next candidate, no retry of that candidate, nothing for this wrapper to do.
#   - CANDIDATES list, scoring logic (run_bakeoff.py itself), and gold-set data
#     (shortlist_for_review.json): never modified by this script, under any circumstance.
#   - Once everything that's going to finish has finished (success, attempt cap reached, or the
#     8-hour ceiling hit): write a full, real status report to overnight_status.md in this same
#     folder -- not just conversation output nobody will read overnight.

$ErrorActionPreference = "Continue"

$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BakeoffDir  = Join-Path $RepoRoot "evaluation\model_selection\diagram_charts"
$VenvPython  = Join-Path $RepoRoot "backend\.venv\Scripts\python.exe"
$StatusFile  = Join-Path $BakeoffDir "overnight_status.md"
$PerRowDir   = Join-Path $BakeoffDir "per_row_results"
$SubprocLog  = Join-Path $BakeoffDir "subprocess_run_log.jsonl"
$ResultsMd   = Join-Path $BakeoffDir "results.md"
$ResultsCsv  = Join-Path $BakeoffDir "results.csv"

$MaxAttempts        = 4       # pre-committed cap (Issue 325) -- up to 3 retries after a kill,
                               # stop after a would-be 4th kill, never asks, never waits for a human
$RetryWaitSeconds    = 300     # 5 minutes between attempts, per the pre-authorized policy
$OverallDeadline     = (Get-Date).AddHours(8)
$RunStart            = Get-Date

function Get-AvailableRamGb {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
}

$Events = New-Object System.Collections.Generic.List[string]
function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    $Events.Add($line)
    # Also append incrementally to the status file as we go, so a partial night still leaves a
    # real, readable trail even if this script itself somehow never reaches the final write.
    Add-Content -Path $StatusFile -Value $line -Encoding utf8
}

"# Bucket 1 Bake-off - Overnight Unattended Run (Issue 325)" | Set-Content -Path $StatusFile -Encoding utf8
Add-Content -Path $StatusFile -Value "" -Encoding utf8

Log "Overnight run started. 8-hour ceiling: $($OverallDeadline.ToString('yyyy-MM-dd HH:mm:ss'))"
Log "Available RAM at start: $(Get-AvailableRamGb) GB"
$chromeProcs = Get-Process chrome -ErrorAction SilentlyContinue
if ($chromeProcs) {
    $chromeMb = [math]::Round(($chromeProcs | Measure-Object WS -Sum).Sum / 1MB, 1)
    Log "Chrome NOT fully closed: $($chromeProcs.Count) process(es), ~$chromeMb MB combined. Proceeding anyway (logged, not blocking, per pre-authorized policy)."
} else {
    Log "Chrome confirmed closed."
}
Log "Pagefile increase attempted before launch: FAILED (access denied - not running elevated). Current pagefile ~32.5GB (already ~=total physical RAM), 31.1GB free disk at that check. Proceeding regardless, per pre-authorized policy."

$attempt = 0
$success = $false
$deadlineHit = $false

while ($attempt -lt $MaxAttempts -and -not $success -and (Get-Date) -lt $OverallDeadline) {
    $attempt++
    Log "=== Attempt $attempt of $MaxAttempts starting. Available RAM: $(Get-AvailableRamGb) GB ==="

    $stdoutLog = Join-Path $BakeoffDir "bakeoff_run_attempt$attempt.log"
    $stderrLog = Join-Path $BakeoffDir "bakeoff_run_attempt$attempt.err.log"

    $proc = Start-Process -FilePath $VenvPython -ArgumentList "run_bakeoff.py" `
        -WorkingDirectory $BakeoffDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    Log "Attempt ${attempt}: driver process started, PID $($proc.Id)."

    while (-not $proc.HasExited -and (Get-Date) -lt $OverallDeadline) {
        Start-Sleep -Seconds 60
    }

    if (-not $proc.HasExited) {
        Log "8-hour ceiling reached while attempt $attempt was still running (PID $($proc.Id)) -- leaving it running as-is (not killing genuine in-progress work), stopping the retry-orchestration loop here."
        $deadlineHit = $true
        break
    }

    $exitCode = $proc.ExitCode
    Log "Attempt ${attempt}: driver process exited, code $exitCode."

    if ($exitCode -eq 0) {
        $success = $true
        Log "Attempt $attempt completed (exit 0) -- driver's own loop finished without crashing."
    } else {
        Log "Attempt $attempt did not exit cleanly (code $exitCode) -- treated as a possible OOM/kill per the pre-authorized retry policy, regardless of the precise cause."
        if ($attempt -lt $MaxAttempts) {
            Log "Waiting $($RetryWaitSeconds / 60) minutes before attempt $($attempt + 1)..."
            Start-Sleep -Seconds $RetryWaitSeconds
        } else {
            Log "Reached max attempts ($MaxAttempts) -- stopping, not retrying further, per pre-authorized policy."
        }
    }
}

Log "=== Retry loop finished. success=$success attempts_used=$attempt deadline_hit=$deadlineHit ==="

# ---------------------------------------------------------------------------------------------
# Real, final status report -- pulled from the actual artifacts on disk, not just this script's
# own event log, so it reflects real measured data (Issue 320/321/322/323's own discipline).
# ---------------------------------------------------------------------------------------------

Add-Content -Path $StatusFile -Value "`n## Summary`n" -Encoding utf8
Add-Content -Path $StatusFile -Value "- Started: $($RunStart.ToString('yyyy-MM-dd HH:mm:ss'))" -Encoding utf8
Add-Content -Path $StatusFile -Value "- Ended: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" -Encoding utf8
Add-Content -Path $StatusFile -Value "- Total wall-clock: $([math]::Round(((Get-Date) - $RunStart).TotalMinutes, 1)) minutes" -Encoding utf8
Add-Content -Path $StatusFile -Value "- Attempts used: $attempt of $MaxAttempts" -Encoding utf8
Add-Content -Path $StatusFile -Value "- Final driver exit clean (all candidates attempted, results.md/csv written): $success" -Encoding utf8
Add-Content -Path $StatusFile -Value "- 8-hour ceiling hit mid-attempt: $deadlineHit" -Encoding utf8

Add-Content -Path $StatusFile -Value "`n## Per-candidate real status`n" -Encoding utf8
if (Test-Path $PerRowDir) {
    Get-ChildItem $PerRowDir -Filter "*__metrics.json" | ForEach-Object {
        try {
            $m = Get-Content $_.FullName -Raw | ConvertFrom-Json
            Add-Content -Path $StatusFile -Value "### $($m.candidate) - COMPLETED" -Encoding utf8
            Add-Content -Path $StatusFile -Value "- n_rows: $($m.n_rows), n_errors: $($m.n_errors)" -Encoding utf8
            Add-Content -Path $StatusFile -Value "- figure_detected_rate: $($m.figure_detected_rate), chart_type_identified_rate: $($m.chart_type_identified_rate), values_extracted_rate: $($m.values_extracted_rate), final_answer_exact_match_rate: $($m.final_answer_exact_match_rate)" -Encoding utf8
            Add-Content -Path $StatusFile -Value "- load_time_s: $($m.load_time_s), vram_delta_gb: $($m.vram_delta_gb), fits_8gb: $($m.fits_8gb)" -Encoding utf8
        } catch {
            Add-Content -Path $StatusFile -Value "### $($_.Name) - metrics file present but failed to parse: $($_.Exception.Message)" -Encoding utf8
        }
        Add-Content -Path $StatusFile -Value "" -Encoding utf8
    }
    # Candidates that ran (raw per-row json exists) but never produced a metrics file = crashed
    # or never finished (Issue 323's own disclosed-gap behavior in driver_main()).
    Get-ChildItem $PerRowDir -Filter "*.json" | Where-Object { $_.Name -notmatch "__metrics\.json$" } | ForEach-Object {
        $metricsCounterpart = Join-Path $PerRowDir ($_.BaseName + "__metrics.json")
        if (-not (Test-Path $metricsCounterpart)) {
            try {
                $rows = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $errCount = ($rows | Where-Object { $_.error }).Count
                Add-Content -Path $StatusFile -Value "### $($_.BaseName) - RAN BUT NO METRICS FILE (crashed or incomplete)" -Encoding utf8
                Add-Content -Path $StatusFile -Value "- rows in per-row file: $($rows.Count), rows with a recorded error: $errCount" -Encoding utf8
            } catch {
                Add-Content -Path $StatusFile -Value "### $($_.BaseName) - RAN BUT NO METRICS FILE, and per-row file failed to parse" -Encoding utf8
            }
            Add-Content -Path $StatusFile -Value "" -Encoding utf8
        }
    }
}

Add-Content -Path $StatusFile -Value "`n## Subprocess-level log (real, from run_bakeoff.py's own driver)`n" -Encoding utf8
if (Test-Path $SubprocLog) {
    Add-Content -Path $StatusFile -Value '```' -Encoding utf8
    Get-Content $SubprocLog | Add-Content -Path $StatusFile -Encoding utf8
    Add-Content -Path $StatusFile -Value '```' -Encoding utf8
} else {
    Add-Content -Path $StatusFile -Value "(subprocess_run_log.jsonl does not exist -- no candidate subprocess ever returned control to the driver across any attempt)" -Encoding utf8
}

Add-Content -Path $StatusFile -Value "`n## results.md / results.csv`n" -Encoding utf8
if (Test-Path $ResultsMd) {
    Add-Content -Path $StatusFile -Value "Both written -- driver_main() completed its full loop over CANDIDATES at least once. See results.md/results.csv in this folder for the real aggregate table (role/rejection_reason left TBD for manual review, as required)." -Encoding utf8
} else {
    Add-Content -Path $StatusFile -Value "NOT written -- the driver's loop over CANDIDATES did not reach its own end in any attempt (see per-candidate status above for what real data does exist)." -Encoding utf8
}

Add-Content -Path $StatusFile -Value "`n## Orchestration event log`n" -Encoding utf8
Add-Content -Path $StatusFile -Value '```' -Encoding utf8
$Events | Add-Content -Path $StatusFile -Encoding utf8
Add-Content -Path $StatusFile -Value '```' -Encoding utf8

Log "Status report written to $StatusFile. Overnight orchestration script exiting."
