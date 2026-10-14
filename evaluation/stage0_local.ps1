# Page-27 Stage 0 (+ Stage 1) on the laptop CPU, gently: max 3 jobs x 2 threads, BelowNormal priority, no GPU.
# Resumable: rerun the script and finished steps are skipped, unfinished runs resume from their state.pt.
$ErrorActionPreference = "Continue"
$root = "E:\own model reseach"
$py = "$root\.venv\Scripts\python.exe"
$out = "$root\scratch_runs\bench"
# Threads per job: 2 when the user is busy on the laptop; 5 when they said the full CPU may be used.
$threads = if ($env:RIYA_THREADS) { [int]$env:RIYA_THREADS } else { 2 }
$evalThreads = [Math]::Max(2, [Math]::Floor(15 / 6))
$env:OMP_NUM_THREADS = "$threads"; $env:MKL_NUM_THREADS = "$threads"; $env:CUDA_VISIBLE_DEVICES = ""
New-Item -ItemType Directory -Force $out | Out-Null

function Run-Batch($jobs, $nThreads = $threads) {
    $procs = @()
    foreach ($j in $jobs) {
        if (Test-Path "$($j.done)") { Write-Host "skip $($j.name) (done)"; continue }
        $argList = "-m " + $j.args + " --threads $nThreads --cpu"
        $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root -NoNewWindow -PassThru `
            -RedirectStandardOutput "$out\$($j.name).out" -RedirectStandardError "$out\$($j.name).err"
        $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        Write-Host "started $($j.name) pid $($p.Id) (BelowNormal, $nThreads threads)"
        $procs += $p
    }
    $procs | ForEach-Object { $_.WaitForExit() }
}

function Train($v) { @{ name = "train-$v"; done = "$out\train-$v\brain.pt";
    args = "evaluation.run_world train --variant $v --episodes 60 --out `"$out\train-$v`" --save_minutes 5" } }
function EvalJob($name, $ckpt, $policy, $extra = "") { @{ name = "eval-$name"; done = "$out\eval-$name\DONE";
    args = "evaluation.run_world eval --ckpt `"$ckpt`" --policy $policy --split test --episodes 60 --out `"$out\eval-$name`" $extra" } }

# 1) frozen brains on train seeds
Run-Batch @((Train "v0"), (Train "v0.1"), (Train "oracle"))
Write-Host "STAGE0_TRAIN_DONE"

# 2) baselines + v0.1 on the 20 locked test seeds (frozen weights)
$b = "$out\train-v0\brain.pt"; $b1 = "$out\train-v0.1\brain.pt"; $bo = "$out\train-oracle\brain.pt"
$evals = @(
    @{ name = "eval-B0-random"; done = "$out\eval-B0-random\DONE"; args = "evaluation.run_world eval --policy random --split test --episodes 60 --out `"$out\eval-B0-random`"" },
    (EvalJob "B1-s1" $b "s1"), (EvalJob "B2-v0" $b "full"), (EvalJob "B3-memory" $b "memory"),
    (EvalJob "v0.1" $b1 "full"), (EvalJob "B4-oracle" $bo "full"))
if ($threads -ge 4) { Run-Batch $evals $evalThreads }  # full CPU allowed: all evals at once
else { for ($i = 0; $i -lt $evals.Count; $i += 3) { Run-Batch $evals[$i..([Math]::Min($i + 2, $evals.Count - 1))] } }
Get-ChildItem "$out\eval-*" -Directory | ForEach-Object { if (Select-String -Path "$out\$($_.Name).out" -Pattern RUN_DONE -Quiet) { New-Item "$($_.FullName)\DONE" -Force | Out-Null } }
Write-Host "STAGE0_EVAL_DONE"

# 3) Stage 1: fit + calibrate the confidence gate (validation seeds), then gated vs always-MPC on test seeds
foreach ($v in "v0", "v0.1") {
    if (-not (Test-Path "$out\gate-$v.pt")) {
        $p = Start-Process -FilePath $py -ArgumentList "-m evaluation.fit_gate --train_dir `"$out\train-$v`" --out `"$out\gate-$v.pt`"" `
            -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$out\gate-$v.out" -RedirectStandardError "$out\gate-$v.err"
        $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        $p.WaitForExit()
    }
}
Run-Batch @((EvalJob "S1gate-v0.1" $b1 "gated" "--gate `"$out\gate-v0.1.pt`""), (EvalJob "mpc-v0.1" $b1 "mpc"),
            (EvalJob "S1gate-v0" $b "gated" "--gate `"$out\gate-v0.pt`""))
Get-ChildItem "$out\eval-*" -Directory | ForEach-Object { if (Select-String -Path "$out\$($_.Name).out" -Pattern RUN_DONE -Quiet) { New-Item "$($_.FullName)\DONE" -Force | Out-Null } }
& $py -m evaluation.metrics (Get-ChildItem "$out\eval-*\episodes.jsonl" | ForEach-Object FullName) --baseline "$out\eval-B2-v0\episodes.jsonl" *> "$out\scorecard.txt"
Get-Content "$out\scorecard.txt"
Write-Host "ALL_DONE"
