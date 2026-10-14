# Recreate CPU sessions after Colab reclaims them and resume world runs from the backed-up checkpoints.
$root = "E:\own model reseach"
$colab = "/root/.local/bin/colab"
$common = "--episodes 400 --shift_every 30 --shift_jitter 0.4 --save_every 20 --seed 1 --context"
$runs = [ordered]@{
    "w3-belief"     = @{ session = "jarvis-cpu2"; args = "$common belief --arousal_plasticity" }
    "w3-belief-sal" = @{ session = "jarvis-cpu2"; args = "$common belief --arousal_plasticity --salience 19" }
    "w3-oracle-sal" = @{ session = "jarvis-cpu3"; args = "$common oracle --salience 19" }
}
$made = @{}
foreach ($run in $runs.Keys) {
    $s = $runs[$run].session
    if (-not $made[$s]) {
        $sessions = wsl -d Arch -e $colab sessions 2>&1 | Out-String
        if ($sessions -notmatch "\[$s\]") { wsl -d Arch -e $colab new -s $s 2>&1 | Select-Object -Last 1 }
        wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/jarvis.zip" /content/jarvis.zip 2>&1 | Select-Object -Last 1
        $made[$s] = $true
    }
    foreach ($f in "world.pt", "log.jsonl") {
        wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/scratch_runs/colab/$run/$f" "/content/${run}_$f" 2>&1 | Select-Object -Last 1
    }
    $script = (Get-Content "$root\colab\resume_world_cpu.py" -Raw).Replace("__RUN__", $run).Replace("__ARGS__", $runs[$run].args)
    $script = $script.Replace('"OMP_NUM_THREADS": "2"', '"OMP_NUM_THREADS": "1"')
    [IO.File]::WriteAllText("$root\colab\_resume_$run.py", $script, (New-Object Text.UTF8Encoding $false))
    wsl -d Arch -e $colab exec -s $s -f "/mnt/e/own model reseach/colab/_resume_$run.py" 2>&1 | Select-Object -Last 1
}
