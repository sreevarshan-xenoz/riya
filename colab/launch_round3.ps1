# Round 3: salience-weighted body prediction. One CPU session per run; same seed and (fixed) flip schedule.
$root = "E:\own model reseach"
$colab = "/root/.local/bin/colab"
$common = "--episodes 400 --shift_every 30 --shift_jitter 0.4 --save_every 20 --seed 1 --context"
$runs = [ordered]@{
    # Free tier allows 4 sessions (1 GPU + 3 CPU), so each round-3 run shares a CPU session with a round-2 run.
    "w3-belief"     = @{ session = "jarvis-cpu";  args = "$common belief --arousal_plasticity" }
    "w3-belief-sal" = @{ session = "jarvis-cpu2"; args = "$common belief --arousal_plasticity --salience 19" }
    "w3-oracle-sal" = @{ session = "jarvis-cpu3"; args = "$common oracle --salience 19" }
}
foreach ($run in $runs.Keys) {
    $s = $runs[$run].session
    wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/jarvis.zip" /content/jarvis.zip 2>&1 | Select-Object -Last 1
    $script = (Get-Content "$root\colab\launch_world_cpu.py" -Raw).Replace("__RUN__", $run).Replace("__ARGS__", $runs[$run].args)
    $tmp = "$root\colab\_launch_$run.py"
    [IO.File]::WriteAllText($tmp, $script, (New-Object Text.UTF8Encoding $false))
    wsl -d Arch -e $colab exec -s $s -f "/mnt/e/own model reseach/colab/_launch_$run.py" 2>&1 | Select-Object -Last 1
}
