# Move the round-2 world runs from the GPU VM to one CPU session each (resume from checkpoint).
$ErrorActionPreference = "Continue"
$root = "E:\own model reseach"
$colab = "/root/.local/bin/colab"
$common = "--episodes 400 --shift_every 30 --shift_jitter 0.4 --save_every 20 --seed 1"
$runs = [ordered]@{
    "w2-none"   = @{ session = "jarvis-cpu";  args = $common }
    "w2-belief" = @{ session = "jarvis-cpu2"; args = "--context belief --arousal_plasticity $common" }
    "w2-oracle" = @{ session = "jarvis-cpu3"; args = "--context oracle $common" }
}
if ($args -notcontains "-skipstop") {
    wsl -d Arch -e $colab exec -s jarvis -f "/mnt/e/own model reseach/colab/stop_world.py"
    & "$root\.venv\Scripts\python.exe" "$root\colab\pull.py"
}
foreach ($run in $runs.Keys) {
    $s = $runs[$run].session
    $sessions = wsl -d Arch -e $colab sessions 2>&1 | Out-String
    if ($sessions -notmatch "\[$s\]") { wsl -d Arch -e $colab new -s $s 2>&1 | Select-Object -Last 1 }
    wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/jarvis.zip" /content/jarvis.zip 2>&1 | Select-Object -Last 1
    foreach ($f in "world.pt", "log.jsonl") {
        $local = "/mnt/e/own model reseach/scratch_runs/colab/$run/$f"
        wsl -d Arch -e $colab upload -s $s $local "/content/${run}_$f" 2>&1 | Select-Object -Last 1
    }
    $script = (Get-Content "$root\colab\resume_world_cpu.py" -Raw).Replace("__RUN__", $run).Replace("__ARGS__", $runs[$run].args)
    $tmp = "$root\colab\_resume_$run.py"
    [IO.File]::WriteAllText($tmp, $script, (New-Object Text.UTF8Encoding $false))  # no BOM (PS 5 utf8 adds one)
    wsl -d Arch -e $colab exec -s $s -f "/mnt/e/own model reseach/colab/_resume_$run.py" 2>&1 | Select-Object -Last 1
}
