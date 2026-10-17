# Launch page-27 Stage 0/1 on three Colab CPU sessions, resuming from the checkpoints saved on this PC.
# (No compute on the laptop: it only uploads files and starts the remote jobs.)
$root = "E:\own model reseach"
$colab = "/root/.local/bin/colab"
$roles = [ordered]@{ "jarvis-cpu" = "v0"; "jarvis-cpu2" = "v0.1"; "jarvis-cpu3" = "oracle" }
& "$root\.venv\Scripts\python.exe" "$root\make_zip.py" | Out-Null
foreach ($s in $roles.Keys) {
    $role = $roles[$s]
    $sessions = wsl -d Arch -e $colab sessions 2>&1 | Out-String
    if ($sessions -notmatch "\[$s\]") { wsl -d Arch -e $colab new -s $s 2>&1 | Select-Object -Last 1 }
    wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/jarvis.zip" /content/jarvis.zip 2>&1 | Select-Object -Last 1
    # slim state (no replay buffer; see colab/slim_state.py) uploads fine; full state.pt (~125 MB) is rejected
    $pairs = @{ "state_slim.pt" = "state.pt"; "episodes.jsonl" = "episodes.jsonl" }
    foreach ($src in $pairs.Keys) {
        $local = "/mnt/e/own model reseach/scratch_runs/bench/train-$role/$src"
        wsl -d Arch -e $colab upload -s $s $local "/content/train-${role}_$($pairs[$src])" 2>&1 | Select-Object -Last 1
    }
    $py = (Get-Content "$root\colab\stage0_colab.py" -Raw).Replace("__ROLE__", $role)
    [IO.File]::WriteAllText("$root\colab\_stage0_colab_$role.py", $py, (New-Object Text.UTF8Encoding $false))
    wsl -d Arch -e $colab exec -s $s -f "/mnt/e/own model reseach/colab/_stage0_colab_$role.py" 2>&1 | Select-Object -Last 1
}
