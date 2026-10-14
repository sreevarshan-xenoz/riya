# Page 27 Stage 0: train frozen brains on the train seeds, one CPU session each (resumable: rerun to resume).
$root = "E:\own model reseach"
$colab = "/root/.local/bin/colab"
$jobs = [ordered]@{
    "jarvis-cpu"  = "v0"
    "jarvis-cpu2" = "v0.1"
    "jarvis-cpu3" = "oracle"
}
& "$root\.venv\Scripts\python.exe" "$root\make_zip.py" | Out-Null
foreach ($s in $jobs.Keys) {
    $variant = $jobs[$s]
    $sessions = wsl -d Arch -e $colab sessions 2>&1 | Out-String
    if ($sessions -notmatch "\[$s\]") { wsl -d Arch -e $colab new -s $s 2>&1 | Select-Object -Last 1 }
    wsl -d Arch -e $colab upload -s $s "/mnt/e/own model reseach/jarvis.zip" /content/jarvis.zip 2>&1 | Select-Object -Last 1
    $py = @"
import os, subprocess, zipfile
zipfile.ZipFile('/content/jarvis.zip').extractall('/content/code')
os.makedirs('/content/runs/bench', exist_ok=True)
cmd = ['python', '-m', 'evaluation.run_world', 'train', '--variant', '$variant', '--episodes', '60',
       '--out', '/content/runs/bench/train-$variant', '--save_minutes', '8']
p = subprocess.Popen(cmd, cwd='/content/code', stdout=open('/content/runs/bench/train-$variant.out', 'a'),
                     stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
print('started train-$variant pid', p.pid)
"@
    $tmp = "$root\colab\_stage0_$variant.py"
    [IO.File]::WriteAllText($tmp, $py, (New-Object Text.UTF8Encoding $false))
    wsl -d Arch -e $colab exec -s $s -f "/mnt/e/own model reseach/colab/_stage0_$variant.py" 2>&1 | Select-Object -Last 1
}
