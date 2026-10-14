import os, subprocess, zipfile
zipfile.ZipFile('/content/jarvis.zip').extractall('/content/code')
os.makedirs('/content/runs/bench', exist_ok=True)
cmd = ['python', '-m', 'evaluation.run_world', 'train', '--variant', 'oracle', '--episodes', '60',
       '--out', '/content/runs/bench/train-oracle', '--save_minutes', '8']
p = subprocess.Popen(cmd, cwd='/content/code', stdout=open('/content/runs/bench/train-oracle.out', 'a'),
                     stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
print('started train-oracle pid', p.pid)