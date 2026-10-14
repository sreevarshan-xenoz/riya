import subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout)
