import os
import subprocess

print("cores:", os.cpu_count())
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
