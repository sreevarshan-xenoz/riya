import subprocess

subprocess.run(["kill", "1643"])
subprocess.run(["pkill", "-f", "gen_riya_data_hf.py"])
print(subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.count("gen_riya_data_hf"), "old teacher processes left")
