lines = open("/content/runs/llamacpp.out", errors="replace").read().splitlines()
print("\n".join(line[:160] for line in lines[-6:]))
