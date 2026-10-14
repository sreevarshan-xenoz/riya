"""Remove any data/ folder that an older code zip may have carried (licensed lexicons must not be copied around)."""
import os
import shutil

p = "/content/code/data"
print("found" if os.path.exists(p) else "clean", p)
shutil.rmtree(p, ignore_errors=True)
