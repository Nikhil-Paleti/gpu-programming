"""Write pyrightconfig.json pointing pyright at Bazel's editor venv.

Run with the venv's own Python, so every .pth file has already been applied
to sys.path. Pyright can't execute .pth code, so the extra folders are
listed for it as extraPaths.
"""
import json
import os
import sys

venv = sys.argv[1]  # venv folder, relative to the repo root
root = os.getcwd()
ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
site = os.path.join(root, venv, "lib", f"python{ver}", "site-packages")

# Folders inside site-packages that .pth files added to sys.path.
extra = []
for path in sys.path:
    rel = os.path.relpath(path, root)
    if path.startswith(site + os.sep) and rel not in extra:
        extra.append(rel)

config = {
    "venvPath": os.path.dirname(venv),
    "venv": os.path.basename(venv),
    "pythonVersion": ver,
    "extraPaths": extra,
}
with open("pyrightconfig.json", "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
print(f"Wrote pyrightconfig.json ({len(extra)} extra path(s))")
