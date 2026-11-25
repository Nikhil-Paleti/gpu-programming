#!/usr/bin/env bash
# Re-lock Python deps with uv, export them for Bazel, rebuild the editor venv,
# and regenerate pyrightconfig.json from it.
set -euo pipefail
cd "$(dirname "$0")/.."

uv lock
uv export --format requirements-txt --no-emit-project --quiet -o requirements_lock.txt
bazel build //tools/py_env

venv=bazel-bin/tools/py_env/py_env.runfiles/_main/tools/py_env/_py_env.venv
"$venv/bin/python3" tools/py_env/gen_pyrightconfig.py "$venv"
