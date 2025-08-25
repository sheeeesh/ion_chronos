"""
Ion Chronos — Project paths configuration.

Defines the project root and workspace directory, with optional environment overrides.
Creates a ".ops" subdirectory under the workspace for logs and other operational files.
"""
from __future__ import annotations
import os

def _project_root() -> str:
    # Repository root = parent of tools/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _resolve_workspace() -> str:
    # 1) Use env override if set (ION_WORKSPACE or ION_WORKSPACE_ROOT)
    env_dir = os.environ.get("ION_WORKSPACE", "").strip() or os.environ.get("ION_WORKSPACE_ROOT", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    # 2) Default to <repo_root>/workspace
    return os.path.join(_project_root(), "workspace")

WORKSPACE = _resolve_workspace()
OPS_DIR = os.path.join(WORKSPACE, ".ops")

# Ensure workspace and ops directories exist
os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs(OPS_DIR, exist_ok=True)
