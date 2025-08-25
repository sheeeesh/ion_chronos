"""
Ion Chronos — Workspace-scoped file utilities.

All file operations are confined to the workspace directory (no access outside).

- Default workspace is <project_root>/workspace (override with env ION_WORKSPACE or ION_WORKSPACE_ROOT).
- Safe on Windows/macOS/Linux (common path checks prevent escaping workspace).
- Provided functions: write_file(path, text), append_file(path, text), read_file(path), list_dir(path=""), ensure_dir(path), delete_file(path).
- All operations are logged to workspace/.ops/fs_access.log for transparency.
"""
from __future__ import annotations

import os
import io
import tempfile
import shutil
import json
import time
from typing import List

from tools.io_paths import WORKSPACE, OPS_DIR

def get_workspace_root() -> str:
    """Return absolute workspace root (ensuring it exists)."""
    os.makedirs(WORKSPACE, exist_ok=True)
    return WORKSPACE

def _safe_path(relpath: str) -> str:
    """
    Resolve a *relative* path inside the workspace safely.
    Reject absolute paths or traversal outside the workspace.
    """
    if relpath is None:
        raise ValueError("Path cannot be empty.")
    relpath = str(relpath).strip()
    if not relpath:
        raise ValueError("Path cannot be empty.")
    if os.path.isabs(relpath):
        raise ValueError("Absolute paths are not allowed.")
    rel_norm = os.path.normpath(relpath)
    base = get_workspace_root()
    abs_path = os.path.abspath(os.path.join(base, rel_norm))
    if os.path.commonpath([base, abs_path]) != base:
        raise ValueError("Refusing to access outside workspace/")
    return abs_path

# Internal logger (appends JSON lines to fs_access.log)
_LOG_PATH = os.path.join(OPS_DIR, "fs_access.log")
def _log(event: str, **kwargs) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kwargs}
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as log_f:
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def ensure_dir(relpath: str) -> str:
    """
    Ensure a directory exists in the workspace. Returns the absolute path.
    """
    path = _safe_path(relpath)
    os.makedirs(path, exist_ok=True)
    _log("ensure_dir", path=path)
    return path

def write_file(filename: str, text: str) -> str:
    """
    Write UTF-8 text to workspace/<filename>, creating parent dirs as needed.
    Uses atomic write (temp file replace) to avoid partial files.
    """
    if not filename:
        return "Usage: write_file relative/path.ext\n---\n<content>"
    out_path = _safe_path(filename)
    os.makedirs(os.path.dirname(out_path) or get_workspace_root(), exist_ok=True)
    data = text if text is not None else ""
    b = len(data.encode("utf-8"))
    tmp_dir = os.path.dirname(out_path) or get_workspace_root()
    tmp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=tmp_dir)
    tmp_file.write(data)
    tmp_path = tmp_file.name
    tmp_file.close()
    shutil.move(tmp_path, out_path)
    _log("write_file", path=out_path, bytes=b)
    return f"[file] wrote → {out_path}"

def append_file(filename: str, text: str) -> str:
    """
    Append UTF-8 text to workspace/<filename>, creating file/dirs if needed.
    """
    if not filename:
        return "Usage: append_file relative/path.ext\n---\n<content>"
    out_path = _safe_path(filename)
    os.makedirs(os.path.dirname(out_path) or get_workspace_root(), exist_ok=True)
    data = text if text is not None else ""
    b = len(data.encode("utf-8"))
    with io.open(out_path, "a", encoding="utf-8") as f:
        f.write(data)
    _log("append_file", path=out_path, bytes=b)
    return f"[file] appended → {out_path}"

def read_file(filename: str) -> str:
    """
    Read UTF-8 text from workspace/<filename>.
    """
    if not filename:
        return "Usage: read_file relative/path.ext"
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"[file] not found: {path}"
    with io.open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _log("read_file", path=path, bytes=len(content.encode("utf-8")))
    return content

def delete_file(filename: str) -> str:
    """
    Delete a file inside the workspace.
    """
    if not filename:
        return "Usage: delete_file relative/path.ext"
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"[file] not found: {path}"
    if os.path.isdir(path):
        return f"[file] refusing to delete directory: {path}"
    os.remove(path)
    _log("delete_file", path=path)
    return f"[file] deleted → {path}"

def list_dir(relpath: str = "") -> str:
    """
    List files/folders under a workspace subdirectory (or root if blank).
    Returns a formatted newline-separated listing.
    """
    root = _safe_path(relpath or ".")
    if not os.path.exists(root):
        return f"[file] not found: {root}"
    lines: List[str] = [f"[dir] {root}"]
    for dirpath, dirnames, filenames in os.walk(root):
        rel_root = os.path.relpath(dirpath, get_workspace_root())
        rel_root = "" if rel_root == "." else rel_root
        for d in sorted(dirnames):
            lines.append(f"  <dir> {os.path.join(rel_root, d)}")
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = -1
            lines.append(f"   file {os.path.join(rel_root, fn)}  ({size} bytes)")
    _log("list_dir", path=root, items=max(0, len(lines) - 1))
    return "\n".join(lines)
