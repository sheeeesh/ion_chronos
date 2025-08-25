"""
Workspace-scoped OS executor for Ion Chronos.

All operations are confined to the project workspace directory for safety.

Capabilities:
- File ops (relative to workspace/):
    mkdir <path>
    ls [path]
    rm <path>            (file)
    rmdir <path>         (empty dir)
    rm -r <path>         (recursive remove dir)
    write <path>\n---\n<content>
    append <path>\n---\n<content>
    read <path>
    touch <path>

- Safe shell exec (cwd=workspace/), allowlisted executables:
    Defaults: python, python3, pip, pip3, git, conda
    Extend via env IONCHRONOS_ALLOWED_BINS (os.pathsep-separated list).

All actions are logged to workspace/.ops/os_exec.log.

Return format:
Returns a plain-text summary and, when relevant, the captured stdout/stderr (truncated for brevity).
"""
from __future__ import annotations

import os
import shlex
import time
import json
import textwrap
import subprocess
import shutil
from typing import Optional, Tuple, List

from tools.io_paths import WORKSPACE, OPS_DIR

LOG_PATH = os.path.join(OPS_DIR, "os_exec.log")
MAX_OUTPUT = int(os.environ.get("IONCHRONOS_MAX_OUTPUT_CHARS", "20000"))
DEFAULT_TIMEOUT = int(os.environ.get("IONCHRONOS_EXEC_TIMEOUT", "60"))

os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs(OPS_DIR, exist_ok=True)

def _log(event: str, **kwargs) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kwargs}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _safe_path(relpath: str) -> str:
    """Resolve a path strictly under workspace/ (no traversal)."""
    relpath = (relpath or "").strip().lstrip("/\\")
    if not relpath:
        raise ValueError("path is empty")
    out = os.path.abspath(os.path.join(WORKSPACE, relpath))
    if not (out == WORKSPACE or out.startswith(WORKSPACE + os.sep)):
        raise ValueError("refusing to access path outside workspace/")
    return out

def _truncate(s: str, n: int = MAX_OUTPUT) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n] + "\n... [output truncated]"

def _result(header: str, body: str = "") -> str:
    return f"{header}\n\n{body}" if body else header

# -------------------------- filesystem ops --------------------------

def _op_mkdir(path: str) -> str:
    p = _safe_path(path)
    os.makedirs(p, exist_ok=True)
    _log("mkdir", path=p)
    return _result("[os_exec] mkdir ✓", f"path: {p}")

def _op_ls(path: Optional[str]) -> str:
    p = _safe_path(path) if path else WORKSPACE
    if not os.path.exists(p):
        return _result("[os_exec] ls ✗", f"not found: {p}")
    if os.path.isfile(p):
        st = os.stat(p)
        info = f"FILE  {p}  ({st.st_size} bytes)"
        _log("ls", path=p, kind="file")
        return _result("[os_exec] ls (file)", info)
    lines: List[str] = []
    for name in sorted(os.listdir(p)):
        full = os.path.join(p, name)
        try:
            st = os.stat(full)
            lines.append(f"[{'DIR' if os.path.isdir(full) else 'FILE'}] {name}{'/' if os.path.isdir(full) else ''}  {st.st_size if not os.path.isdir(full) else ''} bytes".rstrip())
        except Exception:
            lines.append(f"[????] {name}")
    _log("ls", path=p, kind="dir", count=len(lines))
    return _result(f"[os_exec] ls {p}", "\n".join(lines))

def _op_rm(path: str, recursive: bool) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return _result("[os_exec] rm ✗", f"not found: {p}")
    if os.path.isdir(p):
        if recursive:
            shutil.rmtree(p)
            _log("rm -r", path=p)
            return _result("[os_exec] rm -r ✓", f"removed dir: {p}")
        else:
            os.rmdir(p)
            _log("rmdir", path=p)
            return _result("[os_exec] rmdir ✓", f"removed empty dir: {p}")
    else:
        os.remove(p)
        _log("rm", path=p)
        return _result("[os_exec] rm ✓", f"removed file: {p}")

def _op_write(path: str, content: str, append: bool = False) -> str:
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p) or WORKSPACE, exist_ok=True)
    data = content if content is not None else ""
    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8", newline="") as f:
        f.write(data)
    b = len(data.encode("utf-8"))
    _log("append" if append else "write", path=p, bytes=b)
    return _result(f"[os_exec] {'append' if append else 'write'} ✓", f"path: {p}, bytes: {b}")

def _op_read(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return _result("[os_exec] read ✗", f"not found: {p}")
    with open(p, "r", encoding="utf-8", newline="") as f:
        data = f.read()
    _log("read", path=p, bytes=len(data.encode("utf-8")))
    return _result(f"[os_exec] read: {p}", _truncate(data))

def _op_touch(path: str) -> str:
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p) or WORKSPACE, exist_ok=True)
    open(p, "a", encoding="utf-8").close()
    os.utime(p, None)
    _log("touch", path=p)
    return _result("[os_exec] touch ✓", f"path: {p}")

# --------------------------- shell execution ------------------------

def _allowed_bins() -> set:
    """
    Allowed executables:
      - defaults: python, python3, pip, pip3, git, conda
      - plus any from env IONCHRONOS_ALLOWED_BINS
    """
    allowed = {"python", "python3", "pip", "pip3", "git", "conda"}
    extra = os.environ.get("IONCHRONOS_ALLOWED_BINS", "")
    if extra.strip():
        allowed |= {os.path.splitext(os.path.basename(x.strip().lower()))[0] for x in extra.split(os.pathsep) if x.strip()}
    return allowed

def _parse_cmd(cmd: str) -> Tuple[str, list]:
    tokens = shlex.split(cmd, posix=(os.name != "nt"))
    if not tokens:
        raise ValueError("empty command")
    exe = tokens[0]
    base = os.path.splitext(os.path.basename(exe))[0].lower()
    return base, tokens

def _op_shell(cmd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    base, tokens = _parse_cmd(cmd)
    allowed = _allowed_bins()
    if base not in allowed:
        return _result("[os_exec] shell ✗ (blocked)",
                       textwrap.dedent(f"Command not allowed: {base}\nAllowed: {', '.join(sorted(allowed))}\ncwd: {WORKSPACE}"))
    t0 = time.time()
    _log("shell_start", cmd=cmd, cwd=WORKSPACE, base=base, timeout=int(timeout))
    try:
        proc = subprocess.run(tokens, cwd=WORKSPACE, capture_output=True, text=True,
                              timeout=max(1, int(timeout)), shell=False, env=os.environ.copy())
    except subprocess.TimeoutExpired:
        _log("shell_timeout", cmd=cmd, cwd=WORKSPACE, base=base, timeout=int(timeout))
        return _result("[os_exec] shell ✗ timeout", f"timeout: {timeout}s")
    dur = round(time.time() - t0, 3)
    out_text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    _log("shell_done", cmd=cmd, cwd=WORKSPACE, base=base, code=proc.returncode, secs=dur,
         out=len(proc.stdout or ""), err=len(proc.stderr or ""))
    header = f"[os_exec] shell ✓ (exit {proc.returncode}, {dur}s)"
    return _result(header, _truncate(out_text.strip()))

# ------------------------------- tool entry -------------------------------

DELIM = "\n---\n"

def os_exec(command: str) -> str:
    """
    Execute a workspace-scoped operation.

    Accepted forms:
      - mkdir <relpath>
      - ls [relpath]
      - rm <relpath> | rm -r <relpath> | rmdir <relpath>
      - read <relpath> | touch <relpath>
      - write <relpath>\n---\n<content>
      - append <relpath>\n---\n<content>
      - shell: <command>    # runs allowlisted executables in workspace/
    """
    s = (command or "").strip()
    if not s:
        return _result("[os_exec] usage",
                       "mkdir/ls/rm/rmdir/read/touch/write/append/shell — all relative to workspace/")
    if s.startswith("write ") and DELIM in s:
        head, content = s.split(DELIM, 1)
        _, rel = head.split(" ", 1)
        return _op_write(rel.strip(), content, append=False)
    if s.startswith("append ") and DELIM in s:
        head, content = s.split(DELIM, 1)
        _, rel = head.split(" ", 1)
        return _op_write(rel.strip(), content, append=True)
    parts = s.split(" ", 2)
    verb = parts[0].lower()
    try:
        if verb == "mkdir" and len(parts) >= 2:
            return _op_mkdir(parts[1])
        if verb == "ls":
            return _op_ls(parts[1] if len(parts) >= 2 else None)
        if verb in {"rm", "rmdir"} and len(parts) >= 2:
            recursive = (verb == "rm" and parts[1] == "-r")
            if recursive and len(parts) < 3:
                return _result("[os_exec] rm ✗", "usage: rm -r <relpath>")
            path = parts[2] if recursive else parts[1]
            if path == "-r":
                return _result("[os_exec] rm ✗", "usage: rm -r <relpath>")
            return _op_rm(path, recursive)
        if verb == "read" and len(parts) >= 2:
            return _op_read(parts[1])
        if verb == "touch" and len(parts) >= 2:
            return _op_touch(parts[1])
        if verb == "shell:" or s.lower().startswith("shell: "):
            cmd = s[len("shell:"):].strip()
            if not cmd:
                return _result("[os_exec] shell ✗", "usage: shell: <command>")
            if cmd.startswith("[") and "]" in cmd:
                bracket, rest = cmd.split("]", 1)
                kv = bracket.strip("[]").split("=")
                if len(kv) == 2 and kv[0].strip().lower() == "timeout":
                    try:
                        t = int(kv[1].strip())
                        return _op_shell(rest.strip(), timeout=t)
                    except ValueError:
                        pass
            return _op_shell(cmd)
        unknown_lines = [
            "Try one of:",
            "  mkdir <relpath>",
            "  ls [relpath]",
            "  rm <relpath> | rm -r <relpath> | rmdir <relpath>",
            "  read <relpath> | touch <relpath>",
            "  write <relpath>\\n---\\n<content>",
            "  append <relpath>\\n---\\n<content>",
            "  shell: <command>   (allowed bins via IONCHRONOS_ALLOWED_BINS; default: python|python3|pip|pip3|git|conda)"
        ]
        return _result("[os_exec] unknown command", textwrap.dedent("\n".join(unknown_lines)))
    except Exception as e:
        _log("error", inp=s, error=str(e))
        return _result("[os_exec] error", str(e))
