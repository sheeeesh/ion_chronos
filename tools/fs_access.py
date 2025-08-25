"""
Ion Chronos — Workspace-scoped filesystem helpers.

All operations are strictly confined to the project workspace directory.

Features:
- Text/Binary: write_text(path, text), append_text(path, text), read_text(path),
               write_bytes_b64(path, b64_string), read_bytes_b64(path)
- Paths & metadata: mkdirs(path), ls(path=None), exists(path), stat(path), glob_paths(pattern)
- Modify: rm(path), rmdir(path), rm_r(path), cp(src, dst, overwrite=False), mv(src, dst, overwrite=False)
- Archives & hashes: zip_paths(paths, archive_path), unzip(archive_path, dest="."), file_hash(path, algo="sha256")

Notes:
- All path arguments are relative to the workspace/.
- Binary read/write uses Base64 to safely handle content.
- Operations are logged to workspace/.ops/fs_access.log.
- Large writes are limited by IONCHRONOS_MAX_FILE_MB (default 100MB).
"""
from __future__ import annotations

import os
import json
import time
import glob as _glob
import base64
import shutil
import hashlib
import zipfile
import tempfile
from typing import Iterable, List, Optional, Dict

from tools.io_paths import WORKSPACE, OPS_DIR

LOG_PATH = os.path.join(OPS_DIR, "fs_access.log")
MAX_FILE_MB = int(os.environ.get("IONCHRONOS_MAX_FILE_MB", "100"))

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
    relpath = (relpath or "").strip().lstrip("/\\")
    if not relpath:
        raise ValueError("path is empty")
    out_path = os.path.abspath(os.path.join(WORKSPACE, relpath))
    if not (out_path == WORKSPACE or out_path.startswith(WORKSPACE + os.sep)):
        raise ValueError("refusing to access path outside workspace/")
    return out_path

def _enforce_size_limit(byte_len: int) -> None:
    if byte_len > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"refusing write > {MAX_FILE_MB}MB (got {byte_len} bytes)")

def _pretty_ls(path: str) -> str:
    items: List[str] = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        try:
            st = os.stat(full)
            if os.path.isdir(full):
                items.append(f"[DIR ] {name}/")
            else:
                items.append(f"[FILE] {name}  {st.st_size} bytes")
        except Exception:
            items.append(f"[????] {name}")
    return "\n".join(items)

# -------------------------- public API: text / binary --------------------------

def write_text(path: str, text: str) -> str:
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p) or WORKSPACE, exist_ok=True)
    data = text if text is not None else ""
    b = len(data.encode("utf-8"))
    _enforce_size_limit(b)
    tmp_dir = os.path.dirname(p) or WORKSPACE
    tmp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=tmp_dir)
    tmp_file.write(data)
    tmp_path = tmp_file.name
    tmp_file.close()
    shutil.move(tmp_path, p)
    _log("write_text", path=p, bytes=b)
    return f"[fs] wrote text → {p} ({b} bytes)"

def append_text(path: str, text: str) -> str:
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p) or WORKSPACE, exist_ok=True)
    data = text if text is not None else ""
    b = len(data.encode("utf-8"))
    _enforce_size_limit(b)
    with open(p, "a", encoding="utf-8", newline="") as f:
        f.write(data)
    _log("append_text", path=p, bytes=b)
    return f"[fs] appended text → {p} (+{b} bytes)"

def read_text(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return f"[fs] not found: {p}"
    with open(p, "r", encoding="utf-8", newline="") as f:
        data = f.read()
    _log("read_text", path=p, bytes=len(data.encode("utf-8")))
    return data

def write_bytes_b64(path: str, b64_string: str) -> str:
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p) or WORKSPACE, exist_ok=True)
    raw = base64.b64decode(b64_string or "", validate=True)
    _enforce_size_limit(len(raw))
    tmp_dir = os.path.dirname(p) or WORKSPACE
    tmp_file = tempfile.NamedTemporaryFile("wb", delete=False, dir=tmp_dir)
    tmp_file.write(raw)
    tmp_path = tmp_file.name
    tmp_file.close()
    shutil.move(tmp_path, p)
    _log("write_bytes", path=p, bytes=len(raw))
    return f"[fs] wrote bytes → {p} ({len(raw)} bytes)"

def read_bytes_b64(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[fs] not found: {p}")
    with open(p, "rb") as f:
        raw = f.read()
    _log("read_bytes", path=p, bytes=len(raw))
    return base64.b64encode(raw).decode("ascii")

# -------------------------- public API: paths & metadata --------------------------

def mkdirs(path: str) -> str:
    p = _safe_path(path)
    os.makedirs(p, exist_ok=True)
    _log("mkdirs", path=p)
    return f"[fs] mkdirs → {p}"

def ls(path: Optional[str] = None) -> str:
    p = _safe_path(path) if path else WORKSPACE
    if not os.path.exists(p):
        return f"[fs] not found: {p}"
    if os.path.isfile(p):
        st = os.stat(p)
        _log("ls", path=p, kind="file")
        return f"FILE  {p}  ({st.st_size} bytes)"
    out = _pretty_ls(p)
    _log("ls", path=p, kind="dir")
    return out

def exists(path: str) -> bool:
    p = _safe_path(path)
    ok = os.path.exists(p)
    _log("exists", path=p, exists=ok)
    return ok

def stat(path: str) -> Dict:
    p = _safe_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[fs] not found: {p}")
    st = os.stat(p)
    info = {
        "path": p,
        "is_dir": os.path.isdir(p),
        "is_file": os.path.isfile(p),
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
    }
    _log("stat", **info)
    return info

def glob_paths(pattern: str) -> List[str]:
    patt = (pattern or "").strip().lstrip("/\\")
    if not patt:
        return []
    abs_pattern = os.path.join(WORKSPACE, patt)
    matches = [os.path.abspath(m) for m in _glob.glob(abs_pattern, recursive=True)]
    filtered = [m for m in matches if m == WORKSPACE or m.startswith(WORKSPACE + os.sep)]
    _log("glob", pattern=pattern, count=len(filtered))
    return [os.path.relpath(m, WORKSPACE) for m in filtered]

# -------------------------- public API: modify --------------------------

def rm(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return f"[fs] not found: {p}"
    if os.path.isdir(p):
        return f"[fs] is a directory (use rmdir or rm_r): {p}"
    os.remove(p)
    _log("rm", path=p)
    return f"[fs] removed file → {p}"

def rmdir(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return f"[fs] not found: {p}"
    if not os.path.isdir(p):
        return f"[fs] not a directory: {p}"
    os.rmdir(p)
    _log("rmdir", path=p)
    return f"[fs] removed empty dir → {p}"

def rm_r(path: str) -> str:
    p = _safe_path(path)
    if not os.path.exists(p):
        return f"[fs] not found: {p}"
    shutil.rmtree(p)
    _log("rm_r", path=p)
    return f"[fs] removed recursively → {p}"

def _prepare_dst_dir(dst_abs: str) -> None:
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)

def cp(src: str, dst: str, overwrite: bool = False) -> str:
    s = _safe_path(src)
    d = _safe_path(dst)
    if not os.path.exists(s):
        return f"[fs] not found: {s}"
    if os.path.exists(d) and not overwrite:
        return f"[fs] exists (set overwrite=True): {d}"
    _prepare_dst_dir(d)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)
    _log("cp", src=s, dst=d, overwrite=overwrite)
    return f"[fs] copied → {s} -> {d}"

def mv(src: str, dst: str, overwrite: bool = False) -> str:
    s = _safe_path(src)
    d = _safe_path(dst)
    if not os.path.exists(s):
        return f"[fs] not found: {s}"
    if os.path.exists(d) and not overwrite:
        return f"[fs] exists (set overwrite=True): {d}"
    _prepare_dst_dir(d)
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
    shutil.move(s, d)
    _log("mv", src=s, dst=d, overwrite=overwrite)
    return f"[fs] moved → {s} -> {d}"

# -------------------------- archives & hashes --------------------------

def _safe_extract_zip(zf: zipfile.ZipFile, dest_abs: str) -> None:
    for member in zf.infolist():
        out_path = os.path.abspath(os.path.join(dest_abs, member.filename.lstrip("/\\")))
        if not (out_path == WORKSPACE or out_path.startswith(WORKSPACE + os.sep)):
            raise ValueError(f"refusing to extract outside workspace: {member.filename}")
        if member.is_dir():
            os.makedirs(out_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

def zip_paths(paths: Iterable[str], archive_path: str) -> str:
    dest = _safe_path(archive_path)
    os.makedirs(os.path.dirname(dest) or WORKSPACE, exist_ok=True)
    count = 0
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            p = _safe_path(rel)
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        abs_fp = os.path.join(root, fn)
                        arcname = os.path.relpath(abs_fp, WORKSPACE)
                        zf.write(abs_fp, arcname)
                        count += 1
            elif os.path.isfile(p):
                arcname = os.path.relpath(p, WORKSPACE)
                zf.write(p, arcname)
                count += 1
    _log("zip", dst=dest, files=count)
    return f"[fs] zipped {count} file(s) → {dest}"

def unzip(archive_path: str, dest: str = ".") -> str:
    src = _safe_path(archive_path)
    if not os.path.exists(src):
        return f"[fs] not found: {src}"
    dest_abs = _safe_path(dest)
    os.makedirs(dest_abs, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        _safe_extract_zip(zf, dest_abs)
        count = len(zf.infolist())
    _log("unzip", src=src, dest=dest_abs, items=count)
    return f"[fs] unzipped {count} item(s) → {dest_abs}"

def file_hash(path: str, algo: str = "sha256") -> str:
    p = _safe_path(path)
    if not os.path.exists(p) or not os.path.isfile(p):
        return f"[fs] not a file: {p}"
    try:
        h = hashlib.new(algo)
    except Exception:
        raise ValueError(f"unsupported hash algo: {algo}")
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _log("hash", path=p, algo=algo, hex=digest)
    return f"{algo}:{digest}"
