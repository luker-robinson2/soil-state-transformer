"""Shared shard machinery for feature extractors: stable task assignment, provenance, status.

Every extractor writes `sites-<task>-of-<n>.parquet` plus a `.json` manifest into one run
directory, so `status` works the same for all of them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .. import config as C


def shard_of(key: str, ntasks: int) -> int:
    """Stable hash assignment: the same key always lands in the same task."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:7], "big") % ntasks


def sha256(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def shard_path(out: Path, task: int, ntasks: int, kind: str = "sites") -> Path:
    return out / f"{kind}-{task:04d}-of-{ntasks:04d}.parquet"


def write(df, out: Path, task: int, ntasks: int, meta: dict) -> None:
    """Atomic write of a shard (tmp + rename) and its manifest."""
    out.mkdir(parents=True, exist_ok=True)
    path = shard_path(out, task, ntasks)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)
    meta = {**meta, "task": task, "ntasks": ntasks, "code_version": code_version()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def status(ntasks: int, out: Path) -> dict:
    """Which shards are done, which are missing, and how many units failed inside done shards."""
    done, failed = [], []
    for t in range(ntasks):
        p = shard_path(out, t, ntasks)
        m = p.with_suffix(".json")
        if p.exists() and m.exists():
            done.append(t)
            failed += json.loads(m.read_text()).get("failed_blocks", [])
    missing = [t for t in range(ntasks) if t not in done]
    return {"ntasks": ntasks, "done": len(done), "missing_tasks": missing, "failed_blocks": len(failed),
            "resubmit_tasks": ",".join(map(str, missing)) or None}
