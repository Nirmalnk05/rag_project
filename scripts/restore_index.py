"""Restore index artifacts from a backup directory.

    python scripts/restore_index.py <backup_dir>

Copies the chunk store + manifests back into data/. The Qdrant collection itself
is restored from its snapshot via Qdrant's recover API / dashboard (see the
snapshot reference in the backup dir); or simply run `scripts/reembed.py` to
rebuild vectors from the restored chunks.json.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACTS = ["chunks.json", "ingest_manifest.json", "context_cache.json",
             "caption_cache.json", "index_meta.json", "acl.json", "versions.json"]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/restore_index.py <backup_dir>")
        return 2
    src = Path(sys.argv[1])
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 2
    Path("data").mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        f = src / name
        if f.exists():
            shutil.copy2(f, Path("data") / name)
            print(f"  restored {name}")
    print("[restore] artifacts restored. If Qdrant is empty, run: python scripts/reembed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
