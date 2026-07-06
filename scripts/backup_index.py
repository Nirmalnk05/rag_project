"""Back up the index: chunk store + manifests + a Qdrant snapshot.

    python scripts/backup_index.py [backup_dir]

Copies data/*.json artifacts and triggers a Qdrant snapshot of the collection.
Restore with scripts/restore_index.py.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings

ARTIFACTS = ["chunks.json", "ingest_manifest.json", "context_cache.json",
             "caption_cache.json", "index_meta.json", "acl.json", "versions.json"]


def main() -> int:
    s = get_settings()
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else
                f"backups/{datetime.now():%Y%m%d-%H%M%S}")
    dest.mkdir(parents=True, exist_ok=True)

    for name in ARTIFACTS:
        src = Path("data") / name
        if src.exists():
            shutil.copy2(src, dest / name)
            print(f"  copied {name}")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port)
        snap = client.create_snapshot(collection_name=s.qdrant_collection)
        (dest / "qdrant_snapshot.txt").write_text(str(snap), encoding="utf-8")
        print(f"  qdrant snapshot: {snap}")
    except Exception as e:  # Qdrant not reachable — artifacts still backed up
        print(f"  qdrant snapshot skipped: {e}")

    print(f"[backup] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
