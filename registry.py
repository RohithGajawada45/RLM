"""
Dynamic Document Registry — supports uploads from the UI.

Each document has:
    • id            (auto-generated short hash)
    • filename      (original upload name)
    • path          (where the file lives on disk)
    • title         (human-readable, defaults to filename)
    • description   (rich, auto-generated at upload time)
    • text          (full extracted text, kept in memory)
    • char_count, uploaded_at

Persistence:
    • Files saved to ./uploads/
    • Metadata saved to ./uploads/registry.json
    • On startup, the registry rebuilds itself from disk
      (re-extracting text).

Thread-safety: we use a simple lock for mutations.
The web app is single-process so this is sufficient.
"""

import os
import json
import hashlib
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from document_loader import load_document
from advanced_router import generate_rich_description


UPLOAD_DIR = Path("./uploads")
REGISTRY_FILE = UPLOAD_DIR / "registry.json"


class DynamicRegistry:
    def __init__(self, client):
        self.client = client
        self.docs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        UPLOAD_DIR.mkdir(exist_ok=True)
        self._load_persisted()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """
        Reload docs from disk on startup.
        Re-extract text but reuse cached descriptions
        (no LM cost on restart).
        """
        if not REGISTRY_FILE.exists():
            return

        try:
            data = json.loads(REGISTRY_FILE.read_text())
        except json.JSONDecodeError:
            print("[registry] registry.json corrupt — starting fresh")
            return

        for entry in data.get("docs", []):
            path = entry["path"]

            if not os.path.exists(path):
                continue

            try:
                text = load_document(path)
            except Exception as e:
                print(f"[registry] skipping {path}: {e}")
                continue

            entry["text"] = text
            entry["char_count"] = len(text)
            self.docs[entry["id"]] = entry

        print(f"[registry] loaded {len(self.docs)} doc(s) from disk")

    def _save_persisted(self) -> None:
        """
        Write metadata (NOT the text, NOT the file content)
        to JSON.
        """

        payload = {
            "docs": [
                {k: v for k, v in d.items() if k != "text"}
                for d in self.docs.values()
            ]
        }

        REGISTRY_FILE.write_text(
            json.dumps(payload, indent=2)
        )
        
    # ─── Document CRUD ────────────────────────────────────────────────────────

    def add_uploaded_file(
        self,
        filename: str,
        file_bytes: bytes,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save an uploaded file, extract its text, generate a
        description, and register it.
        Returns the registry entry.
        """
        with self._lock:
            doc_id = self._make_doc_id(filename, file_bytes)

            ext = Path(filename).suffix.lower()
            safe_name = f"{doc_id}{ext}"
            path = UPLOAD_DIR / safe_name

            # Write the file to disk
            path.write_bytes(file_bytes)

            # Extract text
            try:
                text = load_document(str(path))
            except Exception as e:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to extract text: {e}")

            # Generate rich description (one LM call)
            description = generate_rich_description(
                self.client,
                filename,
                text,
            )

            entry = {
                "id": doc_id,
                "filename": filename,
                "path": str(path),
                "title": title or filename,
                "description": description,
                "text": text,
                "char_count": len(text),
                "uploaded_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            self.docs[doc_id] = entry
            self._save_persisted()

            return entry

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            entry = self.docs.pop(doc_id, None)

            if entry is None:
                return False

            try:
                Path(entry["path"]).unlink(missing_ok=True)
            except OSError:
                pass

            self._save_persisted()
            return True

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.docs.get(doc_id)

    def list_for_api(self) -> List[Dict[str, Any]]:
        """
        Returns a list view safe to send over HTTP —
        no 'text' field.
        """
        return [
            {
                "id": d["id"],
                "title": d["title"],
                "filename": d["filename"],
                "description": d["description"],
                "char_count": d["char_count"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in self.docs.values()
        ]

    def catalog_for_router(self) -> List[Dict[str, Any]]:
        """
        Returns minimal catalog dicts the advanced
        router needs.
        """
        return [
            {
                "id": d["id"],
                "title": d["title"],
                "description": d["description"],
            }
            for d in self.docs.values()
        ]
    
        # ─── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_doc_id(filename: str, file_bytes: bytes) -> str:
        """
        Short, stable, content-based id.
        """
        h = hashlib.sha256()
        h.update(filename.encode("utf-8"))
        h.update(file_bytes)
        return h.hexdigest()[:12]