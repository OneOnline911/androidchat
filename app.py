from __future__ import annotations

import copy
import html
import json
import re
import requests
from types import SimpleNamespace
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from anthropic import Anthropic
from PySide6.QtCore import QMimeData, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QKeySequence, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

MODELS = ["claude-opus-4-8", "claude-opus-5", "deepseek-v4-flash", "glm-5.3", "moonshotai/kimi-k3"]
DEFAULT_MODEL = "claude-opus-5"
BASE_URL = "https://agentrouter.org"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_TOOL_ROUNDS = 100
MAX_INCOMPLETE_RETRIES = 2
MAX_TRANSPORT_RETRIES = 2
SEARCH_RESULT_GUARD_CHARS = 50_000
KEEP_FULL_ASSISTANT_TURNS = 2
PASTE_FILE_THRESHOLD = 1200
ZIP_MAX_FILES = 1000
ZIP_MAX_UNCOMPRESSED = 512 * 1024 * 1024

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "claude-opus-5": {
        "max_tokens": 128000,
        "context_window": 1_000_000,
        "effort": "medium",
        "adaptive_thinking": True,
        "prompt_cache": True,
    },
    "claude-opus-4-8": {
        "max_tokens": 128000,
        "context_window": 1_000_000,
        "effort": "medium",
        "adaptive_thinking": True,
        "prompt_cache": True,
    },
    # These AgentRouter routes previously worked with effort=max. Keep their
    # known-good request shape separate from Claude instead of sharing one global effort.
    "glm-5.3": {
        "max_tokens": 128000,
        "context_window": None,
        "effort": "max",
        "adaptive_thinking": True,
        "prompt_cache": False,
    },
    "deepseek-v4-flash": {
        "max_tokens": 128000,
        "context_window": None,
        "effort": "max",
        "adaptive_thinking": True,
        "prompt_cache": False,
    },
    "moonshotai/kimi-k3": {
        "provider": "nvidia",
        "max_tokens": 65536,
        "context_window": 1_048_576,
        "effort": "max",
        "adaptive_thinking": False,
        "prompt_cache": False,
    },
}

SYSTEM_PROMPT = (
    "Attachments are [file ID: name; lines=N; chars=N]. The model chooses what to inspect; file_read only retrieves it. "
    "If a task already requires a whole file, read it whole (omit start/end), not page-by-page. "
    "If several files/ranges/searches are already known to be needed, batch them in one file_read call; use another model round only when the next operation depends on prior results. "
    "Ranges are 1-based inclusive. Use file_save for text files or ZIPs from existing file IDs."
)

PROMPT_CACHE_DISABLED_MODELS: set[str] = set()
PROMPT_CACHE_TTL_BY_MODEL: dict[str, str] = {"claude-opus-5": "1h", "claude-opus-4-8": "1h"}
MODEL_CONTROLS_DISABLED_MODELS: set[str] = set()

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
STATE_FILE = ROOT / "chat_state.json"
AGENTROUTER_API_KEY_FILE = ROOT / "api_key_agentrouter.txt"
LEGACY_API_KEY_FILE = ROOT / "api_key.txt"
NVIDIA_API_KEY_FILE = ROOT / "api_key_nvidia.txt"
TRACE_FILE = ROOT / "agent_trace.jsonl"

FILES_DIR.mkdir(exist_ok=True)


def trace_event(event: str, **data: Any) -> None:
    """Append one compact diagnostic event. Logging must never break the chat."""
    try:
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event}
        record.update(data)
        with TRACE_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


def compact_tool_input(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "file_read":
        items = args.get("items")
        if isinstance(items, list):
            return {
                "items": [
                    {
                        key: item[key]
                        for key in ("op", "id", "q", "mode", "full", "context", "case_sensitive", "offset", "match_offset", "start", "end")
                        if isinstance(item, dict) and key in item
                    }
                    for item in items
                ]
            }
        return {"invalid_input_keys": sorted(args)}
    if name == "file_save":
        result: dict[str, Any] = {"name": args.get("name")}
        if "files" in args:
            result["files"] = args.get("files")
        if "text" in args:
            result["text_chars"] = len(str(args.get("text") or ""))
        return result
    return {"keys": sorted(args)}


def compact_tool_result(name: str, args: dict[str, Any], result: dict[str, Any], serialized_chars: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"serialized_chars": serialized_chars}
    if "error" in result:
        summary["error"] = str(result["error"])
    if name == "file_read":
        requested = args.get("items") if isinstance(args.get("items"), list) else []
        returned = result.get("results") if isinstance(result.get("results"), list) else []
        item_summaries: list[dict[str, Any]] = []
        for index, item in enumerate(requested):
            request = item if isinstance(item, dict) else {}
            response = returned[index] if index < len(returned) and isinstance(returned[index], dict) else {}
            entry: dict[str, Any] = {
                "op": request.get("op"),
                "id": request.get("id"),
            }
            if "error" in response:
                entry["error"] = str(response["error"])
            if request.get("op") == "search":
                entry["mode"] = response.get("mode", request.get("mode", "phrase"))
                entry["matches"] = response.get("match_count", len(response.get("matches") or []))
                entry["result_too_large"] = bool(response.get("result_too_large"))
                if response.get("result_chars") is not None:
                    entry["result_chars"] = response.get("result_chars")
                if request.get("full"):
                    entry["full"] = True
            elif request.get("op") == "read":
                entry["requested_start"] = request.get("start")
                entry["requested_end"] = request.get("end")
                entry["returned_start"] = response.get("start")
                entry["returned_end"] = response.get("end")
                entry["text_chars"] = len(str(response.get("text") or ""))
            item_summaries.append(entry)
        summary["items"] = item_summaries
    elif name == "file_save":
        summary["saved"] = bool(result.get("saved"))
        summary["id"] = result.get("id")
        summary["name"] = result.get("name")
    return summary

FILE_READ_TOOL = {
    "name": "file_read",
    "description": (
        "Read or search chat-local text files; put independent operations in items. "
        "For read, omit start/end for the whole file; explicit ranges are 1-based inclusive and end is clamped to EOF. "
        "For search, mode defaults to phrase; all requires every term/quoted phrase, any is OR, regex is regex. "
        "Search is case-insensitive unless case_sensitive=true; context adds merged surrounding lines. "
        "Oversized results report count/locations; full=true retrieves transport-sized pages via offset/next_offset."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "One or more independent file operations, executed together in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"enum": ["search", "read"]},
                        "id": {"type": "integer"},
                        "q": {"type": "string", "description": "Search text or regex pattern."},
                        "mode": {
                            "enum": ["phrase", "all", "any", "regex"],
                            "description": "Search semantics. Defaults to phrase. In all/any, double-quoted groups are phrases."
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Search only. Defaults to false."
                        },
                        "context": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Search only. Surrounding lines before/after each hit; overlapping windows are merged. Defaults to 0."
                        },
                        "full": {
                            "type": "boolean",
                            "description": "Search only. Explicitly retrieve a large result in transport-sized pages rather than only its summary/locations."
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Search only with full=true. Continue from this zero-based result-unit offset returned as next_offset."
                        },
                        "start": {
                            "type": "integer",
                            "description": "Read only. 1-based inclusive first line. Omit for line 1."
                        },
                        "end": {
                            "type": "integer",
                            "description": "Read only. 1-based inclusive last line. Omit for EOF; values beyond EOF are clamped."
                        },
                    },
                    "required": ["op", "id"],
                },
            },
        },
        "required": ["items"],
    },
}

FILE_SAVE_TOOL = {
    "name": "file_save",
    "description": (
        "Create a local artifact in this chat. Provide text to write one text file, or provide files with existing file IDs to create a ZIP; do not provide both. "
        "In ZIP mode the name is normalized to end in .zip. A successful save returns a new file ID that can be read or included in a later ZIP."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Requested output filename."},
            "text": {"type": "string", "description": "Exact text content for a single generated text file."},
            "files": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Existing file IDs to package into a ZIP."
            },
        },
        "required": ["name"],
    },
}

TOOLS = [FILE_READ_TOOL, FILE_SAVE_TOOL]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_api_key() -> str:
    # One-time local migration from the old single-provider filename.
    if not AGENTROUTER_API_KEY_FILE.exists() and LEGACY_API_KEY_FILE.exists():
        LEGACY_API_KEY_FILE.rename(AGENTROUTER_API_KEY_FILE)
    if not AGENTROUTER_API_KEY_FILE.exists():
        raise RuntimeError("api_key_agentrouter.txt not found")
    key = AGENTROUTER_API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("api_key_agentrouter.txt is empty")
    return key


def load_nvidia_api_key() -> str:
    if not NVIDIA_API_KEY_FILE.exists():
        raise RuntimeError("api_key_nvidia.txt not found")
    key = NVIDIA_API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("api_key_nvidia.txt is empty")
    return key


def block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if hasattr(block, "__dict__"):
        return {k: v for k, v in vars(block).items() if v is not None}
    return dict(block)


def markdown_html(text: str) -> str:
    doc = QTextDocument()
    font = QFont("Segoe UI")
    font.setPointSize(10)
    doc.setDefaultFont(font)
    doc.setMarkdown(text or "")
    return doc.toHtml()


class State:
    def __init__(self) -> None:
        self.chats: dict[int, dict[str, Any]] = {}
        self.files: dict[int, dict[str, Any]] = {}
        self.current_chat_id = 0
        self.next_chat_id = 1
        self.next_file_id = 1
        self.next_archive_id = 1
        self.selected_model = DEFAULT_MODEL
        self.text_cache: dict[int, str] = {}
        self.load()

        if not self.chats:
            self.create_chat(save=False)

    def load(self) -> None:
        if not STATE_FILE.exists():
            return

        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        if "chats" not in data:
            messages = data.get("messages", [])
            old_files = {int(k): v for k, v in data.get("files", {}).items()}
            self.chats = {
                1: {
                    "title": "Chat 1",
                    "messages": messages,
                    "file_ids": sorted(old_files),
                    "created": now(),
                }
            }
            for file_id, meta in old_files.items():
                meta = dict(meta)
                meta["chat_id"] = 1
                self.files[file_id] = meta
            self.current_chat_id = 1
            self.next_chat_id = 2
            self.next_file_id = int(data.get("next_file_id", max(old_files, default=0) + 1))
            self.next_archive_id = int(data.get("next_archive_id", 1))
            self.selected_model = data.get("selected_model", DEFAULT_MODEL)
            if self.selected_model not in MODELS:
                self.selected_model = DEFAULT_MODEL
            self.save()
            return

        self.chats = {int(k): v for k, v in data.get("chats", {}).items()}
        self.files = {int(k): v for k, v in data.get("files", {}).items()}
        self.current_chat_id = int(data.get("current_chat_id", 0))
        self.next_chat_id = int(data.get("next_chat_id", max(self.chats, default=0) + 1))
        self.next_file_id = int(data.get("next_file_id", max(self.files, default=0) + 1))
        self.next_archive_id = int(data.get("next_archive_id", 1))
        self.selected_model = data.get("selected_model", DEFAULT_MODEL)
        if self.selected_model not in MODELS:
            self.selected_model = DEFAULT_MODEL

        if self.current_chat_id not in self.chats and self.chats:
            self.current_chat_id = sorted(self.chats)[0]

    def save(self) -> None:
        payload = {
            "current_chat_id": self.current_chat_id,
            "next_chat_id": self.next_chat_id,
            "next_file_id": self.next_file_id,
            "next_archive_id": self.next_archive_id,
            "selected_model": self.selected_model,
            "chats": self.chats,
            "files": self.files,
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(STATE_FILE)

    def create_chat(self, save: bool = True) -> int:
        chat_id = self.next_chat_id
        self.next_chat_id += 1
        self.chats[chat_id] = {
            "title": f"Chat {chat_id}",
            "messages": [],
            "file_ids": [],
            "created": now(),
        }
        self.current_chat_id = chat_id
        if save:
            self.save()
        return chat_id

    def set_current_chat(self, chat_id: int) -> None:
        if chat_id not in self.chats:
            raise ValueError("chat not found")
        self.current_chat_id = chat_id
        self.save()

    def set_model(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError("unsupported model")
        self.selected_model = model
        self.save()

    def rename_chat(self, chat_id: int, title: str) -> None:
        title = re.sub(r"\s+", " ", title).strip()
        if not title:
            return
        self.chats[chat_id]["title"] = title[:80]
        self.save()

    def delete_chat(self, chat_id: int) -> None:
        if chat_id not in self.chats:
            return

        file_ids = list(self.chats[chat_id].get("file_ids", []))
        del self.chats[chat_id]

        for file_id in file_ids:
            self.files.pop(file_id, None)
            self.text_cache.pop(file_id, None)

        if not self.chats:
            self.create_chat(save=False)
        else:
            self.current_chat_id = sorted(self.chats)[0]

        self.save()

    def chat(self, chat_id: int) -> dict[str, Any]:
        return self.chats[chat_id]

    def maybe_title_chat(self, chat_id: int, text: str, file_ids: list[int]) -> None:
        chat = self.chat(chat_id)
        if chat["messages"]:
            return

        candidate = text.strip().splitlines()[0] if text.strip() else ""
        if not candidate and file_ids:
            candidate = self.files[file_ids[0]]["name"]
        if candidate:
            candidate = re.sub(r"\s+", " ", candidate).strip()
            chat["title"] = candidate[:42]

    def new_path(self, original_name: str, kind: str) -> tuple[int, Path]:
        file_id = self.next_file_id
        self.next_file_id += 1

        source = Path(original_name)
        suffix = source.suffix or ".txt"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("_")[:80] or kind
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return file_id, FILES_DIR / f"{stem}_{file_id:04d}_{stamp}{suffix}"

    def register_file(
        self,
        chat_id: int,
        file_id: int,
        target: Path,
        name: str,
        kind: str,
    ) -> None:
        self.files[file_id] = {
            "name": name,
            "path": str(target),
            "kind": kind,
            "chat_id": chat_id,
            "created": now(),
        }
        self.chat(chat_id)["file_ids"].append(file_id)

    def archive_folder(self, source: Path) -> tuple[int, Path]:
        archive_id = self.next_archive_id
        self.next_archive_id += 1
        stem = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            source.stem,
        ).strip("_")[:80] or "archive"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = FILES_DIR / f"{stem}_{archive_id:04d}_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        return archive_id, folder

    def add_zip_attachment(self, chat_id: int, source: Path) -> list[int]:
        archive_id, folder = self.archive_folder(source)
        file_ids: list[int] = []

        try:
            with zipfile.ZipFile(source) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]

                if len(infos) > ZIP_MAX_FILES:
                    raise ValueError(
                        f"ZIP contains too many files ({len(infos)} > {ZIP_MAX_FILES})"
                    )

                total_size = sum(info.file_size for info in infos)
                if total_size > ZIP_MAX_UNCOMPRESSED:
                    raise ValueError(
                        "ZIP uncompressed size exceeds the configured limit"
                    )

                for info in infos:
                    raw_name = info.filename.replace("\\", "/")
                    relative = PurePosixPath(raw_name)

                    if (
                        not raw_name
                        or relative.is_absolute()
                        or ".." in relative.parts
                    ):
                        raise ValueError(f"Unsafe ZIP path: {info.filename}")

                    target = folder.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)

                    with archive.open(info, "r") as source_stream:
                        with target.open("wb") as target_stream:
                            shutil.copyfileobj(source_stream, target_stream)

                    file_id = self.next_file_id
                    self.next_file_id += 1

                    display_name = f"{source.name}/{relative.as_posix()}"
                    self.register_file(
                        chat_id,
                        file_id,
                        target,
                        display_name,
                        "archive_member",
                    )
                    self.files[file_id]["archive_id"] = archive_id
                    self.files[file_id]["archive_name"] = source.name
                    self.files[file_id]["relative_path"] = relative.as_posix()
                    self.files[file_id]["archive_root"] = str(folder)
                    file_ids.append(file_id)

        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            for file_id in file_ids:
                self.files.pop(file_id, None)
                if file_id in self.chat(chat_id)["file_ids"]:
                    self.chat(chat_id)["file_ids"].remove(file_id)
            raise

        self.save()
        return file_ids

    def add_attachment(self, chat_id: int, source: Path) -> list[int]:
        if source.suffix.lower() == ".zip":
            return self.add_zip_attachment(chat_id, source)

        file_id, target = self.new_path(source.name, "file")
        shutil.copy2(source, target)
        self.register_file(chat_id, file_id, target, source.name, "attachment")
        self.save()
        return [file_id]

    def add_paste(self, chat_id: int, text: str) -> int:
        file_id, target = self.new_path("paste.txt", "paste")
        target.write_text(text, encoding="utf-8")
        self.register_file(chat_id, file_id, target, "paste.txt", "paste")
        self.text_cache[file_id] = text
        self.save()
        return file_id

    def save_generated(self, chat_id: int, name: str, text: str) -> int:
        file_id, target = self.new_path(name or "generated.txt", "generated")
        target.write_text(text, encoding="utf-8")
        self.register_file(
            chat_id,
            file_id,
            target,
            name or target.name,
            "generated",
        )
        self.text_cache[file_id] = text
        self.save()
        return file_id

    def save_generated_zip(
        self,
        chat_id: int,
        name: str,
        file_ids: list[int],
    ) -> int:
        if not file_ids:
            raise ValueError("ZIP requires at least one file id")

        for file_id in file_ids:
            if file_id not in self.chat(chat_id)["file_ids"]:
                raise ValueError(f"file {file_id} is not available in this chat")

        zip_name = name if name.lower().endswith(".zip") else f"{name}.zip"
        file_id, target = self.new_path(zip_name, "generated")

        used_names: set[str] = set()

        with zipfile.ZipFile(
            target,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for source_id in file_ids:
                meta = self.files[source_id]
                source_path = Path(meta["path"])

                relative = meta.get("relative_path")
                if relative:
                    arcname = str(relative).replace("\\", "/")
                else:
                    arcname = Path(meta["name"]).name

                original = arcname
                index = 2
                while arcname in used_names:
                    path = PurePosixPath(original)
                    arcname = (
                        f"{path.stem}_{index}{path.suffix}"
                        if str(path.parent) == "."
                        else f"{path.parent.as_posix()}/{path.stem}_{index}{path.suffix}"
                    )
                    index += 1

                used_names.add(arcname)
                archive.write(source_path, arcname=arcname)

        self.register_file(
            chat_id,
            file_id,
            target,
            zip_name,
            "generated_zip",
        )
        self.files[file_id]["contains"] = list(file_ids)
        self.save()
        return file_id

    def file_text(self, chat_id: int, file_id: int) -> str:
        if file_id not in self.chat(chat_id)["file_ids"]:
            raise ValueError("file not available in this chat")

        if file_id in self.text_cache:
            return self.text_cache[file_id]

        meta = self.files.get(file_id)
        if not meta:
            raise ValueError("file not found")

        if meta.get("kind") == "generated_zip":
            contained = meta.get("contains", [])
            raise ValueError(
                "ZIP archive is stored locally; read its contained file ids instead: "
                + ",".join(str(value) for value in contained)
            )

        data = Path(meta["path"]).read_bytes()

        if b"\x00" in data[:4096]:
            raise ValueError("binary file is stored but no text parser is installed")

        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")

        self.text_cache[file_id] = text
        return text

    def file_stats(self, file_id: int) -> dict[str, Any]:
        meta = self.files[file_id]
        cached = meta.get("stats")
        if isinstance(cached, dict):
            return cached

        path = Path(meta["path"])
        size = path.stat().st_size
        stats: dict[str, Any] = {"bytes": size}

        if meta.get("kind") == "generated_zip":
            stats["binary"] = True
            meta["stats"] = stats
            return stats

        with path.open("rb") as handle:
            head = handle.read(4096)
        if b"\x00" in head:
            stats["binary"] = True
            meta["stats"] = stats
            return stats

        lines = 0
        chars = 0
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                for line in handle:
                    lines += 1
                    chars += len(line)
        except OSError:
            stats["binary"] = True
        else:
            stats.update({"lines": lines, "chars": chars, "binary": False})

        meta["stats"] = stats
        return stats

    def file_ref(self, file_id: int) -> str:
        meta = self.files[file_id]
        if meta.get("kind") == "generated_zip" and meta.get("contains"):
            ids = ",".join(str(value) for value in meta["contains"])
            return f"[file {file_id}: {meta['name']}; contains file ids {ids}]"

        stats = self.file_stats(file_id)
        if stats.get("binary"):
            return f"[file {file_id}: {meta['name']}; bytes={stats.get('bytes', 0)}; binary]"
        return (
            f"[file {file_id}: {meta['name']}; "
            f"lines={stats.get('lines', 0)}; chars={stats.get('chars', 0)}]"
        )

    def archive_manifest(self, file_ids: list[int]) -> str:
        archive_groups: dict[int, list[int]] = {}

        for file_id in file_ids:
            meta = self.files.get(file_id, {})
            archive_id = meta.get("archive_id")
            if archive_id is not None:
                archive_groups.setdefault(int(archive_id), []).append(file_id)

        blocks: list[str] = []
        for archive_id, ids in archive_groups.items():
            first = self.files[ids[0]]
            archive_name = first.get("archive_name", "archive.zip")
            blocks.append(f"[archive: {archive_name}]")
            for file_id in sorted(ids, key=lambda value: self.files[value].get("relative_path", "")):
                meta = self.files[file_id]
                relative = meta.get("relative_path", meta["name"])
                stats = self.file_stats(file_id)
                if stats.get("binary"):
                    suffix = f"bytes={stats.get('bytes', 0)}; binary"
                else:
                    suffix = f"lines={stats.get('lines', 0)}; chars={stats.get('chars', 0)}"
                blocks.append(f"{file_id}: {relative}; {suffix}")

        return "\n".join(blocks)

    def existing_zip_names(self, chat_id: int) -> set[str]:
        names: set[str] = set()

        for file_id in self.chat(chat_id).get("file_ids", []):
            meta = self.files.get(file_id, {})

            if meta.get("kind") == "generated_zip":
                names.add(str(meta.get("name", "")).lower())

            archive_name = meta.get("archive_name")
            if archive_name:
                names.add(str(archive_name).lower())

        return names

    def add_user(self, chat_id: int, text: str, file_ids: list[int]) -> str:
        self.maybe_title_chat(chat_id, text, file_ids)

        standalone_ids = [
            file_id for file_id in file_ids
            if self.files.get(file_id, {}).get("archive_id") is None
        ]
        refs = "\n".join(self.file_ref(file_id) for file_id in standalone_ids)
        manifest = self.archive_manifest(file_ids)

        content = text.strip()
        extra = "\n".join(part for part in (refs, manifest) if part)
        if extra:
            content = f"{content}\n{extra}".strip()

        self.chat(chat_id)["messages"].append(
            {"role": "user", "content": content}
        )
        self.save()
        return content

    def add_assistant(
        self,
        chat_id: int,
        text: str,
        api_content: list[dict[str, Any]],
        generated_ids: list[int],
    ) -> str:
        refs = "\n".join(self.file_ref(file_id) for file_id in generated_ids)

        content = text.strip()
        if refs:
            content = f"{content}\n{refs}".strip()
            api_content = list(api_content)
            api_content.append({"type": "text", "text": f"\n{refs}"})

        self.chat(chat_id)["messages"].append(
            {
                "role": "assistant",
                "content": content,
                "api_content": api_content,
            }
        )
        self.save()
        return content

    def add_refusal(self, chat_id: int, text: str) -> None:
        self.chat(chat_id)["messages"].append(
            {"role": "assistant", "content": text}
        )
        self.save()

    def api_messages(self, chat_id: int) -> list[dict[str, Any]]:
        messages = self.chat(chat_id)["messages"]

        assistant_indexes = [
            i for i, message in enumerate(messages)
            if message["role"] == "assistant" and not message.get("client_notice")
        ]
        keep_full = set(assistant_indexes[-KEEP_FULL_ASSISTANT_TURNS:])

        result: list[dict[str, Any]] = []

        for index, message in enumerate(messages):
            if message.get("client_notice") or message.get("client_failed_turn"):
                continue
            if (
                message["role"] == "assistant"
                and index in keep_full
                and message.get("api_content")
            ):
                result.append(
                    {"role": "assistant", "content": message["api_content"]}
                )
            else:
                result.append(
                    {"role": message["role"], "content": message["content"]}
                )

        return result


STATE = State()


def parse_search_terms(query: str, case_sensitive: bool = False) -> list[str]:
    """Split all/any search into bare terms plus double-quoted phrases."""
    terms: list[str] = []
    for quoted, bare in re.findall(r'"([^"\n]+)"|(\S+)', query):
        value = (quoted or bare).strip()
        if not quoted:
            stripped = value.strip('.,;:!?()[]{}')
            if not stripped:
                # Do not silently broaden an all/any query by dropping a term.
                return []
            value = stripped
        if value:
            terms.append(value if case_sensitive else value.casefold())
    return terms


def merged_context_windows(lines: list[str], hit_lines: list[int], context: int) -> list[dict[str, Any]]:
    if context <= 0 or not hit_lines:
        return []
    total = len(lines)
    ranges: list[tuple[int, int]] = []
    for line_no in hit_lines:
        start = max(1, line_no - context)
        end = min(total, line_no + context)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return [
        {
            "start": start,
            "end": end,
            "text": "\n".join(
                f"{line_no}: {lines[line_no - 1]}"
                for line_no in range(start, end + 1)
            ),
        }
        for start, end in ranges
    ]


def search_result_payload(
    file_id: int,
    raw_query: str,
    mode: str,
    case_sensitive: bool,
    context: int,
    lines: list[str],
    matched: list[dict[str, Any]],
) -> dict[str, Any]:
    hit_lines = [int(item["line"]) for item in matched]
    payload: dict[str, Any] = {
        "id": file_id,
        "op": "search",
        "q": raw_query,
        "mode": mode,
        "case_sensitive": case_sensitive,
        "context": context,
        "match_count": len(matched),
    }
    if context > 0:
        payload["match_lines"] = hit_lines
        payload["contexts"] = merged_context_windows(lines, hit_lines, context)
    else:
        payload["matches"] = matched
    return payload


def paged_search_payload(payload: dict[str, Any], offset: int) -> dict[str, Any]:
    key = "contexts" if "contexts" in payload else "matches"
    units = list(payload.get(key) or [])
    base = {k: v for k, v in payload.items() if k not in {"matches", "contexts", "match_lines"}}
    selected: list[dict[str, Any]] = []
    next_offset: int | None = None

    for index in range(offset, len(units)):
        candidate = selected + [units[index]]
        probe = dict(base)
        probe[key] = candidate
        probe["returned_offset"] = offset
        size = search_model_visible_chars(probe)
        if size > SEARCH_RESULT_GUARD_CHARS:
            if not selected:
                return {
                    **base,
                    "returned_offset": offset,
                    "returned_units": 0,
                    "has_more": True,
                    "next_offset": offset,
                    "error": "one requested result unit exceeds the transport target; reduce search context or narrow the query",
                }
            next_offset = index
            break
        selected = candidate

    result = dict(base)
    result[key] = selected
    result["returned_offset"] = offset
    result["returned_units"] = len(selected)
    if next_offset is not None and next_offset < len(units):
        result["next_offset"] = next_offset
        result["has_more"] = True
    else:
        result["has_more"] = False

    # The continuation metadata itself consumes a few characters.  Enforce the
    # transport target against the final model-visible representation rather
    # than only the provisional payload used while selecting units.
    while selected and search_model_visible_chars(result) > SEARCH_RESULT_GUARD_CHARS:
        selected.pop()
        result[key] = selected
        result["returned_units"] = len(selected)
        result["next_offset"] = offset + len(selected)
        result["has_more"] = True

    if not selected and units and offset < len(units):
        return {
            **base,
            "returned_offset": offset,
            "returned_units": 0,
            "has_more": True,
            "next_offset": offset,
            "error": "one requested result unit exceeds the transport target; reduce search context or narrow the query",
        }
    return result


def search_model_visible_chars(payload: dict[str, Any]) -> int:
    excluded = {"matches", "contexts"}
    if payload.get("matches") or payload.get("contexts"):
        excluded.add("match_lines")
    meta = {k: v for k, v in payload.items() if k not in excluded}
    meta["index"] = 0
    total = len(
        "FILE_READ_METADATA "
        + json.dumps({"results": [meta]}, ensure_ascii=False, separators=(",", ":"))
    )
    matches = payload.get("matches")
    if isinstance(matches, list) and matches:
        total += len(f"SEARCH_RESULT index=0 id={payload.get('id')}\n")
        total += sum(
            len(str(match.get("line"))) + 2 + len(str(match.get("text", ""))) + 1
            for match in matches if isinstance(match, dict)
        )
    contexts = payload.get("contexts")
    if isinstance(contexts, list) and contexts:
        total += len(f"SEARCH_CONTEXT index=0 id={payload.get('id')}\n")
        total += sum(
            len(str(window.get("text", ""))) + 5
            for window in contexts if isinstance(window, dict)
        )
    return total


def execute_file_read_one(
    chat_id: int,
    args: dict[str, Any],
) -> dict[str, Any]:
    if "id" not in args:
        return {"error": "missing required field: id"}
    if "op" not in args:
        return {"id": args.get("id"), "error": "missing required field: op"}

    file_id = int(args["id"])
    op = str(args["op"])

    if op == "search":
        raw_query = str(args.get("q", "")).strip()
        if not raw_query:
            return {"id": file_id, "error": "search query required"}

        mode = str(args.get("mode") or "phrase").lower()
        if mode not in {"phrase", "all", "any", "regex"}:
            return {"id": file_id, "error": "search mode must be phrase, all, any, or regex"}

        case_sensitive = bool(args.get("case_sensitive", False))
        context = max(0, int(args.get("context", 0)))
        text = STATE.file_text(chat_id, file_id)
        lines = text.splitlines()
        matched: list[dict[str, Any]] = []

        if mode == "regex":
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(raw_query, flags=flags)
            except re.error as exc:
                return {"id": file_id, "mode": mode, "error": f"invalid regex: {exc}"}
            for line_no, line in enumerate(lines, 1):
                if pattern.search(line):
                    matched.append({"line": line_no, "text": line})
        elif mode == "phrase":
            phrase = raw_query
            if len(phrase) >= 2 and phrase[0] == phrase[-1] == '"':
                phrase = phrase[1:-1].strip()
            if not phrase:
                return {"id": file_id, "mode": mode, "error": "search phrase required"}
            needle = phrase if case_sensitive else phrase.casefold()
            for line_no, line in enumerate(lines, 1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matched.append({"line": line_no, "text": line})
        else:
            terms = parse_search_terms(raw_query, case_sensitive=case_sensitive)
            if not terms:
                return {
                    "id": file_id,
                    "mode": mode,
                    "error": "search terms required; punctuation-only terms are not silently discarded",
                }
            for line_no, line in enumerate(lines, 1):
                haystack = line if case_sensitive else line.casefold()
                hits = [term in haystack for term in terms]
                is_match = all(hits) if mode == "all" else any(hits)
                if is_match:
                    matched.append({"line": line_no, "text": line})

        payload = search_result_payload(
            file_id, raw_query, mode, case_sensitive, context, lines, matched
        )
        result_chars = search_model_visible_chars(payload)

        if result_chars <= SEARCH_RESULT_GUARD_CHARS:
            return payload

        if bool(args.get("full")):
            offset = max(0, int(args.get("offset", args.get("match_offset", 0))))
            paged = paged_search_payload(payload, offset)
            paged["result_too_large"] = True
            paged["full_requested"] = True
            return paged

        locations = [int(item["line"]) for item in matched]
        locations_probe = json.dumps(locations, separators=(",", ":"))
        summary: dict[str, Any] = {
            "id": file_id,
            "op": "search",
            "q": raw_query,
            "mode": mode,
            "case_sensitive": case_sensitive,
            "context": context,
            "match_count": len(matched),
            "result_chars": result_chars,
            "result_too_large": True,
        }
        if len(locations_probe) <= SEARCH_RESULT_GUARD_CHARS:
            summary["match_lines"] = locations
        elif locations:
            summary["first_match_line"] = locations[0]
            summary["last_match_line"] = locations[-1]
        summary["message"] = (
            "The complete search payload exceeds the transport target. No match text was silently selected or discarded. "
            "Narrow the query, request surrounding context selectively, or repeat with full=true; use offset=next_offset to continue if a full result is paged."
        )
        return summary

    if op == "read":
        lines = STATE.file_text(chat_id, file_id).splitlines()
        total = len(lines)
        if not lines:
            return {"id": file_id, "op": "read", "start": 0, "end": 0, "lines": 0, "text": ""}

        requested_start = int(args["start"]) if "start" in args else 1
        requested_end = int(args["end"]) if "end" in args else total
        if requested_start > total:
            return {"id": file_id, "op": "read", "error": "start is beyond EOF", "lines": total}
        if requested_start > requested_end:
            return {"id": file_id, "op": "read", "error": "start must be <= end", "lines": total}

        start = max(1, requested_start)
        end = min(total, max(1, requested_end))
        clamped = start != requested_start or end != requested_end
        return {
            "id": file_id,
            "op": "read",
            "start": start,
            "end": end,
            "lines": total,
            "clamped": clamped,
            "text": "\n".join(
                f"{line_no}: {lines[line_no - 1]}"
                for line_no in range(start, end + 1)
            ),
        }

    return {"id": file_id, "error": "unknown op"}


def execute_file_read(
    chat_id: int,
    args: dict[str, Any],
) -> dict[str, Any]:
    items = args.get("items")
    if not isinstance(items, list) or not items:
        return {"error": "items must contain at least one file operation"}

    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            results.append({"error": "invalid item"})
            continue
        try:
            results.append(execute_file_read_one(chat_id, item))
        except Exception as exc:
            results.append({"id": item.get("id"), "error": str(exc)})

    return {"results": results}


def execute_tool(
    chat_id: int,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    try:
        if name == "file_read":
            return execute_file_read(chat_id, args)

        if name == "file_save":
            filename = str(args.get("name") or "generated.txt")
            source_files = [
                int(value)
                for value in (args.get("files") or [])
            ]

            has_text = "text" in args
            if source_files and has_text:
                return {"error": "provide text or files, not both"}

            if source_files:
                file_id = STATE.save_generated_zip(
                    chat_id,
                    filename,
                    source_files,
                )
                return {
                    "id": file_id,
                    "saved": True,
                    "name": STATE.files[file_id]["name"],
                    "files": source_files,
                }

            if not has_text:
                return {"error": "text or files required"}
            text = str(args.get("text") or "")
            file_id = STATE.save_generated(chat_id, filename, text)
            stats = STATE.file_stats(file_id)
            return {
                "id": file_id,
                "saved": True,
                "name": filename,
                "lines": stats.get("lines", 0),
                "chars": stats.get("chars", 0),
            }

        return {"error": "unknown tool"}

    except Exception as exc:
        return {"error": str(exc)}


def model_profile(model: str) -> dict[str, Any]:
    return MODEL_PROFILES.get(model, {
        "max_tokens": 128000,
        "context_window": None,
        "effort": "max",
        "adaptive_thinking": True,
        "prompt_cache": False,
    })


def request_max_tokens(model: str, previous_input_tokens: int | None = None) -> int:
    profile = model_profile(model)
    cap = int(profile.get("max_tokens") or 128000)
    window = profile.get("context_window")
    if not window or previous_input_tokens is None:
        return cap
    # Only reserve against a measured prior-round input size. Never guess the first request.
    reserve = 8_000
    remaining = int(window) - int(previous_input_tokens) - reserve
    if remaining < 1_024:
        raise RuntimeError(
            f"Context exhausted: previous input used {previous_input_tokens} tokens in a {window}-token window."
        )
    return min(cap, remaining)


def cache_control_value(ttl: str) -> dict[str, str]:
    return {"type": "ephemeral", "ttl": ttl}


def cacheable_tools(enabled: bool, ttl: str = "5m") -> list[dict[str, Any]]:
    tools = copy.deepcopy(TOOLS)
    if enabled and tools:
        tools[-1]["cache_control"] = cache_control_value(ttl)
    return tools


def messages_with_rolling_cache(
    messages: list[dict[str, Any]],
    enabled: bool,
    ttl: str = "5m",
    lag_one_user: bool = False,
) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(messages)
    if not enabled or not prepared:
        return prepared

    # Use a one-round-lag breakpoint during tool loops: cache a prefix only after we
    # already know the model needed another round beyond it. This avoids paying a cache-write
    # premium on the newest tool result when the very next response may be the final answer.
    user_indexes = [
        index for index, message in enumerate(prepared)
        if message.get("role") == "user"
    ]
    if not user_indexes:
        return prepared
    target_index = user_indexes[-2] if lag_one_user and len(user_indexes) >= 2 else user_indexes[-1]
    message = prepared[target_index]
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": cache_control_value(ttl),
            }
        ]
    elif isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_control_value(ttl)
    return prepared


def _nvidia_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        for tool in TOOLS
    ]


def _nvidia_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate our Anthropic-native working chain to OpenAI-compatible chat messages."""
    out: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") in ("thinking", "redacted_thinking"):
                    reasoning = block.get("thinking") or block.get("text")
                    if reasoning:
                        reasoning_parts.append(str(reasoning))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    })
            item: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if reasoning_parts:
                item["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls:
                item["tool_calls"] = tool_calls
            out.append(item)
            continue

        # Anthropic groups all tool_result blocks in one user message; OpenAI-compatible
        # APIs represent each result as its own role=tool message.
        normal_text: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                normal_text.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    parts = [str(x.get("text") or "") for x in tool_content if isinstance(x, dict) and x.get("type") == "text"]
                    tool_content = "\n".join(parts)
                out.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": str(tool_content),
                })
        if normal_text:
            out.append({"role": "user", "content": "".join(normal_text)})
    return out


def call_nvidia_model(
    messages: list[dict[str, Any]],
    *,
    previous_input_tokens: int | None = None,
    force_no_tools: bool = False,
) -> Any:
    model = STATE.selected_model
    profile = model_profile(model)
    payload: dict[str, Any] = {
        "model": model,
        "messages": _nvidia_messages(messages),
        "max_tokens": request_max_tokens(model, previous_input_tokens),
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "reasoning_effort": str(profile.get("effort") or "max"),
    }
    if not force_no_tools:
        payload["tools"] = _nvidia_tools()
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {load_nvidia_api_key()}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    response = requests.post(NVIDIA_BASE_URL, headers=headers, json=payload, stream=True, timeout=(30, 900))
    if not response.ok:
        body = response.text[:2000]
        raise RuntimeError(f"NVIDIA API HTTP {response.status_code}: {body}")

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage_data: dict[str, Any] = {}
    # NVIDIA SSE is UTF-8. requests may otherwise guess ISO-8859-1 when
    # text/event-stream has no explicit charset, which mojibakes Cyrillic.
    response.encoding = "utf-8"
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event.get("usage"), dict):
            usage_data = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}
        if delta.get("content"):
            text_parts.append(str(delta["content"]))
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            reasoning_parts.append(str(reasoning))
        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0))
            entry = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                entry["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["name"] += str(fn["name"])
            if fn.get("arguments"):
                entry["arguments"] += str(fn["arguments"])

    blocks: list[Any] = []
    if reasoning_parts:
        blocks.append(SimpleNamespace(type="thinking", thinking="".join(reasoning_parts), signature=""))
    if text_parts:
        blocks.append(SimpleNamespace(type="text", text="".join(text_parts)))
    for idx in sorted(calls):
        tc = calls[idx]
        try:
            args = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"NVIDIA returned invalid tool arguments for {tc['name']}: {exc}") from exc
        blocks.append(SimpleNamespace(type="tool_use", id=tc["id"] or f"nvidia_tool_{idx}", name=tc["name"], input=args))

    stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
    usage = SimpleNamespace(
        input_tokens=int(usage_data.get("prompt_tokens") or 0),
        output_tokens=int(usage_data.get("completion_tokens") or 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(content=blocks, stop_reason=stop_map.get(finish_reason, finish_reason), usage=usage)


def call_model(
    client: Anthropic,
    messages: list[dict[str, Any]],
    *,
    run_id: str = "",
    round_index: int = 0,
    previous_input_tokens: int | None = None,
    force_no_tools: bool = False,
    cache_messages: bool = True,
    cache_lag_one: bool = False,
) -> Any:
    model = STATE.selected_model
    profile = model_profile(model)
    if profile.get("provider") == "nvidia":
        return call_nvidia_model(
            messages,
            previous_input_tokens=previous_input_tokens,
            force_no_tools=force_no_tools,
        )
    cache_enabled = bool(profile.get("prompt_cache")) and model not in PROMPT_CACHE_DISABLED_MODELS
    cache_ttl = PROMPT_CACHE_TTL_BY_MODEL.get(model, "5m")
    controls_enabled = model not in MODEL_CONTROLS_DISABLED_MODELS

    def make_request(use_cache: bool, use_model_controls: bool = True) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request_max_tokens(model, previous_input_tokens),
            "tools": cacheable_tools(use_cache, cache_ttl),
            "messages": messages_with_rolling_cache(
                messages, use_cache and cache_messages, cache_ttl, lag_one_user=cache_lag_one
            ),
            "system": SYSTEM_PROMPT,
        }
        if use_model_controls and model not in MODEL_CONTROLS_DISABLED_MODELS:
            if profile.get("adaptive_thinking"):
                kwargs["thinking"] = {"type": "adaptive"}
            effort = profile.get("effort")
            if effort:
                kwargs["output_config"] = {"effort": effort}
        if force_no_tools:
            kwargs["tool_choice"] = {"type": "none"}

        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    transport_retry = 0
    while True:
        try:
            return make_request(cache_enabled, controls_enabled)
        except Exception as exc:
            low = str(exc).lower()
            status = getattr(exc, "status_code", None)
            invalid_request = (
                "invalid_request_error" in low
                or "invalid_request" in low
                or "error code: 400" in low
                or "status code: 400" in low
            )

            if cache_enabled and invalid_request:
                if cache_ttl == "1h":
                    cache_ttl = "5m"
                    PROMPT_CACHE_TTL_BY_MODEL[model] = "5m"
                    trace_event(
                        "prompt_cache_ttl_fallback",
                        run=run_id,
                        round=round_index,
                        model=model,
                        from_ttl="1h",
                        to_ttl="5m",
                        reason=str(exc)[:500],
                    )
                    continue
                PROMPT_CACHE_DISABLED_MODELS.add(model)
                cache_enabled = False
                trace_event(
                    "prompt_cache_fallback",
                    run=run_id,
                    round=round_index,
                    model=model,
                    reason=str(exc)[:500],
                )
                continue

            if invalid_request and controls_enabled and not model.startswith("claude-"):
                controls_enabled = False
                MODEL_CONTROLS_DISABLED_MODELS.add(model)
                trace_event(
                    "model_controls_fallback",
                    run=run_id,
                    round=round_index,
                    model=model,
                    disabled=["thinking", "output_config"],
                    reason=str(exc)[:500],
                )
                continue

            # Retry only explicit transient HTTP statuses. Do not automatically retry an
            # unknown/mid-stream exception: the upstream may already have generated and billed it.
            transient_status = status in {408, 409, 429} or (isinstance(status, int) and status >= 500)
            if invalid_request:
                transient_status = False
            if transient_status and transport_retry < MAX_TRANSPORT_RETRIES:
                transport_retry += 1
                delay = min(4.0, 2.0 ** (transport_retry - 1))
                trace_event(
                    "transport_retry",
                    run=run_id,
                    round=round_index,
                    retry=transport_retry,
                    max_retries=MAX_TRANSPORT_RETRIES,
                    status_code=status,
                    delay_seconds=delay,
                    error=str(exc)[:500],
                )
                time.sleep(delay)
                continue
            trace_event(
                "transport_failure_no_retry",
                run=run_id,
                round=round_index,
                status_code=status,
                invalid_request=invalid_request,
                error=str(exc)[:500],
            )
            raise



def tool_result_content(name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return model-visible tool data without JSON-escaping large file/search text."""
    if name != "file_read" or not isinstance(result.get("results"), list):
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False, separators=(",", ":"))}]

    metadata: list[dict[str, Any]] = []
    raw_blocks: list[dict[str, Any]] = []
    for index, item in enumerate(result["results"]):
        if not isinstance(item, dict):
            metadata.append({"index": index, "error": "invalid result"})
            continue
        excluded = {"text", "matches", "contexts"}
        if item.get("matches") or item.get("contexts"):
            excluded.add("match_lines")
        meta = {k: v for k, v in item.items() if k not in excluded}
        meta["index"] = index
        metadata.append(meta)

        if isinstance(item.get("text"), str):
            raw_blocks.append({
                "type": "text",
                "text": f"READ_RESULT index={index} id={item.get('id')}\n{item['text']}",
            })
        matches = item.get("matches")
        if isinstance(matches, list) and matches:
            raw = "\n".join(
                f"{match.get('line')}: {match.get('text', '')}"
                for match in matches
                if isinstance(match, dict)
            )
            raw_blocks.append({
                "type": "text",
                "text": f"SEARCH_RESULT index={index} id={item.get('id')}\n{raw}",
            })
        contexts = item.get("contexts")
        if isinstance(contexts, list) and contexts:
            raw = "\n---\n".join(
                str(window.get("text", ""))
                for window in contexts
                if isinstance(window, dict)
            )
            raw_blocks.append({
                "type": "text",
                "text": f"SEARCH_CONTEXT index={index} id={item.get('id')}\n{raw}",
            })

    return [
        {"type": "text", "text": "FILE_READ_METADATA " + json.dumps({"results": metadata}, ensure_ascii=False, separators=(",", ":"))},
        *raw_blocks,
    ]


def usage_trace_fields(usage: Any) -> dict[str, Any]:
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_write = getattr(usage, "cache_creation_input_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    fields: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }
    cache_creation = getattr(usage, "cache_creation", None)
    nested_creation_tokens = 0
    if cache_creation is not None:
        if hasattr(cache_creation, "model_dump"):
            cache_data = cache_creation.model_dump(exclude_none=True)
            fields["cache_creation"] = cache_data
            nested_creation_tokens = sum(
                int(value) for key, value in cache_data.items()
                if key.endswith("_input_tokens") and isinstance(value, (int, float))
            )
        else:
            fields["cache_creation"] = str(cache_creation)
    base = int(input_tokens or 0)
    read = int(cache_read or 0)
    write = int(cache_write or 0) if cache_write is not None else nested_creation_tokens
    fields["context_input_tokens"] = base + read + write
    return fields


def refusal_text(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    if details is not None and hasattr(details, "model_dump"):
        data = details.model_dump(exclude_none=True)
        if data.get("explanation"):
            return str(data["explanation"])
    return "Request refused."


def run_turn(
    client: Anthropic,
    chat_id: int,
    user_text: str,
    file_ids: list[int],
) -> str:
    STATE.add_user(chat_id, user_text, file_ids)
    working = STATE.api_messages(chat_id)
    generated_ids: list[int] = []
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    seen_tool_calls: set[str] = set()
    incomplete_retries = 0
    previous_input_tokens: int | None = None
    continued_text: list[str] = []

    trace_event(
        "turn_start",
        run=run_id,
        chat=chat_id,
        model=STATE.selected_model,
        user_chars=len(user_text),
        attached_file_ids=file_ids,
        history_messages=len(working),
        effort=model_profile(STATE.selected_model).get("effort"),
        prompt_cache=bool(model_profile(STATE.selected_model).get("prompt_cache")),
    )

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        requested_max_tokens = request_max_tokens(STATE.selected_model, previous_input_tokens)
        started = time.perf_counter()
        response = call_model(
            client,
            working,
            run_id=run_id,
            round_index=round_index,
            previous_input_tokens=previous_input_tokens,
            cache_messages=(round_index > 1 or bool(file_ids)),
            cache_lag_one=(round_index > 1),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        usage_fields = usage_trace_fields(usage)
        previous_input_tokens = usage_fields.get("context_input_tokens")
        trace_event(
            "model_response",
            run=run_id,
            round=round_index,
            elapsed_ms=elapsed_ms,
            stop_reason=getattr(response, "stop_reason", None),
            blocks=[getattr(block, "type", None) for block in response.content],
            working_messages=len(working),
            requested_max_tokens=requested_max_tokens,
            **usage_fields,
        )

        tool_blocks = [block for block in response.content if block.type == "tool_use"]
        visible_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        stop_reason = getattr(response, "stop_reason", None)

        # Never execute a tool call from an unconfirmed/truncated response.
        if tool_blocks and stop_reason != "tool_use":
            if stop_reason is None and not visible_text.strip():
                if incomplete_retries < MAX_INCOMPLETE_RETRIES:
                    incomplete_retries += 1
                    trace_event(
                        "incomplete_response_retry", run=run_id, round=round_index,
                        retry=incomplete_retries, max_retries=MAX_INCOMPLETE_RETRIES,
                        stop_reason=stop_reason, blocks=[getattr(block, "type", None) for block in response.content],
                        visible_chars=len(visible_text),
                    )
                    continue
            partial = "".join(continued_text) + visible_text
            if partial.strip():
                trace_event("turn_end", run=run_id, round=round_index, outcome="unconfirmed_partial_with_tool", stop_reason=stop_reason, answer_chars=len(partial))
                return STATE.add_assistant(chat_id, partial, [{"type": "text", "text": partial}], generated_ids)
            raise RuntimeError(f"Model returned tool_use with unsupported stop_reason={stop_reason!r}")

        if not tool_blocks:
            if stop_reason == "refusal":
                text = refusal_text(response)
                trace_event("turn_end", run=run_id, round=round_index, outcome="refusal")
                STATE.add_refusal(chat_id, text)
                return text

            if stop_reason == "max_tokens":
                if not visible_text.strip():
                    raise RuntimeError("Model hit max_tokens without usable visible text")
                continued_text.append(visible_text)
                trace_event("max_tokens_continue", run=run_id, round=round_index, accumulated_chars=sum(map(len, continued_text)))
                working.append({
                    "role": "assistant",
                    "content": [block_to_dict(block) for block in response.content],
                })
                working.append({
                    "role": "user",
                    "content": "Continue exactly from where the response stopped. Do not repeat completed text.",
                })
                incomplete_retries = 0
                continue

            if stop_reason == "model_context_window_exceeded":
                partial = "".join(continued_text) + visible_text
                if partial.strip():
                    trace_event("turn_end", run=run_id, round=round_index, outcome="context_window_partial", answer_chars=len(partial))
                    return STATE.add_assistant(chat_id, partial, [{"type": "text", "text": partial}], generated_ids)
                raise RuntimeError("Model context window was exhausted before a usable answer was produced")

            if stop_reason is None:
                if visible_text.strip():
                    final_text = "".join(continued_text) + visible_text
                    trace_event("turn_end", run=run_id, round=round_index, outcome="termination_unconfirmed_with_text", answer_chars=len(final_text))
                    return STATE.add_assistant(chat_id, final_text, [{"type": "text", "text": final_text}], generated_ids)
                if incomplete_retries < MAX_INCOMPLETE_RETRIES:
                    incomplete_retries += 1
                    trace_event(
                        "incomplete_response_retry", run=run_id, round=round_index,
                        retry=incomplete_retries, max_retries=MAX_INCOMPLETE_RETRIES,
                        stop_reason=stop_reason, blocks=[getattr(block, "type", None) for block in response.content],
                        visible_chars=0,
                    )
                    continue
                raise RuntimeError("Model returned an incomplete response repeatedly; the turn was not saved as a successful empty answer.")

            if stop_reason == "end_turn" and not visible_text.strip() and not generated_ids and not continued_text:
                if incomplete_retries < MAX_INCOMPLETE_RETRIES:
                    incomplete_retries += 1
                    trace_event(
                        "incomplete_response_retry", run=run_id, round=round_index,
                        retry=incomplete_retries, max_retries=MAX_INCOMPLETE_RETRIES,
                        stop_reason=stop_reason, blocks=[getattr(block, "type", None) for block in response.content],
                        visible_chars=0,
                    )
                    continue
                raise RuntimeError("Model returned an empty end_turn repeatedly")

            if stop_reason not in {"end_turn", None}:
                final_text = "".join(continued_text) + visible_text
                if final_text.strip():
                    trace_event("turn_end", run=run_id, round=round_index, outcome="unexpected_stop_with_text", stop_reason=stop_reason, answer_chars=len(final_text))
                    return STATE.add_assistant(chat_id, final_text, [{"type": "text", "text": final_text}], generated_ids)
                raise RuntimeError(f"Model stopped with unsupported stop_reason={stop_reason!r}")

            incomplete_retries = 0
            final_text = "".join(continued_text) + visible_text

            mentioned_zip_names = {
                match.lower()
                for match in re.findall(r"([A-Za-z0-9_.-]+\.zip)", final_text, flags=re.IGNORECASE)
            }
            existing_zip_names = STATE.existing_zip_names(chat_id)
            missing_zip_names = mentioned_zip_names - existing_zip_names
            if missing_zip_names:
                trace_event("zip_correction", run=run_id, round=round_index, missing=sorted(missing_zip_names))
                working.append({
                    "role": "assistant",
                    "content": [block_to_dict(block) for block in response.content],
                })
                working.append({
                    "role": "user",
                    "content": (
                        "Internal file-state check: the ZIP file(s) "
                        + ", ".join(sorted(missing_zip_names))
                        + " mentioned in your answer do not exist. Create the required ZIP now with file_save using the relevant existing file IDs. "
                        "Do not claim the ZIP exists until file_save succeeds."
                    ),
                })
                continue

            if continued_text:
                api_content = [{"type": "text", "text": final_text}]
            else:
                api_content = [
                    block_to_dict(block)
                    for block in response.content
                    if block.type in ("thinking", "redacted_thinking", "text")
                ]
            answer = STATE.add_assistant(chat_id, final_text, api_content, generated_ids)
            trace_event("turn_end", run=run_id, round=round_index, outcome="end_turn", answer_chars=len(final_text), generated_file_ids=generated_ids)
            return answer

        # Confirmed tool_use path.
        incomplete_retries = 0
        results: list[dict[str, Any]] = []
        for block in tool_blocks:
            compact_input = compact_tool_input(block.name, block.input)
            signature = block.name + ":" + json.dumps(compact_input, ensure_ascii=False, sort_keys=True)
            repeated = signature in seen_tool_calls
            seen_tool_calls.add(signature)
            trace_event("tool_call", run=run_id, round=round_index, name=block.name, input=compact_input, repeated_exact=repeated)

            result = execute_tool(chat_id, block.name, block.input)
            if result.get("saved"):
                generated_ids.append(int(result["id"]))

            serialized = json.dumps(result, ensure_ascii=False)
            trace_event(
                "tool_result", run=run_id, round=round_index, name=block.name,
                result=compact_tool_result(block.name, block.input, result, len(serialized)),
            )
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_content(block.name, result),
            }
            if result.get("error"):
                tool_result["is_error"] = True
            results.append(tool_result)

        working.append({
            "role": "assistant",
            "content": [block_to_dict(block) for block in response.content],
        })
        working.append({"role": "user", "content": results})

    # Emergency fuse only: preserve accumulated work by forcing one final synthesis without tools.
    trace_event("tool_round_fuse", run=run_id, round=MAX_TOOL_ROUNDS)
    response = call_model(
        client,
        working,
        run_id=run_id,
        round_index=MAX_TOOL_ROUNDS + 1,
        previous_input_tokens=previous_input_tokens,
        force_no_tools=True,
        cache_messages=True,
        cache_lag_one=True,
    )
    visible_text = "".join(block.text for block in response.content if block.type == "text")
    final_text = "".join(continued_text) + visible_text
    if not final_text.strip():
        raise RuntimeError("Tool-round fuse reached and final forced synthesis returned no usable text")
    answer = STATE.add_assistant(chat_id, final_text, [{"type": "text", "text": final_text}], generated_ids)
    trace_event("turn_end", run=run_id, round=MAX_TOOL_ROUNDS + 1, outcome="tool_round_fuse_final", answer_chars=len(final_text), generated_file_ids=generated_ids)
    return answer


class MessageInput(QPlainTextEdit):
    paste_as_file = Signal(str)
    send_requested = Signal()

    def keyPressEvent(self, event: Any) -> None:
        if event.matches(QKeySequence.StandardKey.Paste):
            text = QApplication.clipboard().text()
            if not text:
                return

            if len(text) >= PASTE_FILE_THRESHOLD:
                self.paste_as_file.emit(text)
                return

            self.insertPlainText(text)
            return

        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.send_requested.emit()
            return

        super().keyPressEvent(event)


def friendly_error_text(error: str, model: str) -> str:
    low = error.lower()

    if "budget pool quota has been exhausted" in low or "error code: 402" in low:
        return (
            f"{model}: общий лимит AgentRouter для пользователей этой модели исчерпан. "
            "Это ограничение сервиса AgentRouter, а не ошибка чата."
        )

    if "content-blocked" in low:
        return (
            f"{model}: AgentRouter отклонил этот API-запрос как content-blocked. "
            "Файл сохранён локально; это блокировка провайдера, а не ошибка распаковки ZIP."
        )

    if "incomplete response repeatedly" in low:
        return (
            f"{model}: провайдер несколько раз вернул незавершённый ответ. "
            "Пустой ответ не был сохранён как успешный; повторите отправку позже."
        )

    if "unsupported stop_reason" in low:
        return (
            f"{model}: провайдер завершил ответ в неожиданном состоянии. "
            "Частичный ответ не был сохранён как завершённый."
        )

    return f"{model}: {error}"


class Worker(QThread):
    finished_ok = Signal(int, str)
    failed = Signal(int, str)

    def __init__(
        self,
        client: Anthropic,
        chat_id: int,
        text: str,
        file_ids: list[int],
    ) -> None:
        super().__init__()
        self.client = client
        self.chat_id = chat_id
        self.text = text
        self.file_ids = file_ids

    def run(self) -> None:
        try:
            answer = run_turn(
                self.client,
                self.chat_id,
                self.text,
                self.file_ids,
            )
            self.finished_ok.emit(self.chat_id, answer)
        except Exception as exc:
            raw = str(exc)
            print(
                f"[API ERROR] model={STATE.selected_model} "
                f"chat={self.chat_id} error={raw}"
            )
            self.failed.emit(
                self.chat_id,
                friendly_error_text(raw, STATE.selected_model),
            )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Claude Chat")
        self.resize(1050, 760)

        self.client = Anthropic(
            api_key=load_api_key(),
            base_url=BASE_URL,
            max_retries=0,
        )

        self.pending: list[dict[str, Any]] = []
        self.worker: Worker | None = None
        self.sidebar_expanded = True
        self.chat_buttons: list[QPushButton] = []

        self.build_ui()
        self.refresh_sidebar()
        self.render_chat()

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        self.sidebar_toggle = QPushButton("☰")
        self.sidebar_toggle.setFixedHeight(36)
        self.sidebar_toggle.clicked.connect(self.toggle_sidebar)
        sidebar_layout.addWidget(self.sidebar_toggle)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        self.model_combo.setCurrentText(STATE.selected_model)
        self.model_combo.currentTextChanged.connect(self.change_model)
        sidebar_layout.addWidget(self.model_combo)

        self.new_chat_button = QPushButton("+  New chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        sidebar_layout.addWidget(self.new_chat_button)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.chat_list_widget = QWidget()
        self.chat_list_layout = QVBoxLayout(self.chat_list_widget)
        self.chat_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_scroll.setWidget(self.chat_list_widget)
        sidebar_layout.addWidget(self.chat_scroll, 1)

        root_layout.addWidget(self.sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        self.chat_view = QTextBrowser()
        chat_font = QFont("Segoe UI")
        chat_font.setPointSize(10)
        self.chat_view.setFont(chat_font)
        self.chat_view.document().setDefaultFont(chat_font)
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.anchorClicked.connect(self.handle_chat_link)
        main_layout.addWidget(self.chat_view, 1)

        self.attachments_widget = QWidget()
        self.attachments_layout = QHBoxLayout(self.attachments_widget)
        self.attachments_layout.setContentsMargins(0, 0, 0, 0)
        self.attachments_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(self.attachments_widget)

        composer = QHBoxLayout()

        self.attach_button = QPushButton("+")
        self.attach_button.setFixedSize(42, 72)
        self.attach_button.clicked.connect(self.choose_files)
        composer.addWidget(self.attach_button)

        self.input = MessageInput()
        self.input.setPlaceholderText("Message")
        self.input.setFixedHeight(72)
        self.input.paste_as_file.connect(self.add_paste)
        self.input.send_requested.connect(self.send)
        composer.addWidget(self.input, 1)

        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(78, 72)
        self.send_button.clicked.connect(self.send)
        composer.addWidget(self.send_button)

        main_layout.addLayout(composer)

        self.status = QLabel("")
        main_layout.addWidget(self.status)

        root_layout.addWidget(main, 1)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #171717;
                color: #e8e8e8;
            }
            #sidebar {
                background: #111111;
                border-right: 1px solid #2b2b2b;
            }
            QPushButton {
                background: #252525;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 7px;
            }
            QPushButton:hover {
                background: #303030;
            }
            QPushButton:disabled {
                color: #777777;
            }
            QPlainTextEdit, QTextBrowser {
                background: #1d1d1d;
                border: 1px solid #333333;
                border-radius: 7px;
                padding: 8px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

    def clear_layout(self, layout: QVBoxLayout | QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def refresh_sidebar(self) -> None:
        self.clear_layout(self.chat_list_layout)
        self.chat_buttons.clear()

        for chat_id in sorted(STATE.chats):
            chat = STATE.chats[chat_id]

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = chat["title"] if self.sidebar_expanded else str(chat_id)

            button = QPushButton(label)
            button.setToolTip(chat["title"])
            button.clicked.connect(
                lambda checked=False, cid=chat_id: self.switch_chat(cid)
            )

            if chat_id == STATE.current_chat_id:
                button.setStyleSheet("background: #3a3a3a;")

            row_layout.addWidget(button, 1)
            self.chat_buttons.append(button)

            if self.sidebar_expanded:
                menu_button = QPushButton("☰")
                menu_button.setFixedWidth(34)
                menu_button.clicked.connect(
                    lambda checked=False, cid=chat_id, btn=menu_button:
                        self.show_chat_menu(cid, btn)
                )
                row_layout.addWidget(menu_button)

            self.chat_list_layout.addWidget(row)

        self.new_chat_button.setText(
            "+  New chat" if self.sidebar_expanded else "+"
        )

    def change_model(self, model: str) -> None:
        if self.worker is not None:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(STATE.selected_model)
            self.model_combo.blockSignals(False)
            return
        STATE.set_model(model)

    def show_chat_menu(self, chat_id: int, button: QPushButton) -> None:
        menu = QMenu(self)

        rename_action = menu.addAction("Переименовать")
        delete_action = menu.addAction("Удалить чат")

        action = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

        if action == rename_action:
            self.rename_chat_dialog(chat_id)
        elif action == delete_action:
            self.delete_chat_dialog(chat_id)

    def rename_chat_dialog(self, chat_id: int) -> None:
        current = STATE.chats[chat_id]["title"]
        title, ok = QInputDialog.getText(
            self,
            "Переименовать чат",
            "Название:",
            text=current,
        )
        if ok and title.strip():
            STATE.rename_chat(chat_id, title)
            self.refresh_sidebar()

    def delete_chat_dialog(self, chat_id: int) -> None:
        title = STATE.chats[chat_id]["title"]
        answer = QMessageBox.question(
            self,
            "Удалить чат",
            f'Удалить чат "{title}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        STATE.delete_chat(chat_id)
        self.pending.clear()
        self.input.clear()
        self.render_pending()
        self.refresh_sidebar()
        self.render_chat()

    def toggle_sidebar(self) -> None:
        self.sidebar_expanded = not self.sidebar_expanded
        self.sidebar.setFixedWidth(230 if self.sidebar_expanded else 58)
        self.model_combo.setVisible(self.sidebar_expanded)
        self.refresh_sidebar()

    def new_chat(self) -> None:
        if self.worker is not None:
            return
        STATE.create_chat()
        self.pending.clear()
        self.input.clear()
        self.render_pending()
        self.refresh_sidebar()
        self.render_chat()

    def switch_chat(self, chat_id: int) -> None:
        if self.worker is not None or chat_id == STATE.current_chat_id:
            return
        STATE.set_current_chat(chat_id)
        self.pending.clear()
        self.input.clear()
        self.render_pending()
        self.refresh_sidebar()
        self.render_chat()

    def render_chat(self) -> None:
        pieces = []
        messages = STATE.chat(STATE.current_chat_id)["messages"]

        for index, message in enumerate(messages):
            if message["role"] == "user":
                body = (
                    "<div style='margin:10px 0 14px 0;'>"
                    "<b>You</b><br>"
                    f"<div style='white-space:pre-wrap'>{html.escape(message['content'])}</div>"
                    "</div>"
                )
            else:
                body = (
                    "<div style='margin:10px 0 18px 0;'>"
                    "<div style='margin-bottom:6px;'>"
                    "<b>Claude</b>"
                    f"<a href='copymsg:{index}' "
                    "style='float:right;text-decoration:none;"
                    "color:#bdbdbd;background:#2b2b2b;"
                    "padding:3px 8px;border-radius:5px;'>Copy</a>"
                    "</div>"
                    f"{markdown_html(message['content'])}"
                    "</div>"
                )
            pieces.append(body)

        self.chat_view.setHtml("".join(pieces))
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def handle_chat_link(self, url: QUrl) -> None:
        if url.scheme() == "copymsg":
            raw_index = url.path() or url.toString().split(":", 1)[-1]
            try:
                index = int(raw_index.lstrip("/"))
            except ValueError:
                return

            messages = STATE.chat(STATE.current_chat_id)["messages"]
            if not (0 <= index < len(messages)):
                return

            message = messages[index]
            if message.get("role") != "assistant":
                return

            source = str(message.get("content", ""))
            mime = QMimeData()
            mime.setText(source)
            mime.setHtml(markdown_html(source))
            QApplication.clipboard().setMimeData(mime)

            self.status.setText("Ответ скопирован")
            return

        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)


    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Attach files")
        for raw in paths:
            path = Path(raw)
            if path.is_file():
                self.pending.append(
                    {
                        "kind": "attachment",
                        "label": path.name,
                        "source": path,
                    }
                )
        self.render_pending()

    def add_paste(self, text: str) -> None:
        self.pending.append(
            {
                "kind": "paste",
                "label": "paste.txt",
                "text": text,
            }
        )
        self.render_pending()

    def render_pending(self) -> None:
        self.clear_layout(self.attachments_layout)

        for index, item in enumerate(self.pending):
            chip = QFrame()
            chip.setStyleSheet(
                "QFrame {background:#262626;border:1px solid #3a3a3a;"
                "border-radius:6px;}"
            )
            layout = QHBoxLayout(chip)
            layout.setContentsMargins(8, 3, 4, 3)

            label = QLabel(item["label"])
            layout.addWidget(label)

            close = QPushButton("×")
            close.setFixedSize(24, 24)
            close.clicked.connect(
                lambda checked=False, i=index: self.remove_pending(i)
            )
            layout.addWidget(close)

            self.attachments_layout.addWidget(chip)

    def remove_pending(self, index: int) -> None:
        if 0 <= index < len(self.pending):
            self.pending.pop(index)
            self.render_pending()

    def materialize_pending(self, chat_id: int) -> list[int]:
        file_ids: list[int] = []

        for item in self.pending:
            if item["kind"] == "attachment":
                file_ids.extend(
                    STATE.add_attachment(chat_id, item["source"])
                )
            else:
                file_ids.append(
                    STATE.add_paste(chat_id, item["text"])
                )

        self.pending.clear()
        self.render_pending()
        return file_ids

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.attach_button.setEnabled(not busy)
        self.new_chat_button.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        for button in self.chat_buttons:
            button.setEnabled(not busy)

        self.status.setText("Claude is responding..." if busy else "")

    def send(self) -> None:
        if self.worker is not None:
            return

        text = self.input.toPlainText().strip()
        if not text and not self.pending:
            return

        chat_id = STATE.current_chat_id

        try:
            file_ids = self.materialize_pending(chat_id)
        except Exception as exc:
            QMessageBox.critical(self, "File error", str(exc))
            return

        self.input.clear()

        display = text
        if file_ids:
            refs = "\n".join(STATE.file_ref(file_id) for file_id in file_ids)
            display = f"{display}\n{refs}".strip()

        STATE.maybe_title_chat(chat_id, text, file_ids)
        self.append_temporary_user(display)
        self.refresh_sidebar()

        self.worker = Worker(
            self.client,
            chat_id,
            text,
            file_ids,
        )
        self.worker.finished_ok.connect(self.response_ready)
        self.worker.failed.connect(self.response_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.set_busy(True)
        self.worker.start()

    def append_temporary_user(self, text: str) -> None:
        current = self.chat_view.toHtml()
        addition = (
            "<div style='margin:10px 0 14px 0;'>"
            "<b>You</b><br>"
            f"<div style='white-space:pre-wrap'>{html.escape(text)}</div>"
            "</div>"
        )
        self.chat_view.setHtml(current + addition)
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def response_ready(self, chat_id: int, answer: str) -> None:
        self.worker = None
        self.set_busy(False)
        self.refresh_sidebar()

        if chat_id == STATE.current_chat_id:
            self.render_chat()

        self.input.setFocus()

    def response_failed(self, chat_id: int, error: str) -> None:
        self.worker = None
        self.set_busy(False)

        messages = STATE.chat(chat_id)["messages"]
        for message in reversed(messages):
            if message.get("role") == "user" and not message.get("client_failed_turn"):
                message["client_failed_turn"] = True
                break

        messages.append(
            {
                "role": "assistant",
                "content": error,
                "client_notice": True,
            }
        )
        STATE.save()

        if chat_id == STATE.current_chat_id:
            self.render_chat()

        self.input.setFocus()

    def closeEvent(self, event: Any) -> None:
        if self.worker is not None:
            self.worker.wait(3000)
        STATE.save()
        self.client.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app_font = QFont("Segoe UI")
    app_font.setPointSize(10)
    app.setFont(app_font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
