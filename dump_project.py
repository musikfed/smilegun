#!/usr/bin/env python3
"""
dump_project.py — собирает весь проект в один текстовый файл
для анализа в LLM (DeepSeek, GPT, Claude и т.д.).

Использование:
    uv run python dump_project.py
    uv run python dump_project.py --out dump.txt
    uv run python dump_project.py --max-size 500000
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------- Что исключаем ----------
EXCLUDE_DIRS = {
    ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".git", ".github",
    ".acl-reports",
    ".aider.tags.cache.v3", ".aider.tags.cache",
    "node_modules",
    ".idea", ".vscode",
    "dist", "build", "*.egg-info",
    ".dsh", ".ollama",
}

EXCLUDE_FILES = {
    ".DS_Store", "Thumbs.db",
    "dump_project.py",  # не дампить сам себя
}

EXCLUDE_PATTERNS = [
    "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dll", "*.dylib", "*.exe",
    "*.mp3", "*.mp4", "*.wav", "*.ogg", "*.flac",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico", "*.bmp",
    "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
    "*.pdf", "*.docx", "*.xlsx",
    "*.bin", "*.gguf", "*.safetensors", "*.onnx",
    "*.db", "*.sqlite", "*.sqlite3",
    "dump_*.txt", "dump.txt", "*.dump.txt",
]

# ---------- Что включаем (текстовые расширения) ----------
TEXT_EXTS = {
    ".py", ".pyi", ".pyx",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".json5", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".log",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java",
    ".go", ".rs", ".rb", ".php", ".lua",
    ".xml", ".svg",
    ".env", ".env.example", ".gitignore", ".gitattributes",
    ".dockerfile", "Dockerfile", "Makefile",
    ".prj", ".task", ".nfo",
}

# ---------- Размер файла, больше которого не читаем ----------
MAX_FILE_SIZE = 200_000  # 200 КБ на один файл


def should_skip_dir(name: str) -> bool:
    return name in EXCLUDE_DIRS or any(fnmatch.fnmatch(name, p) for p in EXCLUDE_DIRS)


def should_skip_file(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_FILES:
        return True
    if any(fnmatch.fnmatch(name, p) for p in EXCLUDE_PATTERNS):
        return True
    # скрытые файлы типа .DS_Store пропускаем, но .gitignore и .env включаем
    if name.startswith(".") and name not in {".gitignore", ".gitattributes", ".env", ".env.example", ".dockerfile"}:
        return True
    return False


def is_text(path: Path) -> bool:
    """Проверяем, текстовый ли файл — по расширению и по первым байтам."""
    if path.suffix.lower() in TEXT_EXTS:
        return True
    if path.name in {"Dockerfile", "Makefile", ".gitignore", ".env", ".env.example"}:
        return True
    # если расширения нет — пробуем по содержимому
    if not path.suffix:
        try:
            chunk = path.read_bytes()[:1024]
            chunk.decode("utf-8")
            return True
        except (UnicodeDecodeError, OSError):
            return False
    return False


def collect_files(root: Path, args) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # фильтруем директории на месте — чтобы os.walk не спускался
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if should_skip_file(fpath):
                continue
            try:
                if fpath.stat().st_size > args.max_size:
                    continue
                if not is_text(fpath):
                    continue
            except OSError:
                continue
            files.append(fpath)
    return sorted(files)


def dump(root: Path, out_path: Path, args) -> None:
    files = collect_files(root, args)

    total_bytes = 0
    total_lines = 0

    with out_path.open("w", encoding="utf-8") as out:
        # ---------- Заголовок ----------
        out.write("=" * 78 + "\n")
        out.write(f"  PROJECT DUMP: {root.name}\n")
        out.write(f"  Path: {root}\n")
        out.write(f"  Date: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"  Files: {len(files)}\n")
        out.write("=" * 78 + "\n\n")

        # ---------- Содержание ----------
        out.write("--- ОГЛАВЛЕНИЕ ---\n")
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root).as_posix()
            out.write(f"{i:4d}. {rel}\n")
        out.write("\n")

        # ---------- Файлы ----------
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root).as_posix()
            try:
                content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = f.read_text(encoding="cp1251")
                except Exception:
                    content = f"<<< НЕ УДАЛОСЬ ПРОЧИТАТЬ {rel} >>>"

            lines = content.count("\n") + 1
            size = len(content.encode("utf-8"))
            total_bytes += size
            total_lines += lines

            out.write("\n" + "=" * 78 + "\n")
            out.write(f"### FILE {i}/{len(files)}: {rel}\n")
            out.write(f"### Size: {size} bytes | Lines: {lines}\n")
            out.write("=" * 78 + "\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")

        # ---------- Итого ----------
        out.write("\n" + "=" * 78 + "\n")
        out.write(f"  TOTAL: {len(files)} files | {total_lines} lines | {total_bytes} bytes\n")
        out.write("=" * 78 + "\n")

    print(f"[OK] Дамп записан: {out_path}")
    print(f"     Файлов: {len(files)}")
    print(f"     Строк:  {total_lines}")
    print(f"     Байт:   {total_bytes}")


def main():
    parser = argparse.ArgumentParser(description="Собрать проект в один текстовый дамп.")
    parser.add_argument("--root", "-r", default=".", help="Корень проекта (по умолчанию — текущая папка)")
    parser.add_argument("--out", "-o", default=None, help="Имя выходного файла (по умолчанию — dump_<name>.txt)")
    parser.add_argument("--max-size", type=int, default=MAX_FILE_SIZE, help="Макс. размер файла в байтах (по умолчанию 200000)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[X] Не папка: {root}")
        sys.exit(1)

    if args.out is None:
        out_path = root / f"dump_{root.name}.txt"
    else:
        out_path = Path(args.out).resolve()

    dump(root, out_path, args)


if __name__ == "__main__":
    main()