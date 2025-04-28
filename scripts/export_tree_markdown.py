# noqa: INP001
"""export_tree_markdown.py

Usage:
    python export_tree_markdown.py <root_dir> [--output_path OUTPUT_PATH] [--no-binary]

Positional arguments
--------------------
root_dir : str | Path
    走査対象となるディレクトリ。

Optional arguments
------------------
--output_path, -o : str | Path  (default: project_snapshot.md)
    生成される Markdown ファイルのパス。
--no-binary
    画像やアーカイブなど代表的なバイナリをスキップ。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import fire

if TYPE_CHECKING:
    from collections.abc import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PREVIEW_SIZE = 1 * 1024 * 1024  # 1MB
SKIP_BINARY_EXT = {
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.bmp',
    '.ico',
    '.pdf',
    '.zip',
    '.tar',
    '.gz',
    '.xz',
    '.exe',
    '.dll',
}
IGNORE_DIRS = {'__pycache__'}
LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.html': 'html',
    '.css': 'css',
    '.md': 'markdown',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.sh': 'bash',
    '.rs': 'rust',
    '.go': 'go',
}


def _build_tree(root: Path) -> str:
    lines: list[str] = [f'{root.name}/']

    def _walk(dir_path: Path, prefix: str = '') -> None:
        entries = sorted(
            (e for e in dir_path.iterdir() if e.name not in IGNORE_DIRS),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        last = len(entries) - 1
        for idx, entry in enumerate(entries):
            connector = '└── ' if idx == last else '├── '
            lines.append(f'{prefix}{connector}{entry.name}')
            if entry.is_dir():
                extension = '    ' if idx == last else '│   '
                _walk(entry, prefix + extension)

    _walk(root)
    return '\n'.join(lines)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob('*')):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _file_section(root: Path, file_path: Path, no_binary: bool) -> str | None:
    rel = file_path.relative_to(root).as_posix()
    suffix = file_path.suffix.lower()

    if no_binary and suffix in SKIP_BINARY_EXT:
        return None

    try:
        text = file_path.read_text(encoding='utf-8', errors='replace')
    except UnicodeDecodeError:
        return None

    if len(text) > MAX_PREVIEW_SIZE:
        text = text[:MAX_PREVIEW_SIZE] + '\n... (truncated)\n'

    lang = LANG_MAP.get(suffix, '')
    fence = f'```{lang}' if lang else '```'
    return f'### {rel}\n\n{fence}\n{text}\n```'


def export(
    root_dir: str | Path,
    output_path: str | Path = 'project_snapshot.md',
    no_binary: bool = False,
) -> None:
    """ディレクトリツリーとコード全文を Markdown にまとめて出力します。"""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'Error: {root} is not a directory.')

    md: list[str] = [
        f'# Project Snapshot for `{root}`',
        '',
        '## Directory Tree',
        '```',
        _build_tree(root),
        '```',
        '',
    ]

    for fp in _iter_files(root):
        section = _file_section(root, fp, no_binary)
        if section:
            md.extend([section, ''])

    output_path = Path(output_path).expanduser().resolve()
    output_path.write_text('\n'.join(md), encoding='utf-8')
    logger.info(f'✅  Markdown written to {output_path}')


if __name__ == '__main__':
    fire.Fire(export)
