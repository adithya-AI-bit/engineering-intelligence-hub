import ast
import re
from dataclasses import dataclass

FALLBACK_WINDOW_LINES = 40


@dataclass
class Chunk:
    text: str
    source: str
    start_line: int
    end_line: int
    context: str


def chunk_markdown(text, source):
    lines = text.splitlines()
    header_stack = []
    chunks = []
    body = []
    start = 1

    def flush(end_line):
        content = "\n".join(body).strip()
        if content:
            chunks.append(Chunk(content, source, start, end_line, " > ".join(header_stack)))

    for number, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if not match:
            body.append(line)
            continue

        flush(number - 1)
        depth = len(match.group(1))
        header_stack[:] = header_stack[:depth - 1] + [match.group(2).strip()]
        body = []
        start = number

    flush(len(lines))
    return chunks


def _window_chunks(text, source):
    lines = text.splitlines()
    chunks = []
    for offset in range(0, len(lines), FALLBACK_WINDOW_LINES):
        window = lines[offset:offset + FALLBACK_WINDOW_LINES]
        content = "\n".join(window).strip()
        if content:
            chunks.append(Chunk(content, source, offset + 1, offset + len(window), source))
    return chunks


def chunk_python(text, source):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _window_chunks(text, source)

    lines = text.splitlines()
    chunks = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = node.lineno, node.end_lineno
        content = "\n".join(lines[start - 1:end]).strip()
        if content:
            chunks.append(Chunk(content, source, start, end, f"{source}::{node.name}"))

    return chunks or _window_chunks(text, source)


def chunk_file(path, text):
    if path.endswith(".py"):
        return chunk_python(text, path)
    if path.endswith((".md", ".markdown", ".txt", ".rst")):
        return chunk_markdown(text, path)
    return _window_chunks(text, path)
