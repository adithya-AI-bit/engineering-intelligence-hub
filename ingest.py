import argparse
import os
from hub.chunking import chunk_file
from hub.embed import embed_texts
from hub.index import VectorIndex
from hub.config import INDEX_PATH, OPENAI_EMBEDDER, LOCAL_EMBEDDER

IGNORED_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv",
                "dist", "build", ".next", ".pytest_cache", "site-packages"}
SUPPORTED = (".py", ".md", ".markdown", ".txt", ".rst", ".js", ".jsx", ".ts", ".tsx", ".go", ".java")


def collect_files(root):
    found = []
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in IGNORED_DIRS]
        for name in filenames:
            if not name.endswith(SUPPORTED):
                continue
            full = os.path.join(directory, name)
            found.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(found)


def build_index(root, embed_fn=embed_texts, exclude=(), embedder=OPENAI_EMBEDDER):
    chunks = []
    for relative in collect_files(root):
        if any(relative.startswith(prefix) for prefix in exclude):
            continue
        try:
            with open(os.path.join(root, relative), encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            continue
        chunks.extend(chunk_file(relative, text))

    index = VectorIndex(embedder=embedder)
    if chunks:
        index.add(chunks, embed_fn([f"{c.context}\n{c.text}" for c in chunks]))
    return index


def main():
    parser = argparse.ArgumentParser(description="Index a repository for the Engineering Intelligence Hub")
    parser.add_argument("root")
    parser.add_argument("--out", default=INDEX_PATH)
    parser.add_argument("--local", action="store_true",
                        help="use the offline lexical embedder instead of the OpenAI API")
    parser.add_argument("--exclude", action="append", default=[], metavar="PREFIX",
                        help="skip paths starting with PREFIX; repeatable. "
                             "Excluding 'tests/' measurably improves retrieval, see the README")
    args = parser.parse_args()

    embed_fn, embedder = embed_texts, OPENAI_EMBEDDER
    if args.local:
        from hub.embed_local import embed_texts_local
        embed_fn, embedder = embed_texts_local, LOCAL_EMBEDDER

    index = build_index(args.root, embed_fn=embed_fn, exclude=tuple(args.exclude), embedder=embedder)
    index.save(args.out)
    print(f"Indexed {len(index.chunks)} chunks from {args.root} into {args.out} "
          f"using the {embedder} embedder")


if __name__ == "__main__":
    main()
