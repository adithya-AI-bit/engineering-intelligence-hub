import numpy as np
from ingest import collect_files, build_index


def _fake_embed(texts):
    return np.eye(len(texts), dtype=np.float32)


def test_collect_files_skips_ignored_directories(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# G\n\nbody\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("secret")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.md").write_text("# X\n\nnope\n")

    found = collect_files(str(tmp_path))
    assert "docs/guide.md" in found
    assert not any(".git" in f or "node_modules" in f for f in found)


def test_collect_files_skips_unsupported_extensions(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "readme.md").write_text("# R\n\nbody\n")
    assert collect_files(str(tmp_path)) == ["readme.md"]


def test_build_index_embeds_every_chunk(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha\n\n## B\n\nbeta\n")
    index = build_index(str(tmp_path), embed_fn=_fake_embed)
    assert len(index.chunks) == 2
    assert index.vectors.shape[0] == 2


def test_build_index_on_empty_directory_returns_empty_index(tmp_path):
    index = build_index(str(tmp_path), embed_fn=_fake_embed)
    assert index.chunks == []


def test_build_index_skips_undecodable_files(tmp_path):
    (tmp_path / "good.md").write_text("# G\n\nbody\n")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00binary")
    index = build_index(str(tmp_path), embed_fn=_fake_embed)
    assert [c.source for c in index.chunks] == ["good.md"]


def test_build_index_honours_exclude_prefixes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.md").write_text("# A\n\nalpha\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.md").write_text("# T\n\ntest text\n")

    index = build_index(str(tmp_path), embed_fn=_fake_embed, exclude=("tests/",))
    assert [c.source for c in index.chunks] == ["src/a.md"]


def test_build_index_without_exclude_includes_everything(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.md").write_text("# A\n\nalpha\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.md").write_text("# T\n\ntest text\n")

    assert len(build_index(str(tmp_path), embed_fn=_fake_embed).chunks) == 2
