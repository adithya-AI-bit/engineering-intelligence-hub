from hub.chunking import chunk_markdown, chunk_python, chunk_file


def test_markdown_splits_on_headers_and_keeps_header_path():
    text = "# Guide\n\nIntro line.\n\n## Setup\n\nRun the installer.\n\n## Usage\n\nCall the API.\n"
    chunks = chunk_markdown(text, "guide.md")
    assert len(chunks) == 3
    assert chunks[1].context == "Guide > Setup"
    assert "Run the installer." in chunks[1].text
    assert chunks[1].source == "guide.md"


def test_markdown_records_line_numbers():
    text = "# A\n\nfirst\n\n## B\n\nsecond\n"
    chunks = chunk_markdown(text, "a.md")
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 5


def test_python_splits_on_top_level_definitions():
    text = "import os\n\n\ndef alpha():\n    return 1\n\n\nclass Beta:\n    def method(self):\n        return 2\n"
    chunks = chunk_python(text, "mod.py")
    names = [c.context for c in chunks]
    assert "mod.py::alpha" in names
    assert "mod.py::Beta" in names


def test_python_chunk_line_range_covers_the_definition():
    text = "def alpha():\n    return 1\n"
    chunk = chunk_python(text, "mod.py")[0]
    assert chunk.start_line == 1
    assert chunk.end_line == 2


def test_python_falls_back_to_line_windows_on_syntax_error():
    chunks = chunk_python("def broken(\n", "bad.py")
    assert len(chunks) >= 1
    assert chunks[0].source == "bad.py"


def test_chunk_file_dispatches_on_extension():
    assert chunk_file("notes.md", "# T\n\nbody\n")[0].context == "T"
    assert chunk_file("m.py", "def f():\n    pass\n")[0].context == "m.py::f"
