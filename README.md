# Engineering Intelligence Hub

![tests](https://github.com/adithya-AI-bit/engineering-intelligence-hub/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)

Ask questions about a codebase and its docs. Every claim in the answer cites the file and line range it came from, and the retrieval quality is measured rather than asserted.

No vector database. No RAG framework. Retrieval is a matrix multiply.

## How it works

```
Sources                docs, code, incident write-ups
   |
   v
Chunking               headers for prose, AST for Python, line windows otherwise
   |
   v
Embedding              OpenAI, or an offline lexical embedder with no API key
   |
   v
Index                  one normalised numpy matrix, saved with np.savez
   |
   v
Retrieve + answer      cosine top-k, answers carry [n] citations to file:lines
   |
   v
Eval harness           recall@k, groundedness, and which questions missed
```

## Measured results

Indexing this repository itself, 15 hand-written golden questions, offline lexical embedder:

| Configuration | Chunks | Recall@3 | Recall@5 |
|---|---|---|---|
| All files | 112 | 20.0% | 40.0% |
| `--exclude tests/` | 48 | 40.0% | 40.0% |
| `--exclude tests/ --exclude README.md` | 32 | 73.3% | **86.7%** |

Mean groundedness 100%, invalid citations 0, in all three runs.

These figures describe this repository at this commit. Because the corpus is the repository
itself, editing the code moves the numbers: adding the offline embedder and the server's
embedder-matching logic shifted recall@3 from 80.0% to 73.3% between two runs. Re-run the
command below rather than trusting the table after any change.

Reproduce with:

```bash
python ingest.py . --local --exclude tests/ --exclude README.md
python -m evaluation.run_eval --local --k 5
```

Those two exclusions are the entire reason the eval harness exists, and neither was
predictable from reading the code.

The first came from the miss report: test files repeat the vocabulary of the code they
test, so `tests/test_chunking.py` outranked `hub/chunking.py` for questions about chunking.

The second is stranger and was found by accident. An early version of this table was
measured before this README existed. Adding a README that describes the code created a
document that lexically matches every question about the code, and it displaced the source
files it was describing. Recall fell from 86.7% to 46.7% purely because the repository
gained a good README. Prose about code competes with code.

Both numbers come from the lexical embedder, which matches words rather than meaning. Real
OpenAI embeddings should handle paraphrased questions better, and may well be less
distracted by prose restating the same terms. That has not been measured here and no number
is claimed for it.

## Why no vector database

The index is a normalised `(n_chunks, dim)` float32 matrix. Search is `vectors @ query`, then `argsort`. For a repository-sized corpus this is exact, sub-millisecond, and needs nothing running. A vector database would add a service to operate in exchange for approximate results at a scale this project never reaches.

## Why no RAG framework

LangChain or LlamaIndex would install a large dependency tree to hide the parts worth understanding: how text is split, what goes in the prompt, how citations are attached, how retrieval is scored. All of that is a few hundred readable lines here. `hub/index.py` is 40.

## Prompt injection

Indexed content is untrusted. A README or a code comment in a repository can contain text shaped like an instruction. The system prompt in `hub/answer.py` states that source content is data to read and must never be followed, and every chunk is delimited with a numbered header in the user message. `tests/test_answer.py` asserts that a chunk containing "Ignore all previous instructions" is passed through as a source rather than promoted into the instructions.

## Project structure

```
engineering-intelligence-hub/
├── hub/
│   ├── chunking.py       header-aware, AST-aware, and fallback chunkers
│   ├── embed.py          OpenAI embeddings, batched and L2-normalised
│   ├── embed_local.py    offline lexical embedder, no API key
│   ├── index.py          numpy vector index, save and load
│   ├── answer.py         prompt, retrieval, citation parsing
│   └── config.py         models and limits
├── evaluation/
│   ├── metrics.py        recall@k and groundedness
│   ├── run_eval.py       scores the golden set, reports misses
│   └── golden.jsonl      15 questions with known answer locations
├── api/
│   ├── main.py           FastAPI: /, /ask, /health
│   └── static/index.html query UI
├── ingest.py             repository indexing CLI
└── tests/                59 tests
```

## Setup

```bash
git clone https://github.com/adithya-AI-bit/engineering-intelligence-hub.git
cd engineering-intelligence-hub
pip install -r requirements.txt
```

### Try it with no API key

```bash
pip install -r requirements.txt
python ingest.py . --local --exclude tests/ --exclude README.md
uvicorn api.main:app --reload
```

Open `http://localhost:8000` and ask something. The whole loop works without an account.

The index records which embedder built it, so the server matches it automatically. In this
mode retrieval is lexical, and instead of writing an answer the system quotes the top-ranked
chunk verbatim with its citation. You are seeing real retrieval and real citations, not
generated prose.

### With OpenAI

```bash
pip install -r requirements.txt
cp .env.example .env
# put your key in .env, which is loaded automatically
python ingest.py /path/to/some/repo --exclude tests/
python -m evaluation.run_eval --k 5
uvicorn api.main:app --reload
```

Embedding a repository of this size with `text-embedding-3-small` costs well under a cent. Answers use `gpt-5.6-luna` by default; override with `OPENAI_MODEL`.

## Writing your own golden set

`evaluation/golden.jsonl` is one JSON object per line:

```json
{"question": "How is cosine similarity search implemented?", "expected_source": "hub/index.py"}
```

Pick a repository you know well, ask questions whose answer location you are certain of, and let the miss report tell you where retrieval breaks. That report is the point of the harness.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

59 tests, no network calls. Embedders and the language model are stubbed at the call boundary.
CI runs them on Python 3.11, 3.12 and 3.13, and also runs the keyless pipeline end to end so
the quickstart above cannot silently rot.

## What was and wasn't tested

Verified: all 59 tests; indexing this repository through the CLI (21 files, 103 chunks, or 29 with tests and README excluded); index save and reload producing byte-identical vectors; the eval harness end to end producing the table above, re-run from a clean extraction of the packaged zip so the figures reproduce; the API answering over the real persisted index with citations resolving to real file and line ranges; and that the offline embedder produces identical vectors across separate processes, which it did not before `zlib.crc32` replaced Python's per-process-randomised `hash()`.

Not verified: live OpenAI embedding and answering, which the sandbox blocked at the network layer, and the browser UI, which needs a browser. The code around both is tested with stubs at the call boundary; the responses coming back from OpenAI are not.

Tested on Python 3.12. The code targets 3.11 and newer.

## License

MIT. See [LICENSE](LICENSE).

## Stack

Python, FastAPI, numpy, OpenAI API, pytest.
