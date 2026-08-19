import os
from dotenv import load_dotenv

# Every entry point imports this module, so loading here means a .env file works
# for the CLI, the eval harness and the server without each one remembering to.
load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
EMBED_BATCH_SIZE = 100
TOP_K = 6
INDEX_PATH = os.environ.get("INDEX_PATH", "index.npz")

OPENAI_EMBEDDER = "openai"
LOCAL_EMBEDDER = "local"
