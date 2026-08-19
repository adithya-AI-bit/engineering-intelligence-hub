import re

SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def recall_at_k(retrieved_sources, expected_source, k):
    return expected_source in retrieved_sources[:k]


def _split_sentences(answer):
    sentences = [s.strip() for s in SENTENCE_PATTERN.findall(answer) if s.strip()]
    # Text after the final terminator (or an answer with none at all) is still a
    # claim, and scoring it as zero sentences would report 0% groundedness for a
    # perfectly cited one-line answer.
    consumed = sum(len(s) for s in SENTENCE_PATTERN.findall(answer))
    remainder = answer[consumed:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def groundedness(answer, num_sources):
    sentences = _split_sentences(answer)
    cited = [s for s in sentences if CITATION_PATTERN.search(s)]
    markers = [int(m) for m in CITATION_PATTERN.findall(answer)]

    return {
        "sentences": len(sentences),
        "cited": len(cited),
        "ratio": round(len(cited) / len(sentences), 3) if sentences else 0.0,
        "invalid": sorted({m for m in markers if m < 1 or m > num_sources}),
    }
