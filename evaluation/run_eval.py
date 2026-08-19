import argparse
import json
import statistics
from hub.index import VectorIndex
from hub.embed import embed_texts
from hub.answer import answer_question
from evaluation.metrics import recall_at_k, groundedness


def load_golden(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip() and not line.lstrip().startswith("#")]


def evaluate(cases, index, k, embed_fn=embed_texts, llm_fn=None):
    from hub.answer import default_llm
    llm_fn = llm_fn or default_llm

    hits, ratios, invalid_total, misses = 0, [], 0, []
    for case in cases:
        result = answer_question(case["question"], index, embed_fn=embed_fn, llm_fn=llm_fn, k=k)
        sources = [s["source"] for s in result["sources"]]

        if recall_at_k(sources, case["expected_source"], k):
            hits += 1
        else:
            misses.append({"question": case["question"],
                           "expected": case["expected_source"], "retrieved": sources[:k]})

        scored = groundedness(result["answer"], len(result["sources"]))
        ratios.append(scored["ratio"])
        invalid_total += len(scored["invalid"])

    return {
        "cases": len(cases),
        "recall_at_k": hits / len(cases) if cases else 0.0,
        "mean_groundedness": statistics.mean(ratios) if ratios else 0.0,
        "invalid_citations": invalid_total,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser(description="Score retrieval and groundedness against the golden set")
    parser.add_argument("--index", default="index.npz")
    parser.add_argument("--golden", default="evaluation/golden.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--local", action="store_true",
                        help="offline embedder and a stub answerer; scores retrieval only")
    args = parser.parse_args()

    if args.local:
        from hub.embed_local import embed_texts_local
        report = evaluate(load_golden(args.golden), VectorIndex.load(args.index), args.k,
                          embed_fn=embed_texts_local,
                          llm_fn=lambda system, user: "Offline mode, retrieval only [1].")
    else:
        report = evaluate(load_golden(args.golden), VectorIndex.load(args.index), args.k)

    print(f"Cases:              {report['cases']}")
    print(f"Recall@{args.k}:           {report['recall_at_k']:.1%}")
    print(f"Mean groundedness:  {report['mean_groundedness']:.1%}")
    print(f"Invalid citations:  {report['invalid_citations']}")

    if report["misses"]:
        print(f"\nRetrieval misses ({len(report['misses'])}):")
        for miss in report["misses"]:
            print(f"  - {miss['question']}")
            print(f"    expected {miss['expected']}, got {miss['retrieved']}")


if __name__ == "__main__":
    main()
