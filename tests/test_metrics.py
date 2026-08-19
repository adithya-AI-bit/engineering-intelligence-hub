from evaluation.metrics import recall_at_k, groundedness


def test_recall_at_k_true_when_expected_source_in_top_k():
    assert recall_at_k(["a.md", "b.md", "c.md"], "b.md", k=3) is True


def test_recall_at_k_false_when_expected_source_below_k():
    assert recall_at_k(["a.md", "b.md", "c.md"], "c.md", k=2) is False


def test_recall_at_k_on_empty_retrieval():
    assert recall_at_k([], "a.md", k=5) is False


def test_groundedness_counts_cited_sentences():
    result = groundedness("Backoff is used [1]. It doubles each retry [1].", num_sources=2)
    assert result["sentences"] == 2
    assert result["cited"] == 2
    assert result["ratio"] == 1.0


def test_groundedness_flags_uncited_sentences():
    result = groundedness("Backoff is used [1]. This part is invented.", num_sources=2)
    assert result["cited"] == 1
    assert result["ratio"] == 0.5


def test_groundedness_flags_citation_out_of_range():
    assert groundedness("Claim [7].", num_sources=2)["invalid"] == [7]


def test_groundedness_accepts_in_range_citations():
    assert groundedness("Claim [2].", num_sources=2)["invalid"] == []


def test_groundedness_on_empty_answer_is_zero_not_a_crash():
    assert groundedness("", num_sources=2)["ratio"] == 0.0


def test_groundedness_counts_text_after_the_last_terminator():
    result = groundedness("a claim with no full stop [1]", num_sources=1)
    assert result["sentences"] == 1
    assert result["cited"] == 1
    assert result["ratio"] == 1.0


def test_groundedness_counts_trailing_fragment_alongside_full_sentences():
    result = groundedness("First claim [1]. trailing uncited fragment", num_sources=1)
    assert result["sentences"] == 2
    assert result["cited"] == 1
    assert result["ratio"] == 0.5
