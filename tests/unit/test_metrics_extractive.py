import pytest
from tests.evaluation.metrics import RAGMetrics

@pytest.fixture
def m():
    return RAGMetrics()  # no LLM needed

class TestNormalize:
    def test_lowercase(self, m):
        assert m.normalize("Hello World") == "hello world"

    def test_removes_articles(self, m):
        assert m.normalize("the cat sat on an mat") == "cat sat on mat"

    def test_removes_punctuation(self, m):
        assert m.normalize("hello, world!") == "hello world"

    def test_nfkd_accent(self, m):
        assert m.normalize("résumé") == "resume"

    def test_collapses_whitespace(self, m):
        assert m.normalize("  hello   world  ") == "hello world"

    def test_empty_string(self, m):
        assert m.normalize("") == ""


class TestExactMatch:
    def test_identical(self, m):
        assert m.calculate_exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self, m):
        assert m.calculate_exact_match("paris", "Paris") == 1.0

    def test_article_stripped(self, m):
        assert m.calculate_exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_different(self, m):
        assert m.calculate_exact_match("London", "Paris") == 0.0

    def test_empty_both(self, m):
        assert m.calculate_exact_match("", "") == 1.0


class TestTokenF1:
    def test_identical(self, m):
        assert m.calculate_token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_partial_overlap(self, m):
        # "on" is NOT an article, so it survives normalize
        # pred_tokens=["cat","sat"], gt_tokens=["cat","on","mat"]
        # common={"cat":1}=1, precision=1/2, recall=1/3, f1=2/5=0.4
        result = m.calculate_token_f1("cat sat", "cat on mat")
        assert result == pytest.approx(0.4, abs=1e-3)

    def test_counter_based_not_set_based(self, m):
        # pred="cat cat dog", gt="cat dog dog"
        # pred_counts={cat:2, dog:1}, gt_counts={cat:1, dog:2}
        # common = min sums = {cat:1, dog:1} = 2
        # precision = 2/3, recall = 2/3, f1 = 2/3
        result = m.calculate_token_f1("cat cat dog", "cat dog dog")
        assert result == pytest.approx(2 / 3, abs=1e-3)

    def test_empty_prediction(self, m):
        assert m.calculate_token_f1("", "Paris") == 0.0

    def test_empty_ground_truth(self, m):
        assert m.calculate_token_f1("Paris", "") == 0.0

    def test_both_empty(self, m):
        assert m.calculate_token_f1("", "") == 1.0
