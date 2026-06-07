import pytest
from tests.golden import golden_runner as gr

CASES = gr.list_cases()


@pytest.mark.parametrize("name", CASES)
def test_golden_case_matches_expected(name):
    assert CASES, "no replay cases found — run python -m tests.golden._gen_fixtures"
    try:
        expected = gr.load_expected(name)
    except FileNotFoundError:
        pytest.fail(f"no expected file for {name} — run "
                    f"`python -m tests.golden.golden_runner --update {name}` to bless")
    result = gr.run_case(gr.load_case(name))

    assert result["features_hash"] == expected["features_hash"], f"{name}: features drift"
    assert result["signal_hash"] == expected["signal_hash"], f"{name}: signal drift"
    assert result["prompt_hash"] == expected["prompt_hash"], f"{name}: prompt drift"
    assert result["retrieval_hash"] == expected["retrieval_hash"]
    assert result["versions"] == expected["versions"], (
        f"{name}: version changed without re-bless — review and rerun --update")
