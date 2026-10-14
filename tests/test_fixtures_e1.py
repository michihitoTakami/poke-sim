"""E1 tactical fixtures (pokesim.evaluation.fixtures): both the legacy method and the planner must pass the ones whose
answer is clear of the rollouts' noise."""
import pytest

from pokesim.evaluation.fixtures import Worker, load, run

FIXTURES = [fx for fx in load() if not fx.get("known_noisy")]


@pytest.fixture(scope="module")
def worker():
    w = Worker()
    yield w
    w.close()


@pytest.mark.parametrize("method", ["legacy", "planner"])
@pytest.mark.parametrize("fixture", FIXTURES, ids=[fx["name"] for fx in FIXTURES])
def test_fixture(worker, fixture, method):
    (outcome,) = run([fixture], [method], worker)
    assert outcome.passed, (outcome.top, outcome.values, outcome.checks)


def test_fixture_files_are_well_formed():
    for fx in load():
        assert {"name", "category", "note", "samples", "expect"} <= set(fx), fx["file"]
        assert fx["file"] == f"{fx['name']}.json"
