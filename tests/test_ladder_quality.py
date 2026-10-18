import os
import subprocess

from pokesim.config import ROOT
from scripts.ladder import search_quality_issues


def test_terminal_search_errors_stop_collection_but_incomplete_final_trial_is_expected():
    healthy = {'search': {'terminal_search': {'failed': 0, 'unfinished_at_deadline': 1, 'censored': 2}}}
    assert search_quality_issues(healthy, 8.5) == []
    assert search_quality_issues({'fallback': 'search timed out'}, 9) == ['search fallback']
    assert search_quality_issues({'search': {'terminal_search': {'failed': 1}}}, 8) == ['simulation failed']
    assert search_quality_issues(healthy, 10.1) == ['decision exceeded 10 seconds']


def test_collector_does_not_restart_after_a_quality_failure(tmp_path):
    stub = tmp_path / 'python'
    stub.write_text('#!/bin/sh\nexit 78\n')
    stub.chmod(0o700)
    env = dict(os.environ, COLLECT_DIR=str(tmp_path), PYTHON=str(stub), MIN_FREE_MB='0')
    result = subprocess.run(['bash', str(ROOT / 'scripts/collect.sh')], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert (tmp_path / 'STOP').exists()
    assert 'search quality gate failed' in (tmp_path / 'status.log').read_text()
