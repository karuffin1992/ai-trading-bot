from typing import get_args

from app.pipeline.models import ReplayMode


def test_replay_modes_include_memory_only():
    modes = set(get_args(ReplayMode))
    assert {"full_pipeline", "feature_only", "strategy_only", "ai_only",
            "memory_only"} <= modes
