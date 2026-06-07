import json
import os
import time
import textwrap
from uuid import UUID

from app.models.market import MarketData
from app.models.signals import TradeSignal
from app.features.pipeline import FeaturePipeline
from app.strategies.spy_trend import SpyTrendStrategy
from app.ai.analyst import AIAnalyst
from app.config import settings
from tests.golden.canonical import canonicalize, sha256_of

CASES_DIR = os.path.join(os.path.dirname(__file__), "replay_cases")
EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "expected_hashes")
CANONICAL_SCHEMA_VERSION = "1.0.0"
SENTINEL_TRADE_ID = UUID(int=0)
PREVIEW_WIDTH = 300


def _versions() -> dict:
    return {
        "pipeline": settings.pipeline_version,
        "strategy": settings.strategy_version,
        "prompt": settings.prompt_version,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
    }


def run_case(case: dict) -> dict:
    md = MarketData(**case["market_data"])

    t0 = time.perf_counter()
    features = FeaturePipeline.compute(md)
    features_ms = (time.perf_counter() - t0) * 1000.0
    features_dump = features.model_dump(mode="json")
    features_hash = sha256_of(features_dump)

    t1 = time.perf_counter()
    result = SpyTrendStrategy.evaluate(features)
    strategy_ms = (time.perf_counter() - t1) * 1000.0

    if isinstance(result, TradeSignal):
        result = result.model_copy(update={"trade_id": SENTINEL_TRADE_ID})
    signal_dump = result.model_dump(mode="json")
    signal_hash = sha256_of(signal_dump)

    prompt_hash = None
    prompt_preview = None
    prompt = None
    if isinstance(result, TradeSignal):
        analyst = AIAnalyst.__new__(AIAnalyst)  # no client, no network
        ctx = analyst.build_prompt_context(result, features, account_balance=100.0)
        prompt = analyst.build_prompt(ctx)
        prompt_hash = sha256_of(prompt)
        prompt_preview = textwrap.shorten(prompt, width=PREVIEW_WIDTH, placeholder="...")

    return {
        "features_hash": features_hash,
        "signal_hash": signal_hash,
        "prompt_hash": prompt_hash,
        "retrieval_hash": None,
        "prompt_preview": prompt_preview,
        "versions": _versions(),
        "timing": {
            "features_runtime_ms": round(features_ms, 3),
            "strategy_runtime_ms": round(strategy_ms, 3),
        },
        "_snapshots": {
            "features": features_dump,
            "signal": signal_dump,
            "prompt": prompt,
        },
    }


def expected_payload(result: dict) -> dict:
    # The blessed-file shape: drop the ephemeral _snapshots and prompt full text.
    return {
        "features_hash": result["features_hash"],
        "signal_hash": result["signal_hash"],
        "prompt_hash": result["prompt_hash"],
        "retrieval_hash": result["retrieval_hash"],
        "prompt_preview": result["prompt_preview"],
        "versions": result["versions"],
        "timing": result["timing"],
    }


def load_case(name: str) -> dict:
    with open(os.path.join(CASES_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_expected(name: str) -> dict:
    with open(os.path.join(EXPECTED_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def write_expected(name: str, payload: dict) -> None:
    os.makedirs(EXPECTED_DIR, exist_ok=True)
    with open(os.path.join(EXPECTED_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def list_cases() -> list[str]:
    if not os.path.isdir(CASES_DIR):
        return []
    return sorted(n for n in os.listdir(CASES_DIR) if n.endswith(".json"))
