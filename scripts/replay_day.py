"""
Replay a stored pipeline cycle for debugging and prompt tuning.

Usage:
  python scripts/replay_day.py --date 2026-05-16
  python scripts/replay_day.py --cycle-id <uuid>
  python scripts/replay_day.py --date 2026-05-16 --mode ai_only
"""
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.persistence.db import CycleRecord
from app.config import settings

def load_cycle(engine, date_str, cycle_id):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        if cycle_id:
            return s.get(CycleRecord, cycle_id)
        if date_str:
            return s.query(CycleRecord).filter(
                CycleRecord.started_at >= f"{date_str}T00:00:00",
                CycleRecord.started_at <  f"{date_str}T23:59:59",
            ).order_by(CycleRecord.started_at.desc()).first()
    return None

def replay(cycle: CycleRecord, mode: str) -> None:
    print(f"\n=== Replay {cycle.cycle_id} | mode={mode} ===\n")

    if mode == "feature_only":
        from app.models.market import MarketData
        from app.features.pipeline import FeaturePipeline
        fs = FeaturePipeline.compute(MarketData(**cycle.market_data_json))
        print(fs.model_dump_json(indent=2))

    elif mode == "strategy_only":
        from app.models.market import FeatureSet
        from app.strategies.spy_trend import SpyTrendStrategy
        result = SpyTrendStrategy.evaluate(FeatureSet(**cycle.features_json))
        print(result.model_dump_json(indent=2))

    elif mode == "ai_only":
        from app.models.market import FeatureSet
        from app.models.signals import TradeSignal, TradeRejection
        from app.ai.analyst import AIAnalyst
        fs  = FeatureSet(**cycle.features_json)
        sig_data = cycle.signal_json
        sig = (TradeSignal(**sig_data) if sig_data.get("type") == "SIGNAL"
               else TradeRejection(**sig_data))
        print(AIAnalyst().analyze(sig, fs).model_dump_json(indent=2))

    elif mode == "full_pipeline":
        from app.models.market import MarketData
        from app.features.pipeline import FeaturePipeline
        from app.strategies.spy_trend import SpyTrendStrategy
        from app.ai.analyst import AIAnalyst
        md  = MarketData(**cycle.market_data_json)
        fs  = FeaturePipeline.compute(md)
        sig = SpyTrendStrategy.evaluate(fs)
        print(AIAnalyst().analyze(sig, fs).model_dump_json(indent=2))

    print("\n=== Done ===")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date")
    p.add_argument("--cycle-id")
    p.add_argument("--mode", default="full_pipeline",
                   choices=["full_pipeline","feature_only","strategy_only","ai_only"])
    args = p.parse_args()

    engine = create_engine(settings.database_url)
    cycle = load_cycle(engine, args.date, getattr(args, "cycle_id", None))
    if not cycle:
        print("No cycle found."); return
    replay(cycle, args.mode)

if __name__ == "__main__":
    main()
