import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from app.util.clock import now_utc
from app.pipeline.trading_pipeline import TradingPipeline
from app.pipeline.context import PipelineContext
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult

def stub_pipeline():
    p = TradingPipeline.__new__(TradingPipeline)
    p.fetcher   = MagicMock()
    p.strategy  = MagicMock()
    p.analyst   = MagicMock()
    p.risk      = MagicMock()
    p.validator = MagicMock()
    p.executor  = MagicMock()
    p.logger    = MagicMock()
    return p

def wire_happy(p):
    p.fetcher.fetch.return_value = MagicMock(spec=MarketData, vix=18.0, symbol="SPY",
        timestamp=now_utc(), pipeline_version="1.0.0",
        open=520.0, high=522.0, low=519.0, close=521.0, volume=1e6,
        bid=521.0, ask=521.1, spread_proxy=0.0001, news_sentiment=0.1, bars_daily=[])
    p.strategy.evaluate.return_value = MagicMock(spec=TradeSignal, type="SIGNAL",
        trade_id=uuid4(), strategy_confidence=0.74)
    p.strategy.evaluate.return_value.model_dump_json.return_value = "{}"
    p.analyst.analyze.return_value = MagicMock(spec=AIAnalysis, decision="APPROVE",
        ai_confidence=0.81, failed=False)
    p.analyst.analyze.return_value.model_dump_json.return_value = "{}"
    p.risk.evaluate.return_value = MagicMock(spec=RiskDecision, outcome="APPROVED")
    p.validator.validate.return_value = MagicMock(spec=ValidationResult, outcome="PASS")
    p.executor.execute.return_value = None

def test_happy_path(mocker):
    p = stub_pipeline()
    wire_happy(p)
    feats = MagicMock(spec=FeatureSet, vix=18.0)
    feats.model_dump_json.return_value = "{}"
    mocker.patch("app.pipeline.trading_pipeline.FeaturePipeline.compute",
                 return_value=feats)
    ctx = p.run_assessment()
    assert isinstance(ctx, PipelineContext)
    p.logger.write_cycle.assert_called_once()

def test_fetch_error_logged(mocker):
    p = stub_pipeline()
    p.fetcher.fetch.side_effect = RuntimeError("API down")
    ctx = p.run_assessment()
    assert any("API down" in e for e in ctx.errors)
    p.logger.write_cycle.assert_called_once()

def test_kill_skips_execution(mocker):
    p = stub_pipeline()
    wire_happy(p)
    mocker.patch("app.pipeline.trading_pipeline.FeaturePipeline.compute",
                 return_value=MagicMock(spec=FeatureSet, vix=18.0))
    p.risk.evaluate.return_value = MagicMock(spec=RiskDecision, outcome="KILL")
    p.run_assessment()
    p.executor.execute.assert_not_called()

def _wire_features(mocker):
    feats = MagicMock(spec=FeatureSet, vix=18.0)
    feats.model_dump_json.return_value = "{}"
    mocker.patch("app.pipeline.trading_pipeline.FeaturePipeline.compute",
                 return_value=feats)

def test_ai_no_trade_vetoes(mocker):
    p = stub_pipeline()
    wire_happy(p)
    _wire_features(mocker)
    p.analyst.analyze.return_value = MagicMock(spec=AIAnalysis, decision="NO_TRADE",
        ai_confidence=0.9, failed=False)
    p.run_assessment()
    p.risk.evaluate.assert_not_called()
    p.executor.execute.assert_not_called()

def test_ai_low_confidence_vetoes(mocker):
    p = stub_pipeline()
    wire_happy(p)
    _wire_features(mocker)
    p.analyst.analyze.return_value = MagicMock(spec=AIAnalysis, decision="APPROVE",
        ai_confidence=0.5, failed=False)
    p.run_assessment()
    p.risk.evaluate.assert_not_called()
    p.executor.execute.assert_not_called()

def test_balance_passed_to_executor(mocker):
    p = stub_pipeline()
    wire_happy(p)
    _wire_features(mocker)
    mocker.patch.object(p, "_account_balance", return_value=250.0)
    p.run_assessment()
    assert p.executor.execute.call_args.kwargs["account_balance"] == 250.0
