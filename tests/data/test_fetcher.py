import pytest
from app.util.clock import now_utc
from app.data.fetcher import DataFetcher
from app.models.market import MarketData

@pytest.fixture
def fetcher():
    return DataFetcher(alpaca_api_key="test", alpaca_secret_key="test",
                       finnhub_api_key="test")

def _stub_bars(n=25):
    return [{"t": now_utc(), "o": 520.0, "h": 522.0,
             "l": 519.0, "c": 521.0, "v": 1_000_000.0}] * n

def test_fetch_returns_market_data(fetcher, mocker):
    mocker.patch.object(fetcher, "_fetch_bars_alpaca", return_value=_stub_bars())
    mocker.patch.object(fetcher, "_fetch_quote_alpaca", return_value=(521.0, 521.1))
    mocker.patch.object(fetcher, "_fetch_vix", return_value=18.5)
    mocker.patch.object(fetcher, "_fetch_news_sentiment", return_value=0.2)
    result = fetcher.fetch("SPY")
    assert isinstance(result, MarketData)
    assert result.symbol == "SPY"
    assert result.vix == 18.5

def test_spread_proxy_computed(fetcher, mocker):
    mocker.patch.object(fetcher, "_fetch_bars_alpaca", return_value=_stub_bars())
    mocker.patch.object(fetcher, "_fetch_quote_alpaca", return_value=(521.0, 521.2))
    mocker.patch.object(fetcher, "_fetch_vix", return_value=18.5)
    mocker.patch.object(fetcher, "_fetch_news_sentiment", return_value=0.0)
    result = fetcher.fetch("SPY")
    expected = (521.2 - 521.0) / ((521.0 + 521.2) / 2)
    assert abs(result.spread_proxy - expected) < 0.0001
