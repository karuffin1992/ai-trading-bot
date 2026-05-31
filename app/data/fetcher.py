from datetime import datetime, timedelta
from app.models.market import MarketData
from app.config import settings

class DataFetcher:
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, finnhub_api_key: str):
        from alpaca.data.historical import StockHistoricalDataClient
        import finnhub
        self._data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        self._finnhub = finnhub.Client(api_key=finnhub_api_key)

    def fetch(self, symbol: str) -> MarketData:
        bars = self._fetch_bars_alpaca(symbol)
        bid, ask = self._fetch_quote_alpaca(symbol)
        vix = self._fetch_vix()
        sentiment = self._fetch_news_sentiment(symbol)
        mid = (bid + ask) / 2
        spread_proxy = (ask - bid) / mid if mid > 0 else 0.0
        latest = bars[-1] if bars else {}
        return MarketData(
            symbol=symbol, timestamp=datetime.utcnow(),
            pipeline_version=settings.pipeline_version,
            open=latest.get("o", 0.0), high=latest.get("h", 0.0),
            low=latest.get("l", 0.0), close=latest.get("c", 0.0),
            volume=latest.get("v", 0.0),
            bid=bid, ask=ask, spread_proxy=spread_proxy,
            bars_daily=bars, vix=vix, news_sentiment=sentiment,
        )

    def _fetch_bars_alpaca(self, symbol: str) -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=datetime.utcnow() - timedelta(days=30),
            end=datetime.utcnow(),
        )
        bars = self._data_client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return []
        return [{"t": idx[1] if isinstance(idx, tuple) else idx,
                 "o": float(row["open"]), "h": float(row["high"]),
                 "l": float(row["low"]), "c": float(row["close"]),
                 "v": float(row["volume"])}
                for idx, row in df.iterrows()]

    def _fetch_quote_alpaca(self, symbol: str) -> tuple[float, float]:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self._data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        return float(q.bid_price), float(q.ask_price)

    def _fetch_vix(self) -> float:
        try:
            import yfinance as yf
            return float(yf.Ticker("^VIX").fast_info["last_price"])
        except Exception:
            return 20.0

    def _fetch_news_sentiment(self, symbol: str) -> float:
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            news = self._finnhub.company_news(symbol, _from=today, to=today)
            scores = [n["sentiment"]["score"] for n in (news or [])[:10]
                      if "sentiment" in n]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

def make_fetcher() -> DataFetcher:
    return DataFetcher(
        alpaca_api_key=settings.alpaca_api_key,
        alpaca_secret_key=settings.alpaca_secret_key,
        finnhub_api_key=settings.finnhub_api_key,
    )
