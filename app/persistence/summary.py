from app.persistence.db import PositionRecord, DailySummaryRecord
from app.config import settings
from app.util.clock import et_day_start_utc, et_today_iso

# Aggregates today's closed positions into a DailySummaryRecord. This record is
# the source the RiskEngine reads for drawdown / consecutive-loss kill switches,
# so it must be recomputed whenever a position closes.
def compute_daily_summary(session, starting_capital: float | None = None) -> DailySummaryRecord:
    starting_capital = starting_capital if starting_capital is not None else settings.starting_capital
    day_iso = et_today_iso()
    day_start = et_day_start_utc()

    closed = (session.query(PositionRecord)
              .filter(PositionRecord.status == "closed",
                      PositionRecord.opened_at >= day_start)
              .order_by(PositionRecord.closed_at.asc())
              .all())

    pnls = [float(p.pnl or 0.0) for p in closed]
    trade_count = len(pnls)
    total_pnl = round(sum(pnls), 4)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = round(len(wins) / trade_count, 4) if trade_count else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 4) if gross_loss > 0 else (
        float(len(wins)) if gross_win > 0 else 0.0)

    consecutive_losses = 0
    for p in reversed(pnls):
        if p < 0:
            consecutive_losses += 1
        else:
            break

    drawdown_pct = _peak_to_trough_drawdown(pnls, starting_capital)

    rec = DailySummaryRecord(
        date=day_iso, total_pnl=total_pnl, drawdown_pct=drawdown_pct,
        win_rate=win_rate, profit_factor=profit_factor, trade_count=trade_count,
        consecutive_losses=consecutive_losses,
        strategy_performance={settings.trading_symbol: {
            "win_rate": win_rate, "profit_factor": profit_factor, "trades": trade_count}},
        pipeline_version=settings.pipeline_version,
    )
    session.merge(rec)
    session.commit()
    return rec

def _peak_to_trough_drawdown(pnls: list[float], starting_capital: float) -> float:
    if not pnls or starting_capital <= 0:
        return 0.0
    equity = starting_capital
    peak = starting_capital
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return round(max_dd, 4)
