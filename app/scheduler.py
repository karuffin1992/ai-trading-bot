from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR
import pandas_market_calendars as mcal
from datetime import date
import logging

logger = logging.getLogger(__name__)
_DEFAULTS = dict(max_instances=1, coalesce=True, misfire_grace_time=30)

def _market_open() -> bool:
    nyse = mcal.get_calendar("NYSE")
    return not nyse.schedule(start_date=date.today().isoformat(),
                              end_date=date.today().isoformat()).empty

def _guard(fn):
    def wrapper(*a, **kw):
        if not _market_open():
            logger.info("Market closed — skipping %s", fn.__name__)
            return
        return fn(*a, **kw)
    wrapper.__name__ = fn.__name__
    return wrapper

def _assessment_job():
    from app.pipeline.trading_pipeline import TradingPipeline
    ctx = TradingPipeline().run_assessment()
    logger.info("Cycle done: %s errors=%s", ctx.cycle_id, ctx.errors)

def _monitor_job():
    from app.execution.position_manager import PositionManager
    from app.persistence.db import make_session_factory
    PositionManager(make_session_factory()).sync()
    logger.info("Position monitor sync complete")

def _close_positions_job():
    from app.execution.position_manager import PositionManager
    from app.persistence.db import make_session_factory
    PositionManager(make_session_factory()).close_all()
    logger.info("EOD positions flattened")

def _report_job():
    from app.persistence.db import make_session_factory
    from app.persistence.summary import compute_daily_summary
    sf = make_session_factory()
    with sf() as s:
        summary = compute_daily_summary(s)
    logger.info("Daily report: pnl=%.2f trades=%d drawdown=%.2f%%",
                summary.total_pnl, summary.trade_count, summary.drawdown_pct * 100)

def create_scheduler() -> BackgroundScheduler:
    s = BackgroundScheduler(timezone="America/New_York")
    s.add_listener(lambda e: logger.error("Job failed: %s — %s", e.job_id, e.exception),
                   EVENT_JOB_ERROR)
    s.add_job(_guard(_assessment_job), "cron", hour=9, minute=45,
              id="assessment", **_DEFAULTS)
    s.add_job(_guard(_monitor_job), "interval", minutes=5,
              id="position_monitor", **_DEFAULTS)
    s.add_job(_guard(_close_positions_job), "cron", hour=15, minute=55,
              id="close_positions", **_DEFAULTS)
    s.add_job(_guard(_report_job), "cron", hour=16, minute=10,
              id="daily_report", **_DEFAULTS)
    return s
