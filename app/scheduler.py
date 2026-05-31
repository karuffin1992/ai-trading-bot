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

def _report_job():
    logger.info("Daily report job complete")

def create_scheduler() -> BackgroundScheduler:
    s = BackgroundScheduler(timezone="America/New_York")
    s.add_listener(lambda e: logger.error("Job failed: %s — %s", e.job_id, e.exception),
                   EVENT_JOB_ERROR)
    s.add_job(_guard(_assessment_job), "cron", hour=9, minute=45,
              id="assessment", **_DEFAULTS)
    s.add_job(_guard(_report_job), "cron", hour=16, minute=10,
              id="daily_report", **_DEFAULTS)
    return s
