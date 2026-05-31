from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def now_et() -> datetime:
    return datetime.now(ET)

# Start of the current ET trading day, expressed as a naive UTC datetime so it
# can be compared against created_at columns stored via now_utc().
def et_day_start_utc() -> datetime:
    et = datetime.now(ET)
    midnight_et = et.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_et.astimezone(timezone.utc).replace(tzinfo=None)

def et_today_iso() -> str:
    return datetime.now(ET).date().isoformat()
