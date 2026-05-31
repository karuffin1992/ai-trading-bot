import os
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
from app.persistence.db import Base, KillSwitchRecord, make_engine, get_session
engine = make_engine()
Base.metadata.create_all(engine)
with get_session(engine) as s:
    if not s.get(KillSwitchRecord, 1):
        s.add(KillSwitchRecord(id=1, active=False, reason=""))
        s.commit()
print("Database initialized.")
