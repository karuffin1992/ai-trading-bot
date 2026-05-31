import pytest
from datetime import datetime, date
from uuid import uuid4

@pytest.fixture
def cycle_id():
    return uuid4()

@pytest.fixture
def trade_date():
    return date(2026, 5, 16)
