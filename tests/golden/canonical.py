import hashlib
import json
import math
from datetime import datetime, date
from decimal import Decimal

FLOAT_DP = 6


def _normalize(obj):
    # numpy scalars -> python primitives (no hard numpy dependency).
    if type(obj).__module__ == "numpy":
        item = getattr(obj, "item", None)
        if callable(item):
            obj = item()

    if isinstance(obj, bool):
        return obj
    if isinstance(obj, Decimal):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return round(obj, FLOAT_DP)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(obj, dict):
        # Keys coerced to str; values normalized; ordering handled at dump time.
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]  # order preserved
    if obj is None:
        return None
    raise TypeError(f"canonicalize: unsupported type {type(obj)!r}")


def canonicalize(obj) -> str:
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_of(obj) -> str:
    return hashlib.sha256(canonicalize(obj).encode("utf-8")).hexdigest()
