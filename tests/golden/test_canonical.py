import json
import math
from decimal import Decimal
import numpy as np
from tests.golden.canonical import canonicalize, sha256_of


def test_float_rounded_to_6dp():
    assert canonicalize(1.234567891) == json.dumps(1.234568)
    # Sub-6dp noise collapses to the same canonical form.
    assert canonicalize(1.2345670001) == canonicalize(1.2345670002)


def test_nan_and_inf_become_strings():
    assert canonicalize(float("nan")) == json.dumps("NaN")
    assert canonicalize(float("inf")) == json.dumps("Infinity")
    assert canonicalize(float("-inf")) == json.dumps("-Infinity")


def test_decimal_coerced_to_rounded_float():
    assert canonicalize(Decimal("1.2345678")) == json.dumps(1.234568)


def test_numpy_scalars_coerced():
    assert canonicalize(np.float64(1.2345678)) == json.dumps(1.234568)
    assert canonicalize(np.int64(7)) == json.dumps(7)
    assert canonicalize(np.bool_(True)) == json.dumps(True)


def test_dict_keys_sorted_recursively():
    a = canonicalize({"b": 1, "a": {"d": 2, "c": 3}})
    b = canonicalize({"a": {"c": 3, "d": 2}, "b": 1})
    assert a == b
    assert a == '{"a":{"c":3,"d":2},"b":1}'


def test_list_order_preserved():
    assert canonicalize([3, 1, 2]) != canonicalize([1, 2, 3])
    assert canonicalize([3, 1, 2]) == "[3,1,2]"


def test_newlines_normalized():
    assert canonicalize("a\r\nb\rc") == canonicalize("a\nb\nc")


def test_sha256_stable_and_hex():
    h = sha256_of({"x": 1.0000001})
    assert h == sha256_of({"x": 1.00000009})  # same after 6dp round
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
