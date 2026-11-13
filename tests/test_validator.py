import pandas as pd
import pytest

from src.validator import ERROR_PRIORITY, NEEDS_REVIEW, VALID, validate_case
from tests.conftest import COMPLETE, frame

BASE_B = [("DRUG_A", "가상 약물 A", "10", "1", "1", "아침"), ("DRUG_B", "가상 약물 B", "5", "1", "1", "저녁")]
BASE_A = [("DRUG_A", "가상 약물 A", "10", "0.5", "1", "아침"), ("DRUG_B", "가상 약물 B", "5", "1", "1", "저녁")]


def codes(res):
    return [e.code for e in res.errors]


@pytest.mark.parametrize("cid,code", [
    ("CASE_016", "MISSING_DRUG_ID"), ("CASE_017", "DUPLICATE_DRUG_ID"), ("CASE_018", "MISSING_DOSE_COUNT"),
    ("CASE_019", "SCHEDULE_MISMATCH"), ("CASE_020", "INCOMPLETE_AFTER_LIST"),
])
def test_error_cases(run_case, cid, code):
    res = run_case(cid)
    assert res.status == NEEDS_REVIEW
    assert res.error_code == code
    assert res.before == [] and res.after == []  # 실패 시 정규화 처방을 내보내지 않음


@pytest.mark.parametrize("cid", [f"CASE_{i:03d}" for i in range(1, 16)])
def test_normal_cases_valid(run_case, cid):
    res = run_case(cid)
    assert res.status == VALID, res.errors


def test_baseline_valid():
    assert validate_case(frame(BASE_B, "before"), frame(BASE_A, "after"), COMPLETE).is_valid


def test_empty_input():
    assert validate_case(None, None, COMPLETE).error_code == "EMPTY_INPUT"


def test_missing_stage():
    assert validate_case(frame(BASE_B, "before"), None, COMPLETE).error_code == "MISSING_STAGE"


def test_missing_columns():
    b = frame(BASE_B, "before").drop(columns=["dose_unit"])
    assert validate_case(b, frame(BASE_A, "after"), COMPLETE).error_code == "MISSING_COLUMNS"


def test_mixed_case_id():
    assert validate_case(frame(BASE_B, "before", "X"), frame(BASE_A, "after", "Y"), COMPLETE).error_code == "MIXED_CASE_ID"


def test_wrong_stage_label():
    a = frame(BASE_A, "after")
    a.loc[0, "prescription_stage"] = "later"
    assert validate_case(frame(BASE_B, "before"), a, COMPLETE).error_code == "INVALID_STAGE"


@pytest.mark.parametrize("field,value,code", [
    ("dose_count", "abc", "INVALID_DOSE_COUNT"),
    ("dose_count", "0", "INVALID_DOSE_COUNT"),
    ("dose_count", "-1", "INVALID_DOSE_COUNT"),
    ("dose_count", "inf", "INVALID_DOSE_COUNT"),
    ("strength_mg", "", "MISSING_STRENGTH"),
    ("strength_mg", "nan", "INVALID_STRENGTH"),
    ("frequency_per_day", "1.5", "INVALID_FREQUENCY"),
    ("frequency_per_day", "", "MISSING_FREQUENCY"),
    ("dose_unit", "캡슐", "INVALID_DOSE_UNIT"),
    ("administration_time", "", "MISSING_ADMIN_TIME"),
    ("administration_time", "자기전", "INVALID_ADMIN_TIME"),
    ("administration_time", "아침|", "INVALID_ADMIN_TIME"),
    ("administration_time", "아침|아침", "DUPLICATE_ADMIN_TIME"),
    ("drug_name", "", "MISSING_DRUG_NAME"),
])
def test_field_errors(field, value, code):
    a = frame(BASE_A, "after")
    a.loc[0, field] = value
    res = validate_case(frame(BASE_B, "before"), a, COMPLETE)
    assert res.status == NEEDS_REVIEW and code in codes(res)


def test_numeric_formats_accepted():
    a = frame(BASE_A, "after")
    a.loc[0, "dose_count"] = " 0.50 "
    a.loc[1, "frequency_per_day"] = "1.0"
    assert validate_case(frame(BASE_B, "before"), a, COMPLETE).is_valid


def test_schedule_mismatch_both_directions():
    a = frame(BASE_A, "after")
    a.loc[0, "frequency_per_day"] = "1"
    a.loc[0, "administration_time"] = "아침|저녁"
    assert validate_case(frame(BASE_B, "before"), a, COMPLETE).error_code == "SCHEDULE_MISMATCH"


def test_drug_name_mismatch():
    a = frame(BASE_A, "after")
    a.loc[0, "drug_name"] = "가상 약물 Z"
    assert validate_case(frame(BASE_B, "before"), a, COMPLETE).error_code == "DRUG_NAME_MISMATCH"


@pytest.mark.parametrize("meta,code", [
    ({"before_list_complete": False, "after_list_complete": True}, "INCOMPLETE_BEFORE_LIST"),
    ({"before_list_complete": True, "after_list_complete": False}, "INCOMPLETE_AFTER_LIST"),
    ({"before_list_complete": True}, "INCOMPLETE_AFTER_LIST"),   # 외부 입력: 완전성 미확인
    ({}, "INCOMPLETE_BEFORE_LIST"),
    ({"before_list_complete": "TRUE", "after_list_complete": "FALSE"}, "INCOMPLETE_AFTER_LIST"),
])
def test_completeness(meta, code):
    res = validate_case(frame(BASE_B, "before"), frame(BASE_A, "after"), meta)
    assert res.error_code == code


def test_priority_is_fixed_and_reproducible():
    a = frame(BASE_A, "after")
    a.loc[1, "dose_count"] = ""          # MISSING_DOSE_COUNT (B)
    a.loc[0, "drug_id"] = ""             # MISSING_DRUG_ID (A)
    meta = {"before_list_complete": True, "after_list_complete": False}
    first = validate_case(frame(BASE_B, "before"), a, meta)
    second = validate_case(frame(BASE_B, "before"), a.iloc[::-1].reset_index(drop=True), meta)
    assert first.error_code == second.error_code == "MISSING_DRUG_ID"
    order = [c for c, _ in ERROR_PRIORITY]
    assert [order.index(c) for c in codes(first)] == sorted(order.index(c) for c in codes(first))


def test_validator_never_fixes_input():
    a = frame(BASE_A, "after")
    a.loc[0, "dose_count"] = ""
    snapshot = a.copy()
    validate_case(frame(BASE_B, "before"), a, COMPLETE)
    pd.testing.assert_frame_equal(a, snapshot)
