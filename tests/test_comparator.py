import pytest

from src.comparator import ComparisonNotAllowed, compare_case, count_by_class
from src.validator import validate_case
from tests.conftest import COMPLETE, frame


def by_id(rows):
    return {r.drug_id: r for r in rows}


def test_case_001(compared):
    rows = compared("CASE_001")
    c = count_by_class(rows)
    assert c == {"중단": 0, "변경": 1, "추가": 0, "유지": 5}
    a = by_id(rows)["DRUG_A"]
    assert a.classification == "변경" and a.changed_fields == ("dose_count",)
    assert a.change_summary == "1회 복용량 1정→0.5정"


def test_case_012(compared):
    rows = compared("CASE_012")
    assert count_by_class(rows) == {"중단": 1, "변경": 2, "추가": 1, "유지": 4}
    d = by_id(rows)
    assert d["DRUG_B"].classification == "중단"
    assert d["DRUG_C"].changed_fields == ("dose_count",)
    assert d["DRUG_D"].changed_fields == ("frequency_per_day", "administration_time")
    assert d["DRUG_H"].classification == "추가"


def test_case_014(compared):
    assert count_by_class(compared("CASE_014")) == {"중단": 1, "변경": 4, "추가": 0, "유지": 1}


def test_strength_only_change_is_not_dose_change(compared):
    f = by_id(compared("CASE_008"))["DRUG_F"]
    assert f.changed_fields == ("strength_mg",)
    assert f.change_summary == "함량 100mg→50mg"
    assert "정" not in f.change_summary


def test_refuses_invalid(run_case):
    with pytest.raises(ComparisonNotAllowed):
        compare_case(run_case("CASE_020"))


def test_numeric_and_time_normalization():
    b = frame([("DRUG_A", "가상 약물 A", "10", "1", "2", "저녁|아침")], "before")
    a = frame([("DRUG_A", "가상 약물 A", "10.0", "1.00", "2", "아침|저녁")], "after")
    rows = compare_case(validate_case(b, a, COMPLETE))
    assert rows[0].classification == "유지"


def test_order_is_deterministic():
    b = frame([("DRUG_B", "가상 약물 B", "5", "1", "1", "아침"), ("DRUG_A", "가상 약물 A", "10", "1", "1", "아침")], "before")
    a = frame([("DRUG_C", "가상 약물 C", "1", "1", "1", "아침"), ("DRUG_A", "가상 약물 A", "10", "1", "1", "저녁")], "after")
    rows1 = compare_case(validate_case(b, a, COMPLETE))
    rows2 = compare_case(validate_case(b.iloc[::-1], a.iloc[::-1], COMPLETE))
    assert [(r.drug_id, r.classification) for r in rows1] == [(r.drug_id, r.classification) for r in rows2] == [
        ("DRUG_B", "중단"), ("DRUG_A", "변경"), ("DRUG_C", "추가")]


def test_record_structure(compared):
    rec = compared("CASE_001")[0].as_record()
    for key in ("case_id", "drug_id", "classification", "changed_fields", "before_value", "after_value", "change_summary"):
        assert key in rec
