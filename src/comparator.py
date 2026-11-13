"""4단계: 규칙 기반 처방 비교 엔진.

검증(validate_case)을 통과한 정규화 처방만 입력으로 받는다. 의학적 추론을 하지 않으며
ground_truth.csv 를 참조하지 않는다. UI·AI 와 독립된 순수 함수다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .formatting import FIELD_LABELS, fmt_field, fmt_regimen
from .validator import ValidationResult

MAINTAINED, STOPPED, CHANGED, ADDED = "유지", "중단", "변경", "추가"
# 화면·결과 순서: 중단 → 변경 → 추가 → 유지 (같은 유형 안에서는 drug_id 순)
CLASS_ORDER = (STOPPED, CHANGED, ADDED, MAINTAINED)
COMPARE_FIELDS = ("strength_mg", "dose_count", "dose_unit", "frequency_per_day", "administration_time")
FIELD_SEPARATOR = "|"


@dataclass(frozen=True)
class ComparisonRow:
    case_id: str
    drug_id: str
    drug_name: str
    classification: str
    changed_fields: tuple[str, ...]
    before: dict | None        # 정규화된 변경 전 처방 (없으면 None)
    after: dict | None         # 정규화된 변경 후 처방 (없으면 None)
    before_value: str          # 사람이 읽는 요약
    after_value: str
    change_summary: str

    @property
    def changed_fields_str(self) -> str:
        return FIELD_SEPARATOR.join(self.changed_fields)

    def field_changes(self) -> list[tuple[str, str, str, str]]:
        """[(field, 라벨, 이전 표기, 이후 표기)] — 변경 유형에서만 값이 있다."""
        return [(f, FIELD_LABELS[f], fmt_field(f, self.before[f]), fmt_field(f, self.after[f]))
                for f in self.changed_fields]

    def as_record(self) -> dict:
        d = asdict(self)
        d["changed_fields"] = self.changed_fields_str
        return d


class ComparisonNotAllowed(RuntimeError):
    """검증을 통과하지 못한 사례를 비교하려고 할 때 발생."""


def _same(field: str, a, b) -> bool:
    if field in ("strength_mg", "dose_count"):
        return float(a) == float(b)   # 1 == 1.0
    if field == "frequency_per_day":
        return int(a) == int(b)
    return a == b                      # 단위 문자열, 시점 튜플(표준 순서로 정규화됨)


def _summary(classification: str, changes: list[tuple[str, str, str, str]]) -> str:
    if classification == MAINTAINED:
        return "변경 없음"
    if classification == STOPPED:
        return "변경 후 처방에서 중단"
    if classification == ADDED:
        return "새로운 처방에 추가"
    return "; ".join(f"{label} {old}→{new}" for _, label, old, new in changes)


def compare_case(validation: ValidationResult) -> list[ComparisonRow]:
    """VALID 판정을 받은 검증 결과만 비교한다. 그 밖의 경우 예외를 발생시킨다."""
    if not isinstance(validation, ValidationResult) or not validation.is_valid:
        raise ComparisonNotAllowed("검증을 통과하지 못한 처방은 비교하지 않습니다.")
    return compare_prescriptions(validation.before, validation.after, validation.case_id or "")


def compare_prescriptions(before: list[dict], after: list[dict], case_id: str = "") -> list[ComparisonRow]:
    b = {r["drug_id"]: r for r in before}
    a = {r["drug_id"]: r for r in after}
    rows: list[ComparisonRow] = []
    for drug_id in sorted(set(b) | set(a)):
        rb, ra = b.get(drug_id), a.get(drug_id)
        if rb and not ra:
            cls, fields = STOPPED, ()
        elif ra and not rb:
            cls, fields = ADDED, ()
        else:
            fields = tuple(f for f in COMPARE_FIELDS if not _same(f, rb[f], ra[f]))
            cls = CHANGED if fields else MAINTAINED
        name = (ra or rb)["drug_name"]
        changes = [(f, FIELD_LABELS[f], fmt_field(f, rb[f]), fmt_field(f, ra[f])) for f in fields]
        rows.append(ComparisonRow(
            case_id=case_id,
            drug_id=drug_id,
            drug_name=name,
            classification=cls,
            changed_fields=fields,
            before=rb,
            after=ra,
            before_value=fmt_regimen(rb) if rb else "-",
            after_value=fmt_regimen(ra) if ra else "-",
            change_summary=_summary(cls, changes),
        ))
    rows.sort(key=lambda r: (CLASS_ORDER.index(r.classification), r.drug_id))
    return rows


def count_by_class(rows: list[ComparisonRow]) -> dict[str, int]:
    counts = {c: 0 for c in CLASS_ORDER}
    for r in rows:
        counts[r.classification] += 1
    return counts
