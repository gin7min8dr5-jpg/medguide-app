"""3단계: 데이터 전처리·안전성 검증 엔진.

validate_case(before, after, metadata) 는 처방을 고치지 않는다.
문제가 하나라도 있으면 NEEDS_REVIEW 를 반환하고, 그 경우 앱은 비교·AI 설명·퀴즈로
진행하지 않는다. 오류가 여러 개면 ERROR_PRIORITY 순서로 정렬해 첫 번째를 대표 오류로 삼는다
(동일 입력 → 동일 대표 오류, 재현성 보장).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data_loader import PRESCRIPTION_COLUMNS, clean_cell, parse_bool

VALID = "VALID"
NEEDS_REVIEW = "NEEDS_REVIEW"

ALLOWED_UNITS = ("정",)
ALLOWED_TIMES = ("아침", "점심", "저녁")  # 표준 순서
TIME_SEPARATOR = "|"

# 오류 코드 → (우선순위 순서, 기본 설명). 위에 있을수록 먼저 보고된다.
ERROR_PRIORITY: list[tuple[str, str]] = [
    ("EMPTY_INPUT", "처방 입력이 비어 있습니다."),
    ("MISSING_COLUMNS", "필수 열이 없습니다."),
    ("MIXED_CASE_ID", "서로 다른 사례(case_id)의 처방이 섞여 있습니다."),
    ("MISSING_CASE_ID", "사례 번호(case_id)가 비어 있는 행이 있습니다."),
    ("INVALID_STAGE", "처방 단계(prescription_stage)는 before 또는 after만 허용됩니다."),
    ("MISSING_STAGE", "변경 전 또는 변경 후 처방이 없습니다."),
    ("MISSING_DRUG_ID", "약물 식별자(drug_id)가 비어 있습니다."),
    ("MISSING_DRUG_NAME", "약물 이름(drug_name)이 비어 있습니다."),
    ("DUPLICATE_DRUG_ID", "같은 처방 단계에 같은 약물 식별자가 두 번 이상 있습니다."),
    ("DRUG_NAME_MISMATCH", "같은 약물 식별자인데 변경 전후 약물 이름이 다릅니다."),
    ("MISSING_STRENGTH", "정제당 함량(strength_mg)이 비어 있습니다."),
    ("INVALID_STRENGTH", "정제당 함량(strength_mg)은 0보다 큰 숫자여야 합니다."),
    ("MISSING_DOSE_COUNT", "1회 복용량(dose_count)이 비어 있습니다."),
    ("INVALID_DOSE_COUNT", "1회 복용량(dose_count)은 0보다 큰 숫자여야 합니다."),
    ("INVALID_DOSE_UNIT", "복용 단위(dose_unit)는 '정'만 허용됩니다."),
    ("MISSING_FREQUENCY", "1일 복용 횟수(frequency_per_day)가 비어 있습니다."),
    ("INVALID_FREQUENCY", "1일 복용 횟수(frequency_per_day)는 1 이상의 정수여야 합니다."),
    ("MISSING_ADMIN_TIME", "복용 시점(administration_time)이 비어 있습니다."),
    ("INVALID_ADMIN_TIME", "복용 시점은 아침·점심·저녁만 허용됩니다."),
    ("DUPLICATE_ADMIN_TIME", "복용 시점이 중복 기재되어 있습니다."),
    ("SCHEDULE_MISMATCH", "1일 복용 횟수와 복용 시점의 개수가 맞지 않습니다."),
    ("INCOMPLETE_BEFORE_LIST", "변경 전 처방 목록이 완전한지 확인되지 않았습니다."),
    ("INCOMPLETE_AFTER_LIST", "변경 후 처방 목록이 완전한지 확인되지 않았습니다."),
]
_PRIORITY = {code: i for i, (code, _) in enumerate(ERROR_PRIORITY)}
_DEFAULT_MSG = dict(ERROR_PRIORITY)

STAGE_KO = {"before": "변경 전", "after": "변경 후"}


@dataclass
class ValidationError:
    code: str
    message: str
    stage: str | None = None
    drug_id: str | None = None
    drug_name: str | None = None
    row: int | None = None  # 해당 단계 표에서의 행 번호(1부터)

    @property
    def priority(self) -> int:
        return _PRIORITY.get(self.code, len(_PRIORITY))

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "row": self.row,
        }


@dataclass
class ValidationResult:
    status: str
    errors: list[ValidationError] = field(default_factory=list)
    before: list[dict] = field(default_factory=list)  # VALID일 때만 채워지는 정규화 처방
    after: list[dict] = field(default_factory=list)
    case_id: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == VALID

    @property
    def error_code(self) -> str | None:
        return self.errors[0].code if self.errors else None

    @property
    def error_message(self) -> str | None:
        return self.errors[0].message if self.errors else None


# ---------------------------------------------------------------- 정규화 도우미
def parse_positive_number(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def parse_positive_int(raw: str) -> int | None:
    value = parse_positive_number(raw)
    if value is None or value != int(value):
        return None
    return int(value)


def parse_times(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(TIME_SEPARATOR)]


def normalize_times(times: list[str]) -> tuple[str, ...]:
    """검증을 통과한 시점 목록을 표준 순서(아침→점심→저녁) 튜플로 만든다."""
    return tuple(t for t in ALLOWED_TIMES if t in set(times))


def _location(stage: str, row: int, drug_id: str, drug_name: str) -> str:
    who = drug_name or drug_id or "이름 없는 약물"
    return f"{STAGE_KO.get(stage, stage)} 처방 {row}번째 줄({who})"


def _err(errors: list, code: str, detail: str = "", **kw) -> None:
    msg = _DEFAULT_MSG.get(code, code)
    errors.append(ValidationError(code=code, message=f"{detail} — {msg}" if detail else msg, **kw))


# ---------------------------------------------------------------- 메인 함수
def validate_case(before: pd.DataFrame | None, after: pd.DataFrame | None, metadata: dict | None) -> ValidationResult:
    """변경 전·후 처방과 목록 완전성 정보를 검증한다.

    metadata 에는 최소한 before_list_complete / after_list_complete 가 필요하다.
    값이 True 로 확인되지 않으면(False·None·누락) 비교를 보류한다.
    """
    errors: list[ValidationError] = []
    metadata = metadata or {}

    frames = {"before": before, "after": after}
    present = {s: f for s, f in frames.items() if f is not None and len(f) > 0}

    if not present:
        _err(errors, "EMPTY_INPUT")
        return _finish(errors, None)

    # 1) 필수 열
    for stage, df in present.items():
        cols = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        missing = [c for c in PRESCRIPTION_COLUMNS if c not in cols]
        if missing:
            _err(errors, "MISSING_COLUMNS", f"{STAGE_KO[stage]} 처방에 없는 열: {', '.join(missing)}", stage=stage)
    if errors:
        return _finish(errors, None)

    cleaned = {s: _clean(df) for s, df in present.items()}

    # 2) case_id 일관성
    case_ids = set()
    for stage, df in cleaned.items():
        for i, cid in enumerate(df["case_id"], start=1):
            if cid == "":
                _err(errors, "MISSING_CASE_ID", f"{STAGE_KO[stage]} 처방 {i}번째 줄", stage=stage, row=i)
            else:
                case_ids.add(cid)
    if len(case_ids) > 1:
        _err(errors, "MIXED_CASE_ID", f"발견된 사례: {', '.join(sorted(case_ids))}")
    case_id = next(iter(case_ids)) if len(case_ids) == 1 else None

    # 3) 단계 값 / 단계 존재
    for stage, df in cleaned.items():
        for i, st in enumerate(df["prescription_stage"], start=1):
            if st not in ("before", "after"):
                _err(errors, "INVALID_STAGE", f"{STAGE_KO[stage]} 처방 {i}번째 줄 값 '{st}'", stage=stage, row=i)
            elif st != stage:
                _err(errors, "INVALID_STAGE",
                     f"{STAGE_KO[stage]} 처방 표 {i}번째 줄이 '{st}'로 표시됨", stage=stage, row=i)
    for stage in ("before", "after"):
        if stage not in cleaned:
            _err(errors, "MISSING_STAGE", f"{STAGE_KO[stage]} 처방 없음", stage=stage)

    # 4) 행 단위 검사
    normalized: dict[str, list[dict]] = {"before": [], "after": []}
    for stage, df in cleaned.items():
        seen: dict[str, int] = {}
        for i, rec in enumerate(df.to_dict("records"), start=1):
            did, name = rec["drug_id"], rec["drug_name"]
            loc = _location(stage, i, did, name)
            kw = {"stage": stage, "drug_id": did or None, "drug_name": name or None, "row": i}
            row_ok = True

            if did == "":
                _err(errors, "MISSING_DRUG_ID", loc, **kw); row_ok = False
            if name == "":
                _err(errors, "MISSING_DRUG_NAME", loc, **kw); row_ok = False
            if did:
                if did in seen:
                    _err(errors, "DUPLICATE_DRUG_ID", f"{loc}, {seen[did]}번째 줄과 중복", **kw); row_ok = False
                else:
                    seen[did] = i

            strength = _check_number(errors, rec["strength_mg"], "MISSING_STRENGTH", "INVALID_STRENGTH", loc, kw)
            dose = _check_number(errors, rec["dose_count"], "MISSING_DOSE_COUNT", "INVALID_DOSE_COUNT", loc, kw)
            row_ok &= strength is not None and dose is not None

            unit = rec["dose_unit"]
            if unit not in ALLOWED_UNITS:
                _err(errors, "INVALID_DOSE_UNIT", f"{loc} 값 '{unit}'", **kw); row_ok = False

            freq = None
            if rec["frequency_per_day"] == "":
                _err(errors, "MISSING_FREQUENCY", loc, **kw); row_ok = False
            else:
                freq = parse_positive_int(rec["frequency_per_day"])
                if freq is None:
                    _err(errors, "INVALID_FREQUENCY", f"{loc} 값 '{rec['frequency_per_day']}'", **kw); row_ok = False

            times_ok = True
            raw_time = rec["administration_time"]
            if raw_time == "":
                _err(errors, "MISSING_ADMIN_TIME", loc, **kw); times_ok = False
                times = []
            else:
                times = parse_times(raw_time)
                bad = [t for t in times if t not in ALLOWED_TIMES]
                if bad:
                    shown = ", ".join("(빈 칸)" if t == "" else t for t in bad)
                    _err(errors, "INVALID_ADMIN_TIME", f"{loc} 값 '{shown}'", **kw); times_ok = False
                elif len(set(times)) != len(times):
                    _err(errors, "DUPLICATE_ADMIN_TIME", f"{loc} 값 '{raw_time}'", **kw); times_ok = False
            row_ok &= times_ok

            if freq is not None and times_ok and len(times) != freq:
                _err(errors, "SCHEDULE_MISMATCH",
                     f"{loc}: 하루 {freq}회인데 복용 시점은 {len(times)}개({'·'.join(times)})", **kw)
                row_ok = False

            if row_ok:
                normalized[stage].append({
                    "case_id": rec["case_id"],
                    "stage": stage,
                    "drug_id": did,
                    "drug_name": name,
                    "strength_mg": strength,
                    "dose_count": dose,
                    "dose_unit": unit,
                    "frequency_per_day": freq,
                    "administration_time": normalize_times(times),
                })

    # 5) 같은 drug_id의 전후 이름 일치
    if "before" in cleaned and "after" in cleaned:
        names_before = {r["drug_id"]: r["drug_name"] for r in cleaned["before"].to_dict("records") if r["drug_id"]}
        for i, r in enumerate(cleaned["after"].to_dict("records"), start=1):
            did = r["drug_id"]
            if did and did in names_before and r["drug_name"] and names_before[did] != r["drug_name"]:
                _err(errors, "DRUG_NAME_MISMATCH",
                     f"{did}: 변경 전 '{names_before[did]}' / 변경 후 '{r['drug_name']}'",
                     stage="after", drug_id=did, drug_name=r["drug_name"], row=i)

    # 6) 목록 완전성 — True로 확인된 경우에만 비교 가능
    for stage, key, code in (("before", "before_list_complete", "INCOMPLETE_BEFORE_LIST"),
                             ("after", "after_list_complete", "INCOMPLETE_AFTER_LIST")):
        flag = metadata.get(key)
        flag = flag if isinstance(flag, bool) else parse_bool(flag)
        if flag is not True:
            detail = "목록 불완전으로 표시됨" if flag is False else "목록 완전성 미확인"
            _err(errors, code, f"{STAGE_KO[stage]} 처방: {detail}", stage=stage)

    result = _finish(errors, case_id)
    if result.is_valid:
        result.before = normalized["before"]
        result.after = normalized["after"]
    return result


# ---------------------------------------------------------------- 내부 도우미
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    out = out[PRESCRIPTION_COLUMNS]
    for col in PRESCRIPTION_COLUMNS:
        out[col] = out[col].map(clean_cell)
    return out.reset_index(drop=True)


def _check_number(errors, raw, missing_code, invalid_code, loc, kw) -> float | None:
    if raw == "":
        _err(errors, missing_code, loc, **kw)
        return None
    value = parse_positive_number(raw)
    if value is None:
        _err(errors, invalid_code, f"{loc} 값 '{raw}'", **kw)
    return value


def _finish(errors: list[ValidationError], case_id: str | None) -> ValidationResult:
    # 안정 정렬: 우선순위 → 단계 → 행 번호
    stage_order = {"before": 0, "after": 1, None: 2}
    errors = sorted(errors, key=lambda e: (e.priority, stage_order.get(e.stage, 2), e.row or 0))
    return ValidationResult(status=NEEDS_REVIEW if errors else VALID, errors=errors, case_id=case_id)
