"""표시용 표준 문자열 변환 (비교·설명·퀴즈가 같은 표기를 쓰도록 한 곳에 모음)."""
from __future__ import annotations

TIME_JOINER = "·"


def fmt_num(value: float | int) -> str:
    """1.0 → '1', 0.5 → '0.5', 2.50 → '2.5'"""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fmt_strength(v) -> str:
    return f"{fmt_num(v)}mg"


def fmt_dose(v, unit: str = "정") -> str:
    return f"{fmt_num(v)}{unit}"


def fmt_freq(v) -> str:
    return f"{int(v)}회"


def fmt_times(times) -> str:
    return TIME_JOINER.join(times)


def fmt_regimen(rec: dict) -> str:
    """한 줄 요약: '10mg 정제, 1회 1정, 하루 1회, 아침'"""
    return (f"{fmt_strength(rec['strength_mg'])} 정제, 1회 {fmt_dose(rec['dose_count'], rec['dose_unit'])}, "
            f"하루 {fmt_freq(rec['frequency_per_day'])}, {fmt_times(rec['administration_time'])}")


FIELD_LABELS = {
    "strength_mg": "함량",
    "dose_count": "1회 복용량",
    "dose_unit": "복용 단위",
    "frequency_per_day": "1일 복용 횟수",
    "administration_time": "복용 시점",
}

FIELD_FORMATTERS = {
    "strength_mg": fmt_strength,
    "dose_count": lambda v: fmt_dose(v),
    "dose_unit": str,
    "frequency_per_day": fmt_freq,
    "administration_time": fmt_times,
}


def fmt_field(field: str, value) -> str:
    return FIELD_FORMATTERS[field](value)


# ------------------------------------------------------------ 조사(받침) 처리
_LATIN_BATCHIM = {"L": 8, "M": 16, "N": 4, "R": 8}          # 엘·엠·엔·알 (8 = ㄹ)
_DIGIT_BATCHIM = {"0": 21, "1": 8, "3": 16, "6": 1, "7": 8, "8": 8}  # 영·일·삼·육·칠·팔


def _jong(word: str) -> int:
    """마지막 글자의 종성 번호(0 = 받침 없음, 8 = ㄹ)."""
    last = word.rstrip()[-1:] or "가"
    code = ord(last) - 0xAC00
    if 0 <= code <= 11171:
        return code % 28
    if last.upper() in _LATIN_BATCHIM:
        return _LATIN_BATCHIM[last.upper()]
    if last.isdigit():
        return _DIGIT_BATCHIM.get(last, 0)
    if last.lower() == "g":   # mg → 밀리그램
        return 16
    return 0


def josa_eun(word: str) -> str:
    return word + ("은" if _jong(word) else "는")


def josa_ro(word: str) -> str:
    return word + ("로" if _jong(word) in (0, 8) else "으로")
