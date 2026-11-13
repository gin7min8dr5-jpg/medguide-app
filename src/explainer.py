"""6단계: Understand · Repeat — 쉬운 설명.

1) 모든 약물에 규칙 기반 설명(템플릿)을 먼저 만든다. 이것이 항상 기본값이다.
2) AI 기능을 켜면 '확정된 비교 결과'의 최소 정보(약물명·변경 유형·변경 필드·전후 값)만
   LLM에 보내 쉬운 표현을 받는다. 연령·case_id·환자 정보는 보내지 않는다.
3) AI 문장은 check_ai_text()로 검사한다. 약물·수치·횟수·시점이 원본과 어긋나거나
   금지 내용(효과·부작용·적응증·식사/분할 등 임의 용법)이 보이면 그 문장을 버리고
   템플릿을 쓴다. API 키 없음·통신 오류·시간 초과·형식 오류 시에도 템플릿으로 되돌린다.
4) AI 문장은 '보조 표현'일 뿐이며, 화면에는 규칙 기반 원문을 항상 함께 표시한다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from .comparator import ADDED, CHANGED, MAINTAINED, STOPPED, ComparisonRow
from .formatting import fmt_dose, fmt_freq, fmt_num, fmt_strength, fmt_times, josa_eun, josa_ro

SOURCE_TEMPLATE = "template"
SOURCE_AI = "ai"

DEFAULT_MODEL = "claude-haiku-5-5"
AI_TIMEOUT_SECONDS = 20.0
MAX_AI_TEXT_LEN = 300

LLMFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> 응답 문자열


@dataclass
class Explanation:
    drug_id: str
    drug_name: str
    classification: str
    rule_text: str                 # 규칙 기반 원문 (항상 표시)
    easy_text: str                 # 최종 쉬운 설명 (AI 통과 시 AI 문장, 아니면 템플릿)
    template_text: str             # 템플릿 쉬운 설명 (대체용)
    source: str = SOURCE_TEMPLATE  # 'ai' 이면 실제 LLM 생성 문장
    ai_rejected_reason: str | None = None


@dataclass
class CaseExplanation:
    items: list[Explanation]
    ai_requested: bool = False
    ai_called: bool = False        # 실제로 LLM API를 호출했는가
    ai_status: str = "AI 기능 꺼짐 — 규칙 기반 설명"
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (drug_name, 사유)

    @property
    def ai_used_count(self) -> int:
        return sum(1 for e in self.items if e.source == SOURCE_AI)


# ------------------------------------------------------------ 규칙 기반 텍스트
def _ro(word: str) -> str:
    return josa_ro(word)


def rule_text(row: ComparisonRow) -> str:
    """원문형 설명: '가상 약물 A: 1회 복용량 1정 → 0.5정으로 변경'"""
    if row.classification == MAINTAINED:
        return f"{row.drug_name}: 변경 없음 (유지) — {row.after_value}"
    if row.classification == STOPPED:
        return f"{row.drug_name}: 변경 후 처방에서 중단 — 이전 처방 {row.before_value}"
    if row.classification == ADDED:
        return f"{row.drug_name}: 새로 추가 — {row.after_value}"
    parts = [f"{label} {old} → {_ro(new)}" for _, label, old, new in row.field_changes()]
    return f"{row.drug_name}: " + ", ".join(parts) + " 변경"


def template_easy_text(row: ComparisonRow) -> str:
    if row.classification == MAINTAINED:
        return f"{josa_eun(row.drug_name)} 이전과 같습니다. ({row.after_value})"
    if row.classification == STOPPED:
        return (f"{josa_eun(row.drug_name)} 바뀐 처방에 없는 약입니다(중단). "
                f"이전에는 {_ro(row.before_value)} 처방되어 있었습니다.")
    if row.classification == ADDED:
        return f"{josa_eun(row.drug_name)} 새로 추가된 약입니다. {row.after_value}입니다."

    b, a = row.before, row.after
    sentences = []
    for f in row.changed_fields:
        if f == "dose_count":
            sentences.append(f"한 번에 드시는 양이 {fmt_dose(b[f])}에서 {_ro(fmt_dose(a[f]))} 바뀌었습니다.")
        elif f == "strength_mg":
            s = f"약 한 알에 들어 있는 양(함량)이 {fmt_strength(b[f])}에서 {_ro(fmt_strength(a[f]))} 바뀌었습니다."
            if "dose_count" not in row.changed_fields:
                s += f" 한 번에 드시는 알 수는 {fmt_dose(a['dose_count'])} 그대로입니다."
            sentences.append(s)
        elif f == "frequency_per_day":
            sentences.append(f"하루에 드시는 횟수가 {fmt_freq(b[f])}에서 {_ro(fmt_freq(a[f]))} 바뀌었습니다.")
        elif f == "administration_time":
            sentences.append(f"드시는 때가 {fmt_times(b[f])}에서 {_ro(fmt_times(a[f]))} 바뀌었습니다.")
        elif f == "dose_unit":
            sentences.append(f"복용 단위가 {b[f]}에서 {_ro(a[f])} 바뀌었습니다.")
    return f"{row.drug_name}: " + " ".join(sentences) + f" 바뀐 뒤 처방은 {row.after_value}입니다."


def template_explanations(rows: list[ComparisonRow]) -> CaseExplanation:
    items = [Explanation(r.drug_id, r.drug_name, r.classification, rule_text(r),
                         template_easy_text(r), template_easy_text(r)) for r in rows]
    return CaseExplanation(items=items)


# ------------------------------------------------------------ AI 전송 정보 (최소화)
_FIELD_KEYS = {
    "strength_mg": ("함량", fmt_strength),
    "dose_count": ("1회 복용량", fmt_dose),
    "frequency_per_day": ("1일 복용 횟수", fmt_freq),
    "administration_time": ("복용 시점", fmt_times),
    "dose_unit": ("복용 단위", str),
}


def _regimen_payload(rec: dict) -> dict:
    return {label: fn(rec[k]) for k, (label, fn) in _FIELD_KEYS.items() if k != "dose_unit"}


def build_ai_payload(rows: list[ComparisonRow]) -> list[dict]:
    """LLM에 보낼 최소 정보. 유지 약물은 보내지 않는다(템플릿으로 충분)."""
    payload = []
    for r in rows:
        if r.classification == MAINTAINED:
            continue
        item = {"drug_id": r.drug_id, "약물명": r.drug_name, "변경 유형": r.classification}
        if r.classification == CHANGED:
            item["바뀐 항목"] = [
                {"항목": label, "이전": old, "이후": new} for _, label, old, new in r.field_changes()
            ]
            unchanged = {
                _FIELD_KEYS[k][0]: _FIELD_KEYS[k][1](r.after[k])
                for k in ("strength_mg", "dose_count", "frequency_per_day", "administration_time")
                if k not in r.changed_fields
            }
            if unchanged:
                item["그대로인 항목"] = unchanged
        elif r.classification == ADDED:
            item["새 처방"] = _regimen_payload(r.after)
        elif r.classification == STOPPED:
            item["이전 처방"] = _regimen_payload(r.before)
        payload.append(item)
    return payload


SYSTEM_PROMPT = """너는 가상 처방 교육용 프로그램의 '문장 다듬기' 도우미다.
입력 JSON에 있는 확정된 처방 변경 정보만, 고령자가 읽기 쉬운 한국어 존댓말 문장으로 바꿔라.

반드시 지킬 것:
- 입력에 있는 약물명·숫자·단위·복용 시점만 그대로 사용한다. 숫자는 아라비아 숫자로 쓴다(예: 0.5정, 2회, 50mg).
- 바뀐 항목의 '이후' 값은 반드시 문장에 포함한다.
- 함량(mg) 변화와 1회 복용량(정) 변화를 서로 바꿔 말하지 않는다.
- 새로운 약, 약의 효과·효능·적응증·부작용·상호작용, 질병, 식사 전후, 알약 쪼개기, 복용 요령,
  의학적 권고나 지시를 만들어내지 않는다. 입력에 없는 정보는 쓰지 않는다.
- 약물마다 1~2문장, 120자 이내.

출력은 JSON 하나만 쓴다(설명·코드블록 없이):
{"items": [{"drug_id": "...", "text": "..."}]}"""


def _default_llm() -> LLMFn | None:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    try:
        import anthropic  # 선택 의존성
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=key, timeout=AI_TIMEOUT_SECONDS, max_retries=0)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    def call(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=1000, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content)

    return call


def ai_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# ------------------------------------------------------------ AI 문장 안전 검사
FORBIDDEN_TERMS = [
    "부작용", "효과", "효능", "적응증", "상호작용", "질환", "질병", "혈압", "혈당", "당뇨",
    "콜레스테롤", "증상", "치료", "통증", "식사", "식전", "식후", "공복", "음식", "술",
    "쪼개", "자르", "잘라", "분할", "나누어", "갈아", "씹", "권장", "권고", "추천",
    "취침", "자기 전", "잠들기", "오전", "오후", "새벽", "밤에", "건너뛰", "임의로",
]
_NATIVE_COUNT = re.compile(r"(두|세|네|다섯|여섯)\s*(번|회|알|정)")
_DRUG_LETTER = re.compile(r"약물\s*([A-Za-z])(?![A-Za-z])")
_DRUG_ID = re.compile(r"DRUG_[A-Za-z]+")
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|밀리그램|정|알|회|번)?", re.IGNORECASE)
_TIMES = ("아침", "점심", "저녁")


def _letter(drug_id: str) -> str:
    return drug_id.split("_")[-1].upper()


def check_ai_text(row: ComparisonRow, text: str) -> str | None:
    """문제가 없으면 None, 있으면 거부 사유 문자열을 반환한다."""
    if not isinstance(text, str) or not text.strip():
        return "빈 응답"
    t = text.strip()
    if len(t) > MAX_AI_TEXT_LEN:
        return "응답이 너무 김"
    for term in FORBIDDEN_TERMS:
        if term in t:
            return f"금지 표현 포함: '{term}'"
    if _NATIVE_COUNT.search(t):
        return "숫자를 한글로 표기(검증 불가)"

    # 약물 일치
    letters = {m.upper() for m in _DRUG_LETTER.findall(t)}
    if letters - {_letter(row.drug_id)}:
        return f"다른 약물 언급: {', '.join(sorted(letters))}"
    if set(_DRUG_ID.findall(t)) - {row.drug_id}:
        return "다른 약물 식별자 언급"
    if row.drug_name not in t and _letter(row.drug_id) not in letters:
        return "약물명 누락"

    recs = [r for r in (row.before, row.after) if r]
    strengths = {float(r["strength_mg"]) for r in recs}
    doses = {float(r["dose_count"]) for r in recs}
    freqs = {float(r["frequency_per_day"]) for r in recs} | {1.0}  # '1회 복용량' 표현 허용
    allowed_any = strengths | doses | freqs

    for num, unit in _NUM_UNIT.findall(t):
        v = float(num)
        u = (unit or "").lower()
        if u in ("mg", "밀리그램"):
            ok = v in strengths
        elif u in ("정", "알"):
            ok = v in doses
        elif u in ("회", "번"):
            ok = v in freqs
        else:
            ok = v in allowed_any
        if not ok:
            return f"원본에 없는 수치: {num}{unit}"

    allowed_times = set()
    for r in recs:
        allowed_times |= set(r["administration_time"])
    for tw in _TIMES:
        if tw in t and tw not in allowed_times:
            return f"원본에 없는 복용 시점: {tw}"

    # 유형 일관성 + 필수 정보 포함
    cls = row.classification
    if cls == STOPPED:
        if "추가" in t:
            return "유형 불일치(중단인데 추가로 표현)"
        if not any(k in t for k in ("중단", "빠졌", "빠집", "빠진", "없습니다", "드시지 않")):
            return "중단 사실이 드러나지 않음"
    elif cls == ADDED:
        if "중단" in t:
            return "유형 불일치(추가인데 중단으로 표현)"
        a = row.after
        need = [_num_pattern(a["strength_mg"], "mg|밀리그램"), _num_pattern(a["dose_count"], "정|알"),
                _num_pattern(a["frequency_per_day"], "회|번")]
        if not all(re.search(p, t, re.IGNORECASE) for p in need):
            return "새 약의 함량·복용량·횟수 중 누락"
        if not all(tm in t for tm in a["administration_time"]):
            return "새 약의 복용 시점 누락"
    elif cls == CHANGED:
        if "중단" in t or "추가" in t:
            return "유형 불일치(변경인데 중단/추가로 표현)"
        a = row.after
        for f in row.changed_fields:
            if f == "strength_mg" and not re.search(_num_pattern(a[f], "mg|밀리그램"), t, re.IGNORECASE):
                return "새 함량 누락"
            if f == "dose_count" and not re.search(_num_pattern(a[f], "정|알"), t):
                return "새 1회 복용량 누락"
            if f == "frequency_per_day" and not re.search(_num_pattern(a[f], "회|번"), t):
                return "새 복용 횟수 누락"
            if f == "administration_time" and not all(tm in t for tm in a[f]):
                return "새 복용 시점 누락"
        if row.changed_fields == ("strength_mg",) and re.search(r"(알|정)\s*수", t) and "그대로" not in t:
            return "함량 변화를 정 수 변화로 표현"
    return None


def _num_pattern(value, units: str) -> str:
    return rf"(?<![\d.]){re.escape(fmt_num(value))}\s*(?:{units})"


def _parse_ai_json(raw: str) -> dict[str, str]:
    clean = re.sub(r"```(?:json)?", "", raw or "").strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("JSON 없음")
    data = json.loads(clean[start:end + 1])
    out = {}
    for item in data.get("items", []):
        if isinstance(item, dict) and isinstance(item.get("drug_id"), str) and isinstance(item.get("text"), str):
            out.setdefault(item["drug_id"], item["text"].strip())
    return out


# ------------------------------------------------------------ 메인 진입점
def explain_case(rows: list[ComparisonRow], use_ai: bool = False, llm: LLMFn | None = None) -> CaseExplanation:
    """규칙 기반 설명을 만들고, 요청 시 안전 검사를 통과한 AI 문장으로만 쉬운 설명을 교체한다."""
    result = template_explanations(rows)
    if not use_ai:
        return result
    result.ai_requested = True

    payload = build_ai_payload(rows)
    if not payload:
        result.ai_status = "변경된 약이 없어 AI 설명을 만들지 않았습니다 — 규칙 기반 설명"
        return result

    llm = llm or _default_llm()
    if llm is None:
        result.ai_status = "AI 사용 불가(API 키 없음) — 규칙 기반 설명으로 표시"
        return result

    try:
        result.ai_called = True
        raw = llm(SYSTEM_PROMPT, json.dumps({"items": payload}, ensure_ascii=False))
        texts = _parse_ai_json(raw)
    except Exception as exc:  # 통신 실패·시간 초과·형식 오류 모두 동일하게 안전 대체
        result.ai_status = f"AI 응답 실패({type(exc).__name__}) — 규칙 기반 설명으로 표시"
        return result

    by_id = {r.drug_id: r for r in rows}
    for item in result.items:
        row = by_id[item.drug_id]
        if row.classification == MAINTAINED:
            continue
        text = texts.get(item.drug_id)
        reason = "응답에 해당 약물 없음" if text is None else check_ai_text(row, text)
        if reason:
            item.ai_rejected_reason = reason
            result.rejected.append((item.drug_name, reason))
        else:
            item.easy_text = text
            item.source = SOURCE_AI

    n_target = len(payload)
    result.ai_status = (f"AI 쉬운 설명 {result.ai_used_count}/{n_target}건 사용"
                        + (f", {len(result.rejected)}건은 정보 불일치로 폐기하고 규칙 기반 설명 표시"
                           if result.rejected else ""))
    return result
