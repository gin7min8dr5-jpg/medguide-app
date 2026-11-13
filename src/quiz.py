"""7단계: Check — Teach-back 객관식 문항 생성·채점.

문항과 정답은 오직 구조화된 비교 결과(ComparisonRow)에서 만든다. AI는 관여하지 않는다.
- 유지 약물은 필수 문항에서 제외
- 중단 → 중단 여부 / 추가 → 새로 추가된 약 / 변경 → 바뀐 항목별(복용량·횟수·시점·함량) 문항
- 정답은 정확히 하나, 선택지는 중복 없음, 선택지 순서는 사례·문항 ID로 고정(재현 가능)
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

from .comparator import ADDED, CHANGED, MAINTAINED, STOPPED, ComparisonRow
from .formatting import fmt_dose, fmt_freq, fmt_num, fmt_strength, fmt_times, josa_eun

KIND_STOP = "중단 여부"
KIND_ADDED = "새로 추가된 약"
KIND_DOSE = "새 1회 복용량"
KIND_FREQ = "새 1일 복용 횟수"
KIND_TIME = "새 복용 시점"
KIND_STRENGTH = "새 함량"

_FIELD_KIND = {
    "dose_count": KIND_DOSE,
    "frequency_per_day": KIND_FREQ,
    "administration_time": KIND_TIME,
    "strength_mg": KIND_STRENGTH,
}
_FIELD_ORDER = ("dose_count", "frequency_per_day", "administration_time", "strength_mg")

STOP_OPTIONS = ("바뀐 처방에서 빠졌다 (중단)", "이전과 똑같이 처방되어 있다 (유지)", "복용 방법이 바뀌어 계속 처방되어 있다 (변경)")
NO_ADDED = "새로 추가된 약은 없다"
TIME_POOL = (("아침",), ("점심",), ("저녁",), ("아침", "저녁"), ("아침", "점심", "저녁"))
DOSE_POOL = (0.5, 1, 1.5, 2, 3)
FREQ_POOL = (1, 2, 3, 4)


@dataclass(frozen=True)
class Question:
    qid: str
    drug_id: str
    drug_name: str
    kind: str
    prompt: str
    options: tuple[str, ...]
    answer: str

    def is_correct(self, choice: str) -> bool:
        return choice == self.answer


def _rng(seed_key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))


def _pick_distractors(correct, before, pool, k: int = 2) -> list:
    """이전 값(가장 헷갈리기 쉬운 오답)을 먼저, 이어서 정답과 가까운 값을 고른다."""
    out = []
    if before is not None and before != correct:
        out.append(before)
    try:
        ranked = sorted(pool, key=lambda v: (abs(float(v) - float(correct)), float(v)))
    except (TypeError, ValueError):
        ranked = list(pool)
    for v in ranked:
        if len(out) >= k:
            break
        if v != correct and v not in out:
            out.append(v)
    return out


def _make(case_key: str, qid: str, row: ComparisonRow, kind: str, prompt: str,
          correct: str, wrong: list[str]) -> Question:
    opts = [correct] + [w for w in dict.fromkeys(wrong) if w != correct]
    _rng(f"{case_key}:{qid}").shuffle(opts)
    assert opts.count(correct) == 1 and len(set(opts)) == len(opts) >= 2
    return Question(qid, row.drug_id, row.drug_name, kind, prompt, tuple(opts), correct)


def build_questions(rows: list[ComparisonRow], case_key: str = "") -> list[Question]:
    questions: list[Question] = []
    added_ids = {r.drug_id for r in rows if r.classification == ADDED}
    non_added_names = [r.drug_name for r in sorted(rows, key=lambda r: r.drug_id) if r.drug_id not in added_ids]

    for r in rows:
        if r.classification == MAINTAINED:
            continue
        if r.classification == STOPPED:
            questions.append(_make(case_key, f"{r.drug_id}:stop", r, KIND_STOP,
                                   f"{josa_eun(r.drug_name)} 바뀐 처방에서 어떻게 되었나요?",
                                   STOP_OPTIONS[0], list(STOP_OPTIONS[1:])))
        elif r.classification == ADDED:
            pool = list(non_added_names)
            _rng(f"{case_key}:{r.drug_id}:added-pool").shuffle(pool)
            wrong = pool[:2] if len(pool) >= 2 else pool + [NO_ADDED]
            questions.append(_make(case_key, f"{r.drug_id}:added", r, KIND_ADDED,
                                   "다음 중 이번에 새로 추가된 약은 무엇인가요?", r.drug_name, wrong))
        elif r.classification == CHANGED:
            for f in _FIELD_ORDER:
                if f not in r.changed_fields:
                    continue
                b, a = r.before[f], r.after[f]
                if f == "dose_count":
                    wrong = _pick_distractors(float(a), float(b), DOSE_POOL)
                    q = (f"{josa_eun(r.drug_name)} 이제 한 번에 몇 정 드시나요?",
                         f"1회 {fmt_dose(a)}", [f"1회 {fmt_dose(v)}" for v in wrong])
                elif f == "frequency_per_day":
                    wrong = _pick_distractors(int(a), int(b), FREQ_POOL)
                    q = (f"{josa_eun(r.drug_name)} 이제 하루에 몇 번 드시나요?",
                         f"하루 {fmt_freq(a)}", [f"하루 {fmt_freq(v)}" for v in wrong])
                elif f == "administration_time":
                    wrong = [b] + [t for t in TIME_POOL if t not in (a, b)][:1]
                    q = (f"{josa_eun(r.drug_name)} 이제 언제 드시나요?",
                         fmt_times(a), [fmt_times(t) for t in wrong])
                else:  # strength_mg
                    cands = [float(b), float(a) * 2, float(a) / 2, float(b) * 2]
                    wrong = [v for v in dict.fromkeys(cands) if v != float(a)][:2]
                    q = (f"{r.drug_name} 한 알의 함량은 이제 얼마인가요?",
                         fmt_strength(a), [fmt_strength(v) for v in wrong])
                prompt, correct, wrong_txt = q
                questions.append(_make(case_key, f"{r.drug_id}:{f}", r, _FIELD_KIND[f], prompt, correct, wrong_txt))
    return questions


# ------------------------------------------------------------ 채점·기록
@dataclass
class AnswerRecord:
    qid: str
    first_choice: str
    first_correct: bool
    attempts: int = 1
    solved: bool = False
    history: list[str] = field(default_factory=list)


class QuizTracker:
    """st.session_state 에 그대로 저장할 수 있는 단순 채점 기록."""

    def __init__(self) -> None:
        self.records: dict[str, AnswerRecord] = {}

    def submit(self, question: Question, choice: str) -> bool:
        correct = question.is_correct(choice)
        rec = self.records.get(question.qid)
        if rec is None:
            rec = AnswerRecord(question.qid, choice, correct, attempts=1, solved=correct, history=[choice])
            self.records[question.qid] = rec
        elif not rec.solved:
            rec.attempts += 1
            rec.history.append(choice)
            rec.solved = correct
        return correct

    def get(self, qid: str) -> AnswerRecord | None:
        return self.records.get(qid)

    def summary(self, questions: list[Question]) -> dict:
        qids = [q.qid for q in questions]
        answered = [self.records[q] for q in qids if q in self.records]
        first_correct = sum(r.first_correct for r in answered)
        retried = [r for r in answered if not r.first_correct]
        retry_solved = sum(r.solved for r in retried)
        return {
            "total_questions": len(qids),
            "first_answered": len(answered),
            "first_correct": first_correct,
            "first_accuracy": (first_correct / len(answered)) if answered else None,
            "retry_targets": len(retried),
            "retry_solved": retry_solved,
            "retry_accuracy": (retry_solved / len(retried)) if retried else None,
            "total_attempts": sum(r.attempts for r in answered),
        }

    def export_rows(self, questions: list[Question]) -> list[dict]:
        by = {q.qid: q for q in questions}
        rows = []
        for qid, r in self.records.items():
            q = by.get(qid)
            d = asdict(r)
            d["history"] = " → ".join(r.history)
            d.update({"drug_name": q.drug_name if q else "", "kind": q.kind if q else "",
                      "answer": q.answer if q else ""})
            rows.append(d)
        return rows


def fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


__all__ = ["Question", "QuizTracker", "build_questions", "fmt_rate", "fmt_num"]
