import json

import pytest

from src import explainer as ex


def by_id(rows):
    return {r.drug_id: r for r in rows}


class FakeLLM:
    def __init__(self, texts=None, raise_exc=None, raw=None):
        self.texts, self.raise_exc, self.raw, self.calls, self.last_user = texts or {}, raise_exc, raw, 0, None

    def __call__(self, system, user):
        self.calls += 1
        self.last_user = user
        if self.raise_exc:
            raise self.raise_exc
        if self.raw is not None:
            return self.raw
        return json.dumps({"items": [{"drug_id": k, "text": v} for k, v in self.texts.items()]}, ensure_ascii=False)


def test_template_case_001(compared):
    exp = ex.explain_case(compared("CASE_001"))
    a = next(e for e in exp.items if e.drug_id == "DRUG_A")
    assert a.rule_text == "가상 약물 A: 1회 복용량 1정 → 0.5정으로 변경"
    assert "1정" in a.easy_text and "0.5정" in a.easy_text and a.source == "template"
    assert not exp.ai_called


def test_template_strength_not_described_as_tablet_count(compared):
    f = next(e for e in ex.explain_case(compared("CASE_008")).items if e.drug_id == "DRUG_F")
    assert "100mg" in f.easy_text and "50mg" in f.easy_text
    assert "100정" not in f.easy_text and "50정" not in f.easy_text
    assert "1정 그대로" in f.easy_text


def test_payload_is_minimal(compared):
    payload = ex.build_ai_payload(compared("CASE_012"))
    text = json.dumps(payload, ensure_ascii=False)
    assert "CASE_" not in text and "age" not in text and "세" not in text
    assert {p["drug_id"] for p in payload} == {"DRUG_B", "DRUG_C", "DRUG_D", "DRUG_H"}  # 유지 약물 제외


def test_no_key_falls_back(monkeypatch, compared):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exp = ex.explain_case(compared("CASE_001"), use_ai=True)
    assert not exp.ai_called and all(e.source == "template" for e in exp.items)
    assert "API 키" in exp.ai_status


@pytest.mark.parametrize("exc", [TimeoutError("t"), ConnectionError("c"), RuntimeError("x")])
def test_ai_failure_keeps_original(compared, exc):
    rows = compared("CASE_001")
    before = [r.as_record() for r in rows]
    exp = ex.explain_case(rows, use_ai=True, llm=FakeLLM(raise_exc=exc))
    assert exp.ai_called and exp.ai_used_count == 0
    assert [r.as_record() for r in rows] == before
    assert "실패" in exp.ai_status


def test_malformed_response(compared):
    exp = ex.explain_case(compared("CASE_001"), use_ai=True, llm=FakeLLM(raw="죄송합니다"))
    assert exp.ai_used_count == 0


def test_good_ai_text_accepted(compared):
    llm = FakeLLM({"DRUG_A": "가상 약물 A는 이제 한 번에 0.5정만 드시면 됩니다. 이전에는 1정이었어요."})
    exp = ex.explain_case(compared("CASE_001"), use_ai=True, llm=llm)
    a = next(e for e in exp.items if e.drug_id == "DRUG_A")
    assert a.source == "ai" and llm.calls == 1
    assert a.rule_text.startswith("가상 약물 A: 1회 복용량 1정")  # 원문은 그대로


@pytest.mark.parametrize("case,drug,text,reason_part", [
    ("CASE_001", "DRUG_A", "가상 약물 A는 이제 한 번에 2정 드세요.", "수치"),
    ("CASE_001", "DRUG_A", "가상 약물 A는 0.5정으로 줄었고 가상 약물 B도 같이 드세요.", "다른 약물"),
    ("CASE_001", "DRUG_A", "가상 약물 A는 부작용 때문에 0.5정으로 줄었습니다.", "금지"),
    ("CASE_001", "DRUG_A", "가상 약물 A는 반으로 쪼개서 0.5정 드세요.", "금지"),
    ("CASE_001", "DRUG_A", "가상 약물 A는 아침 식사 후 0.5정 드세요.", "금지"),
    ("CASE_001", "DRUG_A", "가상 약물 A의 복용량이 바뀌었습니다.", "누락"),
    ("CASE_008", "DRUG_F", "가상 약물 F는 100정에서 50정으로 바뀌었습니다.", "수치"),
    ("CASE_008", "DRUG_F", "가상 약물 F는 함량이 50mg으로 바뀌어 알 수가 줄었습니다.", "정 수"),
    ("CASE_012", "DRUG_D", "가상 약물 D는 하루 두 번 드세요.", "한글"),
    ("CASE_012", "DRUG_D", "가상 약물 D는 하루 2회, 아침과 점심에 드세요.", "시점"),
    ("CASE_012", "DRUG_B", "가상 약물 B가 새로 추가되었습니다.", "유형"),
    ("CASE_012", "DRUG_H", "가상 약물 H가 새로 추가되었습니다.", "누락"),
])
def test_bad_ai_text_rejected(compared, case, drug, text, reason_part):
    rows = compared(case)
    exp = ex.explain_case(rows, use_ai=True, llm=FakeLLM({drug: text}))
    item = next(e for e in exp.items if e.drug_id == drug)
    assert item.source == "template"
    assert item.easy_text == item.template_text
    assert reason_part in item.ai_rejected_reason


def test_check_accepts_valid_variants(compared):
    d = {r.drug_id: r for r in compared("CASE_012")}
    assert ex.check_ai_text(d["DRUG_D"], "가상 약물 D는 하루 1회에서 2회로 늘어, 아침과 저녁에 드십니다.") is None
    assert ex.check_ai_text(d["DRUG_B"], "가상 약물 B는 이번 처방에서 빠졌습니다.") is None
    assert ex.check_ai_text(d["DRUG_H"], "가상 약물 H가 새로 생겼어요. 40mg 1정을 하루 1회 아침에 드십니다.") is None


def test_maintained_drugs_never_sent(compared):
    llm = FakeLLM({"DRUG_A": "가상 약물 A 그대로"})
    ex.explain_case(compared("CASE_001"), use_ai=True, llm=llm)
    assert "DRUG_B" not in llm.last_user


def test_explanation_is_deterministic(compared):
    rows = compared("CASE_014")
    assert [e.easy_text for e in ex.explain_case(rows).items] == [e.easy_text for e in ex.explain_case(rows).items]
