"""Streamlit 화면 흐름 테스트 (streamlit.testing.AppTest, 브라우저 불필요)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    return app


def md_text(at):
    return "\n".join(m.value for m in at.markdown)


def button(at, key):
    return next(b for b in at.button if b.key == key)


def select_case(at, cid):
    at.selectbox(key="case_select").select(cid).run()


def test_initial_screen_without_api_key(at):
    assert not at.exception
    assert "가상 처방 기반 교육용 프로토타입" in md_text(at)
    assert at.selectbox(key="case_select").value == "CASE_001"


def test_case_012_flow(at):
    select_case(at, "CASE_012")
    assert any("입력 확인 완료" in s.value for s in at.success)
    button(at, "btn_compare").click().run()
    html = md_text(at)
    for label in ("⛔ 중단", "🔄 변경", "➕ 추가", "✅ 유지"):
        assert label in html
    assert "가상 약물 B" in html and "가상 약물 H" in html
    button(at, "btn_explain").click().run()
    assert "0.5정" in md_text(at)
    texts_before = [e.easy_text for e in at.session_state["explanation"].items]
    button(at, "btn_repeat").click().run()
    button(at, "btn_repeat").click().run()
    assert "다시 보기 2회째" in md_text(at)
    assert [e.easy_text for e in at.session_state["explanation"].items] == texts_before  # 같은 설명 재표시
    assert at.session_state["explanation"].ai_called is False
    assert not at.exception


def test_case_020_blocked(at):
    select_case(at, "CASE_020")
    assert any("INCOMPLETE_AFTER_LIST" in e.value for e in at.error)
    assert button(at, "btn_compare").disabled
    assert button(at, "btn_explain_off").disabled and button(at, "btn_quiz_off").disabled
    assert "⛔ 중단" not in md_text(at)  # 허위 중단 카드 없음
    assert "comparison" not in at.session_state


@pytest.mark.parametrize("cid", ["CASE_016", "CASE_017", "CASE_018", "CASE_019"])
def test_other_error_cases_blocked(at, cid):
    select_case(at, cid)
    assert at.error and button(at, "btn_compare").disabled


def test_quiz_wrong_then_retry_and_reset(at):
    button(at, "btn_compare").click().run()      # CASE_001
    button(at, "btn_quiz").click().run()
    q = at.session_state["questions"][0]
    assert q.answer == "1회 0.5정"
    wrong = next(o for o in q.options if o != q.answer)
    at.radio(key=f"q_{q.qid}").set_value(wrong).run()
    button(at, f"q_submit_{q.qid}").click().run()
    assert at.error and "다시 보는 설명" in md_text(at)
    at.radio(key=f"q_{q.qid}").set_value(q.answer).run()
    button(at, f"q_submit_{q.qid}").click().run()
    assert any("맞았습니다" in s.value for s in at.success)
    summ = at.session_state["tracker"].summary(at.session_state["questions"])
    assert summ["first_accuracy"] == 0 and summ["retry_solved"] == 1
    select_case(at, "CASE_002")                   # 사례 변경 → 상태 초기화
    for k in ("comparison", "questions", "tracker", "explanation"):
        assert k not in at.session_state


def test_editor_mode_requires_completeness(at):
    at.sidebar.radio[1].set_value("표에 직접 입력").run()
    next(b for b in at.button if "예시 처방" in b.label).click().run()
    assert any("INCOMPLETE_BEFORE_LIST" in e.value for e in at.error)
    at.checkbox(key="ed_bc").check().run()
    at.checkbox(key="ed_ac").check().run()
    assert any("입력 확인 완료" in s.value for s in at.success)
    button(at, "btn_compare").click().run()
    assert "가상 약물 H" in md_text(at)


def test_ai_called_once_and_repeat_uses_cache(monkeypatch):
    import json
    from src import explainer
    calls = []

    def fake_llm(system, user):
        calls.append(user)
        return json.dumps({"items": [
            {"drug_id": "DRUG_A", "text": "가상 약물 A는 이제 한 번에 0.5정 드십니다. 전에는 1정이었습니다."}]},
            ensure_ascii=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(explainer, "_default_llm", lambda: fake_llm)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    button(at, "btn_compare").click().run()
    button(at, "btn_explain").click().run()
    for _ in range(3):
        button(at, "btn_repeat").click().run()
    assert len(calls) == 1
    assert "CASE_001" not in calls[0] and "72" not in calls[0]   # 사례 번호·나이 미전송
    assert "AI 생성 쉬운 설명" in md_text(at)
    assert "원문 정보: 가상 약물 A: 1회 복용량 1정 → 0.5정으로 변경" in md_text(at)
