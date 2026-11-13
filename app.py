"""고령 환자의 외래 처방 변경 이해를 위한 맞춤형 복약 안내 — 가상 처방 기반 교육용 프로토타입.

실행:  streamlit run app.py

흐름: 처방 입력 → 사전 검증(validator) → 비교(comparator) → See(카드)
      → Understand·Repeat(explainer) → Check(quiz)
정답 파일(ground_truth.csv)은 이 앱에서 읽지 않는다 (src/evaluator.py 전용).
"""
from __future__ import annotations

import hashlib
import html

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv 는 선택 사항
    pass

from src import data_loader as dl
from src.comparator import ADDED, CHANGED, CLASS_ORDER, MAINTAINED, STOPPED, compare_case, count_by_class
from src.explainer import SOURCE_AI, ai_available, explain_case, rule_text, template_easy_text
from src.quiz import QuizTracker, build_questions, fmt_rate
from src.validator import validate_case

APP_TITLE = "처방이 이렇게 바뀌었어요"
DISCLAIMER = "가상 처방 기반 교육용 프로토타입 · 실제 환자 처방이나 복약 지도에 사용하지 않습니다"

CLASS_STYLE = {
    STOPPED: {"icon": "⛔", "label": "중단", "desc": "바뀐 처방에서 빠진 약", "css": "stop"},
    CHANGED: {"icon": "🔄", "label": "변경", "desc": "먹는 방법이 바뀐 약", "css": "change"},
    ADDED: {"icon": "➕", "label": "추가", "desc": "새로 생긴 약", "css": "add"},
    MAINTAINED: {"icon": "✅", "label": "유지", "desc": "그대로인 약", "css": "keep"},
}
FONT_SIZES = {"보통": 18, "크게": 21, "아주 크게": 25}
DISPLAY_COLUMNS = {
    "drug_name": "약 이름", "strength_mg": "함량(mg)", "dose_count": "1회 복용량",
    "dose_unit": "단위", "frequency_per_day": "하루 횟수", "administration_time": "복용 시점",
}

st.set_page_config(page_title="처방 변경 복약 안내 (가상 교육용)", page_icon="💊", layout="wide")


# ------------------------------------------------------------ 스타일
def inject_css(base_px: int) -> None:
    st.markdown(f"""
<style>
:root {{ --fs: {base_px}px; }}
.block-container {{ max-width: 1150px; padding-top: 3.2rem; }}
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown, .stApp [data-testid="stMarkdownContainer"] p,
.stApp div[role="radiogroup"] label p, .stApp .stAlert p {{ font-size: var(--fs) !important; line-height: 1.6; }}
.stApp button p {{ font-size: calc(var(--fs) * 0.95) !important; font-weight: 600; }}
.stApp .stButton > button {{ min-height: 3rem; border-radius: 12px; padding: 0.4rem 1.2rem; }}
.stApp [data-testid="stDataFrame"] {{ font-size: calc(var(--fs) * 0.8); }}
.disclaimer {{ background:#fff4d6; border:2px solid #b7791f; color:#5a3d00; border-radius:12px;
  padding:0.7rem 1rem; font-size:calc(var(--fs)*0.95); font-weight:700; margin-bottom:0.6rem; }}
.step-title {{ font-size: calc(var(--fs) * 1.35); font-weight: 800; margin: 1.6rem 0 0.4rem; color:#1a2b49; }}
.sum-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin: 0.4rem 0 1rem; }}
.sum-card {{ border-radius:14px; padding:0.9rem 1rem; border:3px solid; background:#fff; }}
.sum-card .lab {{ font-size:calc(var(--fs)*1.05); font-weight:800; }}
.sum-card .num {{ font-size:calc(var(--fs)*2.1); font-weight:900; line-height:1.1; }}
.sum-card .sub {{ font-size:calc(var(--fs)*0.8); color:#333; }}
.stop  {{ border-color:#b42318; color:#7a1610; background:#fff1f0; }}
.change{{ border-color:#1d4ed8; color:#0b2f8a; background:#eef4ff; }}
.add   {{ border-color:#15803d; color:#0b4d24; background:#ecfdf3; }}
.keep  {{ border-color:#667085; color:#344054; background:#f6f7f9; }}
.drug-card {{ border-radius:14px; border:3px solid; padding:0.9rem 1.1rem; margin:0.6rem 0; }}
.drug-card .head {{ font-size:calc(var(--fs)*1.2); font-weight:800; }}
.badge {{ display:inline-block; border-radius:999px; padding:0.1rem 0.7rem; margin-right:0.5rem;
  border:2px solid currentColor; font-size:calc(var(--fs)*0.9); font-weight:800; background:#fff; }}
.chg-row {{ display:flex; flex-wrap:wrap; gap:0.4rem 1rem; align-items:baseline; margin-top:0.45rem;
  font-size:var(--fs); color:#111; }}
.chg-row .f {{ min-width:7.5em; font-weight:700; }}
.chg-row .old {{ text-decoration: line-through; color:#555; }}
.chg-row .arrow {{ font-weight:900; }}
.chg-row .new {{ font-weight:900; font-size:calc(var(--fs)*1.15); background:#fff; padding:0 0.35rem; border-radius:6px; }}
.regimen {{ margin-top:0.45rem; color:#222; font-size:calc(var(--fs)*0.92); }}
.expl {{ border-left:6px solid #1d4ed8; background:#f8fafc; padding:0.7rem 1rem; margin:0.5rem 0; border-radius:8px; color:#111; }}
.expl .src {{ font-size:calc(var(--fs)*0.75); font-weight:700; color:#475467; }}
.expl .rule {{ font-size:calc(var(--fs)*0.85); color:#344054; margin-top:0.3rem; }}
.expl .easy {{ font-size:calc(var(--fs)*1.05); font-weight:600; }}
.repeat-box {{ border:3px dashed #1d4ed8; border-radius:14px; padding:0.6rem 1rem; margin-top:0.6rem; }}
</style>""", unsafe_allow_html=True)


def step_title(text: str) -> None:
    st.markdown(f'<div class="step-title">{html.escape(text)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------ 상태 관리
RESULT_KEYS = ("comparison", "explanation", "explanation_ai", "repeat_count", "questions", "tracker", "quiz_started")


def reset_results() -> None:
    for k in RESULT_KEYS:
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state if str(k).startswith("q_")]:
        st.session_state.pop(k, None)


def sync_input_key(key: str) -> None:
    """입력이 바뀌면 이후 결과(비교·설명·퀴즈)를 모두 초기화한다."""
    if st.session_state.get("input_key") != key:
        reset_results()
        st.session_state["input_key"] = key


def frame_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p.to_csv(index=False) if isinstance(p, pd.DataFrame) else str(p)).encode("utf-8"))
    return h.hexdigest()[:16]


@st.cache_data
def load_base_data():
    return dl.load_prescriptions(), dl.load_metadata()


# ------------------------------------------------------------ 입력 화면
def show_tables(before: pd.DataFrame | None, after: pd.DataFrame | None) -> None:
    c1, c2 = st.columns(2)
    for col, title, df in ((c1, "변경 전 처방", before), (c2, "변경 후 처방", after)):
        with col:
            st.markdown(f"**{title}** ({0 if df is None else len(df)}개)")
            if df is None or df.empty:
                st.info("처방이 없습니다.")
            else:
                cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
                view = df[cols].rename(columns=DISPLAY_COLUMNS) if cols else df
                st.dataframe(view, hide_index=True, width="stretch")


def input_from_case(pres, meta):
    case_ids = dl.list_case_ids(pres)
    labels = {}
    for cid in case_ids:
        m = dl.case_metadata(meta, cid)
        labels[cid] = f"{cid} · {m.get('scenario_level', '')} · 가상 환자 {m.get('age_years', '?')}세"
    cid = st.selectbox("살펴볼 가상 사례를 고르세요", case_ids, format_func=lambda c: labels[c], key="case_select")
    m = dl.case_metadata(meta, cid)
    sync_input_key(f"case:{cid}")
    before, after = dl.split_case(pres, cid)
    st.caption(f"난이도: {m.get('scenario_level', '-')} · 가상 환자 나이: {m.get('age_years', '-')}세 · "
               f"변경 전 {len(before)}개 / 변경 후 {len(after)}개 · "
               f"목록 완전성: 변경 전 {'확인됨' if m['before_list_complete'] else '미확인'}, "
               f"변경 후 {'확인됨' if m['after_list_complete'] else '미확인(불완전)'}")
    show_tables(before, after)
    return before, after, m


def completeness_inputs(prefix: str) -> dict:
    st.markdown("**목록 완전성 확인** — 빠진 약이 있으면 ‘중단’으로 잘못 안내할 수 있어 확인이 필요합니다.")
    b = st.checkbox("변경 전 처방 목록이 빠짐없이 모두 들어 있습니다", key=f"{prefix}_bc")
    a = st.checkbox("변경 후 처방 목록이 빠짐없이 모두 들어 있습니다", key=f"{prefix}_ac")
    return {"before_list_complete": b, "after_list_complete": a}


def input_from_upload():
    st.info("업로드 파일에는 이름·생년월일·등록번호 등 **환자를 알아볼 수 있는 정보를 넣지 마세요.** "
            f"필수 열: {', '.join(dl.PRESCRIPTION_COLUMNS)}")
    c1, c2 = st.columns(2)
    with c1:
        up_b = st.file_uploader("변경 전 처방 CSV", type=["csv"], key="up_before")
    with c2:
        up_a = st.file_uploader("변경 후 처방 CSV", type=["csv"], key="up_after")
    before = after = None
    for up, which in ((up_b, "변경 전"), (up_a, "변경 후")):
        if up is None:
            continue
        df, err = dl.read_uploaded_prescription(up)
        if err:
            st.error(f"{which} 파일: {err}")
        elif which == "변경 전":
            before = df
        else:
            after = df
    meta = completeness_inputs("up")
    sync_input_key("upload:" + frame_key(before, after, meta))
    show_tables(before, after)
    return before, after, meta


def input_from_editor(pres):
    st.info("표를 직접 고치거나 줄을 추가할 수 있습니다. **환자를 알아볼 수 있는 정보는 입력하지 마세요.** "
            "복용 시점은 아침, 점심, 저녁 중에서 ‘아침|저녁’처럼 | 로 구분해 적습니다.")
    if "editor_seed" not in st.session_state:
        st.session_state["editor_seed"] = (dl.empty_template("before"), dl.empty_template("after"))
        st.session_state["editor_ver"] = 0
    c0a, c0b = st.columns(2)
    if c0a.button("예시 처방 채우기 (CASE_012 복사본)"):
        b, a = dl.split_case(pres, "CASE_012")
        b, a = b.copy(), a.copy()
        b["case_id"] = a["case_id"] = "MY_CASE"
        st.session_state["editor_seed"] = (b, a)
        st.session_state["editor_ver"] += 1
    if c0b.button("빈 표로 다시 시작"):
        st.session_state["editor_seed"] = (dl.empty_template("before"), dl.empty_template("after"))
        st.session_state["editor_ver"] += 1
    seed_b, seed_a = st.session_state["editor_seed"]
    ver = st.session_state["editor_ver"]
    cfg = {c: st.column_config.TextColumn(c) for c in dl.PRESCRIPTION_COLUMNS}
    st.markdown("**변경 전 처방**")
    ed_b = st.data_editor(seed_b, num_rows="dynamic", column_config=cfg, width="stretch",
                          hide_index=True, key=f"ed_before_{ver}")
    st.markdown("**변경 후 처방**")
    ed_a = st.data_editor(seed_a, num_rows="dynamic", column_config=cfg, width="stretch",
                          hide_index=True, key=f"ed_after_{ver}")
    before, after = dl.clean_frame(ed_b), dl.clean_frame(ed_a)
    meta = completeness_inputs("ed")
    sync_input_key("editor:" + frame_key(before, after, meta))
    return (before if len(before) else None), (after if len(after) else None), meta


def pre_warnings(before, after) -> bool:
    """검증 전 단계 경고. 계속 진행할 수 있으면 True."""
    if before is None and after is None:
        st.warning("아직 처방이 입력되지 않았습니다. 변경 전·후 처방을 모두 넣어 주세요.")
        return False
    if before is None or after is None:
        st.warning(f"{'변경 전' if before is None else '변경 후'} 처방이 비어 있습니다. 두 처방이 모두 있어야 비교할 수 있습니다.")
    ids = set()
    for df in (before, after):
        if df is not None and "case_id" in df.columns:
            ids |= {v for v in df["case_id"].map(dl.clean_cell) if v}
    if len(ids) > 1:
        st.warning(f"서로 다른 사례 번호가 섞여 있습니다: {', '.join(sorted(ids))}")
    return True


# ------------------------------------------------------------ 결과 화면
def show_validation(v) -> None:
    if v.is_valid:
        st.success("✅ 입력 확인 완료 — 처방 정보에 빠지거나 맞지 않는 부분이 없습니다.")
        return
    first = v.errors[0]
    st.error(f"⚠️ **확인 필요 (NEEDS_REVIEW)** · 오류 코드 `{first.code}`\n\n{first.message}\n\n"
             "처방 정보가 불완전하거나 맞지 않아 **변경 내용 분석, 쉬운 설명, 확인 문제를 진행하지 않습니다.** "
             "처방전을 다시 확인하거나 약사·의료진에게 문의해 주세요.")
    if len(v.errors) > 1:
        with st.expander(f"발견된 문제 모두 보기 ({len(v.errors)}건)"):
            st.dataframe(pd.DataFrame([e.as_dict() for e in v.errors]), hide_index=True, width="stretch")


def show_summary_cards(rows) -> None:
    counts = count_by_class(rows)
    cards = []
    for cls in CLASS_ORDER:
        s = CLASS_STYLE[cls]
        cards.append(f'<div class="sum-card {s["css"]}"><div class="lab">{s["icon"]} {s["label"]}</div>'
                     f'<div class="num">{counts[cls]}개</div><div class="sub">{s["desc"]}</div></div>')
    st.markdown(f'<div class="sum-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def drug_card_html(r) -> str:
    s = CLASS_STYLE[r.classification]
    name = html.escape(r.drug_name)
    body = ""
    if r.classification == CHANGED:
        for _, label, old, new in r.field_changes():
            body += (f'<div class="chg-row"><span class="f">{html.escape(label)}</span>'
                     f'<span class="old">{html.escape(old)}</span><span class="arrow">→</span>'
                     f'<span class="new">{html.escape(new)}</span></div>')
        body += f'<div class="regimen">바뀐 뒤 전체: {html.escape(r.after_value)}</div>'
    elif r.classification == STOPPED:
        body = (f'<div class="chg-row"><span class="new">바뀐 처방에 없습니다</span></div>'
                f'<div class="regimen">이전 처방: {html.escape(r.before_value)}</div>')
    elif r.classification == ADDED:
        body = f'<div class="regimen">새 처방: {html.escape(r.after_value)}</div>'
    else:
        body = f'<div class="regimen">{html.escape(r.after_value)}</div>'
    return (f'<div class="drug-card {s["css"]}"><div class="head"><span class="badge">{s["icon"]} {s["label"]}</span>'
            f'{name}</div>{body}</div>')


def show_see(rows) -> None:
    show_summary_cards(rows)
    important = [r for r in rows if r.classification != MAINTAINED]
    kept = [r for r in rows if r.classification == MAINTAINED]
    if important:
        st.markdown("".join(drug_card_html(r) for r in important), unsafe_allow_html=True)
    else:
        st.info("바뀐 약이 없습니다. 모든 약이 이전과 같습니다.")
    if kept:
        with st.expander(f"✅ 그대로인 약 {len(kept)}개 보기"):
            st.markdown("".join(drug_card_html(r) for r in kept), unsafe_allow_html=True)
    with st.expander("상세 비교표 (전체 항목)"):
        table = pd.DataFrame([{
            "약 이름": r.drug_name, "변경 유형": f"{CLASS_STYLE[r.classification]['icon']} {r.classification}",
            "바뀐 항목": r.changed_fields_str or "-", "변경 전": r.before_value, "변경 후": r.after_value,
            "요약": r.change_summary} for r in rows])
        st.dataframe(table, hide_index=True, width="stretch")


def explanation_html(e) -> str:
    src = "🤖 AI 생성 쉬운 설명 (아래 원문과 대조 검증됨)" if e.source == SOURCE_AI else "📋 규칙 기반 쉬운 설명"
    s = CLASS_STYLE[e.classification]
    return (f'<div class="expl"><div class="src"><span class="badge {s["css"]}">{s["icon"]} {s["label"]}</span>{src}</div>'
            f'<div class="easy">{html.escape(e.easy_text)}</div>'
            f'<div class="rule">원문 정보: {html.escape(e.rule_text)}</div></div>')


def _make_explanation(rows, use_ai: bool) -> None:
    """설명 생성 콜백. 같은 AI 설정으로 이미 만든 설명이 있으면 다시 만들지 않는다(캐시)."""
    if "explanation" in st.session_state and st.session_state.get("explanation_ai") == use_ai:
        return
    st.session_state["explanation"] = explain_case(rows, use_ai=use_ai)
    st.session_state["explanation_ai"] = use_ai
    st.session_state["repeat_count"] = 0


def _repeat() -> None:
    """Repeat: 캐시된 설명을 다시 보여 줄 뿐, 새로 생성하거나 AI를 호출하지 않는다."""
    st.session_state["repeat_count"] = st.session_state.get("repeat_count", 0) + 1


def show_understand(rows, valid: bool, use_ai: bool) -> None:
    c1, c2 = st.columns([1, 1])
    c1.button("💬 쉬운 설명 보기", disabled=not valid, width="stretch", key="btn_explain",
              on_click=_make_explanation, args=(rows, use_ai))
    c2.button("🔁 다시 보기 (Repeat)", disabled=not (valid and "explanation" in st.session_state), width="stretch",
              key="btn_repeat", on_click=_repeat,
              help="이미 만든 설명을 그대로 다시 보여 줍니다. AI를 다시 부르지 않습니다.")
    if "explanation" in st.session_state and st.session_state.get("explanation_ai") != use_ai:
        st.caption("AI 설정이 바뀌었습니다. ‘쉬운 설명 보기’를 다시 누르면 새 설정으로 설명을 만듭니다.")

    exp = st.session_state.get("explanation")
    if exp is None:
        st.caption("‘쉬운 설명 보기’를 누르면 바뀐 약마다 쉬운 말로 풀어 드립니다.")
        return
    status = exp.ai_status
    (st.info if exp.ai_used_count else st.caption)(f"설명 방식: {status}")
    if exp.rejected:
        with st.expander("폐기된 AI 문장 사유 (개발자 확인용)"):
            for name, reason in exp.rejected:
                st.write(f"- {name}: {reason}")
    items = [e for e in exp.items if e.classification != MAINTAINED]
    if not items:
        st.info("바뀐 약이 없어 이전 처방 그대로입니다.")
    st.markdown("".join(explanation_html(e) for e in items), unsafe_allow_html=True)

    n = st.session_state.get("repeat_count", 0)
    if n:
        st.markdown(f'<div class="repeat-box"><b>🔁 다시 보기 {n}회째</b> — 같은 내용을 한 번 더 천천히 읽어 보세요.</div>',
                    unsafe_allow_html=True)
        for i, e in enumerate(items, start=1):
            st.markdown(f"**{i}. {e.easy_text}**")


def show_check(rows, valid: bool) -> None:
    start = st.button("📝 확인 문제 시작 (Teach-back)", disabled=not valid, key="btn_quiz")
    if start and "questions" not in st.session_state:
        st.session_state["questions"] = build_questions(rows, case_key=st.session_state.get("input_key", ""))
        st.session_state["tracker"] = QuizTracker()
    questions = st.session_state.get("questions")
    if questions is None:
        st.caption("바뀐 약에 대해 몇 가지 질문으로 이해한 내용을 확인합니다. 그대로인 약은 묻지 않습니다.")
        return
    if not questions:
        st.info("바뀐 약이 없어 확인할 문제가 없습니다.")
        return
    tracker: QuizTracker = st.session_state["tracker"]
    by_drug = {r.drug_id: r for r in rows}

    for i, q in enumerate(questions, start=1):
        rec = tracker.get(q.qid)
        with st.container(border=True):
            st.markdown(f"**문제 {i}. {q.prompt}**  \n<small>({q.kind})</small>", unsafe_allow_html=True)
            choice = st.radio("답을 고르세요", q.options, index=None, key=f"q_{q.qid}",
                              disabled=bool(rec and rec.solved), label_visibility="collapsed")
            if rec is not None and rec.solved:
                msg = "⭕ 맞았습니다!" if rec.first_correct else f"⭕ 맞았습니다! ({rec.attempts}번째 시도)"
                st.success(f"{msg} 정답: **{q.answer}**")
                continue
            if rec is not None:  # 오답 → 검증된 설명으로 재안내 후 재시도
                st.error(f"❌ 다시 확인해 볼까요? 고른 답: {rec.history[-1]}")
                st.markdown(explanation_html_from_row(by_drug[q.drug_id]), unsafe_allow_html=True)
                st.caption("위 설명을 읽고 다시 답을 고른 뒤 ‘다시 풀기’를 눌러 주세요.")
            label = "답 확인" if rec is None else "다시 풀기"
            st.button(label, key=f"q_submit_{q.qid}", disabled=choice is None,
                      on_click=tracker.submit, args=(q, choice))

    summ = tracker.summary(questions)
    st.markdown("#### 확인 결과")
    m1, m2, m3 = st.columns(3)
    m1.metric("처음 답한 기준 정답률", fmt_rate(summ["first_accuracy"]),
              help="최초 응답 정답 수 ÷ 최초 응답 문항 수")
    m1.caption(f"{summ['first_correct']} / {summ['first_answered']}문항 (전체 {summ['total_questions']}문항)")
    m2.metric("다시 풀어 맞힌 문항", f"{summ['retry_solved']} / {summ['retry_targets']}")
    m2.caption("처음 틀린 뒤 재시도로 맞힌 수 (최초 정답률과 따로 기록)")
    m3.metric("총 답한 횟수", summ["total_attempts"])
    if tracker.records:
        csv = pd.DataFrame(tracker.export_rows(questions)).to_csv(index=False).encode("utf-8-sig")
        st.download_button("학습 기록 CSV 저장", csv, file_name="teachback_record.csv", mime="text/csv")


def explanation_html_from_row(r) -> str:
    """오답 시 재안내: 검증된 규칙 기반 설명만 사용한다."""
    s = CLASS_STYLE[r.classification]
    return (f'<div class="expl"><div class="src"><span class="badge {s["css"]}">{s["icon"]} {s["label"]}</span>'
            f'📋 다시 보는 설명</div><div class="easy">{html.escape(template_easy_text(r))}</div>'
            f'<div class="rule">원문 정보: {html.escape(rule_text(r))}</div></div>')


# ------------------------------------------------------------ 메인
def main() -> None:
    with st.sidebar:
        st.header("화면 설정")
        font = st.radio("글자 크기", list(FONT_SIZES), index=1, horizontal=True)
        mode = st.radio("처방 입력 방법", ["가상 사례 선택", "CSV 파일 올리기", "표에 직접 입력"], index=0)
        st.divider()
        st.subheader("AI 쉬운 설명 (선택)")
        has_key = ai_available()
        use_ai = st.toggle("AI로 더 쉬운 말 만들기", value=False, disabled=not has_key)
        if has_key:
            st.caption("AI에는 약 이름·바뀐 항목·전후 값만 보냅니다. 나이·사례 번호는 보내지 않습니다. "
                       "원문과 다른 숫자·약·시점이 나오면 AI 문장을 버리고 규칙 기반 설명을 보여 줍니다.")
        else:
            st.caption("API 키가 설정되지 않아 규칙 기반 설명만 사용합니다. (.env.example 참고)")
        st.divider()
        st.caption("🔒 개인정보: 이 앱은 처방 정보를 저장하지 않으며 로그인·서버 저장 기능이 없습니다. "
                   "업로드·직접 입력 시 환자 식별 정보를 넣지 마세요.")

    inject_css(FONT_SIZES[font])
    st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)
    st.title(f"💊 {APP_TITLE}")
    st.markdown("외래 진료 뒤 **바뀐 처방**을 ‘보고(See) → 이해하고(Understand) → 다시 보고(Repeat) → "
                "확인하는(Check)’ 순서로 안내합니다.")

    pres, meta = load_base_data()

    step_title("1. 처방 불러오기")
    if mode == "가상 사례 선택":
        before, after, m = input_from_case(pres, meta)
    elif mode == "CSV 파일 올리기":
        before, after, m = input_from_upload()
    else:
        before, after, m = input_from_editor(pres)

    if not pre_warnings(before, after):
        return

    step_title("2. 입력 확인 (안전 검사)")
    v = validate_case(before, after, m)
    show_validation(v)
    valid = v.is_valid

    step_title("3. 무엇이 바뀌었나요? (See)")
    if st.button("🔍 바뀐 내용 보기", disabled=not valid, type="primary", key="btn_compare"):
        st.session_state["comparison"] = compare_case(v)
    rows = st.session_state.get("comparison") if valid else None
    if not valid:
        st.caption("입력 확인이 끝나지 않아 변경 카드를 만들지 않습니다.")
        c1, c2 = st.columns(2)
        c1.button("💬 쉬운 설명 보기", disabled=True, width="stretch", key="btn_explain_off")
        c2.button("📝 확인 문제 시작 (Teach-back)", disabled=True, width="stretch", key="btn_quiz_off")
        return
    if rows is None:
        st.caption("버튼을 누르면 변경 전·후 처방을 비교해 보여 드립니다.")
        return
    show_see(rows)

    step_title("4. 쉬운 말로 알아보기 (Understand · Repeat)")
    show_understand(rows, valid, use_ai)

    step_title("5. 잘 이해했는지 확인하기 (Check)")
    show_check(rows, valid)

    st.divider()
    st.caption("이 프로그램은 가상 경구 정제 데이터로 처방 ‘변화’만 구조적으로 비교합니다. 약의 효과·부작용·상호작용, "
               "정제 분할 가능 여부, 복용 적정성은 판단하지 않습니다. 실제 복약은 반드시 의사·약사의 안내를 따르세요.")


main()
