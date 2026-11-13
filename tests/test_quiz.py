from src.quiz import KIND_ADDED, KIND_DOSE, KIND_FREQ, KIND_STOP, KIND_TIME, QuizTracker, build_questions


def test_case_001_answer(compared):
    qs = build_questions(compared("CASE_001"), "case:CASE_001")
    assert len(qs) == 1
    q = qs[0]
    assert q.kind == KIND_DOSE and q.answer == "1회 0.5정"
    assert "1회 1정" in q.options


def test_case_012_covers_all_change_types(compared):
    kinds = {q.kind for q in build_questions(compared("CASE_012"), "k")}
    assert {KIND_STOP, KIND_DOSE, KIND_FREQ, KIND_ADDED} <= kinds
    assert KIND_TIME in kinds


def test_no_question_for_maintained(compared):
    rows = compared("CASE_012")
    kept = {r.drug_id for r in rows if r.classification == "유지"}
    assert not any(q.drug_id in kept for q in build_questions(rows, "k"))


def test_options_unique_and_single_answer(compared):
    for i in range(1, 16):
        cid = f"CASE_{i:03d}"
        rows = compared(cid)
        added = {r.drug_name for r in rows if r.classification == "추가"}
        for q in build_questions(rows, cid):
            assert len(q.options) == len(set(q.options)) >= 2
            assert q.options.count(q.answer) == 1
            if q.kind == KIND_ADDED:
                assert len(set(q.options) & added) == 1


def test_options_deterministic(compared):
    rows = compared("CASE_014")
    assert build_questions(rows, "x") == build_questions(rows, "x")


def test_tracker_first_vs_retry(compared):
    qs = build_questions(compared("CASE_012"), "k")
    t = QuizTracker()
    wrong = next(o for o in qs[0].options if o != qs[0].answer)
    assert t.submit(qs[0], wrong) is False
    assert t.submit(qs[0], qs[0].answer) is True
    t.submit(qs[1], qs[1].answer)
    s = t.summary(qs)
    assert s["first_answered"] == 2 and s["first_correct"] == 1 and s["first_accuracy"] == 0.5
    assert s["retry_targets"] == 1 and s["retry_solved"] == 1
    rec = t.get(qs[0].qid)
    assert rec.first_choice == wrong and rec.attempts == 2 and rec.solved
    t.submit(qs[0], wrong)  # 이미 푼 문항은 기록이 바뀌지 않음
    assert t.get(qs[0].qid).attempts == 2
