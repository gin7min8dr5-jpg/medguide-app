"""8단계: 자동 평가 (개발자 전용).

ground_truth.csv 는 이 모듈에서만 읽는다. 예측은 앱과 똑같은 경로
(data_loader → validate_case → compare_case)로 먼저 계산한 뒤, 그 다음에 정답과 대조한다.
정답 값은 예측 함수에 전달되지 않는다.

실행:  python -m src.evaluator            (결과를 reports/ 에 저장)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import data_loader as dl
from .comparator import compare_case
from .validator import validate_case

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"


def predict_all(data_dir: Path | None = None) -> tuple[dict, dict]:
    """정답을 보지 않고 모든 사례를 예측한다. (validation 결과, 비교 결과) 반환."""
    data_dir = data_dir or dl.DATA_DIR
    pres = dl.load_prescriptions(data_dir / "prescriptions.csv")
    meta = dl.load_metadata(data_dir / "case_metadata.csv")   # expected_* 제외본
    validations, comparisons = {}, {}
    for cid in dl.list_case_ids(pres):
        before, after = dl.split_case(pres, cid)
        v = validate_case(before, after, dl.case_metadata(meta, cid))
        validations[cid] = v
        if v.is_valid:
            comparisons[cid] = compare_case(v)
    return validations, comparisons


def evaluate(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or dl.DATA_DIR
    validations, comparisons = predict_all(data_dir)

    gt = pd.read_csv(data_dir / "ground_truth.csv", encoding=dl.ENCODING, dtype=str, keep_default_na=False)
    drug_gt = gt[gt["record_type"] == "drug"]
    case_gt = gt[gt["record_type"] == "case"]

    # --- 정상 사례: 약물별 분류 ---
    pred_rows = {(cid, r.drug_id): r for cid, rows in comparisons.items() for r in rows}
    drug_results = []
    for g in drug_gt.to_dict("records"):
        key = (g["case_id"], g["drug_id"])
        p = pred_rows.get(key)
        v = validations.get(g["case_id"])
        drug_results.append({
            "case_id": g["case_id"], "drug_id": g["drug_id"],
            "expected_class": g["expected_class"],
            "predicted_class": p.classification if p else ("(검증 실패: %s)" % v.error_code if v else "(예측 없음)"),
            "class_match": bool(p and p.classification == g["expected_class"]),
            "expected_changed_fields": g["changed_fields"],
            "predicted_changed_fields": p.changed_fields_str if p else "",
            "fields_match": bool(p and p.changed_fields_str == g["changed_fields"]),
            "expected_summary": g["change_summary"],
            "predicted_summary": p.change_summary if p else "",
            "summary_match": bool(p and p.change_summary == g["change_summary"]),
        })
    gt_keys = {(g["case_id"], g["drug_id"]) for g in drug_gt.to_dict("records")}
    extra_predictions = sorted(f"{c}/{d}" for (c, d) in pred_rows if (c, d) not in gt_keys)

    # --- 오류 사례 ---
    case_results = []
    for g in case_gt.to_dict("records"):
        v = validations.get(g["case_id"])
        case_results.append({
            "case_id": g["case_id"],
            "expected_status": g["expected_validation_status"],
            "predicted_status": v.status if v else "(없음)",
            "expected_error_code": g["expected_error_code"],
            "predicted_error_code": v.error_code if v else "",
            "match": bool(v and v.status == g["expected_validation_status"]
                          and v.error_code == g["expected_error_code"]),
            "compare_called": g["case_id"] in comparisons,
            "message": v.error_message if v else "",
        })

    # 정상 사례가 잘못 NEEDS_REVIEW 로 막히지 않았는지
    normal_cases = sorted(drug_gt["case_id"].unique())
    false_blocks = [c for c in normal_cases if not validations[c].is_valid]

    n_drug = len(drug_results)
    n_case = len(case_results)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "drug_total": n_drug,
        "drug_class_correct": sum(r["class_match"] for r in drug_results),
        "drug_fields_correct": sum(r["fields_match"] for r in drug_results),
        "drug_summary_correct": sum(r["summary_match"] for r in drug_results),
        "case_total": n_case,
        "case_correct": sum(r["match"] for r in case_results),
        "error_cases_compared": [r["case_id"] for r in case_results if r["compare_called"]],
        "normal_cases_falsely_blocked": false_blocks,
        "extra_predictions": extra_predictions,
        "misclassified": [r for r in drug_results if not r["class_match"]],
        "field_mismatches": [r for r in drug_results if r["class_match"] and not r["fields_match"]],
        "case_results": case_results,
        "drug_results": drug_results,
    }
    report["drug_class_accuracy"] = report["drug_class_correct"] / n_drug * 100 if n_drug else 0.0
    report["case_accuracy"] = report["case_correct"] / n_case * 100 if n_case else 0.0
    return report


def format_report(r: dict) -> str:
    lines = [
        "# 자동 평가 결과",
        "",
        f"- 실행 시각: {r['generated_at']}",
        f"- 정상 약물 분류 일치: **{r['drug_class_correct']}/{r['drug_total']}** "
        f"({r['drug_class_accuracy']:.1f}%)",
        f"- (보조) 변경 필드 일치: {r['drug_fields_correct']}/{r['drug_total']}, "
        f"변경 요약 문구 일치: {r['drug_summary_correct']}/{r['drug_total']}",
        f"- 오류 사례 탐지(NEEDS_REVIEW + 오류 코드) 일치: **{r['case_correct']}/{r['case_total']}** "
        f"({r['case_accuracy']:.1f}%)",
        f"- 오류 사례에서 비교 엔진이 호출된 사례: {', '.join(r['error_cases_compared']) or '없음'}",
        f"- 정상 사례가 잘못 차단된 사례: {', '.join(r['normal_cases_falsely_blocked']) or '없음'}",
        f"- 정답에 없는 추가 예측: {', '.join(r['extra_predictions']) or '없음'}",
        "",
        "## 오류 사례 판정",
        "",
        "| 사례 | 기대 | 예측 | 일치 | 메시지 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in r["case_results"]:
        lines.append(f"| {c['case_id']} | {c['expected_error_code']} | {c['predicted_status']} / "
                     f"{c['predicted_error_code']} | {'O' if c['match'] else 'X'} | {c['message']} |")
    lines += ["", "## 오분류 목록", ""]
    if r["misclassified"]:
        for m in r["misclassified"]:
            lines.append(f"- {m['case_id']} {m['drug_id']}: 기대 {m['expected_class']} / 예측 {m['predicted_class']}")
    else:
        lines.append("- 없음")
    lines += ["", "## 사례별 분류 건수 (예측)", "", "| 사례 | 유지 | 중단 | 변경 | 추가 |", "| --- | --- | --- | --- | --- |"]
    per = {}
    for d in r["drug_results"]:
        per.setdefault(d["case_id"], {"유지": 0, "중단": 0, "변경": 0, "추가": 0})
        if d["predicted_class"] in per[d["case_id"]]:
            per[d["case_id"]][d["predicted_class"]] += 1
    for cid, c in per.items():
        lines.append(f"| {cid} | {c['유지']} | {c['중단']} | {c['변경']} | {c['추가']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    report = evaluate()
    text = format_report(report)
    print(text)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation_report.md").write_text(text, encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "drug_results"}
    (REPORT_DIR / "evaluation_report.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(report["drug_results"]).to_csv(REPORT_DIR / "evaluation_drug_level.csv", index=False, encoding="utf-8-sig")
    ok = (report["drug_class_correct"] == report["drug_total"] and report["case_correct"] == report["case_total"]
          and not report["error_cases_compared"] and not report["normal_cases_falsely_blocked"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
