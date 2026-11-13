import ast
import re
from pathlib import Path

from src.evaluator import evaluate

ROOT = Path(__file__).resolve().parent.parent


def test_evaluation_targets():
    r = evaluate()
    assert r["drug_total"] == 108 and r["case_total"] == 5
    assert r["drug_class_correct"] == 108, r["misclassified"]
    assert r["case_correct"] == 5, r["case_results"]
    assert r["error_cases_compared"] == []
    assert r["normal_cases_falsely_blocked"] == []


def _code_tokens(path: Path) -> list[str]:
    """docstring·주석을 제외한 실제 코드의 문자열 상수·이름·import 대상."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                docstrings.add(id(node.body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out += [getattr(node, "module", "") or ""] + [a.name for a in node.names]
    return out


def test_ground_truth_only_in_evaluator():
    """정답 파일·정답 열은 evaluator.py 밖의 앱/분석 코드에서 사용되지 않아야 한다."""
    offenders = []
    for p in [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]:
        if p.name == "evaluator.py":
            continue
        for tok in _code_tokens(p):
            if "ground_truth" in tok or "evaluator" in tok or re.fullmatch(r"expected_(class|error_code)", tok):
                offenders.append(f"{p.name}:{tok}")
    assert offenders == []
