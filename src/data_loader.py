"""CSV 로딩 모듈.

- 모든 CSV는 utf-8-sig(UTF-8 BOM)로 읽는다.
- 값은 문자열(dtype=str)로 읽고 keep_default_na=False 로 두어,
  "빈 문자열"이 NaN으로 바뀌지 않도록 한다. 결측 판단은 validator가
  문자열을 직접 보고 수행한다(빈 문자열 = 결측).
- case_metadata.csv의 expected_* 열과 scenario_description(정답 요약이 담긴 설명)은
  앱 입력 경로에서 분리한다. 정답 파일(ground_truth.csv)은 이 모듈에서 읽지 않는다
  (evaluator.py 전용).
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENCODING = "utf-8-sig"

PRESCRIPTION_COLUMNS = [
    "case_id",
    "prescription_stage",
    "drug_id",
    "drug_name",
    "strength_mg",
    "dose_count",
    "dose_unit",
    "frequency_per_day",
    "administration_time",
]

# 앱이 사용해도 되는 메타데이터 열 (입력·표시용)
METADATA_INPUT_COLUMNS = [
    "case_id",
    "age_years",
    "scenario_level",
    "before_drug_count",
    "after_drug_count",
    "before_list_complete",
    "after_list_complete",
]

# 정답(expected_* 로 시작하는 열)과 정답 요약이 담긴 설명 열 → 앱 입력 경로에서 제외.
# load_metadata()는 위 METADATA_INPUT_COLUMNS 만 남기는 화이트리스트 방식이다.
METADATA_ANSWER_PREFIX = "expected_"
METADATA_HINT_COLUMNS = ["scenario_description"]

STAGES = ("before", "after")


def _read_csv(source: Any) -> pd.DataFrame:
    """경로 또는 업로드 파일 객체를 문자열 DataFrame으로 읽는다."""
    if hasattr(source, "getvalue"):  # Streamlit UploadedFile / BytesIO
        source = io.BytesIO(source.getvalue())
    return pd.read_csv(source, encoding=ENCODING, dtype=str, keep_default_na=False)


def clean_cell(value: Any) -> str:
    """None / NaN / 공백을 빈 문자열로 통일하고 나머지는 strip한 문자열로 반환."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """외부 입력(편집 테이블 등)의 셀 값을 문자열로 정규화한다. 열 이름은 바꾸지 않는다."""
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    for col in out.columns:
        out[col] = out[col].map(clean_cell)
    # 모든 값이 빈 행(편집 테이블의 빈 줄)은 입력이 아닌 것으로 본다
    if len(out.columns):
        out = out[~(out == "").all(axis=1)]
    return out.reset_index(drop=True)


def load_prescriptions(path: Path | None = None) -> pd.DataFrame:
    return _read_csv(path or DATA_DIR / "prescriptions.csv")


def load_metadata_full(path: Path | None = None) -> pd.DataFrame:
    """정답 관련 열까지 포함한 전체 메타데이터 (점검 스크립트·evaluator 용)."""
    return _read_csv(path or DATA_DIR / "case_metadata.csv")


def load_metadata(path: Path | None = None) -> pd.DataFrame:
    """앱에서 사용하는 메타데이터: expected_* 및 설명 열을 제외한다."""
    df = load_metadata_full(path)
    return df[[c for c in METADATA_INPUT_COLUMNS if c in df.columns]].copy()


def parse_bool(value: Any) -> bool | None:
    v = clean_cell(value).upper()
    if v in {"TRUE", "T", "1", "Y", "YES"}:
        return True
    if v in {"FALSE", "F", "0", "N", "NO"}:
        return False
    return None


def list_case_ids(prescriptions: pd.DataFrame) -> list[str]:
    return sorted(prescriptions["case_id"].unique().tolist())


def split_case(prescriptions: pd.DataFrame, case_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 사례의 처방을 before / after 로 분리한다 (원본 행 순서 유지)."""
    case = prescriptions[prescriptions["case_id"] == case_id]
    before = case[case["prescription_stage"] == "before"].reset_index(drop=True)
    after = case[case["prescription_stage"] == "after"].reset_index(drop=True)
    return before, after


def case_metadata(metadata: pd.DataFrame, case_id: str) -> dict:
    """앱용 메타데이터 한 행을 dict로 반환한다 (완전성 플래그는 bool로 변환)."""
    rows = metadata[metadata["case_id"] == case_id]
    if rows.empty:
        return {"case_id": case_id, "before_list_complete": None, "after_list_complete": None}
    row = rows.iloc[0].to_dict()
    row["before_list_complete"] = parse_bool(row.get("before_list_complete"))
    row["after_list_complete"] = parse_bool(row.get("after_list_complete"))
    return row


def read_uploaded_prescription(upload: Any) -> tuple[pd.DataFrame | None, str | None]:
    """업로드된 CSV를 읽는다. 실패하면 (None, 오류 메시지)를 반환하고 예외를 올리지 않는다."""
    try:
        df = _read_csv(upload)
    except pd.errors.EmptyDataError:
        return None, "빈 파일입니다. 처방 내용이 담긴 CSV를 올려 주세요."
    except UnicodeDecodeError:
        return None, "UTF-8 형식의 CSV가 아닙니다. 엑셀에서 'CSV UTF-8'로 저장해 주세요."
    except Exception as exc:  # 파싱 오류 등
        return None, f"CSV를 읽을 수 없습니다: {exc}"
    df = clean_frame(df)
    if df.empty:
        return None, "파일에 처방 행이 없습니다."
    return df, None


def empty_template(stage: str, case_id: str = "MY_CASE", rows: int = 3) -> pd.DataFrame:
    """직접 입력용 빈 편집 테이블."""
    data = {c: [""] * rows for c in PRESCRIPTION_COLUMNS}
    data["case_id"] = [case_id] * rows
    data["prescription_stage"] = [stage] * rows
    data["dose_unit"] = ["정"] * rows
    return pd.DataFrame(data)
