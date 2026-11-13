from src import data_loader as dl


def test_row_counts_and_cases(pres, meta):
    assert len(pres) == 257
    assert len(meta) == 20
    assert dl.list_case_ids(pres) == [f"CASE_{i:03d}" for i in range(1, 21)]


def test_ground_truth_shape_for_evaluator_only():
    import pandas as pd
    gt = pd.read_csv(dl.DATA_DIR / "ground_truth.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert len(gt) == 113
    assert (gt["record_type"] == "drug").sum() == 108
    assert (gt["record_type"] == "case").sum() == 5


def test_required_columns_exact(pres):
    assert list(pres.columns) == dl.PRESCRIPTION_COLUMNS  # BOM 제거 확인 포함


def test_metadata_for_app_excludes_answers(meta):
    full = dl.load_metadata_full()
    hidden = [c for c in full.columns if c.startswith(dl.METADATA_ANSWER_PREFIX)] + dl.METADATA_HINT_COLUMNS
    assert len(hidden) == 4
    for col in hidden:
        assert col not in meta.columns
    assert "after_list_complete" in meta.columns


def test_empty_string_kept_not_nan(pres):
    row = pres[(pres.case_id == "CASE_018") & (pres.prescription_stage == "after") & (pres.drug_id == "DRUG_B")]
    assert row.iloc[0]["dose_count"] == ""
    row16 = pres[(pres.case_id == "CASE_016") & (pres.drug_name == "가상 약물 E") & (pres.prescription_stage == "after")]
    assert row16.iloc[0]["drug_id"] == ""


def test_split_case_counts(pres):
    b, a = dl.split_case(pres, "CASE_001")
    assert (len(b), len(a)) == (6, 6)
    b, a = dl.split_case(pres, "CASE_020")
    assert (len(b), len(a)) == (7, 5)


def test_case_metadata_flags(meta):
    assert dl.case_metadata(meta, "CASE_020")["after_list_complete"] is False
    assert dl.case_metadata(meta, "CASE_001")["after_list_complete"] is True


def test_clean_cell_handles_none_nan():
    import math
    assert dl.clean_cell(None) == ""
    assert dl.clean_cell(float("nan")) == ""
    assert dl.clean_cell(" 1 ") == "1"
    assert not math.isnan(float(dl.clean_cell(1.0)))


def test_uploaded_empty_and_bad_file():
    import io
    df, err = dl.read_uploaded_prescription(io.BytesIO(b""))
    assert df is None and "빈 파일" in err
    df, err = dl.read_uploaded_prescription(io.BytesIO("a,b\n".encode("utf-8")))
    assert df is None and err
