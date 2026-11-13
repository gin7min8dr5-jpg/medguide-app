import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_loader as dl  # noqa: E402
from src.validator import validate_case  # noqa: E402
from src.comparator import compare_case  # noqa: E402


@pytest.fixture(scope="session")
def pres():
    return dl.load_prescriptions()


@pytest.fixture(scope="session")
def meta():
    return dl.load_metadata()


@pytest.fixture
def run_case(pres, meta):
    def _run(cid):
        b, a = dl.split_case(pres, cid)
        return validate_case(b, a, dl.case_metadata(meta, cid))
    return _run


@pytest.fixture
def compared(run_case):
    def _cmp(cid):
        return compare_case(run_case(cid))
    return _cmp


COMPLETE = {"before_list_complete": True, "after_list_complete": True}


def frame(rows, stage, case_id="T1"):
    """간단한 테스트용 처방 표. rows: (drug_id, name, strength, dose, freq, time)"""
    recs = []
    for did, name, strength, dose, freq, time in rows:
        recs.append({"case_id": case_id, "prescription_stage": stage, "drug_id": did, "drug_name": name,
                     "strength_mg": strength, "dose_count": dose, "dose_unit": "정",
                     "frequency_per_day": freq, "administration_time": time})
    return pd.DataFrame(recs, columns=dl.PRESCRIPTION_COLUMNS)
