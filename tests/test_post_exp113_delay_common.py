import pandas as pd
import pytest

from backend.app.ml.experiments import post_exp113_delay_common as common


class _ImmediateFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _ImmediateExecutor:
    last_workers = None

    def __init__(self, *, max_workers, mp_context, initializer, initargs):
        type(self).last_workers = max_workers
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args):
        try:
            return _ImmediateFuture(value=function(*args))
        except Exception as exc:
            return _ImmediateFuture(error=exc)


def _context():
    return {
        "train": pd.DataFrame({"canonical_project_id": ["p"]}),
        "full_data": pd.DataFrame({"value": [1]}),
        "identity": pd.DataFrame({"value": [1]}),
    }


def test_production_oof_parallelizes_and_restores_chronological_order(monkeypatch):
    folds = [
        (None, pd.DataFrame({"actual_delay_days": [1.0]}), 2020),
        (None, pd.DataFrame({"actual_delay_days": [1.0]}), 2019),
    ]
    monkeypatch.setenv("POST_EXP113_OOF_WORKERS", "2")
    monkeypatch.setattr(common, "forward_folds", lambda frame, max_folds: folds)
    monkeypatch.setattr(common, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(common, "get_context", lambda name: name)
    monkeypatch.setattr(common, "as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr(
        common,
        "_production_oof_fold",
        lambda validation, year, data=None, identity=None: validation.assign(
            production_prediction=0.0,
            residual=1.0,
            oof_year=year,
        ),
    )

    result = common.production_oof(_context())

    assert _ImmediateExecutor.last_workers == 2
    assert result["oof_year"].tolist() == [2019, 2020]


def test_production_oof_reports_fold_failures(monkeypatch):
    folds = [
        (None, pd.DataFrame({"actual_delay_days": [1.0]}), 2019),
        (None, pd.DataFrame({"actual_delay_days": [1.0]}), 2020),
    ]
    monkeypatch.setenv("POST_EXP113_OOF_WORKERS", "1")
    monkeypatch.setattr(common, "forward_folds", lambda frame, max_folds: folds)

    def fail(validation, year, data=None, identity=None):
        raise RuntimeError(f"boom-{year}")

    monkeypatch.setattr(common, "_production_oof_fold", fail)

    with pytest.raises(ValueError, match=r"completed=0.*boom-2019.*boom-2020"):
        common.production_oof(_context())
