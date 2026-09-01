from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.ml.experiments import adapters


def _patch_modules(monkeypatch, modules: dict[str, object]):
    package = SimpleNamespace(__path__=["fake-experiments"])
    items = [SimpleNamespace(name=name) for name in modules]

    def fake_import(name: str):
        if name == adapters.PACKAGE:
            return package
        prefix = f"{adapters.PACKAGE}."
        if name.startswith(prefix):
            return modules[name[len(prefix):]]
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(adapters.importlib, "import_module", fake_import)
    monkeypatch.setattr(adapters.pkgutil, "iter_modules", lambda path: items)


def test_batch_adapter_file_is_ignored_even_without_interactive_contract(monkeypatch):
    batch_module = SimpleNamespace(__name__="adapter_exp999")
    _patch_modules(monkeypatch, {"adapter_exp999": batch_module})

    assert adapters.discover_experiment_adapters() == []


def test_interactive_adapter_must_explicitly_opt_in(monkeypatch):
    module = SimpleNamespace(
        __name__="adapter_exp998",
        INTERACTIVE_ADAPTER=True,
        EXPERIMENT_ID="exp_998",
        EXPERIMENT_SEQUENCE=998,
        EXPERIMENT_NAME="Interactive test",
        EXPERIMENT_SCOPE="cost_delay",
        fit_against_production=lambda **kwargs: {},
        filter_comparable_rows=lambda frame, state: frame,
        predict_project=lambda row, state: {},
    )
    _patch_modules(monkeypatch, {"adapter_exp998": module})

    discovered = adapters.discover_experiment_adapters()
    assert [item.experiment_id for item in discovered] == ["exp_998"]


def test_malformed_interactive_adapter_still_fails_fast(monkeypatch):
    module = SimpleNamespace(
        __name__="adapter_exp997",
        INTERACTIVE_ADAPTER=True,
        EXPERIMENT_ID="exp_997",
        EXPERIMENT_SEQUENCE=997,
        EXPERIMENT_NAME="Broken interactive test",
        EXPERIMENT_SCOPE="cost",
    )
    _patch_modules(monkeypatch, {"adapter_exp997": module})

    with pytest.raises(ValueError, match="fit_against_production"):
        adapters.discover_experiment_adapters()


def test_current_batch_experiments_are_not_in_interactive_catalog():
    assert all(item["experiment_id"] != "exp_61" for item in adapters.available_experiments())
