"""Discovery contract for interactive experiment-vs-production adapters.

Batch experiment PRs may keep files named ``adapter_exp*.py`` for workflow-local
compatibility, but they are not automatically exposed through the lifecycle
comparison catalog. A module is registered here only when it explicitly sets
``INTERACTIVE_ADAPTER = True``.

This keeps offline scientific experiments independent from the single-project
interactive prediction contract. Batch workflows can continue importing their
experiment implementation directly and can report production-vs-experiment MAE
without becoming selectable application challengers.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from types import ModuleType

PACKAGE = "backend.app.ml.experiments"


@dataclass(frozen=True)
class ExperimentAdapter:
    experiment_id: str
    sequence: int
    name: str
    scope: str
    module: ModuleType

    def public_metadata(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "sequence": self.sequence,
            "name": self.name,
            "scope": self.scope,
        }


def _validate(module: ModuleType) -> ExperimentAdapter:
    required_values = ("EXPERIMENT_ID", "EXPERIMENT_SEQUENCE", "EXPERIMENT_NAME", "EXPERIMENT_SCOPE")
    missing = [name for name in required_values if not getattr(module, name, None)]
    required_calls = ("fit_against_production", "filter_comparable_rows", "predict_project")
    missing += [name for name in required_calls if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"Invalid experiment adapter {module.__name__}; missing: {', '.join(sorted(missing))}")
    return ExperimentAdapter(
        experiment_id=str(module.EXPERIMENT_ID),
        sequence=int(module.EXPERIMENT_SEQUENCE),
        name=str(module.EXPERIMENT_NAME),
        scope=str(module.EXPERIMENT_SCOPE),
        module=module,
    )


def discover_experiment_adapters() -> list[ExperimentAdapter]:
    """Return only experiments explicitly opting into interactive prediction."""
    package = importlib.import_module(PACKAGE)
    adapters: list[ExperimentAdapter] = []
    seen: set[str] = set()
    for item in pkgutil.iter_modules(package.__path__):
        if not item.name.startswith("adapter_exp"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{item.name}")
        if getattr(module, "INTERACTIVE_ADAPTER", False) is not True:
            continue
        adapter = _validate(module)
        if adapter.experiment_id in seen:
            raise ValueError(f"Duplicate experiment adapter id: {adapter.experiment_id}")
        seen.add(adapter.experiment_id)
        adapters.append(adapter)
    return sorted(adapters, key=lambda item: (item.sequence, item.experiment_id))


def available_experiments() -> list[dict]:
    return [adapter.public_metadata() for adapter in discover_experiment_adapters()]


def default_experiment_adapter() -> ExperimentAdapter | None:
    adapters = discover_experiment_adapters()
    return adapters[-1] if adapters else None


def get_experiment_adapter(experiment_id: str | None = None) -> ExperimentAdapter:
    adapters = discover_experiment_adapters()
    if not adapters:
        raise ValueError("No interactive experiment comparison adapter is installed.")
    if experiment_id in (None, ""):
        return adapters[-1]
    for adapter in adapters:
        if adapter.experiment_id == experiment_id:
            return adapter
    raise ValueError(f"Unknown interactive experiment adapter '{experiment_id}'.")
