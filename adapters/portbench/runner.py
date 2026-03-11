"""Local workflow runner for PortfolioBench.

Provides a lightweight implementation of the LumidStack ``LocalWorkflowRunner``
API so that the workflow mode can run without the full LumidStack package.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class WorkflowMetadata:
    name: str | None = None


@dataclass
class WorkflowSpec:
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    data: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class WorkflowResult:
    stages: dict[str, StageResult] = field(default_factory=dict)
    total_duration_s: float = 0.0


class LocalWorkflowRunner:
    """Minimal runner that resolves stage dependencies and executes handlers."""

    def __init__(self, workflow_dict: dict[str, Any]) -> None:
        metadata_raw = workflow_dict.get("metadata", {})
        self.workflow = WorkflowMetadata(name=metadata_raw.get("name"))

        spec = workflow_dict.get("spec", {})
        stages_raw = spec.get("stages", {})

        self._stages: dict[str, dict[str, Any]] = stages_raw
        self._handlers: dict[str, Callable] = {}

        # Everything in spec that isn't "stages" is extra (e.g. backtest config).
        self.extra_spec: dict[str, Any] = {
            k: v for k, v in spec.items() if k != "stages"
        }

    # -- Construction helpers ------------------------------------------------

    @classmethod
    def from_json(cls, json_str: str) -> LocalWorkflowRunner:
        return cls(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str | Path) -> LocalWorkflowRunner:
        return cls(json.loads(Path(path).read_text()))

    # -- Handler registration ------------------------------------------------

    def register_handler(self, template: str, handler: Callable) -> None:
        self._handlers[template] = handler

    # -- Execution -----------------------------------------------------------

    def run(self, *, context: dict[str, Any] | None = None) -> WorkflowResult:
        if context is None:
            context = {}

        # Topological sort based on dependsOn.
        order = self._resolve_order()

        result = WorkflowResult()
        t_start = time.monotonic()

        for stage_name in order:
            stage_def = self._stages[stage_name]
            template = stage_def.get("template", stage_name)
            params = stage_def.get("params", {})

            handler = self._handlers.get(template)
            if handler is None:
                raise RuntimeError(
                    f"No handler registered for template {template!r} "
                    f"(stage {stage_name!r})"
                )

            s_start = time.monotonic()
            data = handler(stage_name, params, context)
            s_end = time.monotonic()

            result.stages[stage_name] = StageResult(
                data=data or {},
                duration_s=s_end - s_start,
            )

        result.total_duration_s = time.monotonic() - t_start
        return result

    # -- Internal helpers ----------------------------------------------------

    def _resolve_order(self) -> list[str]:
        """Return stage names in dependency order (simple topological sort)."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            stage_def = self._stages.get(name, {})
            for dep in stage_def.get("dependsOn", []):
                visit(dep)
            order.append(name)

        for name in self._stages:
            visit(name)

        return order
