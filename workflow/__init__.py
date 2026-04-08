"""Workflow-driven CLI mode for PortfolioBench.

This package provides a CLI mode where the alpha -> strategy -> portfolio ->
backtest pipeline is orchestrated by a lumid/v1 workflow JSON, executed via
LumidOS's :class:`LocalWorkflowRunner`.

Stages with ``portbench.*`` templates run locally; stages with ``flowmesh.*``
templates are dispatched to a FlowMesh instance for GPU-accelerated execution
when ``--flowmesh-url`` is provided.
"""
