# PortfolioBench — Development & Testing Makefile
# Usage: make help

# ───────────────────── Configuration ─────────────────────

PYTHON       := python3
FLOWMESH_URL ?= http://localhost:8000
FLOWMESH_KEY ?= flm-demo-00000000000000000000000000000000
LUMIDOS_DIR  := $(shell cd .. && pwd)/LumidOS

WORKFLOW_DIR := workflow/examples
DATA_DIR     := user_data/data/usstock

.PHONY: help install install-dev test lint \
        data generate-data \
        workflow-local workflow-echo workflow-inference workflow-ai-analysis \
        benchmark benchmark-quick \
        check-flowmesh check-data check-lumidos \
        clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ───────────────────── Setup ─────────────────────

install: ## Install PortfolioBench and dependencies
	$(PYTHON) -m pip install -e .
	cd freqtrade && $(PYTHON) -m pip install -e .

install-dev: ## Install with test dependencies
	$(PYTHON) -m pip install -e ".[dev]" 2>/dev/null || $(PYTHON) -m pip install -e .
	cd freqtrade && $(PYTHON) -m pip install -e .

# ───────────────────── Preflight Checks ─────────────────────

check-data: ## Verify market data exists
	@test -d $(DATA_DIR) && echo "Data dir OK: $$(ls $(DATA_DIR)/*.feather 2>/dev/null | wc -l) feather files" \
		|| (echo "No data at $(DATA_DIR). Run: make generate-data" && exit 1)

check-lumidos: ## Verify LumidOS is available as sibling directory
	@test -d $(LUMIDOS_DIR)/adapters/portbench && echo "LumidOS OK at $(LUMIDOS_DIR)" \
		|| (echo "LumidOS not found at $(LUMIDOS_DIR)" && exit 1)

check-flowmesh: ## Verify FlowMesh is reachable
	@curl -sf $(FLOWMESH_URL)/healthz >/dev/null 2>&1 \
		&& echo "FlowMesh OK at $(FLOWMESH_URL)" \
		|| (echo "FlowMesh not reachable at $(FLOWMESH_URL). Start it with: cd ../FlowMesh_dev && make up" && exit 1)

# ───────────────────── Data ─────────────────────

generate-data: ## Generate synthetic OHLCV test data for all 119 instruments
	portbench generate-data

data: generate-data ## Alias for generate-data

download-data: ## Download real market data from Google Drive
	portbench download-data --exchange portfoliobench

# ───────────────────── Tests ─────────────────────

test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

test-workflow: check-data check-lumidos ## Run workflow unit tests (local only, no FlowMesh)
	$(PYTHON) -m pytest tests/test_workflow.py -v

# ───────────────────── Local Workflows (no FlowMesh) ─────────────────────

workflow-local: check-data check-lumidos ## Run default EMA blend workflow (local only)
	portbench workflow $(WORKFLOW_DIR)/ema_ons_blend.json

workflow-local-rsi: check-data check-lumidos ## Run RSI equal-weight workflow (local only)
	portbench workflow $(WORKFLOW_DIR)/rsi_equal_weight.json

workflow-local-macd: check-data check-lumidos ## Run MACD ONS workflow (local only)
	portbench workflow $(WORKFLOW_DIR)/macd_ons_pure.json

workflow-local-all: check-data check-lumidos ## Run all local workflows
	@echo "=== EMA Blend ===" && portbench workflow $(WORKFLOW_DIR)/ema_ons_blend.json --output-json results/ema_blend.json
	@echo ""
	@echo "=== RSI Equal Weight ===" && portbench workflow $(WORKFLOW_DIR)/rsi_equal_weight.json --output-json results/rsi_equal.json
	@echo ""
	@echo "=== MACD ONS ===" && portbench workflow $(WORKFLOW_DIR)/macd_ons_pure.json --output-json results/macd_ons.json
	@echo ""
	@echo "=== Bollinger Blend ===" && portbench workflow $(WORKFLOW_DIR)/bollinger_blend.json --output-json results/bollinger.json
	@echo ""
	@echo "All local workflows complete. Results in results/"

# ───────────────────── FlowMesh Workflows (GPU) ─────────────────────

workflow-echo: check-lumidos check-flowmesh ## Run echo connectivity test via FlowMesh
	portbench workflow $(WORKFLOW_DIR)/flowmesh_echo_test.json \
		--flowmesh-url $(FLOWMESH_URL) \
		--flowmesh-key $(FLOWMESH_KEY)

workflow-inference: check-lumidos check-flowmesh ## Run pure GPU inference via FlowMesh
	portbench workflow $(WORKFLOW_DIR)/flowmesh_inference_only.json \
		--flowmesh-url $(FLOWMESH_URL) \
		--flowmesh-key $(FLOWMESH_KEY) \
		--output-json results/inference_only.json

workflow-ai-analysis: check-data check-lumidos check-flowmesh ## Run full pipeline: local alpha/strategy/portfolio + FlowMesh AI analysis
	portbench workflow $(WORKFLOW_DIR)/ema_blend_ai_analysis.json \
		--flowmesh-url $(FLOWMESH_URL) \
		--flowmesh-key $(FLOWMESH_KEY) \
		--output-json results/ai_analysis.json

# ───────────────────── Benchmarking ─────────────────────

benchmark-quick: ## Run quick benchmark suite (subset of strategies)
	portbench benchmark --quick

benchmark: ## Run full benchmark suite
	portbench benchmark

benchmark-all: ## Run complete benchmark matrix (all strategies x all asset classes)
	portbench benchmark-all

# ───────────────────── E2E Pipeline ─────────────────────

e2e-local: check-data check-lumidos ## Run full local E2E pipeline (no FlowMesh)
	@echo "================================================================"
	@echo "E2E LOCAL PIPELINE"
	@echo "================================================================"
	@mkdir -p results
	portbench workflow $(WORKFLOW_DIR)/ema_ons_blend.json --output-json results/e2e_local.json
	@echo ""
	@echo "E2E local pipeline PASSED"
	@echo "Results: results/e2e_local.json"

e2e-flowmesh: check-data check-lumidos check-flowmesh ## Run full E2E pipeline through FlowMesh GPU
	@echo "================================================================"
	@echo "E2E FLOWMESH PIPELINE"
	@echo "================================================================"
	@mkdir -p results
	@echo "Step 1: Echo test..."
	portbench workflow $(WORKFLOW_DIR)/flowmesh_echo_test.json \
		--flowmesh-url $(FLOWMESH_URL) --flowmesh-key $(FLOWMESH_KEY)
	@echo ""
	@echo "Step 2: GPU inference test..."
	portbench workflow $(WORKFLOW_DIR)/flowmesh_inference_only.json \
		--flowmesh-url $(FLOWMESH_URL) --flowmesh-key $(FLOWMESH_KEY) \
		--output-json results/e2e_inference.json
	@echo ""
	@echo "Step 3: Full pipeline (local compute + FlowMesh AI)..."
	portbench workflow $(WORKFLOW_DIR)/ema_blend_ai_analysis.json \
		--flowmesh-url $(FLOWMESH_URL) --flowmesh-key $(FLOWMESH_KEY) \
		--output-json results/e2e_ai_analysis.json
	@echo ""
	@echo "E2E FlowMesh pipeline PASSED"
	@echo "Results in results/"

# ───────────────────── Cleanup ─────────────────────

clean: ## Remove build artifacts and result files
	rm -rf build/ dist/ *.egg-info .pytest_cache results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
