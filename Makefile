.PHONY: install check test test-network lint typecheck doctor dry-run data synth pull-data label-check s1 s2 s3 s4 mteb release finetune mteb-ft release-ft tune

# Every step below skips work already done locally or on the Hub, so after a lost pod
# `make pull-data s1 s2 s3 s4` picks up where it stopped.
# private/<path> overrides the public example at <path> (src/code_embed_v3/paths.py).
cfg = $(if $(wildcard private/$(1)),private/$(1),$(1))
DATA_REPO ?= divakantpandey/code-embed-v3-data
MODEL_REPO ?= divakantpandey/code-embed-v3
TEACHER ?= Qwen/Qwen3-Reranker-4B
# Rows per source sent to the teacher: the largest cost of the pipeline.
LABEL_ROWS ?= 100000

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -q -m "not network"

test-network:
	uv run pytest -q -m network

check: lint typecheck test

doctor:
	uv run ce3 doctor

# Cluster only (refuses without CUDA): tiny end-to-end rehearsal incl. a killed-and-resumed run.
dry-run:
	uv run ce3 dry-run

# --- Data, shared by both pipelines. CPU-heavy (the audit loads every benchmark): use a CPU pod. ---
data:
	uv run ce3 build-data $(call cfg,configs/data/mix_v1.yaml)
	uv run ce3 audit data/mix_v1
	uv run ce3 push-data data/mix_v1 --repo $(DATA_REPO)

# Synthetic pairs for families the zero-shot mix lacks (needs GPUs, after `make data`).
SEEDS := $(foreach l,python java javascript go php ruby,data/mix_v1/cornstack_$(l))
synth:
	uv run ce3 synth nl2code --seed-dirs $(SEEDS) --rows 300000 --out data/synth/nl2code.parquet
	uv run ce3 synth techqa --seed-dirs $(SEEDS) --rows 100000 --out data/synth/techqa.parquet
	uv run ce3 synth code_edit --seed-dirs $(SEEDS) --rows 100000 --out data/synth/code_edit.parquet
	uv run ce3 synth multiturn --seed-dirs $(SEEDS) --rows 100000 --out data/synth/multiturn.parquet
	uv run ce3 build-data $(call cfg,configs/data/synth_v1.yaml)
	uv run ce3 audit data/mix_v1
	uv run ce3 push-data data/mix_v1 --repo $(DATA_REPO)

pull-data:
	uv run ce3 pull-data data/mix_v1 data/mix_v1_gte_mined data/mix_v1_mined data/mix_v1_labelled --repo $(DATA_REPO)

GTE := Alibaba-NLP/gte-modernbert-base
STAGE ?= $(call cfg,configs/stages/150m_ft_minimal.yaml)

# --- Option A: fine-tune GTE-ModernBERT, no pretraining ---
finetune:
	uv run ce3 zero-shot data/mix_v1 --base-model $(GTE)
	uv run ce3 mine data/mix_v1 --model $(GTE) --no-query-prompts --max-rows-per-source 250000 --output-dir data/mix_v1_gte_mined
	uv run ce3 push-data data/mix_v1_gte_mined --repo $(DATA_REPO)
	uv run ce3 train $(STAGE)

mteb-ft:
	uv run ce3 mteb runs/150m/ft/final --name $(MODEL_REPO) --data-dirs data/mix_v1 --base-model $(GTE)

# Model files and a functional card only: no recipe, data mix or hyperparameters (see release.py).
release-ft:
	uv run ce3 release runs/150m/ft/final --repo $(MODEL_REPO) --base-model $(GTE) --data-dir data/mix_v1 \
		--results results/mteb --contamination-report data/mix_v1/contamination_report.json

# Optuna on the dev suite; writes tune/<study>/best_stage.yaml. Then: make finetune STAGE=tune/ce3-150m-ft-v1/best_stage.yaml
tune:
	uv run ce3 tune $(call cfg,configs/tune/150m_ft.yaml)

# --- Full 150M pipeline ---

s1:
	uv run ce3 train $(call cfg,configs/stages/150m_s1_pretrain.yaml)

s2:
	uv run ce3 mine data/mix_v1 --model runs/150m/s1/final --output-dir data/mix_v1_mined
	uv run ce3 push-data data/mix_v1_mined --repo $(DATA_REPO)
	uv run ce3 train $(call cfg,configs/stages/150m_s2_finetune.yaml)

label-check:
	uv run ce3 label-check data/mix_v1_mined --teacher $(TEACHER)

s3:
	uv run ce3 label data/mix_v1_mined --teacher $(TEACHER) --output-dir data/mix_v1_labelled --max-rows-per-source $(LABEL_ROWS)
	uv run ce3 push-data data/mix_v1_labelled --repo $(DATA_REPO)
	uv run ce3 train $(call cfg,configs/stages/150m_s3_distill.yaml)

s4:
	uv run ce3 train $(call cfg,configs/stages/150m_s4_longctx.yaml)

mteb:
	uv run ce3 mteb runs/150m/s4/final --name $(MODEL_REPO) --data-dirs data/mix_v1

release:
	uv run ce3 release runs/150m/s4/final --repo $(MODEL_REPO) --contamination-report data/mix_v1/contamination_report.json
