PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: setup data tables figures test models
setup:
	$(UV) venv --python 3.11 .venv
	$(UV) pip sync --python $(PYTHON) requirements.lock
	$(UV) pip install --python $(PYTHON) --no-deps --no-build-isolation -e .
data:
	$(PYTHON) scripts/download_data.py $(DATA_ARGS)
tables:
	$(PYTHON) scripts/make_paper_tables.py
figures:
	$(PYTHON) scripts/make_paper_figures.py
test:
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 $(PYTHON) -m pytest -q
models:
	$(PYTHON) scripts/convert_models.py --model-key qwen_4bit
