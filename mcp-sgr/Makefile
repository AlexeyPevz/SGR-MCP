.PHONY: venv install lint test test-unit run-http run-stdio format typecheck

PY?=python3
PIP?=$(PY) -m pip
VENV=.venv
ACT=source $(VENV)/bin/activate

venv:
	$(PY) -m venv $(VENV) || true
	$(ACT); $(PY) -m pip install --upgrade pip setuptools wheel

install: venv
	$(ACT); $(PIP) install -e '.[dev]'

lint:
	$(ACT); black --check src tests
	$(ACT); isort --check-only src tests
	$(ACT); flake8 src tests --max-line-length=100 --extend-ignore=E203,W503

format:
	$(ACT); black src tests
	$(ACT); isort src tests

typecheck:
	$(ACT); mypy src --ignore-missing-imports

test:
	$(ACT); pytest -v --cov=src --cov-report=term-missing:skip-covered

test-unit:
	$(ACT); pytest -q -m 'not integration' --disable-warnings --maxfail=1

run-http:
	$(ACT); $(PY) -m src.http_server

run-stdio:
	$(ACT); $(PY) -m src.server