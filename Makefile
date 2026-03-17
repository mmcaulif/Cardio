.PHONY: precommit_setup
precommit_setup:
	uv run pre-commit --version
	uv run pre-commit install
	uv run pre-commit install -t commit-msg

.PHONY: precommit
precommit:
	uv run pre-commit run --all-files

.PHONY: setup
setup:
	uv sync --group dev
	make precommit_setup
