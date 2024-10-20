.PHONY: precommit_setup
precommit_setup:
	pre-commit --version
	pre-commit install
	pre-commit install -t commit-msg

.PHONY: precommit
precommit_run:
	pre-commit run --all-files

.PHONY: setup
setup:
	poetry install --with dev
	make precommit_setup
