.PHONY: precommit_setup
precommit_setup:
	pre-commit --version
	pre-commit install

.PHONY: precommit_run
precommit_run:
	pre-commit run --all-files
