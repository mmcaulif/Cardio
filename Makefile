.PHONY: precommit
precommit_run:
	pre-commit run --all-files

.ONESHELL: setup
setup:
	poetry install --with dev
	make precommit_setup
