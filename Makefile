.PHONY: setup_precommmit
setup_precommmit:
	pre-commit --version
	pre-commit install

.PHONY: run_precommit
run_precommmit:
	pre-commit run --all-files
