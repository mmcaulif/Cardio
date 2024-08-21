.PHONY: precommit_setup
precommit_setup:
	pre-commit --version
	pre-commit install

.PHONY: precommit_run
precommit_run:
	pre-commit run --all-files

.PHONY: install_cpu
install_cpu:
	pip install -e ".[dev,exp,cpu]"
	make precommit_setup

.PHONY: install_gpu
install_gpu:
	pip install -e ".[dev,exp,gpu]"
	make precommit_setup
