.PHONY: precommit_setup
precommit_setup:
	pre-commit --version
	pre-commit install
	pre-commit install -t commit-msg

.PHONY: precommit_run
precommit_run:
	pre-commit run --all-files

.PHONY: install_cpu
install_cpu:
	pip install -e ".[dev,cpu]"
	make precommit_setup

.PHONY: install_gpu
install_gpu:
	pip install -e ".[dev,gpu]"
	make precommit_setup

.PHONY: profile
profile:
	python -m cProfile -s cumtime run_dummy.py > results.txt
