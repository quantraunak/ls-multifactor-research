install:
	pip install -r requirements.txt

test:
	pytest -q

run:
	python project/run_backtest.py --config project/configs/default.yaml

