.PHONY: install lint test visualize demo

install:
	python -m pip install -r requirements.txt

install-ml:
	python -m pip install -r requirements-ml.txt

install-dev:
	python -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest

visualize:
	python scripts/make_visualisations.py

demo:
	streamlit run app/streamlit_app.py
