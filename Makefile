.PHONY: install lint test visualize demo

install:
	python -m pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest

visualize:
	python scripts/make_visualisations.py

demo:
	streamlit run app/streamlit_app.py
