.PHONY: install test serve
install:
	pip install -r requirements.txt
test:
	python -m pytest -q
serve:
	uvicorn app.main:app --reload
