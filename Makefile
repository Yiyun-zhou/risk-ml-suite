PY=python
data:
	$(PY) -m src.data_prep --dataset synthetic --seed 42
eda:
	$(PY) -m src.eda
train:
	$(PY) -m src.train --models lr xgb lgbm --cv 5
eval:
	$(PY) -m src.eval
drift:
	$(PY) -m src.drift --simulate_incident 1
app:
	streamlit run app/streamlit_app.py
all: data eda train eval drift
