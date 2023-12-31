.PHONY: local_setup preprocess train model_viz evaluate

run_model:
	python scripts_thesis/main.py

local_setup:
	python -c "from scripts_thesis.main import local_setup; local_setup()"

preprocess:
	python -c "from scripts_thesis.main import ModelFlow; ml = ModelFlow(); ml.preprocess()"

optimise:
	python -c "from scripts_thesis.main import ModelFlow; ml = ModelFlow(); ml.optimise(classifiers='$(classifier)', n_iter=$(n_iter))"

train:
	python -c "from scripts_thesis.main import ModelFlow; ml = ModelFlow(); ml.train(n_splits=$(n_splits), classifiers='$(classifier)')"

model_viz:
	python -c "from scripts_thesis.main import ModelFlow; ml = ModelFlow(); ml.model_viz()"

evaluate:
	python -c "from scripts_thesis.main import ModelFlow; ml = ModelFlow(); ml.evaluate()"
