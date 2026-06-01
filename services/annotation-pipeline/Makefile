.PHONY: install preprocess annotate visualize export status clean

install:
	uv sync

preprocess:
	uv run python run.py preprocess $(SCENE)

annotate:
	uv run python run.py annotate $(T1) $(T2)

visualize:
	uv run streamlit run visualizer.py

export:
	uv run python run.py export

status:
	uv run python run.py status

clean:
	rm -rf data/annotation/tiles data/annotation/masks_auto \
		data/annotation/masks_refined data/annotation/export
