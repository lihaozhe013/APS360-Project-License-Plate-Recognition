
default: all

train-recognition:
	cd src/recognition_model && uv run recognition_train.py

train-bbox:
	cd src/bbox_model && uv run train_bbox.py

infer:
	uv run inference_pipeline.py

all: 
	make train-recognition && make train-bbox && make infer
