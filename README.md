# APS360 Project - License Plate Recognition

## Install Libraries

I use `uv` as the package manager, to install all libraries

```bash
uv sync
```

## Steps to Reproduce the Model

1. Setup config file

```bash
cp src/aps360_lpr/data_generation/config.yaml.example src/aps360_lpr/data_generation/config.yaml
```

2. All in one command

```bash
uv run main.py
```

This will generate all data and train the model.

3. Qualitative Test

```bash
uv run predict.py "path/to/my_random_image.jpg"
```

This will save the result into a jpg file.