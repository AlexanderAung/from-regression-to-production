# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Structure

This is a machine learning project repository with two main components:
- `01-housing-market-prediciton/` - Main housing market prediction project
- `test-file/` - Secondary project for testing/experimentation

Both projects follow a similar structure with Python packaging via `uv` dependency manager.

## Development Commands

### Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment (from project directory)
cd 01-housing-market-prediciton
uv venv
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# Install dependencies
uv sync
```

### Running Code
```bash
# Run main application
python main.py

# Run specific modules (from project root)
python -m src.train_model
python -m src.data_prep  
python -m src.evaluate
```

### Jupyter Notebooks
```bash
# Start Jupyter Lab (jupyter is included in dependencies)
jupyter lab

# Or start classic notebook
jupyter notebook
```

### Testing
```bash
# Run tests (from project directory)
python -m pytest tests/
```

## Architecture Overview

### Housing Market Prediction Project (`01-housing-market-prediciton/`)

**Core Structure:**
- `src/` - Main source code modules
  - `data_prep.py` - Data preprocessing and cleaning
  - `train_model.py` - Model training logic  
  - `evaluate.py` - Model evaluation and metrics
- `data/` - Contains `Housing.csv` dataset
- `notebooks/` - Jupyter notebooks for EDA and experimentation
  - `EDA.ipynb` - Exploratory Data Analysis
- `models/` - Saved model artifacts
- `tests/` - Unit tests
- `logs/` - Application logs

**Dependencies:** 
- Core: `numpy`, `pandas`, `matplotlib` for data science
- `jupyter` for interactive development
- Uses Python 3.12+

### Test Project (`test-file/`)
- Simpler structure with basic Python project layout
- Uses `requests` library
- Same organizational pattern but minimal implementation

## Key Files

- `pyproject.toml` - Project configuration and dependencies (uses uv)
- `uv.lock` - Locked dependency versions
- `.python-version` - Python version specification
- `main.py` - Entry point for each project

## Development Workflow

1. Work primarily in the `01-housing-market-prediciton/` directory
2. Use notebooks for exploration and prototyping
3. Implement production code in `src/` modules
4. Test data processing with the `Housing.csv` dataset
5. Use virtual environments managed by `uv`