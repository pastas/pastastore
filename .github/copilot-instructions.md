# Copilot Instructions for pastastore

## Project Overview

- **Purpose:** `pastastore` provides a database-like interface for storing, managing, and analyzing [Pastas](https://pastas.readthedocs.io/latest/) time series and models.
- **Core Abstractions:**
  - `PastaStore`: Main API for users, manages time series and models.
  - **Connectors**: Pluggable backends for storage (`PasConnector`, `DictConnector`, `ArcticDBConnector`).
  - **Extensions**: Optional features in `pastastore/extensions` (e.g., HPD, accessor).
  - **Plotting**: Rich plotting and mapping via `plotting.py` and `store.py`.

## Key Files & Directories

- `pastastore/`: Main package code.
  - `store.py`: Main `PastaStore` class.
  - `base.py`, `connectors.py`: Connector logic and base classes.
  - `util.py`: Utilities and custom exceptions.
  - `plotting.py`, `styling.py`: Visualization and DataFrame styling.
  - `yaml_interface.py`: YAML import/export for models.
  - `datasets.py`: Example/test data loading.
  - `extensions/`: Optional add-ons.
- `tests/`: Pytest-based test suite, with data in `tests/data`.
- `docs/`: Sphinx documentation and example notebooks.

## Developer Workflows

- **Install (dev):** `pip install -e .`
- **Run tests:** `pytest` (all tests in `tests/`)
- **Lint/Format:** Uses `ruff` for linting and formatting (`pyproject.toml`, `ruff.toml`)
- **Docs:** Built with Sphinx (`docs/conf.py`), example notebooks in `examples/`
- **CI:** GitHub Actions in `.github/workflows/` (matrix for Python and Pastas versions)

## Project-Specific Conventions

- **Path Handling:** Use `pathlib.Path` for all filesystem operations (avoid `os.path`).
- **Data Storage:** All time series and models are stored via connectors; never access files directly in user code.
- **Metadata:** Metadata is always a dictionary, stored alongside time series.
- **Error Handling:** Custom exceptions in `util.py` (e.g., `PastastoreException`, `ModelNotFoundError`).
- **Logging:** Use the `logging` module, not `print`, for all non-test output.
- **Type Hints:** Used throughout; maintain and extend as needed.
- **Testing:** Use fixtures in `tests/conftest.py` for reusable test setup.

## Integration & Extensibility

- **Connectors:** To add a new backend, subclass `BaseConnector` and implement all abstract methods.
- **Extensions:** Place optional features in `extensions/` and import conditionally in `__init__.py`.
- **YAML/JSON:** Use `pyyaml` for YAML, and built-in JSON for serialization.

## Examples

- See `datasets.py` for typical usage patterns.
- Example: Creating a store and adding a time series:
  ```python
  import pastastore as pst
  conn = pst.PasConnector(name="mydb", path=".")
  pstore = pst.PastaStore(conn)
  pstore.add_oseries(df, "name", metadata={"x": 1, "y": 2})
  ```
