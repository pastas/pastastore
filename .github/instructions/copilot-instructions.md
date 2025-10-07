# Copilot Instructions for `pastastore`

## Project Overview

- **Purpose:** `pastastore` provides a database-like interface for storing, managing, and analyzing [Pastas](https://pastas.readthedocs.io/latest/) time series and models.
- **Core Abstractions:**
  - `PastaStore`: Main API for users, manages time series and models.
  - **Connectors**: Pluggable backends for storage (`PasConnector`, `ArcticDBConnector`, `DictConnector`).
  - **Extensions**: Optional features in `pastastore/extensions/` (e.g., HPD, accessor).
  - **Plotting**: Rich plotting and mapping via `pastastore/plotting.py` and `styling.py`.

## Key Files & Directories

- `pastastore/`: Main package code.
  - `store.py`: Main `PastaStore` class.
  - `connectors.py`, `base.py`: Connector logic and base classes.
  - `util.py`: Utilities and custom exceptions.
  - `plotting.py`, `styling.py`: Visualization and DataFrame styling.
  - `yaml_interface.py`: YAML import/export for models.
  - `datasets.py`: Example/test data loading.
  - `extensions/`: Optional add-ons.
- `tests/`: Pytest-based test suite, with data in `tests/data/`.
- `docs/`: Sphinx documentation and example notebooks.

## Developer Workflows

- **Install (dev):** `pip install -e .`
- **Run tests:** `pytest` (all tests in `tests/`)
- **Lint/Format:** Uses `ruff` for linting and formatting (`ruff check .`, `ruff format .`)
- **Docs:** Built with Sphinx (`docs/`), example notebooks in `docs/examples/`
- **CI:** GitHub Actions in `.github/workflows/ci.yml` (matrix for Python and Pastas versions)

## Project-Specific Conventions

- **Path Handling:** Use `pathlib.Path` for all filesystem operations (avoid `os.path`).
- **Data Storage:** All time series and models are stored via connectors; never access files directly in user code.
- **Metadata:** Metadata is always a dictionary, stored alongside time series.
- **Error Handling:** Custom exceptions in `util.py` (e.g., `ItemInLibraryException`, `SeriesUsedByModel`).
- **Logging:** Use the `logging` module, not `print`, for all non-test output.
- **Type Hints:** Used throughout; maintain and extend as needed.
- **Testing:** Use fixtures in `tests/conftest.py` for reusable test setup.

## Integration & Extensibility

- **Connectors:** To add a new backend, subclass `BaseConnector` and implement all abstract methods.
- **Extensions:** Place optional features in `pastastore/extensions/` and import conditionally in `__init__.py`.
- **YAML/JSON:** Use `yaml_interface.py` for YAML, and built-in JSON for serialization.

## Examples

- See `readme.md` for typical usage patterns.
- Example: Creating a store and adding a time series:
  ```python
  import pastastore as pst
  conn = pst.PasConnector(name="mydb", path=".")
  pstore = pst.PastaStore(conn)
  pstore.add_oseries(df, "name", metadata={"x": 1, "y": 2})
  ```

---

**Feedback Request:**  
Please review these instructions. Let me know if any project-specific patterns, workflows, or integration details are missing or unclear, so I can further refine this guide for AI agents.
