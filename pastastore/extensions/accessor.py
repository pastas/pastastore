# ruff: noqa: D100
from pastas.extensions.accessor import _register_accessor


def register_pastastore_accessor(name: str):
    """Register an extension in the PastaStore class.

    Parameters
    ----------
    name : str
        name of the extension to register
    """
    from pastastore.store import PastaStore

    return _register_accessor(name, PastaStore)
