from pastas.extensions.accessor import _register_accessor


def register_pastastore_accessor(name: str):
    from pastastore.store import PastaStore

    return _register_accessor(name, PastaStore)