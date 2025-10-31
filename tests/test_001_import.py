# ruff: noqa: D100 D103
import warnings


def test_import():
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        import pastastore
    pastastore.show_versions(optional=True)
