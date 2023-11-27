import warnings


def test_import():
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        import pastastore  # noqa: F401
