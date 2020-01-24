import warnings


def test_import():
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        import pastas_projects as pp
    return
