import warnings

from tqdm import TqdmWarning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=TqdmWarning,
        module="tqdm",
    )
    from tqdm.auto import tqdm  # noqa: F401
    from tqdm.contrib.concurrent import process_map  # noqa: F401
