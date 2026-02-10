import warnings

from tqdm import TqdmWarning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=TqdmWarning,
        module="tqdm",
    )
