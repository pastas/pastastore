"""Typing definitions for PastasStore."""

from typing import Union

import pandas as pd

FrameOrSeriesUnion = Union[pd.DataFrame, pd.Series]
