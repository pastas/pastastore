"""Typing definitions for PastasStore."""

from typing import Literal, Union

import pandas as pd

FrameOrSeriesUnion = Union[pd.DataFrame, pd.Series]

# Literal types for library names
TimeSeriesLibs = Literal["oseries", "stresses"]
PastasLibs = Literal["oseries", "stresses", "models"]
AllLibs = Literal["oseries", "stresses", "models", "oseries_models", "stresses_models"]
