"""Typing definitions for PastasStore."""

from typing import Literal

from pandas import DataFrame, Series

DataFrameOrSeries = DataFrame | Series

# Literal types for library names
TimeSeriesLibs = Literal["oseries", "stresses"]
PastasLibs = Literal["oseries", "stresses", "models"]
AllLibs = Literal["oseries", "stresses", "models", "oseries_models", "stresses_models"]
