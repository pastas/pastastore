"""Module containing dataframe styling functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex


def float_styler(val, norm, cmap=None):
    """Style float values in DataFrame.

    Parameters
    ----------
    val : float
        value in cell
    norm : matplotlib.colors.Normalize
        normalizer to map values to range(0, 1)
    cmap : colormap, optional
        colormap to use, by default None, which uses RdYlBu

    Returns
    -------
    str
        css value pairs for styling dataframe

    Usage
    -----
    Given some dataframe

    >>> df.style.map(float_styler, subset=["some column"], norm=norm, cmap=cmap)
    """
    if cmap is None:
        cmap = plt.get_cmap("RdYlBu")
    bg = cmap(norm(val))
    color = rgb2hex(bg)
    c = "White" if np.mean(bg[:3]) < 0.4 else "Black"
    return f"background-color: {color}; color: {c}"


def boolean_styler(b):
    """Style boolean values in DataFrame.

    Parameters
    ----------
    b : bool
        value in cell

    Returns
    -------
    str
        css value pairs for styling dataframe

    Usage
    -----
    Given some dataframe

    >>> df.style.map(boolean_styler, subset=["some column"])
    """
    if b is True or b is np.True_:
        return (
            f"background-color: {rgb2hex((231 / 255, 255 / 255, 239 / 255))}; "
            "color: darkgreen"
        )
    elif b is False or b is np.False_:
        return (
            f"background-color: {rgb2hex((255 / 255, 238 / 255, 238 / 255))}; "
            "color: darkred"
        )
    else:
        return "background-color: White; color: Black"


def boolean_row_styler(row, column):
    """Styler function to color rows based on the value in column.

    Parameters
    ----------
    row : pd.Series
        row in dataframe
    column : str
        column name to get boolean value for styling

    Returns
    -------
    str
        css for styling dataframe row

    Usage
    -----
    Given some dataframe

    >>> df.style.apply(boolean_row_styler, column="boolean_column", axis=1)
    """
    if row[column]:
        return (
            f"background-color: {rgb2hex((231 / 255, 255 / 255, 239 / 255))}; "
            "color: darkgreen",
        ) * row.size
    else:
        return (
            f"background-color: {rgb2hex((255 / 255, 238 / 255, 238 / 255))}; "
            "color: darkred",
        ) * row.size
