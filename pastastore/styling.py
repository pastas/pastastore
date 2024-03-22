import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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

    >>> df.map(float_styler, subset=["some column"], norm=norm, cmap=cmap)

    """
    if cmap is None:
        cmap = plt.get_cmap("RdYlBu")
    bg = cmap(norm(val))
    color = mpl.colors.rgb2hex(bg)
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

    >>> df.map(boolean_styler, subset=["some column"])
    """
    if b:
        return (
            f"background-color: {mpl.colors.rgb2hex((231/255, 255/255, 239/255))}; "
            "color: darkgreen"
        )
    else:
        return (
            f"background-color: {mpl.colors.rgb2hex((255/255, 238/255, 238/255))}; "
            "color: darkred"
        )
