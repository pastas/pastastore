"""Module containing all the plotting methods for PastaStore.

Pastastore comes with a number helpful plotting methods to quickly
visualize time series or the locations of the time series contained in the
store. Plotting time series or data availability is available through the
`plots` attribute of the PastaStore object. Plotting locations of time series
or model statistics on maps is available through the `maps` attribute.
For example, if we have a :class:`pastastore.PastaStore` called `pstore`
linking to an existing database, the plot and map methods are available as
follows::

    pstore.plots.oseries()

    ax = pstore.maps.oseries()
    pstore.maps.add_background_map(ax)  # for adding a background map
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pastas as ps
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


class Maps:
    """Map Class for PastaStore.

    Allows plotting locations and model statistics on maps.

    Usage
    -----
    Example usage of the maps methods: :

    >> > ax = pstore.maps.oseries()  # plot oseries locations
    >> > pstore.maps.add_background_map(ax)  # add background map
    """

    def __init__(self, pstore):
        """Initialize Plots class for Pastastore.

        Parameters
        ----------
        pstore: pastastore.Pastastore
            Pastastore object
        """
        self.pstore = pstore

    def __repr__(self):
        """Return string representation of Maps submodule."""
        methods = "".join([
            f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")
        ])
        return "Mapping submodule, available methods:" + methods

    def stresses(
        self,
        names=None,
        kind=None,
        extent=None,
        labels=True,
        adjust=False,
        figsize=(10, 8),
        backgroundmap=False,
        label_kwargs=None,
        show_legend: bool = True,
        **kwargs,
    ):
        """Plot stresses locations on map.

        Parameters
        ----------
        names : list of str, optional
            list of names to plot
        kind: str, optional
            if passed, only plot stresses of a specific kind, default is None
            which plots all stresses.
        extent : list of float, optional
            plot only stresses within extent [xmin, xmax, ymin, ymax]
        labels: bool, optional
            label models, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figure size, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        label_kwargs: dict, optional
            dictionary with keyword arguments to pass to add_labels method
        show_legend : bool, optional
            add legend with each kind of stress and associated color, only possible
            if colors are not explicitly passed. Default is True.

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        names = self.pstore.conn.parse_names(names, "stresses")
        if extent is not None:
            names = self.pstore.within(extent, names=names, libname="stresses")
        df = self.pstore.stresses.loc[names]

        if kind is not None:
            if isinstance(kind, str):
                kind = [kind]
            mask = df["kind"].isin(kind)
            stresses = df[mask]
        else:
            stresses = df

        mask0 = (stresses["x"] != 0.0) | (stresses["y"] != 0.0)

        if "c" in kwargs:
            c = kwargs.pop("c")
            kind_to_color = None
        else:
            c = stresses.loc[mask0, "kind"]
            kind_to_color = {k: f"C{i}" for i, k in enumerate(c.unique())}
            c = c.apply(lambda k: kind_to_color[k])

        r = self.dataframe_scatter(stresses.loc[mask0], c=c, figsize=figsize, **kwargs)
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = r
        if labels:
            if label_kwargs is None:
                label_kwargs = {}
            self.add_labels(stresses, ax, adjust=adjust, **label_kwargs)

        if show_legend and kind_to_color is not None:
            for k, color in kind_to_color.items():
                ax.plot([], [], color=color, label=k, **kwargs, marker="o", ls="none")
            ax.legend(loc=(0, 1), frameon=False, ncol=5)

        if backgroundmap:
            self.add_background_map(ax)

        return ax

    def oseries(
        self,
        names=None,
        extent=None,
        labels=True,
        adjust=False,
        figsize=(10, 8),
        backgroundmap=False,
        label_kwargs=None,
        **kwargs,
    ):
        """Plot oseries locations on map.

        Parameters
        ----------
        names: list, optional
            oseries names, by default None which plots all oseries locations
        extent : list of float, optional
            plot only oseries within extent [xmin, xmax, ymin, ymax]
        labels: bool or str, optional
            label models, by default True, if passed as "grouped", only the first
            label for each x,y-location is shown.
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        figsize: tuple, optional
            figure size, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        label_kwargs: dict, optional
            dictionary with keyword arguments to pass to add_labels method

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        names = self.pstore.conn.parse_names(names, "oseries")
        if extent is not None:
            names = self.pstore.within(extent, names=names)
        oseries = self.pstore.oseries.loc[names]
        mask0 = (oseries["x"] != 0.0) | (oseries["y"] != 0.0)
        r = self.dataframe_scatter(oseries.loc[mask0], figsize=figsize, **kwargs)
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = r
        if labels:
            if label_kwargs is None:
                label_kwargs = {}
            if labels == "grouped":
                gr = oseries.sort_index().reset_index().groupby(["x", "y"])
                oseries = oseries.loc[gr["index"].first().tolist()]
            self.add_labels(oseries, ax, adjust=adjust, **label_kwargs)

        if backgroundmap:
            self.add_background_map(ax)

        return ax

    def models(
        self, labels=True, adjust=False, figsize=(10, 8), backgroundmap=False, **kwargs
    ):
        """Plot model locations on map.

        Parameters
        ----------
        labels: bool, optional
            label models, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figure size, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        model_oseries = [
            self.pstore.get_models(m, return_dict=True)["oseries"]["name"]
            for m in self.pstore.model_names
        ]

        models = self.pstore.oseries.loc[model_oseries]
        models.index = self.pstore.model_names

        # mask out 0.0 coordinates
        mask0 = (models["x"] != 0.0) | (models["y"] != 0.0)
        r = self.dataframe_scatter(models.loc[mask0], figsize=figsize, **kwargs)
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = r
        if labels:
            self.add_labels(models, ax, adjust=adjust)

        if backgroundmap:
            self.add_background_map(ax)

        return ax

    def series(
        self,
        series,
        name=None,
        labels=True,
        adjust=False,
        cmap="viridis",
        colorbar=True,
        legend=False,
        norm=None,
        vmin=None,
        vmax=None,
        ax=None,
        figsize=(10, 8),
        backgroundmap=False,
        **kwargs,
    ):
        """Plot the values of a series on a map.

        Parameters
        ----------
        series: str
            Pandas.Series with index that (partly) matches the pstore.oseries_names and
            values to plot on the map. The locations of the oseries are used to plot
            the values on the map.
        name: str, optional
            name of the series to use for labeling, by default None, which uses the
            name of the series itself or "value" if the series has no name.
        labels: bool, optional
            label models, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        cmap: str or colormap, optional
            (name of) the colormap, by default "viridis"
        colorbar : bool, optional
            show colorbar, by default True.
        legend : bool, optional
            show legend, only possible if the Series data type is int/int64, by default
            False.
        norm: norm, optional
            normalization for colorbar, by default None
        vmin: float, optional
            vmin for colorbar, by default None
        vmax: float, optional
            vmax for colorbar, by default None
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figure size, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        **kwargs: dict, optional
            additional keyword arguments to pass to dataframe_scatter method.

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        self.dataframe_scatter

        Notes
        -----
        The index of the `series` should match the names of the oseries in the store.
        Only the oseries with names matching the index of the `series` will be plotted.

        """
        # A few quick checks on the input series
        if not isinstance(series, pd.Series):
            raise ValueError("series should be a pandas Series.")
        if not series.index.isin(self.pstore.oseries_names).any():
            raise ValueError(
                "The index of the series should match the names of the oseries in the "
                "store."
            )

        # Ensure series has a name for labeling and colorbar, if not use "value"
        if name is not None:
            if not isinstance(name, str):
                raise ValueError("name should be a string.")
            series.rename(name, inplace=True)
        elif not series.name and name is None:
            name = "value"
            series.rename("value", inplace=True)

        df = self.pstore.oseries.join(series, how="left")

        return self.dataframe(
            df,
            column=series.name,
            labels=labels,
            adjust=adjust,
            cmap=cmap,
            colorbar=colorbar,
            legend=legend,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            ax=ax,
            backgroundmap=backgroundmap,
            **kwargs,
        )

    def dataframe(
        self,
        df,
        column,
        label=None,
        labels=True,
        adjust=False,
        cmap="viridis",
        colorbar=True,
        legend=False,
        norm=None,
        vmin=None,
        vmax=None,
        ax=None,
        figsize=(10, 8),
        backgroundmap=False,
        **kwargs,
    ):
        """Plot dataframe on a map.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe containing plotting information
        column: str
            column with values to plot
        label: bool, optional
            label points, by default True, Deprecated since Pastastore 1.13.0, use
            labels instead.
        labels: bool, optional
            label the points, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        cmap: str or colormap, optional
            (name of) the colormap, by default "viridis"
        colorbar : bool, optional
            show colorbar, only if column is provided, by default True.
        legend : bool, optional
            show legend, only possible if the column data type is int/int64, by default
            False.
        norm: norm, optional
            normalization for colorbar, by default None
        vmin: float, optional
            vmin for colorbar, by default None
        vmax: float, optional
            vmax for colorbar, by default None
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figuresize, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        progressbar: bool, optional
            show progressbar, default is True.

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        self.dataframe_scatter

        Notes
        -----
        The DataFrame `df` should contain columns "x" and "y" for the coordinates, and
        a column specified by `column` for the values to plot. The index of the
        DataFrame is used for labeling if `label` is True.

        """
        scatter_kwargs = {
            "cmap": cmap,
            "norm": norm,
            "vmin": vmin,
            "vmax": vmax,
            "edgecolors": "w",
            "linewidths": 0.7,
        }
        scatter_kwargs.update(kwargs)

        if label is not None:
            DeprecationWarning(
                "label argument is deprecated since Pastastore 1.13.0. "
                "Please use labels instead."
            )
            labels = label

        _ = self.dataframe_scatter(
            df,
            column=column,
            figsize=figsize,
            ax=ax,
            colorbar=colorbar,
            legend=legend,
            **scatter_kwargs,
        )

        if labels:
            if "index" in df:
                df.set_index("index", inplace=True)
            self.add_labels(df, ax, adjust=adjust)

        if backgroundmap:
            self.add_background_map(ax)

        return ax

    def modelstat(
        self,
        statistic,
        modelnames=None,
        label=True,
        adjust=False,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        figsize=(10, 8),
        backgroundmap=False,
        progressbar=True,
        **kwargs,
    ):
        """Plot model statistic on map.

        Parameters
        ----------
        statistic: str
            name of the statistic, e.g. "evp" or "aic"
        modelnames : list of str, optional
            list of modelnames to include
        label: bool, optional
            label points, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        cmap: str or colormap, optional
            (name of) the colormap, by default "viridis"
        norm: norm, optional
            normalization for colorbar, by default None
        vmin: float, optional
            vmin for colorbar, by default None
        vmax: float, optional
            vmax for colorbar, by default None
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figuresize, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        progressbar: bool, optional
            show progressbar, default is True.

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        statsdf = self.pstore.get_statistics(
            [statistic], modelnames=modelnames, progressbar=progressbar
        ).to_frame()

        statsdf["oseries"] = [
            self.pstore.get_models(m, return_dict=True)["oseries"]["name"]
            for m in statsdf.index
        ]
        statsdf = statsdf.reset_index().set_index("oseries")
        df = statsdf.join(self.pstore.oseries, how="left")

        return self.dataframe(
            df,
            column=statistic,
            label=label,
            adjust=adjust,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            backgroundmap=backgroundmap,
            **kwargs,
        )

    def modelparam(
        self,
        parameter,
        param_value="optimal",
        modelnames=None,
        label=True,
        adjust=False,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        figsize=(10, 8),
        backgroundmap=False,
        progressbar=True,
        **kwargs,
    ):
        """Plot model parameter value on map.

        Parameters
        ----------
        parameter: str
            name of the parameter, e.g. "rech_A" or "river_a"
        param_value: str, optional
            which parameter value to plot, by default "optimal", other options
            are "initial", "pmin", "pmax"
        modelnames : list of str, optional
            list of modelnames to include
        label: bool, optional
            label points, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        cmap: str or colormap, optional
            (name of) the colormap, by default "viridis"
        norm: norm, optional
            normalization for colorbar, by default None
        vmin: float, optional
            vmin for colorbar, by default None
        vmax: float, optional
            vmax for colorbar, by default None
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figuresize, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        progressbar: bool, optional
            show progressbar, default is True

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        paramdf = self.pstore.get_parameters(
            [parameter],
            param_value=param_value,
            modelnames=modelnames,
            progressbar=progressbar,
            ignore_errors=True,
        ).to_frame()

        paramdf["oseries"] = [
            self.pstore.get_models(m, return_dict=True)["oseries"]["name"]
            for m in paramdf.index
        ]
        paramdf = paramdf.reset_index().set_index("oseries")
        df = paramdf.join(self.pstore.oseries, how="left")

        return self.dataframe(
            df,
            column=parameter,
            label=label,
            adjust=adjust,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            backgroundmap=backgroundmap,
            **kwargs,
        )

    def signature(
        self,
        signature,
        names=None,
        label=True,
        adjust=False,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        figsize=(10, 8),
        backgroundmap=False,
        progressbar=True,
        **kwargs,
    ):
        """Plot signature value on map.

        Parameters
        ----------
        signature: str
            name of the signature, e.g. "mean_annual_maximum" or "duration_curve_slope"
        names : list of str, optional
            list of observation well names to include
        label: bool, optional
            label points, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        cmap: str or colormap, optional
            (name of) the colormap, by default "viridis"
        norm: norm, optional
            normalization for colorbar, by default None
        vmin: float, optional
            vmin for colorbar, by default None
        vmax: float, optional
            vmax for colorbar, by default None
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figuresize, by default(10, 8)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.
        progressbar: bool, optional
            show progressbar, default is True

        Returns
        -------
        ax: matplotlib.Axes
            axes object

        See Also
        --------
        self.add_background_map
        """
        signature_df = self.pstore.get_signatures(
            names=names,
            signatures=[signature],
            progressbar=progressbar,
            ignore_errors=True,
        ).transpose()
        df = signature_df.join(self.pstore.oseries, how="left")

        return self.dataframe(
            df,
            column=signature,
            label=label,
            adjust=adjust,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            backgroundmap=backgroundmap,
            **kwargs,
        )

    def _plotmap_dataframe(self, *args, **kwargs):
        """Deprecated, use dataframe method."""  # noqa: D401
        warnings.warn(
            "maps._plotmap_dataframe is deprecated, use maps.dataframe_scatter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dataframe_scatter(*args, **kwargs)

    @staticmethod
    def dataframe_scatter(
        df,
        x="x",
        y="y",
        label=True,
        column=None,
        colorbar=True,
        legend=False,
        ax=None,
        figsize=(10, 8),
        **kwargs,
    ):
        """Plot dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing coordinates and data to plot, with index providing
            names for each location.
        x : str, optional
            name of the column with x - coordinate data, by default "x".
        y : str, optional
            name of the column with y - coordinate data, by default "y".
        column : str, optional
            name of the column containing data used for determining the color of each
            point, by default None (all one color).
                label: bool, optional
            label points, by default True
        adjust: bool, optional
            automated smart label placement using adjustText, by default False
        colorbar : bool, optional
            show colorbar, only if column is provided, by default True.
        legend : bool, optional
            show legend, only possible if the column data type is int/int64, by default
            False.
        progressbar: bool, optional
            show progressbar, default is True.
        ax : matplotlib Axes
            axes handle to plot dataframe, optional, default is None which creates a
            new figure.
        figsize : tuple, optional
            figure size, by default(10, 8)
        **kwargs :
            dictionary containing keyword arguments for ax.scatter, by default None.

        Returns
        -------
        ax : matplotlib.Axes
            axes object, returned if ax is None
        sc : scatter handle
            scatter plot handle, returned if ax is not None

        """
        if ax is None:
            return_scatter = False
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        else:
            return_scatter = True
            fig = ax.figure

        # set default size and marker if not passed
        if kwargs:
            s = kwargs.pop("s", 70)
            marker = kwargs.pop("marker", "o")
        else:
            s = 70
            marker = "o"
            kwargs = {}

        # if column is passed for coloring pts
        if column:
            c = df.loc[:, column]
            if "cmap" not in kwargs:
                kwargs["cmap"] = "viridis"
        else:
            c = kwargs.pop("c", None)

        sc = ax.scatter(df.loc[:, x], df.loc[:, y], marker=marker, s=s, c=c, **kwargs)
        # add colorbar
        if column and colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = fig.colorbar(sc, ax=ax, cax=cax)
            cbar.set_label(column)

        # add legend if column is categorical (int) and legend is True
        if legend and column and pd.api.types.is_integer_dtype(df[column]):
            uniques = df[column].dropna().unique()
            for u in uniques:
                ax.scatter([], [], c=sc.cmap(sc.norm(u)), label=str(u), **kwargs)
            ax.legend(title=column, loc="best")

        # set axes properties
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        for label in ax.get_yticklabels():
            label.set_rotation(90)
            label.set_verticalalignment("center")

        fig.tight_layout()
        if return_scatter:
            return sc
        else:
            return ax

    def model(
        self,
        ml,
        label=True,
        metadata_source="model",
        offset=0.0,
        ax=None,
        figsize=(10, 10),
        backgroundmap=False,
    ):
        """Plot oseries and stresses from one model on a map.

        Parameters
        ----------
        ml: str or pastas.Model
            pastas model or name of pastas model to plot on map
        label: bool, optional, default is True
            add labels to points on map
        metadata_source: str, optional
            one of "model" or "store", pick whether to obtain metadata from model
            Timeseries or from metadata in pastastore, default is "model"
        offset : float, optional
            add offset to current extent of model time series, useful
            for zooming out around models
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize: tuple, optional
            figsize, default is (10, 10)
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.

        Returns
        -------
        ax: axes object
            axis handle of the resulting figure

        See Also
        --------
        self.add_background_map
        """
        if isinstance(ml, str):
            ml = self.pstore.get_models(ml)
        elif not isinstance(ml, ps.Model):
            raise TypeError("Pass model name as string or pastas.Model!")

        stresses = pd.DataFrame(columns=["x", "y", "stressmodel", "color"])
        count = 0
        for name, sm in ml.stressmodels.items():
            for istress in sm.stress:
                if metadata_source == "model":
                    xi = istress.metadata["x"]
                    yi = istress.metadata["y"]
                elif metadata_source == "store":
                    imeta = self.pstore.get_metadata(
                        "stresses", istress.name, as_frame=False
                    )
                    xi = imeta.pop("x", np.nan)
                    yi = imeta.pop("y", np.nan)
                else:
                    raise ValueError(
                        "metadata_source must be either 'model' or 'store'!"
                    )
                if np.isnan(xi) or np.isnan(yi):
                    logger.warning("No x,y-data for %s!", istress.name)
                    continue
                if xi == 0.0 or yi == 0.0:
                    logger.warning(
                        "x,y-data is 0.0 for %s, not plotting!", istress.name
                    )
                    continue

                stresses.loc[istress.name, :] = (xi, yi, name, f"C{count % 10}")
            count += 1

        # create figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # add oseries
        osize = 50
        oserieslabel = ml.oseries.name

        if metadata_source == "model":
            xm = float(ml.oseries.metadata["x"])
            ym = float(ml.oseries.metadata["y"])
        elif metadata_source == "store":
            ometa = self.pstore.get_metadata("oseries", ml.oseries.name, as_frame=False)
            xm = float(ometa.pop("x", np.nan))
            ym = float(ometa.pop("y", np.nan))
        else:
            raise ValueError("metadata_source must be either 'model' or 'store'!")

        po = ax.scatter(xm, ym, s=osize, marker="o", label=oserieslabel, color="k")
        legend_list = [po]

        # add stresses
        ax.scatter(
            stresses["x"],
            stresses["y"],
            s=50,
            c=stresses.color,
            marker="o",
            edgecolors="k",
            linewidths=0.75,
        )

        # label oseries
        if label:
            stroke = [patheffects.withStroke(linewidth=3, foreground="w")]
            txt = ax.annotate(
                text=oserieslabel,
                xy=(xm, ym),
                fontsize=8,
                textcoords="offset points",
                xytext=(10, 10),
            )
            txt.set_path_effects(stroke)

        # get legend entries for stressmodels
        uniques = stresses.loc[:, ["stressmodel", "color"]].drop_duplicates(
            keep="first"
        )
        for _, row in uniques.iterrows():
            (h,) = ax.plot(
                [],
                [],
                marker="o",
                label=row.stressmodel,
                ls="",
                mec="k",
                ms=10,
                color=row.color,
            )
            legend_list.append(h)

        # add legend
        ax.legend(legend_list, [i.get_label() for i in legend_list], loc="best")

        # set axes properties
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        for label in ax.get_yticklabels():
            label.set_rotation(90)
            label.set_verticalalignment("center")

        if offset > 0.0:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(xmin - offset, xmax + offset)
            ax.set_ylim(ymin - offset, ymax + offset)

        # label stresses
        if label:
            for name, row in stresses.iterrows():
                namestr = str(name)
                namestr += f"\n({row.stressmodel})"
                txt = ax.annotate(
                    text=namestr,
                    xy=(row.x, row.y),
                    fontsize=8,
                    textcoords="offset points",
                    xytext=(10, 10),
                )
                txt.set_path_effects(stroke)

        if backgroundmap:
            self.add_background_map(ax)

        fig.tight_layout()

        return ax

    def stresslinks(
        self,
        kinds=None,
        model_names=None,
        color_lines=False,
        alpha=0.4,
        ax=None,
        figsize=(10, 8),
        legend=True,
        labels=False,
        adjust=False,
        backgroundmap=False,
    ):
        """Create a map linking models with their stresses.

        Parameters
        ----------
        kinds: list, optional
            kinds of stresses to plot, defaults to None, which selects
            all kinds.
        model_names: list, optional
            list of model names to plot, substrings of model names
            are also accepted, defaults to None, which selects all
            models.
        color_lines: bool, optional
            if True, connecting lines have the same colors as the stresses,
            defaults to False, which uses a black line.
        alpha: float, optional
            alpha value for the connecting lines, defaults to 0.4.
        ax : matplotlib.Axes, optional
            axes handle, if not provided a new figure is created.
        figsize : tuple, optional
            figure size, by default (10, 8)
        legend: bool, optional
            create a legend for all unique kinds, defaults to True.
        labels: bool, optional
            add labels for stresses and oseries, defaults to False.
        adjust: bool, optional
            automated smart label placement using adjustText, by
            default False
        backgroundmap: bool, optional
            if True, add background map (default CRS is EPSG:28992) with default tiles
            by OpenStreetMap.Mapnik. Default option is False.

        Returns
        -------
        ax: axes object
            axis handle of the resulting figure

        See Also
        --------
        self.add_background_map
        """
        if model_names:
            m_idx = self.pstore.search(libname="models", s=model_names)
        else:
            m_idx = self.pstore.model_names
        struct = self.pstore.get_model_time_series_names(progressbar=False).loc[m_idx]

        oseries = self.pstore.oseries
        stresses = self.pstore.stresses
        skind = stresses.kind.unique()
        stused = np.array([])
        if kinds is None:
            kinds = skind

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        segments = []
        segment_colors = []
        scatter_colors = []
        ax.scatter(
            oseries.loc[struct["oseries"], "x"],
            oseries.loc[struct["oseries"], "y"],
            color="C0",
        )
        for m in struct.index:
            os = oseries.loc[struct.loc[m, "oseries"]]
            mstresses = struct.loc[m].drop("oseries").dropna().index
            st = stresses.loc[mstresses]
            for s in mstresses:
                if np.isin(st.loc[s, "kind"], kinds):
                    (c,) = np.where(skind == st.loc[s, "kind"])
                    if color_lines:
                        color = f"C{c[0] + 1}"
                    else:
                        color = "k"
                    segments.append([
                        [os["x"], os["y"]],
                        [st.loc[s, "x"], st.loc[s, "y"]],
                    ])
                    segment_colors.append(color)
                    scatter_colors.append(f"C{c[0] + 1}")

                    stused = np.append(stused, s)

        if labels:
            self.add_labels(oseries.loc[struct["oseries"].unique()], ax, adjust=adjust)
            self.add_labels(stresses.loc[np.unique(stused)], ax, adjust=adjust)

        ax.scatter(
            [x[1][0] for x in segments],
            [y[1][1] for y in segments],
            color=scatter_colors,
        )
        ax.add_collection(
            LineCollection(segments, colors=segment_colors, linewidths=0.5, alpha=alpha)
        )

        if legend:
            legend_elements = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    markerfacecolor="C0",
                    label="oseries",
                    markersize=10,
                )
            ]
            for kind in kinds:
                (c,) = np.where(skind == kind)
                legend_elements.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        color="w",
                        markerfacecolor=f"C{c[0] + 1}",
                        label=kind,
                        markersize=10,
                    )
                )
            ax.legend(handles=legend_elements)

        if backgroundmap:
            self.add_background_map(ax)

        return ax

    @staticmethod
    def _list_contextily_providers():
        """List contextily providers.

        Taken from contextily notebooks.

        Returns
        -------
        providers : dict
            dictionary containing all providers. See keys for names
            that can be passed as map_provider arguments.
        """
        import contextily as ctx

        providers = {}

        def get_providers(provider):
            if "url" in provider:
                providers[provider["name"]] = provider
            else:
                for prov in provider.values():
                    get_providers(prov)

        get_providers(ctx.providers)
        return providers

    @staticmethod
    def add_background_map(
        ax, proj="epsg:28992", map_provider="OpenStreetMap.Mapnik", **kwargs
    ):
        """Add background map to axes using contextily.

        Parameters
        ----------
        ax: matplotlib.Axes
            axes to add background map to
        map_provider: str, optional
            name of map provider, see `contextily.providers` for options.
            Default is 'OpenStreetMap.Mapnik'
        proj: pyproj.Proj or str, optional
            projection for background map, default is 'epsg:28992'
            (RD Amersfoort, a projection for the Netherlands)
        """
        import contextily as ctx

        if isinstance(proj, str):
            import pyproj

            proj = pyproj.Proj(proj)

        providers = Maps._list_contextily_providers()
        ctx.add_basemap(ax, source=providers[map_provider], crs=proj.srs, **kwargs)

    @staticmethod
    def add_labels(
        df, ax, adjust=False, objects=None, adjust_text_kwargs=None, **kwargs
    ):
        """Add labels to points on plot.

        Uses dataframe index to label points.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing x, y - data. Index is used as label
        ax : matplotlib.Axes
            axes object to label points on
        adjust : bool
            automated smart label placement using adjustText
        objects : list of matplotlib objects
            use to avoid labels overlapping markers
        adjust_text_kwargs
            keyword arguments to adjust_text function, only used if adjust=True
        **kwargs
            keyword arguments to ax.annotate or ax.text
        """
        stroke = [patheffects.withStroke(linewidth=3, foreground="w")]
        fontsize = kwargs.pop("fontsize", 10)

        if adjust:
            from adjustText import adjust_text

            texts = []
            for name, row in df.iterrows():
                texts.append(
                    ax.text(
                        row["x"],
                        row["y"],
                        name,
                        fontsize=fontsize,
                        **{"path_effects": stroke},
                        **kwargs,
                    )
                )
            if adjust_text_kwargs is None:
                adjust_text_kwargs = {}
            adjust_text(
                texts,
                objects=objects,
                force_text=(0.05, 0.10),
                **{
                    "arrowprops": {
                        "arrowstyle": "-",
                        "color": "k",
                        "alpha": 0.5,
                    }
                },
                **adjust_text_kwargs,
            )

        else:
            textcoords = kwargs.pop("textcoords", "offset points")
            xytext = kwargs.pop("xytext", (10, 10))

            for name, row in df.iterrows():
                namestr = str(name)
                ax.annotate(
                    text=namestr,
                    xy=(row["x"], row["y"]),
                    fontsize=fontsize,
                    textcoords=textcoords,
                    xytext=xytext,
                    **{"path_effects": stroke},
                    **kwargs,
                )

