"""Module containing all the plotting methods for PastaStore.

Pastastore comes with a number helpful plotting methods to quickly
visualize time series contained in the store. Plotting time series or data availability
is available through the `plots` attribute of the PastaStore object. For example, if we
have a :class:`pastastore.PastaStore` called `pstore` linking to an existing database,
the plot methods are available as follows::

    pstore.plots.oseries()

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pastas as ps
from matplotlib.colors import BoundaryNorm, LogNorm

logger = logging.getLogger(__name__)


class Plots:
    """Plot class for Pastastore.

    Allows plotting of time series and data availability.
    """

    def __init__(self, pstore):
        """Initialize Plots class for Pastastore.

        Parameters
        ----------
        pstore : pastastore.Pastastore
            Pastastore object
        """
        self.pstore = pstore

    def __repr__(self):
        """Return string representation of Plots submodule."""
        methods = "".join(
            [f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")]
        )
        return "Plotting submodule, available methods:" + methods

    def _timeseries(
        self,
        libname,
        names=None,
        tmin=None,
        tmax=None,
        ax=None,
        split=False,
        figsize=(10, 5),
        progressbar=True,
        show_legend=True,
        labelfunc=None,
        legend_kwargs=None,
        **kwargs,
    ):
        """Plot time series from pastastore (internal method).

        Parameters
        ----------
        libname : str
            name of the library to obtain time series from (oseries
            or stresses)
        names : list[str], optional
            list of time series names to plot, by default None
        tmin : str or pd.Timestamp, optional
            minimum time to plot, by default None, which uses the minimum
            time of the time series
        tmax : str or pd.Timestamp, optional
            maximum time to plot, by default None, which uses the maximum
            time of the time series
        ax : matplotlib.Axes, optional
            pass axes object to plot on existing axes, by default None,
            which creates a new figure
        split : bool, optional
            create a separate subplot for each time series, by default False.
            A maximum of 20 time series is supported when split=True.
        figsize : tuple, optional
            figure size, by default (10, 5)
        progressbar : bool, optional
            show progressbar when loading time series from store,
            by default True
        show_legend : bool, optional
            show legend, default is True.
        labelfunc : callable, optional
            function to create custom labels, function should take name of time series
            as input
        legend_kwargs : dict, optional
            additional arguments to pass to legend

        Returns
        -------
        ax : matplotlib.Axes
            axes handle

        Raises
        ------
        ValueError
            split=True is only supported if there are less than 20 time series
            to plot.
        """
        names = self.pstore.conn.parse_names(names, libname)

        if len(names) > 20 and split:
            raise ValueError(
                "More than 20 time series leads to too many subplots, set split=False."
            )

        if ax is None:
            if split:
                _, axes = plt.subplots(len(names), 1, sharex=True, figsize=figsize)
            else:
                _, axes = plt.subplots(1, 1, figsize=figsize)
        else:
            axes = ax

        tsdict = self.pstore.conn._get_series(  # noqa: SLF001
            libname, names, progressbar=progressbar, squeeze=False
        )
        for i, (n, ts) in enumerate(tsdict.items()):
            if split and ax is None:
                iax = axes[i]
            elif ax is None:
                iax = axes
            else:
                iax = ax
            if labelfunc is not None:
                n = labelfunc(n)
            if tmin is not None:
                ts = ts.loc[tmin:]
            if tmax is not None:
                ts = ts.loc[:tmax]
            iax.plot(ts.index, ts.squeeze(), label=n, **kwargs)

            if split and show_legend:
                iax.legend(loc="best", fontsize="x-small")

        if not split and show_legend:
            if legend_kwargs is None:
                legend_kwargs = {}
            ncol = legend_kwargs.pop("ncol", 7)
            fontsize = legend_kwargs.pop("fontsize", "x-small")
            axes.legend(loc=(0, 1), frameon=False, ncol=ncol, fontsize=fontsize)

        return axes

    def oseries(
        self,
        names=None,
        tmin=None,
        tmax=None,
        ax=None,
        split=False,
        figsize=(10, 5),
        show_legend=True,
        labelfunc=None,
        legend_kwargs=None,
        **kwargs,
    ):
        """Plot oseries.

        Parameters
        ----------
        names : list[str], optional
            list of oseries names to plot, by default None, which loads
            all oseries from store
        tmin : str or pd.Timestamp, optional
            minimum time to plot, by default None, which uses the minimum
            time of the time series
        tmax : str or pd.Timestamp, optional
            maximum time to plot, by default None, which uses the maximum
            time of the time series
        ax : matplotlib.Axes, optional
            pass axes object to plot oseries on existing figure,
            by default None, in which case a new figure is created
        split : bool, optional
            create a separate subplot for each time series, by default False.
            A maximum of 20 time series is supported when split=True.
        figsize : tuple, optional
            figure size, by default (10, 5)
        show_legend : bool, optional
            show legend, default is True.
        labelfunc : callable, optional
            function to create custom labels, function should take name of time series
            as input
        legend_kwargs : dict, optional
            additional arguments to pass to legend

        Returns
        -------
        ax : matplotlib.Axes
            axes handle
        """
        return self._timeseries(
            "oseries",
            names=names,
            tmin=tmin,
            tmax=tmax,
            ax=ax,
            split=split,
            figsize=figsize,
            show_legend=show_legend,
            labelfunc=labelfunc,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def stresses(
        self,
        names=None,
        kind=None,
        tmin=None,
        tmax=None,
        ax=None,
        split=False,
        figsize=(10, 5),
        show_legend=True,
        labelfunc=None,
        legend_kwargs=None,
        **kwargs,
    ):
        """Plot stresses.

        Parameters
        ----------
        names : list[str], optional
            list of oseries names to plot, by default None, which loads
            all oseries from store
        kind : str, optional
            only plot stresses of a certain kind, by default None, which
            includes all stresses
        tmin : str or pd.Timestamp, optional
            minimum time to plot, by default None, which uses the minimum
            time of the time series
        tmax : str or pd.Timestamp, optional
            maximum time to plot, by default None, which uses the maximum
            time of the time series
        ax : matplotlib.Axes, optional
            pass axes object to plot oseries on existing figure,
            by default None, in which case a new figure is created
        split : bool, optional
            create a separate subplot for each time series, by default False.
            A maximum of 20 time series is supported when split=True.
        figsize : tuple, optional
            figure size, by default (10, 5)
        show_legend : bool, optional
            show legend, default is True.
        labelfunc : callable, optional
            function to create custom labels, function should take name of time series
            as input
        legend_kwargs : dict, optional
            additional arguments to pass to legend

        Returns
        -------
        ax : matplotlib.Axes
            axes handle
        """
        names = self.pstore.conn.parse_names(names, "stresses")
        masknames = self.pstore.stresses.index.isin(names)
        stresses = self.pstore.stresses.loc[masknames]

        if kind:
            mask = stresses["kind"] == kind
            names = stresses.loc[mask].index.to_list()

        return self._timeseries(
            "stresses",
            names=names,
            tmin=tmin,
            tmax=tmax,
            ax=ax,
            split=split,
            figsize=figsize,
            show_legend=show_legend,
            labelfunc=labelfunc,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def data_availability(
        self,
        libname,
        names=None,
        kind=None,
        intervals=None,
        ignore=("second", "minute", "14 days"),
        ax=None,
        cax=None,
        normtype="log",
        cmap="viridis_r",
        set_yticks=False,
        figsize=(10, 8),
        progressbar=True,
        dropna=True,
        **kwargs,
    ):
        """Plot the data-availability for multiple time series in pastastore.

        Parameters
        ----------
        libname : str
            name of library to get time series from (oseries or stresses)
        names : list, optional
            specify names in a list to plot data availability for certain
            time series
        kind : str, optional
            if library is stresses, kind can be specified to obtain only
            stresses of a specific kind
        intervals: dict, optional
            A dict with frequencies as keys and number of seconds as values
        ignore : list, optional
            A list with frequencies in intervals to ignore
        ax: matplotlib Axes, optional
            pass axes object to plot data availability on existing figure. by
            default None, in which case a new figure is created
        cax: matplotlib Axes, optional
            pass object axes to plot the colorbar on. by default None, which
            gives default Maptlotlib behavior
        normtype : str, optional
            Determines the type of color normalisations, default is 'log'
        cmap : str, optional
            A reference to a matplotlib colormap
        set_yticks : bool, optional
            Set the names of the series as yticks
        figsize : tuple, optional
            The size of the new figure in inches (h,v)
        progressbar : bool
            Show progressbar
        dropna : bool
            Do not show NaNs as available data
        kwargs : dict, optional
            Extra arguments are passed to matplotlib.pyplot.subplots()

        Returns
        -------
        ax : matplotlib Axes
            The axes in which the data-availability is plotted
        """
        names = self.pstore.conn.parse_names(names, libname)

        if libname == "stresses":
            masknames = self.pstore.stresses.index.isin(names)
            stresses = self.pstore.stresses.loc[masknames]
            if kind:
                mask = stresses["kind"] == kind
                names = stresses.loc[mask].index.to_list()

        series = self.pstore.conn._get_series(  # noqa: SLF001
            libname, names, progressbar=progressbar, squeeze=False
        ).values()

        ax = self._data_availability(
            series,
            names=names,
            intervals=intervals,
            ignore=ignore,
            ax=ax,
            cax=cax,
            normtype=normtype,
            cmap=cmap,
            set_yticks=set_yticks,
            figsize=figsize,
            dropna=dropna,
            **kwargs,
        )
        return ax

    @staticmethod
    def _data_availability(
        series,
        names=None,
        intervals=None,
        ignore=("second", "minute", "14 days"),
        ax=None,
        cax=None,
        normtype="log",
        cmap="viridis_r",
        set_yticks=False,
        ylabels=None,
        figsize=(10, 8),
        dropna=True,
        **kwargs,
    ):
        """Plot the data-availability for a list of time series.

        Parameters
        ----------
        libname : list of pandas.Series
            list of series to plot data availability for
        names : list, optional
            specify names of series, default is None in which case names
            will be taken from series themselves.
        kind : str, optional
            if library is stresses, kind can be specified to obtain only
            stresses of a specific kind
        intervals: dict, optional
            A dict with frequencies as keys and number of seconds as values
        ignore : list, optional
            A list with frequencies in intervals to ignore
        ax: matplotlib Axes, optional
            pass axes object to plot data availability on existing figure. by
            default None, in which case a new figure is created
        cax: matplotlib Axes, optional
            pass object axes to plot the colorbar on. by default None, which
            gives default Maptlotlib behavior
        normtype : str, optional
            Determines the type of color normalisations, default is 'log'
        cmap : str, optional
            A reference to a matplotlib colormap
        set_yticks : bool, optional
            Set the names of the series as yticks
        ylabels : list, optional
            Set the names of the series as yticks, default is None in which case
            names will be used as ylabels
        figsize : tuple, optional
            The size of the new figure in inches (h,v)
        progressbar : bool
            Show progressbar
        dropna : bool
            Do not show NaNs as available data
        kwargs : dict, optional
            Extra arguments are passed to matplotlib.pyplot.subplots()

        Returns
        -------
        ax : matplotlib Axes
            The axes in which the data-availability is plotted
        """
        # a good colormap is cmap='RdYlGn_r' or 'cubehelix'
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
        else:
            fig = ax.get_figure()

        ax.invert_yaxis()
        if intervals is None:
            intervals = {
                "second": 1,
                "minute": 60,
                "hour": 60 * 60,
                "day": 60 * 60 * 24,
                "week": 60 * 60 * 24 * 7,
                "14 days": 60 * 60 * 24 * 14,
                "month": 60 * 60 * 24 * 31,
                "quarter": 60 * 60 * 24 * 31 * 4,
                "year": 60 * 60 * 24 * 366,
            }
            for i in ignore:
                if i in intervals:
                    intervals.pop(i)

        bounds = np.array([intervals[i] for i in intervals])
        bounds = bounds.astype(float) * (10**9)
        labels = intervals.keys()
        if normtype == "log":
            norm = LogNorm(vmin=bounds[0], vmax=bounds[-1])
        else:
            norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.get_cmap(cmap, 256)
        cmap = cmap.with_extremes(over=(1.0, 1.0, 1.0))

        pc = None
        for i, s in enumerate(series):
            if not s.empty:
                if dropna:
                    s = s.dropna()
                pc = ax.pcolormesh(
                    s.index,
                    [i, i + 1],
                    [np.diff(s.index).astype(float)],
                    norm=norm,
                    cmap=cmap,
                    linewidth=0,
                    rasterized=True,
                )

        # make a colorbar in an ax on the
        # right side, then set the current axes to ax again
        if pc is not None:
            cb = fig.colorbar(pc, ax=ax, cax=cax, extend="both")
            cb.set_ticks(bounds)
            cb.ax.set_yticklabels(labels)
            cb.ax.minorticks_off()
        else:
            # nothing was plotted; skip colorbar to avoid UnboundLocalError
            cb = None

        if set_yticks:
            ax.set_yticks(np.arange(0.5, len(series) + 0.5), minor=False)
            ax.set_yticks(np.arange(0, len(series) + 1), minor=True)
            if names is None:
                names = [s.name for s in series]
            if ylabels is not None:
                ax.set_yticklabels(ylabels)
            else:
                ax.set_yticklabels(names)

            for tick in ax.yaxis.get_major_ticks():  # don't show major ytick marker
                tick.tick1line.set_visible(False)

            ax.grid(True, which="minor", axis="y")
            ax.grid(True, which="major", axis="x")

        else:
            ax.set_ylabel("Timeseries (-)")
            ax.grid(True, which="both")
            ax.grid(True, which="both")

        return ax

    def cumulative_hist(
        self,
        statistic="rsq",
        modelnames=None,
        extend=False,
        ax=None,
        figsize=(6, 6),
        label=None,
        legend=True,
        progressbar=True,
    ):
        """Plot a cumulative step histogram for a model statistic.

        Parameters
        ----------
        statistic: str
            name of the statistic, e.g. "evp" or "rmse", by default "rsq"
        modelnames: list[str], optional
            modelnames to plot statistic for, by default None, which
            uses all models in the store
        extend: bool, optional
            force extend the stats Series with a dummy value to move the
            horizontal line outside figure bounds. If True the results
            are skewed a bit, especially if number of models is low.
        ax: matplotlib.Axes, optional
            axes to plot histogram, by default None which creates an Axes
        figsize: tuple, optional
            figure size, by default (6,6)
        label: str, optional
            label for the legend, by default None, which shows the number
            of models
        legend: bool, optional
            show legend, by default True
        progressbar: bool, optional
            show progressbar, default is True.

        Returns
        -------
        ax : matplotlib Axes
            The axes in which the cumulative histogram is plotted
        """
        statsdf = self.pstore.get_statistics(
            [statistic], modelnames=modelnames, progressbar=progressbar
        )

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_xticks(np.linspace(0, 1, 11))
            ax.set_xlim(0, 1)
            ax.set_ylabel(statistic)
            ax.set_xlabel("Density")
            ax.set_title("Cumulative Step Histogram")
        if statistic == "evp":
            ax.set_yticks(np.linspace(0, 100, 11))
            if extend:
                statsdf = statsdf.append(pd.Series(100, index=["dummy"]))
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, statsdf.max())
        elif statistic in ("rsq", "nse", "kge_2012"):
            ax.set_yticks(np.linspace(0, 1, 11))
            if extend:
                statsdf = statsdf.append(pd.Series(1, index=["dummy"]))
                statsdf[statsdf < 0] = 0
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, statsdf.max())
        elif statistic in ("aic", "bic"):
            ax.set_ylim(statsdf.min(), statsdf.max())
        else:
            if extend:
                statsdf = statsdf.append(pd.Series(0, index=["dummy"]))
            ax.set_ylim(0, statsdf.max())

        if label is None:
            if extend:
                label = f"No. Models = {len(statsdf) - 1}"
            else:
                label = f"No. Models = {len(statsdf)}"

        statsdf.hist(
            ax=ax,
            bins=len(statsdf),
            density=True,
            cumulative=True,
            histtype="step",
            orientation="horizontal",
            label=label,
        )

        if legend:
            ax.legend(loc=4)

        return ax

    def compare_models(self, modelnames, ax=None, **kwargs):
        """Compare multiple models and plot the results.

        Parameters
        ----------
        modelnames : list
            A list of model names to compare.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the comparison. If not provided, a new figure
            and axes will be created.
        **kwargs : dict
            Additional keyword arguments to pass to the plot function.

        Returns
        -------
        cm : pastastore.CompareModels
            The CompareModels object containing the comparison results.
        """
        models = self.pstore.get_models(modelnames)
        names = kwargs.pop("names", [])
        onames = [iml.oseries.name for iml in models]
        if len(np.unique(onames)) == 1 and len(names) == 0:
            for modelname in modelnames:
                if onames[0] in modelname:
                    names.append(modelname.replace(onames[0], ""))
                else:
                    names.append(modelname)
        elif len(np.unique(onames)) > 1:
            names = modelnames
        cm = ps.CompareModels(models, names=names)
        if ax is not None:
            kwargs.setdefault("ax", ax)
        cm.plot(**kwargs)
        return cm
