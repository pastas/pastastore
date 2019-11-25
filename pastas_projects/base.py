import numpy as np
import pandas as pd
import pastas as ps


class BaseProject:
    """BaseProject class that holds methods
    that work for both Arctic-based and Pystore-based
    projects.
    """

    def get_distances(self, oseries=None, stresses=None, kind=None):
        """Method to obtain the distances in meters between the stresses and
        oseries.

        Parameters
        ----------
        oseries: str or list of str
        stresses: str or list of str
        kind: str

        Returns
        -------
        distances: pandas.DataFrame
            Pandas DataFrame with the distances between the oseries (index)
            and the stresses (columns).

        """
        oseries_df = self.oseries
        stresses_df = self.stresses

        if isinstance(oseries, str):
            oseries = [oseries]
        elif oseries is None:
            oseries = oseries_df.index

        if stresses is None and kind is None:
            stresses = stresses_df.index
        elif stresses is None:
            stresses = stresses_df[stresses_df.kind == kind].index
        elif stresses is not None and kind is not None:
            mask = stresses_df.kind == kind
            stresses = stresses_df.loc[stresses].loc[mask].index

        xo = pd.to_numeric(oseries_df.loc[oseries, "x"])
        xt = pd.to_numeric(stresses_df.loc[stresses, "x"])
        yo = pd.to_numeric(oseries_df.loc[oseries, "y"])
        yt = pd.to_numeric(stresses_df.loc[stresses, "y"])

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                 index=oseries, columns=stresses)

        return distances

    def get_nearest_stresses(self, oseries=None, stresses=None, kind=None,
                             n=1):
        """Method to obtain the nearest (n) stresses of a specific kind.

        Parameters
        ----------
        oseries: str
            String with the name of the oseries
        stresses: str or list of str
            String with the name of the stresses
        kind:
            String with the name of the stresses
        n: int
            Number of stresses to obtain

        Returns
        -------
        stresses:
            List with the names of the stresses.

        """

        distances = self.get_distances(oseries, stresses, kind)

        data = pd.DataFrame(columns=np.arange(n))

        for series in distances.index:
            series = pd.Series(distances.loc[series].sort_values().index[:n],
                               name=series)
            data = data.append(series)
        return data

    def get_oseries_w_comments(self, comments, return_series=False,
                               progressbar=False):
        """Get oseries that contain certain comments in the 'comment'
        column.

        Parameters
        ----------
        comments : list of str
            list of str containing comments to look for
        return_series : bool, optional
            return series, by default False
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        dict
            dictionary containing either names or series that contain
            one or more of the comments.

        """
        have_comments = {}
        series = {}
        for c in comments:
            have_comments[c] = []
            series[c] = []
        for o in (tqdm(self.oseries.index) if progressbar else self.oseries.index):
            oseries = self.get_oseries(o)
            comment_series = oseries.comment
            for c in comments:
                if comment_series.str.contains(c).any():
                    have_comments[c].append(o)
                    if return_series:
                        series[c].append(oseries)
        if return_series:
            return series
        else:
            return have_comments

    def model_results(self, mls=None, progressbar=True):
        """Get pastas model results

        Parameters
        ----------
        mls : list of str, optional
            list of model names, by default None which means results for
            all models will be calculated
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        results : pd.DataFrame
            dataframe containing parameters and other statistics
            for each model

        Raises
        ------
        ModuleNotFoundError
            if the art_tools module is not available

        """
        try:
            from art_tools import pastas_get_model_results, pastas_model_checks
        except:
            raise ModuleNotFoundError(
                "You need 'art_tools' to use this method!")

        if mls is None:
            mls = self.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        results_list = []
        for mlname in (tqdm(mls) if progressbar else mls):
            iml = self.get_models(mlname)
            iresults = pastas_get_model_results(
                iml, parameters='all', stats=('evp',), stderrors=True)
            results_list.append(iresults)

        return pd.concat(results_list, axis=1).transpose()
