import plotly.graph_objects as go
from sklearn.metrics import auc
import sklearn
import numpy as np

class RocCurveDisplayPlotly():
    """
    A reconstruction of the sklearn.metrics.RocCurveDisplay class to output Plotly plots rather thant plt.plots.

    Converts RocCurveDisplay inner .plot() method to work with plotly instead.

    Small modifications to the class constructor .from_predictions() as well
    """
    def __init__(self, *, fpr, tpr, roc_auc=None, estimator_name=None, pos_label=None):
        self.estimator_name = estimator_name
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label

    def plot(self, fig: go.Figure=None, plot_chance_level: bool=False, chance_level_kw: dict=None, **kwargs):
        """
        The class method used to plot the ROC curve with plotly

        Parameters
        ----------
        fig : go.Figure, optional
            A plotly go.Figure, by default None
        plot_chance_level : bool, optional
            Whether to plot the chance level, by default False
        chance_level_kw : dict, optional
            Additional parameters for the chance level curve, by default None
            Only valid if plot_chance_level=True

        Returns
        -------
        self
            Returns the class RocCurveDisplayPlotly
        """

        if fig is None:
            fig = go.Figure()

        fold = kwargs.get('fold', None)
        n_splits = kwargs.get('n_splits', None)

        name=f"ROC curve - AUC = {self.roc_auc:0.2f}"
        if (type(fold)==int) & (type(n_splits)==int):
            name = f"ROC fold {1 + fold // n_splits} - {fold % n_splits + 1} - AUC = {self.roc_auc:0.2f}"

        fig.add_trace(go.Scatter(x=self.fpr,
                                 y=self.tpr,
                                 opacity=0.3,
                                 mode='lines',
                                 name=name,
                                 showlegend=True
                                )
                    )

        if plot_chance_level:
            chance_level_line_kw = dict(width=2, dash='dash')
            if chance_level_kw is not None:
                chance_level_line_kw.update(**chance_level_kw)

            fig.add_trace(go.Scatter(x=[0, 1],
                            y=[0, 1],
                            mode='lines',
                            name="Chance Level - AUC = 0.50",
                            line = chance_level_line_kw
                            )
                )

        fig.update_layout(
                xaxis=dict(range=[-0.05, 1.05], title="False Positive Rate"),
                yaxis=dict(range=[-0.05, 1.05], title="True Positive Rate"),
                title="ROC curve",
                legend=dict(yanchor="bottom",
                            y=0.0,
                            xanchor="right",
                            x=0.99,
                            )
                )


        if kwargs.get("show_fig", True):
            fig.show()
        #self.fig = fig

        return self

    @classmethod
    def from_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        drop_intermediate: bool=True,
        pos_label: int=None,
        name: str=None,
        fig: go.Figure=None,
        plot_chance_level: bool=False,
        chance_level_kw: dict=None,
        **kwargs):
        """
        A class constructor based on one array of actual value and one array of predictions (binary or probabilities)

        Parameters
        ----------
        y_true : np.ndarray
            The array of test values, needs to be of dimensions (n, m) with n the number of folds and m the length of the test data
        y_pred : np.ndarray
            The array of predicted values, can be binary or probabilities
        drop_intermediate : bool, optional
            Whether to smooth the ROC curve, by default True
        pos_label : int, optional
            The positive class label in the binary classification, by default None
        name : str, optional
            The name of the graph, by default None
        fig : go.Figure, optional
            A plotly figure, by default None
        plot_chance_level : bool, optional
            Whether to plot the chance level, by default False
        chance_level_kw : dict, optional
            Additional arguments for the chance_level curve, by default None

        Returns
        -------
        self
            The class instance via its .plot() method
        """

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_true,
            y_pred,
            pos_label=pos_label,
            drop_intermediate=drop_intermediate,
        )
        roc_auc = auc(fpr, tpr)

        viz = RocCurveDisplayPlotly(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=name,
            pos_label=pos_label,
        )

        return viz.plot(fig=fig,
                        name=name,
                        plot_chance_level=plot_chance_level,
                        chance_level_kw=chance_level_kw,
                        **kwargs)
