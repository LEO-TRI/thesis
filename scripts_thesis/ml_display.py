import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import sklearn
import numpy as np
from collections import Counter

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


class PrecisionRecallDisplayPlotly():
    """
    Precision Recall visualization.

    Parameters
    ----------
    precision : ndarray
        Precision values.

    recall : ndarray
        Recall values.

    average_precision : float, default=None
        Average precision. If None, the average precision is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, then the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    prevalence_pos_label : float, default=None
        The prevalence of the positive label. It is used for plotting the
        chance level line. If None, the chance level line will not be plotted
        even if `plot_chance_level` is set to True when plotting.

    Attributes
    ----------
    fig : Plotly's Figure
        Precision recall curve.
    """

    def __init__(
        self,
        precision,
        recall,
        *,
        average_precision=None,
        estimator_name=None,
        pos_label=None,
        prevalence_pos_label=None,
    ):
        self.estimator_name = estimator_name
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.pos_label = pos_label
        self.prevalence_pos_label = prevalence_pos_label


    def plot(
        self,
        fig=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to plotly's `add_trace` method.

        Parameters
        ----------
        fig : Plotly's Figure, default=None
            Figure object to plot on. If `None`, a new Figure is
            created.

        name : str, default=None
            Name of precision recall curve for labeling. If `None`, use
            `estimator_name` if not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=False
            Whether to plot the chance level. The chance level is the prevalence
            of the positive label computed from the data passed during
            :meth:`from_predictions` call.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to plotly's `add_trace` for rendering
            the chance level line.

        **kwargs : dict
            Keyword arguments to be passed to plotly's `add_trace`.

        Returns
        -------
        display : :class:`~scripts_thesis.ml_display.PrecisionRecallDisplayPlotly`
            Object that stores computed values.
        """

        #Setting up the params of the graphs
        fold = kwargs.get('fold', None)
        n_splits = kwargs.get('n_splits', None)

        if (type(fold)==int) & (type(n_splits)==int) & (name is None):
            name = f"Fold {1 + fold // n_splits} - {fold % n_splits + 1} - (AP = {self.average_precision:0.2f})"
        elif name is None:
            name=f"Precision Recall curve - Recall = {self.average_precision:0.2f}"

        #Setting up the graph
        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.recall,
                                 y=self.precision,
                                 opacity=0.3,
                                 mode='lines',
                                 name=name,
                                 showlegend=True
                                )
                    )

        #Add a chance level variable
        if plot_chance_level:

            if self.prevalence_pos_label is None:
                raise ValueError(
                    "You must provide prevalence_pos_label when constructing the "
                    "PrecisionRecallDisplay object in order to plot the chance "
                    "level line. Alternatively, you may use "
                    "PrecisionRecallDisplay.from_estimator or "
                    "PrecisionRecallDisplay.from_predictions "
                    "to automatically set prevalence_pos_label"
                )

            chance_level_line_kw = dict(width=2, dash='dash')

            if chance_level_kw is not None:
                chance_level_line_kw.update(**chance_level_kw)

            fig.add_trace(go.Scatter(x=[0, 1],
                                     y=[self.prevalence_pos_label, self.prevalence_pos_label],
                                     mode='lines',
                                     name=f"Chance Level - AC = {self.prevalence_pos_label:0.2f}",
                                     line = chance_level_line_kw
                            )
                )

        info_pos_label = (f"- Positive label: {self.pos_label}" if self.pos_label is not None else "")
        xlabel = "Recall" + info_pos_label
        ylabel = "Precision" + info_pos_label

        fig.update_layout(
                xaxis=dict(range=[-0.05, 1.05], title=xlabel),
                yaxis=dict(range=[-0.05, 1.05], title=ylabel),
                title="Precision Recall Curve",
                legend=dict(yanchor="bottom",
                            y=0.0,
                            xanchor="right",
                            x=0.99,
                            )
                )

        self.fig = fig

        if kwargs.get("show_fig", True):
            fig.show()

        return self

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        pos_label=None,
        drop_intermediate=False,
        name=None,
        fig=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        """
        Plot precision-recall curve given binary class predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.

        y_pred : array-like of shape (n_samples,)
            Estimated probabilities or output of decision function.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=False
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        name : str, default=None
            Name for labeling curve. If `None`, name will be set to
            `"Classifier"`.

        fig : plotly.graph_objects.Figure, default=None
            Figure object to plot on. If `None`, a new go.Figure is created.

        plot_chance_level : bool, default=False
            Whether to plot the chance level. The chance level is the prevalence
            of the positive label computed from the data passed during
           :meth:`from_predictions` call.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to plotly's `plot` for rendering
            the chance level line.

        Returns
        -------
        display : :class:`~scripts_thesis.ml_display.PrecisionRecallDisplayPlotly`
        """

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
        )
        average_precision = average_precision_score(
            y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
        )

        class_count = Counter(y_true)
        prevalence_pos_label = class_count[pos_label] / sum(class_count.values())

        viz = PrecisionRecallDisplayPlotly(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_name=name,
            pos_label=pos_label,
            prevalence_pos_label=prevalence_pos_label,
        )

        return viz.plot(fig=fig,
                        name=name,
                        plot_chance_level=plot_chance_level,
                        chance_level_kw=chance_level_kw,
                        **kwargs)
