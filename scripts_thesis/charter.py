import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# color constants and palettes
CORAL = ["#D94535", "#ed8a84", "#f3b1ad", "#f9d8d6"]
DARK_GREY = ["#323C46", "#77949d", "#a4b7be", "#d2dbde"]
STONE = ["#7d8791", "#a5aaaa", "#d2d2d2", "#ebebeb"]
BLUE_GREY = ["#94b7bb", "#bfd4d6", "#d4e2e4", "#eaf1f1"]
BROWN = ["#7a6855", "#b4a594", "#cdc3b8", "#e6e1db"]
PURPLE = ["#8d89a5", "#bbb8c9", "#d1d0db", "#e8e7ed"]

PALETTE = np.array([CORAL, DARK_GREY, STONE, BLUE_GREY, BROWN, PURPLE])

def charter_graph(width:int =12, length:int =8) -> plt.Figure:
    """
    A convenience function to create a unified visual identity

    Needs to be used before creating the sns or plt plot

    Parameters
    ----------
    width : int, optional
        Width of the plot, by default 12
    length : int, optional
        Length of the plot, by default 8

    Returns
    -------
    fig : Figure
        A matplotlib figure
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(width, length)
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["font.family"] = "FuturaTOT"
    ax.patch.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(DARK_GREY[1])
    ax.spines["left"].set_color(DARK_GREY[1])
    ax.tick_params(colors=DARK_GREY[1])
    # ax.tick_params(axis='x', labelrotation=90)
    ax.spines["bottom"].set_color(DARK_GREY[2])
    ax.spines["left"].set_color(DARK_GREY[2])
    return fig, ax

def charter_multiple(width: int=12, length: int=8, rows: int=1, cols: int=1) -> plt.Figure:
    """
    A convenience function to create a unified visual identity.

    Needs to be used before creating the sns or plt plot

    Parameters
    ----------
    width : int, optional
        Width of the plot, by default 12
    length : int, optional
        Length of the plot, by default 8
    rows : int, optional
        Number of rows, by default 1
    cols : int, optional
        Number of columns, by default 1

    Returns
    -------
    fig : Figure
        A matplotlib figure
    """
    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(width, length)
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["font.family"] = "FuturaTOT"
    for ax in axes:
        ax.patch.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(DARK_GREY[0])
        ax.spines["left"].set_color(DARK_GREY[0])
        ax.tick_params(colors=DARK_GREY[0])
        ax.spines["bottom"].set_color(DARK_GREY[2])
        ax.spines["left"].set_color(DARK_GREY[2])
    return fig, axes


# plotly :
def charter_plotly(fig: go.Figure, width: int=950, height: int=600, title_place: float=0.3) -> go.Figure:
    """
    A convenience function to create a unified visual identity.

    Needs to be used after creating the plotly plot

    Parameters
    ----------
    fig : go.Figure
        A instantiated go.Figure
    width : int, optional
        Width of the plot, by default 1000
    height : int, optional
        Height of the plot, by default 600
    title_place : float, optional
        The location of the title, by default 0.3
    Returns
    -------
    go.Figure
        A plotly go.Figure with an updated layout
    """

    fig.update_xaxes(
        title_font=dict(size=20, family="FuturaTOT", color=DARK_GREY[0]),
        title_font_family="FuturaTOT",
        color=DARK_GREY[2],
        linecolor=DARK_GREY[2],
        tickfont=dict(size=12),
        showgrid=True,
    )
    fig.update_yaxes(
        title_font=dict(size=20, family="FuturaTOT", color=DARK_GREY[0]),
        title_font_family="FuturaTOT",
        color=DARK_GREY[2],
        linecolor=DARK_GREY[2],
        tickfont=dict(size=12),
        showgrid=True,
    )
    fig.update_layout(
        paper_bgcolor="white",
        #plot_bgcolor="white",
        template="plotly_white",
        width=width,
        height=height,
        title=dict(
            x=title_place, font=dict(family="FuturaTOT", size=24, color=DARK_GREY[0]),
        ),
        legend=dict(font=dict(family="FuturaTOT", size=14, color=DARK_GREY[0],)),
    )
    return fig
