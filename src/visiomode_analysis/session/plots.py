#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import plotly.express as px
import plotly.graph_objects as go

import numpy as np


def plot_success_pie(num_correct, num_incorrect, num_miss, as_html=False) -> str | go.Figure:
    labels = ["Correct", "Incorrect", "No response"]
    values = [num_correct, num_incorrect, num_miss]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
    )
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_cued_pie(num_cued, num_precued, as_html=False) -> str | go.Figure:
    labels = ["Cued", "Uncued"]
    values = [num_cued, num_precued]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker={
                "colors": [
                    "darkgreen",
                    "darkorange",
                ]
            },
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
    )
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_correction_pie(num_random, num_correction, as_html=False) -> str | go.Figure:
    labels = ["Random", "Correction"]
    values = [num_random, num_correction]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker={
                "colors": [
                    "lightgreen",
                    "lightred",
                ]
            },
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
    )
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_rt_median(rts, stimulus_duration=4, as_html=False) -> str | go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=[np.median(rts)],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[np.percentile(rts, 75)],
                arrayminus=[
                    np.percentile(rts, 25),
                ],
            ),
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[0, stimulus_duration],
    )
    fig.update_xaxes(showticklabels=False)

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_rt_medians_from_dict(rt_dict: dict, stimulus_duration: int = 4, as_html=True) -> str | go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=[np.median(rt) for rt in rt_dict.values()],
            x=[key for key in rt_dict.keys()],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[np.percentile(rt, 75) for rt in rt_dict.values()],
                arrayminus=[np.percentile(rt, 25) for rt in rt_dict.values()],
            ),
            mode="markers",
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[0, stimulus_duration],
    )

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_rt_distribution(rts, as_html=False) -> str | go.Figure: ...


def plot_single_yvalue(value, ymin=0, ymax=1, as_html=False):
    fig = go.Figure(
        go.Scatter(y=[value], marker={"symbol": "x", "size": 12}),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[ymin, ymax],
    )
    fig.update_xaxes(showticklabels=False)

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_roc(hit_rate, fa_rate, as_html=False) -> str | go.Figure: ...


def plot_sdt_pie(num_hits, num_false_alarms, num_correct_rejections, num_misses, as_html=False) -> str | go.Figure: ...
