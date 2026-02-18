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
            marker={"colors": ["green", "salmon", "gold"]},
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
                    "skyblue",
                    "violet",
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
                    "slateblue",
                    "orange",
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


def plot_single_yvalue(value, ymin=0.0, ymax=1.0, as_html=False):
    fig = go.Figure(
        go.Scatter(
            y=[value],
            marker={"symbol": "x", "size": 12},
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[
            ymin if ymin < value else value * 1.25,
            ymax if value < ymax else value * 1.25,
        ],
    )
    fig.update_xaxes(showticklabels=False)

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_roc(hit_rate, fa_rate, hit_rate_wc=None, fa_rate_wc=None, as_html=False) -> str | go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
        ),
        layout_yaxis_range=[0, 1],
        layout_xaxis_range=[0, 1],
    )
    fig.add_trace(
        go.Scatter(
            x=[fa_rate],
            y=[hit_rate],
            marker={"symbol": "x", "size": 12, "color": "slateblue"},
            name="Random",
        ),
    )

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                yref="y",
                y0=0,
                y1=1,
                xref="x",
                x0=0,
                x1=1,
                line_dash="dash",
                opacity=0.8,
                fillcolor="grey",
            )
        ],
        xaxis={"title": "FA rate"},
        yaxis={"title": "Hit rate"},
    )

    if hit_rate_wc and fa_rate_wc:
        fig.add_trace(
            go.Scatter(
                x=[fa_rate_wc],
                y=[hit_rate_wc],
                marker={"symbol": "x", "size": 12, "color": "orange"},
                name="All",
            ),
        )

    fig.update_layout(showlegend=True if hit_rate_wc else False)
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_dprime(d_prime, d_prime_wc=None, as_html=False):
    fig = go.Figure(
        go.Scatter(
            y=[d_prime],
            marker={"symbol": "x", "size": 12, "color": "slateblue"},
            name="d'",
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[-0.25, 4.25],
    )

    fig.add_hline(y=1.5, line_color="grey", opacity=0.8, line_dash="dash")
    fig.add_hline(y=0.0, line_color="darkred", opacity=0.8)
    fig.update_xaxes(showticklabels=False)

    if d_prime_wc:
        fig.add_trace(
            go.Scatter(y=[d_prime_wc], marker={"symbol": "x", "size": 12, "color": "orange"}, name="d' (all)"),
        )

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_criterion(criterion, criterion_wc=None, as_html=False):
    fig = go.Figure(
        go.Scatter(
            y=[criterion],
            marker={"symbol": "x", "size": 12, "color": "slateblue"},
            name="C",
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
        layout_yaxis_range=[-3.25, 3.25],
    )

    fig.add_hline(y=0.0, line_color="grey", opacity=0.8, line_dash="dash")
    fig.update_xaxes(showticklabels=False)

    if criterion_wc:
        fig.add_trace(
            go.Scatter(y=[criterion_wc], marker={"symbol": "x", "size": 12, "color": "orange"}, name="C (all)"),
        )

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_sdt_pie(num_hits, num_false_alarms, num_correct_rejections, num_misses, as_html=False) -> str | go.Figure:
    labels = ["Hits", "False alarms", "Correct rejections", "Misses"]
    values = [num_hits, num_false_alarms, num_correct_rejections, num_misses]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker={"colors": ["darkgreen", "darksalmon", "lightgreen", "gold"]},
        ),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
    )
    if as_html:
        return fig.to_html(full_html=False)
    return fig
