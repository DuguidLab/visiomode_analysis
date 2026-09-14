import numpy as np
import pandas as pd
import pytest

from visiomode_analysis.session import plots


def _assert_embeddable_html(html):
    assert isinstance(html, str)
    assert "<div" in html
    assert "plotly" in html.lower()


# -- Pie charts: labels/values/colors should pass straight through from the arguments. --


def test_plot_success_pie_encodes_labels_values_and_colors():
    fig = plots.plot_success_pie(10, 3, 2)
    pie = fig.data[0]

    assert list(pie.labels) == ["Correct", "Incorrect", "No response"]
    assert list(pie.values) == [10, 3, 2]
    assert list(pie.marker.colors) == ["green", "salmon", "gold"]


def test_plot_cued_pie_encodes_labels_and_values():
    fig = plots.plot_cued_pie(num_cued=7, num_precued=4)
    pie = fig.data[0]

    assert list(pie.labels) == ["Cued", "Uncued"]
    assert list(pie.values) == [7, 4]


def test_plot_correction_pie_encodes_labels_and_values():
    fig = plots.plot_correction_pie(num_random=9, num_correction=6)
    pie = fig.data[0]

    assert list(pie.labels) == ["Random", "Correction"]
    assert list(pie.values) == [9, 6]


def test_plot_sdt_pie_encodes_labels_and_values():
    fig = plots.plot_sdt_pie(num_hits=1, num_false_alarms=2, num_correct_rejections=3, num_misses=4)
    pie = fig.data[0]

    assert list(pie.labels) == ["Hits", "False alarms", "Correct rejections", "Misses"]
    assert list(pie.values) == [1, 2, 3, 4]


# -- Reaction time plots: verify the actual median/IQR values, not just that a figure is built. --


def test_plot_rt_median_reports_median_and_iqr_of_rts():
    rts = [0.2, 0.4, 0.6, 0.8, 1.0]

    fig = plots.plot_rt_median(rts, stimulus_duration=5)
    scatter = fig.data[0]

    assert scatter.y[0] == pytest.approx(np.median(rts))
    assert scatter.error_y.array[0] == pytest.approx(np.percentile(rts, 75))
    assert scatter.error_y.arrayminus[0] == pytest.approx(np.percentile(rts, 25))
    assert fig.layout.yaxis.range == pytest.approx((0, 5))


def test_plot_rt_medians_from_dict_preserves_key_order_and_per_key_stats():
    rt_dict = {"all": [0.2, 0.6, 1.0], "hits": [0.3, 0.5]}

    fig = plots.plot_rt_medians_from_dict(rt_dict, stimulus_duration=4, as_html=False)
    scatter = fig.data[0]

    assert list(scatter.x) == ["all", "hits"]
    assert scatter.y[0] == pytest.approx(np.median(rt_dict["all"]))
    assert scatter.y[1] == pytest.approx(np.median(rt_dict["hits"]))
    assert fig.layout.yaxis.range == pytest.approx((0, 4))


def test_plot_rt_median_handles_no_reaction_times():
    # A session in which no trial had a response has no RTs at all; np.percentile raises on an
    # empty array, so the plot must degrade to NaN markers rather than blow up report generation.
    fig = plots.plot_rt_median([], stimulus_duration=5)
    scatter = fig.data[0]

    assert np.isnan(scatter.y[0])
    assert np.isnan(scatter.error_y.array[0])
    assert np.isnan(scatter.error_y.arrayminus[0])
    _assert_embeddable_html(plots.plot_rt_median([], stimulus_duration=5, as_html=True))


def test_plot_rt_medians_from_dict_handles_empty_entries():
    rt_dict = {"all": [0.2, 0.6, 1.0], "hits": [], "false_alarms": []}

    fig = plots.plot_rt_medians_from_dict(rt_dict, stimulus_duration=4, as_html=False)
    scatter = fig.data[0]

    assert list(scatter.x) == ["all", "hits", "false_alarms"]
    assert scatter.y[0] == pytest.approx(np.median(rt_dict["all"]))
    assert np.isnan(scatter.y[1]) and np.isnan(scatter.y[2])
    assert np.isnan(scatter.error_y.array[1]) and np.isnan(scatter.error_y.arrayminus[2])


@pytest.mark.parametrize("stimulus_duration", [-0.001, 0, None])
def test_rt_plots_leave_yaxis_unbounded_when_stimulus_duration_is_unknown(stimulus_duration):
    # Legacy sessions without a `spec` report stimulus_duration as -1 ms; a [0, -0.001] axis range
    # would render an empty plot, so fall back to plotly's auto-range instead.
    fig = plots.plot_rt_median([0.2, 0.4], stimulus_duration=stimulus_duration)
    fig_dict = plots.plot_rt_medians_from_dict({"all": [0.2, 0.4]}, stimulus_duration=stimulus_duration, as_html=False)

    assert fig.layout.yaxis.range is None
    assert fig_dict.layout.yaxis.range is None


# -- plot_single_yvalue: the y-axis range clamps asymmetrically when the value falls outside [ymin, ymax]. --


def test_plot_single_yvalue_range_stays_at_bounds_when_value_is_inside():
    fig = plots.plot_single_yvalue(0.5, ymin=0.0, ymax=1.0)

    assert fig.layout.yaxis.range == pytest.approx((0.0, 1.0))


def test_plot_single_yvalue_range_expands_below_ymin():
    fig = plots.plot_single_yvalue(-0.5, ymin=0.0, ymax=1.0)

    assert fig.layout.yaxis.range == pytest.approx((-0.625, 1.0))


def test_plot_single_yvalue_range_expands_above_ymax():
    fig = plots.plot_single_yvalue(1.5, ymin=0.0, ymax=1.0)

    assert fig.layout.yaxis.range == pytest.approx((0.0, 1.875))


# -- ROC / d' / criterion plots: the "_wc" comparison trace is only added when a value is supplied,
# and must treat an explicit 0.0 as a real value rather than "missing". --


def test_plot_roc_without_wc_omits_second_trace_and_legend():
    fig = plots.plot_roc(hit_rate=0.8, fa_rate=0.2)

    assert len(fig.data) == 1
    assert fig.layout.showlegend is False


def test_plot_roc_with_wc_adds_second_trace_and_shows_legend():
    fig = plots.plot_roc(hit_rate=0.8, fa_rate=0.2, hit_rate_wc=0.9, fa_rate_wc=0.3)

    assert len(fig.data) == 2
    assert fig.data[1].x[0] == pytest.approx(0.3)
    assert fig.data[1].y[0] == pytest.approx(0.9)
    assert fig.layout.showlegend is True


def test_plot_roc_treats_zero_wc_rates_as_real_values():
    fig = plots.plot_roc(hit_rate=0.5, fa_rate=0.5, hit_rate_wc=0.0, fa_rate_wc=0.0)

    assert len(fig.data) == 2
    assert fig.layout.showlegend is True


def test_plot_dprime_without_wc_has_single_trace():
    fig = plots.plot_dprime(d_prime=1.2)

    assert len(fig.data) == 1


def test_plot_dprime_treats_zero_wc_as_a_real_value():
    fig = plots.plot_dprime(d_prime=1.2, d_prime_wc=0.0)

    assert len(fig.data) == 2
    assert fig.data[1].y[0] == pytest.approx(0.0)


def test_plot_criterion_without_wc_has_single_trace():
    fig = plots.plot_criterion(criterion=0.3)

    assert len(fig.data) == 1


def test_plot_criterion_treats_zero_wc_as_a_real_value():
    fig = plots.plot_criterion(criterion=0.3, criterion_wc=0.0)

    assert len(fig.data) == 2
    assert fig.data[1].y[0] == pytest.approx(0.0)


# -- plot_trial_timeseries: groups by outcome or sdt_type, colors by the lookup tables, and
# falls back to violet for unrecognised categories. --


@pytest.fixture
def trials_df():
    return pd.DataFrame(
        {
            "outcome": ["correct", "correct", "incorrect"],
            "sdt_type": ["hit", "hit", "false_alarm"],
            "stop_time": [1.0, 2.0, 3.0],
        }
    )


def test_plot_trial_timeseries_groups_by_outcome_by_default(trials_df):
    fig = plots.plot_trial_timeseries(trials_df, use_sdt=False, as_html=False)

    traces_by_name = {trace.name: trace for trace in fig.data}
    assert set(traces_by_name) == {"correct", "incorrect"}
    assert list(traces_by_name["correct"].x) == [1.0, 2.0]
    assert traces_by_name["correct"].marker.color == "green"
    assert traces_by_name["incorrect"].marker.color == "salmon"
    assert fig.layout.xaxis.range == pytest.approx((0, 3.0))


def test_plot_trial_timeseries_groups_by_sdt_type_when_requested(trials_df):
    fig = plots.plot_trial_timeseries(trials_df, use_sdt=True, as_html=False)

    assert {trace.name for trace in fig.data} == {"hit", "false_alarm"}


def test_plot_trial_timeseries_falls_back_to_violet_for_unknown_category():
    trials = pd.DataFrame({"outcome": ["mystery"], "sdt_type": [None], "stop_time": [5.0]})

    fig = plots.plot_trial_timeseries(trials, use_sdt=False, as_html=False)

    assert fig.data[0].marker.color == "violet"


# -- Every plotting function also supports as_html=True, returning an embeddable HTML div. --


@pytest.mark.parametrize(
    "render",
    [
        lambda: plots.plot_success_pie(1, 2, 3, as_html=True),
        lambda: plots.plot_cued_pie(5, 2, as_html=True),
        lambda: plots.plot_correction_pie(5, 2, as_html=True),
        lambda: plots.plot_sdt_pie(1, 2, 3, 4, as_html=True),
        lambda: plots.plot_rt_median([0.1, 0.2, 0.3], as_html=True),
        lambda: plots.plot_rt_medians_from_dict({"all": [0.1, 0.2]}, as_html=True),
        lambda: plots.plot_single_yvalue(0.5, as_html=True),
        lambda: plots.plot_roc(0.5, 0.5, as_html=True),
        lambda: plots.plot_dprime(1.0, as_html=True),
        lambda: plots.plot_criterion(0.0, as_html=True),
    ],
)
def test_plot_functions_render_as_embeddable_html(render):
    _assert_embeddable_html(render())


def test_plot_trial_timeseries_renders_as_embeddable_html(trials_df):
    _assert_embeddable_html(plots.plot_trial_timeseries(trials_df, as_html=True))
