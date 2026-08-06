"""Tests for `session._flatten_trials`'s handling of older Visiomode JSON formats: sessions
recorded before an explicit `stimulus`/`sdt_type` field existed on each trial, where the presented
stimulus and signal-detection classification instead have to be reconstructed from the trial's
`outcome`/`response` and the session-level `stimuli` metadata.
"""

import numpy as np

BASE_TRIAL = dict(
    timestamp="2022-01-01T00:00:01",
    iti=5.0,
    response_time=0.5,
    outcome="correct",
    correction=False,
)


def test_unknown_response_name_resolves_via_hf_environment(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": None, "timestamp": "2022-01-01T00:00:02"}, "sdt_type": "hit"},
        metadata_overrides={"environment": "hf"},
    )

    assert trial["response"] == "leverpush"
    # An explicit `sdt_type` on the trial (newer format) is used as-is rather than re-derived.
    assert trial["sdt_type"] == "hit"


def test_unknown_response_name_resolves_via_freelymoving_environment(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": None, "timestamp": "2022-01-01T00:00:02"}},
        metadata_overrides={"environment": "freelymoving"},
    )

    assert trial["response"] == "touch"


def test_legacy_gonogo_hit_reconstructs_target_stimulus_from_metadata(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": "leverpush", "timestamp": "2022-01-01T00:00:02"}, "outcome": "correct"}
    )

    assert trial["stim_id"] == "movinggrating"
    assert trial["stim_contrast"] == "1.0"


def test_legacy_gonogo_false_alarm_reconstructs_distractor_stimulus_from_metadata(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": "leverpush", "timestamp": "2022-01-01T00:00:02"}, "outcome": "incorrect"}
    )

    assert trial["stim_id"] == "isoluminantgray"


def test_legacy_gonogo_correct_rejection_reconstructs_distractor_stimulus(flatten_trial):
    trial = flatten_trial({**BASE_TRIAL, "response": None, "outcome": "correct"})

    assert trial["sdt_type"] == "correct_rejection"
    assert trial["stim_id"] == "isoluminantgray"


def test_legacy_gonogo_miss_reconstructs_target_stimulus_and_classifies_as_miss(flatten_trial):
    trial = flatten_trial({**BASE_TRIAL, "response": None, "outcome": "incorrect"})

    assert trial["sdt_type"] == "miss"
    assert trial["stim_id"] == "movinggrating"


def test_stimulus_literal_none_string_is_treated_as_no_stimulus(flatten_trial):
    trial = flatten_trial({**BASE_TRIAL, "response": None, "stimulus": "None"})

    assert "stim_id" not in trial
    # With no stimulus, there's nothing to cue, so cue_onset is left undefined.
    assert np.isnan(trial["cue_onset"])


def test_stimulus_with_common_name_uses_single_stimulus_format(flatten_trial):
    trial = flatten_trial({**BASE_TRIAL, "response": None, "stimulus": {"common_name": "grating", "contrast": 0.5}})

    assert trial["stim_common_name"] == "grating"
    assert trial["stim_contrast"] == 0.5


def test_stimulus_with_unrecognised_shape_yields_no_stimulus(flatten_trial):
    # Neither a "common_name" nor a "target" key: an unrecognised stimulus dict shape is left
    # unhandled and produces no stimulus fields, same as if none were presented at all.
    trial = flatten_trial({**BASE_TRIAL, "response": None, "stimulus": {"unexpected_key": "value"}})

    assert not any(key.startswith(("stim_", "target_", "distractor_")) for key in trial)
    assert np.isnan(trial["cue_onset"])


def test_stimulus_with_target_and_distractor_keys_uses_2afc_format(flatten_trial):
    trial = flatten_trial(
        {
            **BASE_TRIAL,
            "response": None,
            "stimulus": {"target": {"id": "left_shape"}, "distractor": {"id": "right_shape"}},
        }
    )

    assert trial["target_id"] == "left_shape"
    assert trial["distractor_id"] == "right_shape"


def test_legacy_targetonly_hit_reconstructs_target_stimulus(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": "touch", "timestamp": "2022-01-01T00:00:02"}, "outcome": "correct"},
        metadata_overrides={"protocol": "targetonly"},
    )

    assert trial["stim_id"] == "movinggrating"


def test_legacy_targetonly_no_response_reconstructs_target_stimulus(flatten_trial):
    trial = flatten_trial(
        {**BASE_TRIAL, "response": None, "outcome": "no_response"},
        metadata_overrides={"protocol": "targetonly"},
    )

    assert trial["stim_id"] == "movinggrating"


def test_legacy_targetonly_unmatched_response_outcome_combo_yields_no_stimulus(flatten_trial):
    # Neither "response present and correct" nor "no_response": e.g. an incorrect trial with a
    # response isn't reconstructed under the legacy targetonly branch, and yields no stimulus.
    trial = flatten_trial(
        {**BASE_TRIAL, "response": {"name": "touch", "timestamp": "2022-01-01T00:00:02"}, "outcome": "incorrect"},
        metadata_overrides={"protocol": "targetonly"},
    )

    assert "stim_id" not in trial
    assert np.isnan(trial["cue_onset"])


def test_legacy_other_protocol_uses_raw_stimuli_dict_unchanged(flatten_trial):
    trial = flatten_trial({**BASE_TRIAL, "response": None}, metadata_overrides={"protocol": "afc2"})

    # Unlike the gonogo/targetonly branches, this fallback doesn't rename keys to "stim_*".
    assert trial["target_id"] == "movinggrating"
    assert trial["distractor_id"] == "isoluminantgray"
    assert "stim_id" not in trial
