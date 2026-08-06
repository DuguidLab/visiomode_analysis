import math

import pytest

from visiomode_analysis.session import metrics


def test_d_prime_of_perfect_and_zero_rates_cancel_out():
    # Symmetric hit/false-alarm rates around 0.5 give equidistant z-scores, so d' is zero.
    assert metrics.d_prime(0.5, 0.5) == pytest.approx(0.0)


def test_d_prime_matches_known_value():
    # z(0.8) - z(0.2) = 0.8416... - (-0.8416...) = 1.6832...
    assert metrics.d_prime(0.8, 0.2) == pytest.approx(1.6832424671458286)


def test_d_prime_applies_afc_correction():
    uncorrected = metrics.d_prime(0.8, 0.2, afc_correction=False)
    corrected = metrics.d_prime(0.8, 0.2, afc_correction=True)

    assert corrected == pytest.approx(uncorrected * (1 / math.sqrt(2)))


@pytest.mark.parametrize("hit_rate,fa_rate", [(0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)])
def test_d_prime_rejects_boundary_rates(hit_rate, fa_rate):
    with pytest.raises(ValueError):
        metrics.d_prime(hit_rate, fa_rate)


def test_criterion_is_zero_when_rates_are_symmetric():
    assert metrics.criterion(0.8, 0.2) == pytest.approx(0.0)
    assert metrics.criterion(0.5, 0.5) == pytest.approx(0.0)


def test_criterion_rejects_both_rates_zero():
    with pytest.raises(ValueError):
        metrics.criterion(0.0, 0.0)


def test_perseveration_is_ratio_of_correction_to_incorrect_trials():
    assert metrics.perseveration(num_correction_trials=5, num_incorrect=10) == pytest.approx(0.5)


def test_perseveration_is_zero_when_no_incorrect_trials():
    assert metrics.perseveration(num_correction_trials=5, num_incorrect=0) == 0.0
