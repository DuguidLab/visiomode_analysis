import math

from scipy.stats import norm


def d_prime(hit_rate: float, fa_rate: float, afc_correction: bool = False) -> float:
    if hit_rate == 0 or fa_rate == 0:
        raise ValueError("Cannot calculate a d' with hit or false alarm rate of zero.")

    if hit_rate == 1 or fa_rate == 1:
        raise ValueError("Cannot calculate a d' with hit or false alarm rate of one.")

    z_H = norm.ppf(hit_rate)
    z_FA = norm.ppf(fa_rate)

    d_prime = float(z_H - z_FA)

    if afc_correction:
        d_prime = (1 / math.sqrt(2)) * (d_prime)

    return d_prime


def bias(hit_rate: float, fa_rate: float) -> float:
    if hit_rate == 0 and fa_rate == 0:
        raise ValueError("Cannot calculate C with hit and false alarm rates of zero.")

    z_H = norm.ppf(hit_rate)
    z_FA = norm.ppf(fa_rate)

    decision_criterion = -(z_H + z_FA) / 2

    return float(decision_criterion)


def perseveration(num_correction_trials: int, num_incorrect: int) -> float:
    if num_incorrect == 0:
        return 0.0

    return float(num_correction_trials / num_incorrect)


def percentage_correct(num_correct: int, num_cued: int) -> float:
    return float((num_correct / num_cued) * 100)
