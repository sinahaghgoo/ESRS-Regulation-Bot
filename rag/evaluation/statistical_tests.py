"""
Statistical significance testing for RQ1 configuration comparisons.

This module provides bootstrap confidence intervals and paired permutation
tests to determine whether observed performance differences between
configurations are statistically significant. These tests are essential
for a master's thesis to avoid drawing conclusions from random variation.

Methods:
- Bootstrap confidence intervals (percentile method, 10,000 resamples)
- Paired permutation test (two-sided, 10,000 permutations)
- Effect size (Cohen's d)

References:
    - Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
    - Noreen (1989) "Computer Intensive Methods for Testing Hypotheses"
    - Sakai (2014) "Statistical Reform in Information Retrieval?"
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class BootstrapResult:
    """Result of a bootstrap confidence interval estimation."""
    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int
    n_resamples: int
    confidence_level: float


@dataclass
class PermutationTestResult:
    """Result of a paired permutation test."""
    metric: str
    config_a: str
    config_b: str
    mean_a: float
    mean_b: float
    observed_diff: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float  # Cohen's d
    n_samples: int
    n_permutations: int


def bootstrap_ci(
    values: list[float],
    metric_name: str = "metric",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a bootstrap confidence interval using the percentile method.

    Parameters
    ----------
    values : list[float]
        Per-sample metric values (e.g., Recall@15 for each of 204 samples).
    metric_name : str
        Name of the metric (for reporting).
    n_resamples : int
        Number of bootstrap resamples (default 10,000).
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Mean, confidence interval bounds, and sample statistics.
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return BootstrapResult(
            metric=metric_name, mean=0.0, ci_lower=0.0, ci_upper=0.0,
            std=0.0, n_samples=0, n_resamples=n_resamples,
            confidence_level=confidence,
        )

    # Generate bootstrap distribution of the mean.
    boot_means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    alpha = 1 - confidence
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1

    return BootstrapResult(
        metric=metric_name,
        mean=mean(values),
        ci_lower=boot_means[lo_idx],
        ci_upper=boot_means[hi_idx],
        std=stdev(values) if n > 1 else 0.0,
        n_samples=n,
        n_resamples=n_resamples,
        confidence_level=confidence,
    )


def paired_permutation_test(
    values_a: list[float],
    values_b: list[float],
    config_a: str = "A",
    config_b: str = "B",
    metric_name: str = "metric",
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PermutationTestResult:
    """Two-sided paired permutation test for comparing two configurations.

    Tests the null hypothesis that the two configurations have equal
    mean performance. Uses paired samples (same evaluation queries) to
    control for query difficulty variation.

    Parameters
    ----------
    values_a, values_b : list[float]
        Per-sample metric values for configurations A and B.
        Must have the same length (paired samples).
    n_permutations : int
        Number of random permutations (default 10,000).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    PermutationTestResult
        Test statistic, p-value, significance flag, and effect size.
    """
    assert len(values_a) == len(values_b), (
        f"Paired test requires equal-length inputs: {len(values_a)} vs {len(values_b)}"
    )

    rng = random.Random(seed)
    n = len(values_a)

    # Observed difference in means.
    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed_diff = mean(diffs)

    # Permutation distribution: randomly swap each pair.
    perm_diffs: list[float] = []
    for _ in range(n_permutations):
        perm = []
        for d in diffs:
            if rng.random() < 0.5:
                perm.append(d)
            else:
                perm.append(-d)
        perm_diffs.append(mean(perm))

    # Two-sided p-value: proportion of permutations with |diff| >= |observed|.
    abs_observed = abs(observed_diff)
    extreme_count = sum(1 for pd in perm_diffs if abs(pd) >= abs_observed)
    p_value = (extreme_count + 1) / (n_permutations + 1)  # +1 for continuity

    # Effect size: Cohen's d.
    diff_std = stdev(diffs) if n > 1 else 1.0
    effect_size = observed_diff / diff_std if diff_std > 0 else 0.0

    return PermutationTestResult(
        metric=metric_name,
        config_a=config_a,
        config_b=config_b,
        mean_a=mean(values_a),
        mean_b=mean(values_b),
        observed_diff=observed_diff,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=effect_size,
        n_samples=n,
        n_permutations=n_permutations,
    )


def compare_top_configs(
    per_sample_results: dict[str, list[float]],
    primary_metric: str = "recall@15",
    top_n: int = 5,
    alpha: float = 0.05,
) -> list[PermutationTestResult]:
    """Compare the top-N configurations pairwise.

    Parameters
    ----------
    per_sample_results : dict[str, list[float]]
        Mapping from config_id to list of per-sample metric values.
    primary_metric : str
        Metric name for reporting.
    top_n : int
        Number of top configs to compare.

    Returns
    -------
    list[PermutationTestResult]
        Pairwise test results for all (top_n choose 2) pairs.
    """
    # Rank configs by mean.
    ranked = sorted(
        per_sample_results.items(),
        key=lambda x: mean(x[1]),
        reverse=True,
    )[:top_n]

    results = []
    for i, (id_a, vals_a) in enumerate(ranked):
        for j, (id_b, vals_b) in enumerate(ranked):
            if j <= i:
                continue
            result = paired_permutation_test(
                vals_a, vals_b,
                config_a=id_a, config_b=id_b,
                metric_name=primary_metric,
                alpha=alpha,
            )
            results.append(result)

    return results
