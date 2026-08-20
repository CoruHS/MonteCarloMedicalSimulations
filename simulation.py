#!/usr/bin/env python3
"""Monte Carlo allocation simulation for "Who Should the Algorithm Save?".

The module deliberately uses ordinary least squares. The learned policy receives
neither group membership nor latent true need; its only training target is cost.
Run this file to reproduce every result, table, and figure in the repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression


# This is the single source of truth for every modelling and presentation choice.
# Command-line overrides are written into the emitted config.json before a run.
CONFIG: dict[str, Any] = {
    "study": "Who Should the Algorithm Save?",
    "base_seed": 20260805,
    "n_draws": 200,
    "n_train": 4000,
    "n_score": 1000,
    "diagnostic_n_train": 12000,
    "diagnostic_n_score": 5000,
    "scarcity_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "moderate_scarcity": 0.5,
    "p_disadvantaged": 0.40,
    "age": {
        "distribution": "truncated_normal",
        "mean": 57.0,
        "sd": 18.0,
        "min": 18.0,
        "max": 95.0,
    },
    "severity": {
        "distribution": "beta_scaled_plus_structural_penalty_and_noise",
        "beta_alpha": 2.2,
        "beta_beta": 3.0,
        "min": 0.0,
        "max": 24.0,
        "structural_penalty": 2.0,
        "noise_sd": 1.25,
        "robustness_observed_attenuation": 0.15,
    },
    "survival": {
        "age_center": 50.0,
        "treated": {"intercept": 3.2, "severity": -0.16, "age": -0.025},
        "untreated": {"intercept": 2.0, "severity": -0.18, "age": -0.027},
    },
    "life_expectancy": {
        "intercept_age": 84.0,
        "severity_penalty_per_point": 0.35,
        "noise_sd": 4.0,
        "min": 1.0,
        "max": 70.0,
    },
    "arrival_window_hours": 24.0,
    "true_need": {
        "severity_weight": 0.60,
        "benefit_weight": 0.40,
        "benefit_scale": 0.40,
        "noise_sd": 0.05,
        "min": 0.0,
        "max": 1.20,
    },
    "access": {
        "attenuation_delta": 0.30,
        "prior_utilisation_scale": 12.0,
        "prior_utilisation_noise_sd": 1.0,
        "cost_scale_dollars": 24000.0,
        "cost_noise_sd_dollars": 1800.0,
    },
    "need_matching_bins": 20,
    "conditions": {
        "primary": {"attenuate_observed_severity": False},
        "attenuated_severity_robustness": {"attenuate_observed_severity": True},
    },
    "figures": {
        "font_family": "DejaVu Serif",
        "png_dpi": 300,
        "diagnostic_scatter_points": 3500,
    },
}


FEATURES: tuple[str, ...] = (
    "age",
    "severity",
    "p_survive_treated",
    "p_survive_untreated",
    "prior_utilisation",
)

POLICY_ORDER: tuple[str, ...] = (
    "Lottery",
    "First come, first served",
    "Sickest first",
    "Save the most lives",
    "Youngest first",
    "Life-years",
    "Learned cost model",
)

POLICY_COLORS: dict[str, str] = {
    "Lottery": "#56B4E9",
    "First come, first served": "#000000",
    "Sickest first": "#0072B2",
    "Save the most lives": "#009E73",
    "Youngest first": "#CC79A7",
    "Life-years": "#E69F00",
    "Learned cost model": "#D55E00",
}

POLICY_LINESTYLES: dict[str, str | tuple[Any, ...]] = {
    "Lottery": (0, (1, 1)),
    "First come, first served": (0, (5, 2)),
    "Sickest first": "-",
    "Save the most lives": (0, (3, 1, 1, 1)),
    "Youngest first": (0, (6, 2, 1, 2)),
    "Life-years": (0, (2, 2)),
    "Learned cost model": "-",
}

POLICY_MARKERS: dict[str, str] = {
    "Lottery": "o",
    "First come, first served": "s",
    "Sickest first": "^",
    "Save the most lives": "D",
    "Youngest first": "v",
    "Life-years": "P",
    "Learned cost model": "X",
}

GROUP_LABELS = {0: "Advantaged", 1: "Disadvantaged"}
CONDITION_LABELS = {
    "primary": "Clean observed severity",
    "attenuated_severity_robustness": "Attenuated observed severity",
}


@dataclass(frozen=True)
class FittedCostModel:
    model: LinearRegression
    coefficients: np.ndarray
    standard_errors: np.ndarray
    intercept: float
    intercept_standard_error: float
    train_r2: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert_model_features(frame)
        return self.model.predict(frame.loc[:, FEATURES].to_numpy(dtype=float))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory receiving results.csv, config.json, README.md, and figures/.",
    )
    parser.add_argument("--draws", type=int, help="Override the default Monte Carlo draw count.")
    parser.add_argument("--train-size", type=int, help="Override patients in each training population.")
    parser.add_argument("--score-size", type=int, help="Override patients in each scored population.")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Skip the attenuated-severity robustness condition (Figure 6 is then omitted).",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    if args.draws is not None:
        if args.draws < 2:
            raise ValueError("--draws must be at least 2 so intervals can be estimated")
        config["n_draws"] = args.draws
    if args.train_size is not None:
        if args.train_size <= len(FEATURES) + 2:
            raise ValueError("--train-size is too small for the regression")
        config["n_train"] = args.train_size
    if args.score_size is not None:
        if args.score_size < 100:
            raise ValueError("--score-size must be at least 100")
        config["n_score"] = args.score_size
    if args.primary_only:
        config["conditions"] = {"primary": config["conditions"]["primary"]}
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    scarcity = config["scarcity_levels"]
    if sorted(scarcity) != list(scarcity):
        raise ValueError("scarcity_levels must be sorted")
    if not all(0 < value <= 1 for value in scarcity):
        raise ValueError("scarcity levels must be in (0, 1]")
    if 1.0 not in scarcity:
        raise ValueError("scarcity_levels must include 1.0 for the end-to-end check")
    if config["moderate_scarcity"] not in scarcity:
        raise ValueError("moderate_scarcity must be one of scarcity_levels")
    delta = config["access"]["attenuation_delta"]
    if not 0 < delta < 1:
        raise ValueError("attenuation_delta must lie in (0, 1)")
    need = config["true_need"]
    if not math.isclose(need["severity_weight"] + need["benefit_weight"], 1.0):
        raise ValueError("true-need weights must sum to one")


def config_id(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def truncated_normal(
    rng: np.random.Generator,
    size: int,
    mean: float,
    sd: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Sample a normal distribution by rejection, not boundary clipping."""
    output = np.empty(size, dtype=float)
    remaining = np.arange(size)
    while remaining.size:
        proposals = rng.normal(mean, sd, size=remaining.size)
        accepted = (proposals >= low) & (proposals <= high)
        output[remaining[accepted]] = proposals[accepted]
        remaining = remaining[~accepted]
    return output


def generate_population(
    n: int,
    seed: int,
    config: Mapping[str, Any],
    *,
    attenuate_observed_severity: bool,
) -> pd.DataFrame:
    """Generate one population; true_need is retained only for evaluation."""
    rng = np.random.default_rng(seed)
    group = rng.binomial(1, config["p_disadvantaged"], n).astype(np.int8)

    age_cfg = config["age"]
    age = truncated_normal(
        rng,
        n,
        age_cfg["mean"],
        age_cfg["sd"],
        age_cfg["min"],
        age_cfg["max"],
    )

    severity_cfg = config["severity"]
    base_severity = rng.beta(
        severity_cfg["beta_alpha"], severity_cfg["beta_beta"], n
    ) * (severity_cfg["max"] - severity_cfg["min"])
    clinical_severity = np.clip(
        base_severity
        + severity_cfg["structural_penalty"] * group
        + rng.normal(0.0, severity_cfg["noise_sd"], n),
        severity_cfg["min"],
        severity_cfg["max"],
    )
    severity_attenuation = (
        severity_cfg["robustness_observed_attenuation"]
        if attenuate_observed_severity
        else 0.0
    )
    observed_severity = clinical_severity * (1.0 - severity_attenuation * group)

    survival_cfg = config["survival"]
    centered_age = age - survival_cfg["age_center"]
    treated_cfg = survival_cfg["treated"]
    untreated_cfg = survival_cfg["untreated"]
    treated_logit = (
        treated_cfg["intercept"]
        + treated_cfg["severity"] * clinical_severity
        + treated_cfg["age"] * centered_age
    )
    untreated_logit = (
        untreated_cfg["intercept"]
        + untreated_cfg["severity"] * clinical_severity
        + untreated_cfg["age"] * centered_age
    )
    p_survive_treated = sigmoid(treated_logit)
    p_survive_untreated = sigmoid(untreated_logit)
    if not np.all(p_survive_untreated < p_survive_treated):
        raise AssertionError("untreated survival must be strictly below treated survival")
    benefit = p_survive_treated - p_survive_untreated

    life_cfg = config["life_expectancy"]
    life_expectancy = np.clip(
        life_cfg["intercept_age"]
        - age
        - life_cfg["severity_penalty_per_point"] * clinical_severity
        + rng.normal(0.0, life_cfg["noise_sd"], n),
        life_cfg["min"],
        life_cfg["max"],
    )

    need_cfg = config["true_need"]
    true_need = np.clip(
        need_cfg["severity_weight"]
        * (clinical_severity / severity_cfg["max"])
        + need_cfg["benefit_weight"]
        * (benefit / need_cfg["benefit_scale"])
        + rng.normal(0.0, need_cfg["noise_sd"], n),
        need_cfg["min"],
        need_cfg["max"],
    )

    access_cfg = config["access"]
    attenuation = 1.0 - access_cfg["attenuation_delta"] * group
    prior_utilisation = np.clip(
        access_cfg["prior_utilisation_scale"] * true_need * attenuation
        + rng.normal(0.0, access_cfg["prior_utilisation_noise_sd"], n),
        0.0,
        None,
    )
    cost = np.clip(
        access_cfg["cost_scale_dollars"] * true_need * attenuation
        + rng.normal(0.0, access_cfg["cost_noise_sd_dollars"], n),
        0.0,
        None,
    )

    frame = pd.DataFrame(
        {
            "patient_id": np.arange(n, dtype=np.int64),
            "group": group,
            "age": age,
            "base_severity": base_severity,
            "clinical_severity": clinical_severity,
            "severity": observed_severity,
            "p_survive_treated": p_survive_treated,
            "p_survive_untreated": p_survive_untreated,
            "benefit": benefit,
            "life_expectancy": life_expectancy,
            "arrival_time": rng.uniform(0.0, config["arrival_window_hours"], n),
            "true_need": true_need,
            "attenuation": attenuation,
            "prior_utilisation": prior_utilisation,
            "cost": cost,
            "lottery_key": rng.random(n),
            "tie_breaker": rng.random(n),
            "outcome_uniform": rng.random(n),
        }
    )
    return frame


def assert_model_features(frame: pd.DataFrame) -> None:
    feature_set = set(FEATURES)
    if "group" in feature_set or "true_need" in feature_set or "cost" in feature_set:
        raise AssertionError("protected/evaluation/label columns leaked into FEATURES")
    if tuple(frame.loc[:, FEATURES].columns) != FEATURES:
        raise AssertionError("learned-model feature order changed")


def fit_cost_model(frame: pd.DataFrame) -> FittedCostModel:
    """Fit standard unweighted OLS and compute conventional analytic SEs."""
    assert_model_features(frame)
    x = frame.loc[:, FEATURES].to_numpy(dtype=float)
    y = frame["cost"].to_numpy(dtype=float)
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    prediction = model.predict(x)
    residual = y - prediction
    design = np.column_stack([np.ones(len(x)), x])
    degrees_freedom = len(x) - design.shape[1]
    if degrees_freedom <= 0:
        raise ValueError("training set is too small for coefficient standard errors")
    residual_variance = float(residual @ residual / degrees_freedom)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return FittedCostModel(
        model=model,
        coefficients=np.asarray(model.coef_, dtype=float),
        standard_errors=standard_errors[1:],
        intercept=float(model.intercept_),
        intercept_standard_error=float(standard_errors[0]),
        train_r2=float(model.score(x, y)),
    )


def policy_priority_scores(frame: pd.DataFrame, learned_scores: np.ndarray) -> dict[str, np.ndarray]:
    if len(learned_scores) != len(frame):
        raise ValueError("learned score length differs from population length")
    return {
        "Lottery": frame["lottery_key"].to_numpy(),
        "First come, first served": -frame["arrival_time"].to_numpy(),
        "Sickest first": frame["severity"].to_numpy(),
        "Save the most lives": frame["benefit"].to_numpy(),
        "Youngest first": -frame["age"].to_numpy(),
        "Life-years": (frame["benefit"] * frame["life_expectancy"]).to_numpy(),
        "Learned cost model": np.asarray(learned_scores, dtype=float),
    }


def priority_order(scores: np.ndarray, tie_breaker: np.ndarray) -> np.ndarray:
    """Return indices ordered from highest to lowest priority."""
    return np.lexsort((tie_breaker, -np.asarray(scores, dtype=float)))


def priority_percentile(order: np.ndarray) -> np.ndarray:
    n = len(order)
    percentile = np.empty(n, dtype=float)
    if n == 1:
        percentile[order] = 1.0
        return percentile
    percentile[order] = 1.0 - np.arange(n, dtype=float) / (n - 1)
    return percentile


def matched_need_priority_gap(
    frame: pd.DataFrame,
    priority: np.ndarray,
    bins: int,
) -> float:
    """Advantaged minus disadvantaged priority, within true-need strata.

    Priority is scaled from zero (last) to one (first). A positive returned value
    therefore means the disadvantaged group is ranked lower at matched need.
    Strata are pooled true-need quantiles and are weighted by overlap (the smaller
    group count in each stratum), so non-overlapping need regions do not identify
    the conditional contrast.
    """
    need = frame["true_need"].to_numpy(dtype=float)
    group = frame["group"].to_numpy(dtype=int)
    edges = np.unique(np.quantile(need, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    strata = np.digitize(need, edges[1:-1], right=True)
    differences: list[float] = []
    weights: list[float] = []
    for stratum in range(len(edges) - 1):
        in_stratum = strata == stratum
        group0 = in_stratum & (group == 0)
        group1 = in_stratum & (group == 1)
        n0 = int(group0.sum())
        n1 = int(group1.sum())
        if n0 and n1:
            differences.append(float(priority[group0].mean() - priority[group1].mean()))
            weights.append(float(min(n0, n1)))
    if not weights:
        return float("nan")
    return 100.0 * float(np.average(differences, weights=weights))


def allocation_metrics(
    frame: pd.DataFrame,
    order: np.ndarray,
    scarcity: float,
    *,
    policy: str,
    condition: str,
    draw: int,
    train_seed: int,
    score_seed: int,
    model_train_r2: float,
    heldout_cost_correlation: float,
    need_priority_gap_pp: float,
    current_config_id: str,
) -> dict[str, Any]:
    n = len(frame)
    resources = int(round(scarcity * n))
    resources = max(0, min(resources, n))
    treated = np.zeros(n, dtype=bool)
    treated[order[:resources]] = True

    p_treated = frame["p_survive_treated"].to_numpy(dtype=float)
    p_untreated = frame["p_survive_untreated"].to_numpy(dtype=float)
    expected_survival = np.where(treated, p_treated, p_untreated)
    life_expectancy = frame["life_expectancy"].to_numpy(dtype=float)
    outcomes = frame["outcome_uniform"].to_numpy(dtype=float) < expected_survival
    group = frame["group"].to_numpy(dtype=int)
    scale = 1000.0 / n

    result: dict[str, Any] = {
        "config_id": current_config_id,
        "condition": condition,
        "draw": draw,
        "train_seed": train_seed,
        "score_seed": score_seed,
        "scarcity": scarcity,
        "n_patients": n,
        "resources": resources,
        "policy": policy,
        "lives_saved": float(expected_survival.sum()),
        "lives_saved_per_1000": float(expected_survival.sum() * scale),
        "life_years_saved": float((expected_survival * life_expectancy).sum()),
        "life_years_saved_per_1000": float(
            (expected_survival * life_expectancy).sum() * scale
        ),
        "realized_survivors": int(outcomes.sum()),
        "realized_survivors_per_1000": float(outcomes.sum() * scale),
        "need_priority_gap_pp": need_priority_gap_pp,
        "group_population_share_disadvantaged": float((group == 1).mean()),
        "model_train_r2": model_train_r2,
        "heldout_cost_correlation": heldout_cost_correlation,
    }
    for group_value, group_name in ((0, "advantaged"), (1, "disadvantaged")):
        mask = group == group_value
        if not mask.any():
            raise AssertionError(f"generated population contains no {group_name} patients")
        result[f"allocation_rate_{group_name}"] = float(treated[mask].mean())
        result[f"survival_rate_{group_name}"] = float(expected_survival[mask].mean())
    result["allocation_gap_disadv_minus_adv_pp"] = 100.0 * (
        result["allocation_rate_disadvantaged"] - result["allocation_rate_advantaged"]
    )
    result["survival_gap_disadv_minus_adv_pp"] = 100.0 * (
        result["survival_rate_disadvantaged"] - result["survival_rate_advantaged"]
    )
    return result


def seeds_for_draw(base_seed: int, draw: int) -> tuple[int, int]:
    # Fixed arithmetic makes seeds transparent in results.csv and independent of
    # how many conditions are run.
    return base_seed + 1000 + draw * 2, base_seed + 1001 + draw * 2


def run_monte_carlo(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_config_id = config_id(config)
    for condition, condition_cfg in config["conditions"].items():
        print(f"Running condition: {condition}", flush=True)
        attenuate = bool(condition_cfg["attenuate_observed_severity"])
        for draw in range(config["n_draws"]):
            train_seed, score_seed = seeds_for_draw(config["base_seed"], draw)
            train = generate_population(
                config["n_train"],
                train_seed,
                config,
                attenuate_observed_severity=attenuate,
            )
            score = generate_population(
                config["n_score"],
                score_seed,
                config,
                attenuate_observed_severity=attenuate,
            )
            fitted = fit_cost_model(train)
            learned_scores = fitted.predict(score)
            heldout_cost_correlation = float(np.corrcoef(learned_scores, score["cost"])[0, 1])
            if not np.isfinite(heldout_cost_correlation) or heldout_cost_correlation <= 0:
                raise AssertionError("learned predictions do not correlate positively with held-out cost")

            scores_by_policy = policy_priority_scores(score, learned_scores)
            orders: dict[str, np.ndarray] = {}
            gaps: dict[str, float] = {}
            for policy in POLICY_ORDER:
                order = priority_order(scores_by_policy[policy], score["tie_breaker"].to_numpy())
                orders[policy] = order
                gaps[policy] = matched_need_priority_gap(
                    score,
                    priority_percentile(order),
                    config["need_matching_bins"],
                )

            for scarcity in config["scarcity_levels"]:
                draw_rows: dict[str, dict[str, Any]] = {}
                for policy in POLICY_ORDER:
                    result = allocation_metrics(
                        score,
                        orders[policy],
                        scarcity,
                        policy=policy,
                        condition=condition,
                        draw=draw,
                        train_seed=train_seed,
                        score_seed=score_seed,
                        model_train_r2=fitted.train_r2,
                        heldout_cost_correlation=heldout_cost_correlation,
                        need_priority_gap_pp=gaps[policy],
                        current_config_id=current_config_id,
                    )
                    rows.append(result)
                    draw_rows[policy] = result

                if (
                    draw_rows["Save the most lives"]["lives_saved"]
                    + 1e-10
                    < draw_rows["Lottery"]["lives_saved"]
                ):
                    raise AssertionError("save-the-most-lives underperformed lottery")

                if math.isclose(scarcity, 1.0):
                    reference = draw_rows[POLICY_ORDER[0]]
                    for policy in POLICY_ORDER[1:]:
                        candidate = draw_rows[policy]
                        for metric in (
                            "lives_saved",
                            "life_years_saved",
                            "allocation_rate_advantaged",
                            "allocation_rate_disadvantaged",
                            "survival_rate_advantaged",
                            "survival_rate_disadvantaged",
                        ):
                            if not math.isclose(
                                reference[metric], candidate[metric], rel_tol=0.0, abs_tol=1e-10
                            ):
                                raise AssertionError(
                                    f"policies differ at scarcity 1.0 for {metric}"
                                )
            progress_interval = max(1, config["n_draws"] // 10)
            if (draw + 1) % progress_interval == 0 or draw + 1 == config["n_draws"]:
                print(
                    f"  completed {draw + 1}/{config['n_draws']} draws",
                    flush=True,
                )
    return pd.DataFrame(rows)


def percentile_summary(
    results: pd.DataFrame,
    metric: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    return (
        results.groupby(list(group_columns), sort=False)[metric]
        .agg(
            mean="mean",
            lower=lambda values: np.percentile(values, 2.5),
            upper=lambda values: np.percentile(values, 97.5),
        )
        .reset_index()
    )


def aggregate_sanity_checks(results: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    lottery = results[results["policy"] == "Lottery"].copy()
    lottery["selected_share_disadvantaged"] = np.where(
        lottery["resources"] > 0,
        lottery["allocation_rate_disadvantaged"]
        * lottery["group_population_share_disadvantaged"]
        / lottery["scarcity"],
        np.nan,
    )
    lottery_gap = (
        lottery.groupby(["condition", "scarcity"], sort=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "mean_selected_minus_population": float(
                        (
                            frame["selected_share_disadvantaged"]
                            - frame["group_population_share_disadvantaged"]
                        ).mean()
                    ),
                    "se": float(
                        (
                            frame["selected_share_disadvantaged"]
                            - frame["group_population_share_disadvantaged"]
                        ).std(ddof=1)
                        / math.sqrt(len(frame))
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    lottery_gap["tolerance"] = 4.0 * lottery_gap["se"].fillna(0.0) + 0.005
    lottery_ok = bool(
        (
            lottery_gap["mean_selected_minus_population"].abs()
            <= lottery_gap["tolerance"]
        ).all()
    )
    if not lottery_ok:
        raise AssertionError("lottery selected group share outside aggregate sampling error")
    checks["lottery_group_balance_passed"] = lottery_ok
    checks["lottery_max_absolute_share_error_pp"] = float(
        100.0 * lottery_gap["mean_selected_minus_population"].abs().max()
    )

    full = results[np.isclose(results["scarcity"], 1.0)]
    spread = (
        full.groupby(["condition", "draw"])[
            ["lives_saved", "life_years_saved", "allocation_rate_advantaged", "allocation_rate_disadvantaged"]
        ]
        .agg(lambda values: float(values.max() - values.min()))
        .to_numpy()
    )
    full_ok = bool(np.nanmax(np.abs(spread)) < 1e-9)
    if not full_ok:
        raise AssertionError("not all policies agree when every patient is treated")
    checks["full_allocation_identity_passed"] = full_ok

    pivot = results.pivot_table(
        index=["condition", "draw", "scarcity"],
        columns="policy",
        values="lives_saved",
    )
    save_lives_ok = bool(
        (pivot["Save the most lives"] + 1e-10 >= pivot["Lottery"]).all()
    )
    if not save_lives_ok:
        raise AssertionError("save-the-most-lives sanity check failed")
    checks["save_lives_not_below_lottery_passed"] = save_lives_ok

    min_correlation = float(results["heldout_cost_correlation"].min())
    if min_correlation <= 0:
        raise AssertionError("a learned model failed the positive held-out correlation check")
    checks["heldout_cost_correlation_positive_passed"] = True
    checks["minimum_heldout_cost_correlation"] = min_correlation

    moderate = results[
        np.isclose(results["scarcity"], config["moderate_scarcity"])
        & (results["condition"] == "primary")
    ]
    expected_rank = (
        moderate.groupby("policy")["lives_saved_per_1000"].mean().rank(method="average")
    )
    realized_rank = (
        moderate.groupby("policy")["realized_survivors_per_1000"].mean().rank(method="average")
    )
    rank_correlation = float(expected_rank.corr(realized_rank, method="spearman"))
    checks["expected_vs_bernoulli_rank_spearman"] = rank_correlation
    checks["expected_vs_bernoulli_ranking_check_passed"] = bool(rank_correlation >= 0.80)
    if not checks["expected_vs_bernoulli_ranking_check_passed"]:
        raise AssertionError("expected and Bernoulli policy rankings are not stable")
    return checks


def diagnostic_analysis(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, FittedCostModel, pd.DataFrame, dict[str, Any]]:
    base = config["base_seed"]
    train = generate_population(
        config["diagnostic_n_train"],
        base + 900_001,
        config,
        attenuate_observed_severity=False,
    )
    score = generate_population(
        config["diagnostic_n_score"],
        base + 900_002,
        config,
        attenuate_observed_severity=False,
    )
    fitted = fit_cost_model(train)
    score = score.copy()
    score["predicted_cost"] = fitted.predict(score)
    diagnostics = {
        "heldout_cost_correlation": float(score["predicted_cost"].corr(score["cost"])),
        "heldout_r2": float(
            fitted.model.score(
                score.loc[:, FEATURES].to_numpy(dtype=float),
                score["cost"].to_numpy(dtype=float),
            )
        ),
    }
    if diagnostics["heldout_cost_correlation"] <= 0:
        raise AssertionError("diagnostic model has non-positive held-out cost correlation")

    table_rows = [
        {
            "feature": "Intercept",
            "coefficient": fitted.intercept,
            "standard_error": fitted.intercept_standard_error,
            "correlation_with_group": np.nan,
        }
    ]
    for feature, coefficient, standard_error in zip(
        FEATURES, fitted.coefficients, fitted.standard_errors
    ):
        table_rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "standard_error": float(standard_error),
                "correlation_with_group": float(score[feature].corr(score["group"])),
            }
        )
    coefficient_table = pd.DataFrame(table_rows)
    return (
        train,
        score,
        fitted,
        coefficient_table,
        diagnostics | {"config_id": config_id(config)},
    )


def configure_matplotlib(config: Mapping[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": config["figures"]["font_family"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    figure_number: int,
    figures_dir: Path,
    config: Mapping[str, Any],
    caption: str,
) -> dict[str, Any]:
    current_config_id = config_id(config)
    stem = f"figure{figure_number}"
    fig.text(
        0.995,
        0.005,
        f"config {current_config_id}",
        ha="right",
        va="bottom",
        fontsize=5,
        color="#777777",
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(
        png_path,
        dpi=config["figures"]["png_dpi"],
        facecolor="white",
        metadata={"Title": stem, "Description": caption, "config_id": current_config_id},
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": stem,
            "Subject": f"{caption} Config {current_config_id}",
            "Creator": "simulation.py",
        },
    )
    plt.close(fig)
    return {
        "figure": figure_number,
        "png": str(png_path.relative_to(figures_dir.parent)),
        "pdf": str(pdf_path.relative_to(figures_dir.parent)),
        "caption": caption,
        "config_id": current_config_id,
    }


def create_grayscale_check(png_path: Path, destination: Path) -> None:
    image = mpimg.imread(png_path)
    rgb = image[..., :3]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(float) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if image.shape[-1] == 4:
        alpha = image[..., 3]
        luminance = luminance * alpha + (1.0 - alpha)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(destination, luminance, cmap="gray", vmin=0.0, vmax=1.0)


def _primary_moderate(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    return results[
        (results["condition"] == "primary")
        & np.isclose(results["scarcity"], config["moderate_scarcity"])
    ].copy()


def _spread_label_positions(values: Mapping[str, float], minimum_gap: float) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    positions: dict[str, float] = {}
    previous = -np.inf
    for label, value in ordered:
        position = max(value, previous + minimum_gap)
        positions[label] = position
        previous = position
    if ordered:
        raw_center = float(np.mean([value for _, value in ordered]))
        adjusted_center = float(np.mean(list(positions.values())))
        shift = raw_center - adjusted_center
        positions = {key: value + shift for key, value in positions.items()}
    return positions


def figure1_lives_at_moderate_scarcity(
    results: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    moderate = _primary_moderate(results, config)
    summary = percentile_summary(moderate, "lives_saved_per_1000", ["policy"])
    summary["policy"] = pd.Categorical(summary["policy"], POLICY_ORDER, ordered=True)
    summary = summary.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.7))
    y = np.arange(len(summary))
    for position, row in zip(y, summary.itertuples(index=False)):
        policy = str(row.policy)
        ax.errorbar(
            float(row.mean),
            position,
            xerr=np.array(
                [[float(row.mean - row.lower)], [float(row.upper - row.mean)]]
            ),
            fmt=POLICY_MARKERS[policy],
            color=POLICY_COLORS[policy],
            ecolor=POLICY_COLORS[policy],
            markeredgecolor="#222222",
            markeredgewidth=0.45,
            markersize=5.8,
            elinewidth=1.25,
            capsize=2.2,
            zorder=3,
        )
    ax.set_yticks(y, [str(policy) for policy in summary["policy"]])
    interval_span = float(summary["upper"].max() - summary["lower"].min())
    x_padding = max(1.5, interval_span * 0.025)
    ax.set_xlim(
        float(summary["lower"].min()) - x_padding,
        float(summary["upper"].max()) + x_padding,
    )
    ax.set_xlabel("Expected survivors\n(per 1,000 arrivals)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))

    means = summary.set_index("policy")["mean"]
    best = str(means.idxmax())
    learned = float(means.loc["Learned cost model"])
    advantage = 100.0 * (float(means.max()) - learned) / learned
    caption = (
        f"Figure 1. {best} yields {advantage:.1f}% more expected survivors than the "
        f"learned cost model when resources cover {config['moderate_scarcity']:.0%} of arrivals; "
        "points are means and horizontal lines are 95% Monte Carlo percentile intervals."
    )
    return save_figure(fig, 1, figures_dir, config, caption)


def paired_difference_from_policy(
    frame: pd.DataFrame,
    metric: str,
    reference_policy: str,
    output_column: str,
) -> pd.DataFrame:
    """Subtract a policy's outcome within the same draw and scarcity level."""
    pair_keys = [
        column for column in ("condition", "draw", "scarcity") if column in frame.columns
    ]
    if not pair_keys:
        raise ValueError("Paired policy differences require draw/scarcity identifiers")
    reference_column = "_reference_policy_value"
    reference = frame.loc[
        frame["policy"] == reference_policy, [*pair_keys, metric]
    ].rename(columns={metric: reference_column})
    if reference.empty:
        raise ValueError(f"Reference policy {reference_policy!r} is absent")
    if reference.duplicated(pair_keys).any():
        raise ValueError("Reference policy has duplicate rows within a pairing stratum")
    paired = frame.merge(
        reference,
        on=pair_keys,
        how="left",
        validate="many_to_one",
    )
    if paired[reference_column].isna().any():
        raise ValueError("At least one row has no matching reference-policy outcome")
    paired[output_column] = paired[metric] - paired[reference_column]
    return paired.drop(columns=reference_column)


def _policy_crossovers(summary: pd.DataFrame) -> list[tuple[float, float, str]]:
    policy_series = {
        policy: (
            summary[summary["policy"] == policy]
            .sort_values("scarcity")
            .set_index("scarcity")["mean"]
        )
        for policy in POLICY_ORDER
    }
    crossovers: list[tuple[float, float, str]] = []
    for first_policy, second_policy in combinations(POLICY_ORDER, 2):
        # Lottery and FCFS are statistically identical in this data-generating
        # process: arrival time is uniform and independent of every outcome.
        # Sign changes between their Monte Carlo means are therefore sampling
        # noise, not scarcity-dependent changes in policy performance.
        if {first_policy, second_policy} == {
            "Lottery",
            "First come, first served",
        }:
            continue
        first = policy_series[first_policy]
        second = policy_series[second_policy]
        common = first.index.intersection(second.index)
        x = common.to_numpy(dtype=float)
        difference = (first.loc[common] - second.loc[common]).to_numpy(dtype=float)
        for index in range(len(x) - 1):
            left, right = difference[index], difference[index + 1]
            if left == 0.0:
                crossover_x = x[index]
            elif left * right < 0.0:
                fraction = abs(left) / (abs(left) + abs(right))
                crossover_x = x[index] + fraction * (x[index + 1] - x[index])
            else:
                continue
            crossover_y = np.interp(crossover_x, x, first.loc[common].to_numpy(dtype=float))
            label = f"{first_policy} / {second_policy}"
            duplicate = any(
                abs(crossover_x - existing[0]) < 0.004
                and abs(crossover_y - existing[1]) < 0.5
                for existing in crossovers
            )
            if not duplicate:
                crossovers.append((float(crossover_x), float(crossover_y), label))
    return crossovers


def figure2_scarcity_lines(
    results: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    primary = results[
        (results["condition"] == "primary") & (results["scarcity"] < 1.0)
    ].copy()
    difference_column = "survivors_above_lottery_per_1000"
    relative = paired_difference_from_policy(
        primary,
        "lives_saved_per_1000",
        reference_policy="Lottery",
        output_column=difference_column,
    )
    summary = percentile_summary(
        relative, difference_column, ["scarcity", "policy"]
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
    endpoint_values: dict[str, float] = {}
    x_endpoint = float(summary["scarcity"].max())
    for policy in POLICY_ORDER:
        policy_data = summary[summary["policy"] == policy].sort_values("scarcity")
        x = policy_data["scarcity"].to_numpy(dtype=float)
        mean = policy_data["mean"].to_numpy(dtype=float)
        lower = policy_data["lower"].to_numpy(dtype=float)
        upper = policy_data["upper"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES[policy],
            marker=POLICY_MARKERS[policy],
            markersize=3.4,
            linewidth=1.4 if policy != "Learned cost model" else 2.0,
            markevery=2,
            zorder=3 if policy == "Learned cost model" else 2,
        )
        ax.fill_between(x, lower, upper, color=POLICY_COLORS[policy], alpha=0.10, linewidth=0)
        endpoint_values[policy] = float(mean[-1])

    y_range = float(summary["upper"].max() - summary["lower"].min())
    label_positions = _spread_label_positions(endpoint_values, max(2.0, y_range * 0.04))
    for policy in POLICY_ORDER:
        ax.plot(
            [x_endpoint, x_endpoint + 0.012],
            [endpoint_values[policy], label_positions[policy]],
            color=POLICY_COLORS[policy],
            linewidth=0.6,
            clip_on=False,
        )
        ax.text(
            x_endpoint + 0.016,
            label_positions[policy],
            policy,
            color=POLICY_COLORS[policy],
            va="center",
            fontsize=8,
            clip_on=False,
        )

    crossovers = _policy_crossovers(summary)
    for crossover_x, crossover_y, _ in crossovers:
        ax.scatter(
            crossover_x,
            crossover_y,
            marker="x",
            s=24,
            linewidth=0.9,
            color="#333333",
            zorder=5,
        )
    learned_sickest_crossovers = sorted(
        [
            crossover
            for crossover in crossovers
            if "Sickest first" in crossover[2]
            and "Learned cost model" in crossover[2]
        ],
        key=lambda value: value[0],
    )
    annotation_offsets = [(-0.12, 7.0), (0.055, -8.0)]
    annotation_labels = ["Learned / sickest", "Crosses back"]
    for index, crossover in enumerate(learned_sickest_crossovers[:2]):
        dx, dy = annotation_offsets[index]
        ax.annotate(
            f"{annotation_labels[index]}\n{crossover[0]:.0%} resources",
            xy=(crossover[0], crossover[1]),
            xytext=(crossover[0] + dx, crossover[1] + dy),
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6},
            fontsize=7.5,
            color="#555555",
        )
    y_padding = max(1.0, y_range * 0.025)
    ax.set_ylim(
        min(float(summary["lower"].min()), min(label_positions.values())) - y_padding,
        max(float(summary["upper"].max()), max(label_positions.values())) + y_padding,
    )
    ax.set_xlim(float(summary["scarcity"].min()), x_endpoint)
    ax.set_xlabel("Resources available (share of arrivals)")
    ax.set_ylabel("Expected survivors above lottery\n(per 1,000 arrivals)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.70, bottom=0.17, top=0.96)
    learned_crossing_text = " and ".join(
        f"{crossover[0]:.0%}" for crossover in learned_sickest_crossovers
    )
    caption = (
        f"Figure 2. Subtracting each draw's lottery outcome reveals {len(crossovers)} substantive "
        f"crossover location{'s' if len(crossovers) != 1 else ''}; the learned cost model crosses "
        f"sickest-first near {learned_crossing_text}. Lottery is zero by construction, and "
        "lottery–first-come sign flips are treated as sampling noise between statistically "
        "identical policies. Bands are 95% percentile intervals for paired within-draw differences."
    )
    return save_figure(fig, 2, figures_dir, config, caption)


def figure3_lives_life_years(
    results: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    moderate = _primary_moderate(results, config)
    means = moderate.groupby("policy")[["lives_saved_per_1000", "life_years_saved_per_1000"]].mean()
    reference = means.loc["Lottery"]
    indexed = 100.0 * means / reference
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    low = float(indexed.min().min()) - 0.8
    high = float(indexed.max().max()) + 0.8
    ax.plot([low, high], [low, high], color="#BBBBBB", linewidth=0.8, linestyle="--", zorder=0)
    offsets = {
        "Sickest first": (5, 4),
        "Save the most lives": (7, 6),
        "Youngest first": (5, 5),
        "Life-years": (5, 5),
        "Learned cost model": (-6, 10),
    }
    horizontal_alignment = {
        policy: ("right" if policy == "Learned cost model" else "left")
        for policy in POLICY_ORDER
    }
    coincident_policies = ("Lottery", "First come, first served")
    coincident_x = float(
        indexed.loc[list(coincident_policies), "lives_saved_per_1000"].mean()
    )
    coincident_y = float(
        indexed.loc[list(coincident_policies), "life_years_saved_per_1000"].mean()
    )
    ax.scatter(
        coincident_x,
        coincident_y,
        color=POLICY_COLORS["Lottery"],
        edgecolor=POLICY_COLORS["First come, first served"],
        linewidth=0.9,
        marker="o",
        s=46,
        zorder=3,
    )
    ax.annotate(
        "Lottery and first come,\nfirst served",
        (coincident_x, coincident_y),
        xytext=(5, -3),
        textcoords="offset points",
        fontsize=7.2,
        color="#333333",
        ha="left",
        va="top",
    )
    for policy in POLICY_ORDER:
        if policy in coincident_policies:
            continue
        x = float(indexed.loc[policy, "lives_saved_per_1000"])
        y = float(indexed.loc[policy, "life_years_saved_per_1000"])
        ax.scatter(
            x,
            y,
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            s=42,
            zorder=3,
        )
        ax.annotate(
            policy,
            (x, y),
            xytext=offsets[policy],
            textcoords="offset points",
            fontsize=7.2,
            color=POLICY_COLORS[policy],
            ha=horizontal_alignment[policy],
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Expected survivors (lottery = 100)")
    ax.set_ylabel("Expected survivor-years (lottery = 100)")
    ax.grid(color="#E2E2E2", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    correlation = float(indexed.iloc[:, 0].corr(indexed.iloc[:, 1]))
    if correlation >= 0.70:
        finding = "largely move together"
    elif correlation <= -0.20:
        finding = "trade off across the policies"
    else:
        finding = "show little policy-level association"
    caption = (
        f"Figure 3. Expected lives and survivor-years {finding} at moderate scarcity "
        f"(policy-level r = {correlation:.2f}); the dashed diagonal marks equal proportional "
        "change from lottery. Lottery and first come, first served are shown as one point because "
        "they are statistically equivalent and visually coincident."
    )
    return save_figure(fig, 3, figures_dir, config, caption)


def figure4_group_outcomes(
    results: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    moderate = _primary_moderate(results, config)
    columns = [
        "allocation_rate_advantaged",
        "allocation_rate_disadvantaged",
        "survival_rate_advantaged",
        "survival_rate_disadvantaged",
    ]
    means = moderate.groupby("policy")[columns].mean().reindex(POLICY_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.35), sharey=True)
    y = np.arange(len(POLICY_ORDER))[::-1]
    panels = (
        ("allocation_rate_advantaged", "allocation_rate_disadvantaged", "Allocation rate (%)"),
        ("survival_rate_advantaged", "survival_rate_disadvantaged", "Expected survival rate (%)"),
    )
    axes[0].axvline(
        100.0 * config["moderate_scarcity"],
        color="#AAAAAA",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )
    for ax, (adv_col, disadv_col, label) in zip(axes, panels):
        for position, policy in zip(y, POLICY_ORDER):
            advantaged = 100.0 * float(means.loc[policy, adv_col])
            disadvantaged = 100.0 * float(means.loc[policy, disadv_col])
            ax.plot(
                [advantaged, disadvantaged],
                [position, position],
                color=POLICY_COLORS[policy],
                linewidth=1.5,
                alpha=0.85,
            )
            ax.scatter(
                advantaged,
                position,
                color=POLICY_COLORS[policy],
                marker="o",
                s=34,
                zorder=3,
            )
            ax.scatter(
                disadvantaged,
                position,
                facecolor="white",
                edgecolor=POLICY_COLORS[policy],
                marker="D",
                linewidth=1.2,
                s=34,
                zorder=3,
            )
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, POLICY_ORDER)
    axes[0].text(
        0.0,
        1.04,
        "● Advantaged     ◇ Disadvantaged",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.98), w_pad=2.0)
    learned_allocation_gap = 100.0 * (
        means.loc["Learned cost model", "allocation_rate_disadvantaged"]
        - means.loc["Learned cost model", "allocation_rate_advantaged"]
    )
    learned_survival_gap = 100.0 * (
        means.loc["Learned cost model", "survival_rate_disadvantaged"]
        - means.loc["Learned cost model", "survival_rate_advantaged"]
    )
    caption = (
        f"Figure 4. At moderate scarcity, the learned cost model's disadvantaged-minus-advantaged "
        f"gap is {learned_allocation_gap:+.1f} percentage points in allocation and "
        f"{learned_survival_gap:+.1f} points in expected survival. Marker shapes retain the group "
        f"distinction in grayscale; the allocation panel's {config['moderate_scarcity']:.0%} "
        "reference line marks equal allocation rates."
    )
    return save_figure(fig, 4, figures_dir, config, caption)


def diagnostic_group_gap(score: pd.DataFrame) -> dict[str, Any]:
    coefficients: dict[int, np.ndarray] = {}
    for group_value in (0, 1):
        group_frame = score[score["group"] == group_value]
        coefficients[group_value] = np.polyfit(
            group_frame["true_need"], group_frame["predicted_cost"], 1
        )
    need_reference = float(score["true_need"].median())
    prediction_at_reference = {
        group_value: float(np.polyval(coefficients[group_value], need_reference))
        for group_value in (0, 1)
    }
    return {
        "line_coefficients": coefficients,
        "need_reference": need_reference,
        "advantaged_prediction": prediction_at_reference[0],
        "disadvantaged_prediction": prediction_at_reference[1],
        "advantaged_minus_disadvantaged_dollars": (
            prediction_at_reference[0] - prediction_at_reference[1]
        ),
    }


def figure5_priority_vs_need(
    score: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gap = diagnostic_group_gap(score)
    rng = np.random.default_rng(config["base_seed"] + 900_003)
    requested = min(config["figures"]["diagnostic_scatter_points"], len(score))
    sample_indices = rng.choice(len(score), size=requested, replace=False)
    sample = score.iloc[sample_indices]

    fig = plt.figure(figsize=(7.0, 5.25))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(5.2, 1.0),
        height_ratios=(1.0, 4.7),
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    group_styles = {
        0: ("#0072B2", "o"),
        1: ("#D55E00", "^"),
    }
    x_line = np.linspace(score["true_need"].quantile(0.01), score["true_need"].quantile(0.99), 100)
    for group_value in (0, 1):
        color, marker = group_styles[group_value]
        group_sample = sample[sample["group"] == group_value]
        group_full = score[score["group"] == group_value]
        ax.scatter(
            group_sample["true_need"],
            group_sample["predicted_cost"],
            color=color,
            marker=marker,
            s=9,
            alpha=0.22,
            linewidths=0,
        )
        y_line = np.polyval(gap["line_coefficients"][group_value], x_line)
        ax.plot(x_line, y_line, color=color, linewidth=2.2)
        ax.text(
            x_line[-1],
            y_line[-1],
            GROUP_LABELS[group_value],
            color=color,
            fontsize=8,
            ha="left",
            va="center",
            clip_on=False,
        )
        ax_top.hist(
            group_full["true_need"],
            bins=35,
            density=True,
            histtype="step",
            color=color,
            linewidth=1.2,
        )
        ax_right.hist(
            group_full["predicted_cost"],
            bins=35,
            density=True,
            histtype="step",
            orientation="horizontal",
            color=color,
            linewidth=1.2,
        )

    x_reference = gap["need_reference"]
    y_advantaged = gap["advantaged_prediction"]
    y_disadvantaged = gap["disadvantaged_prediction"]
    ax.axvline(x_reference, color="#666666", linewidth=0.95, linestyle="--", zorder=0)
    ax.annotate(
        "",
        xy=(x_reference, y_disadvantaged),
        xytext=(x_reference, y_advantaged),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.0},
    )
    direction = "lower" if gap["advantaged_minus_disadvantaged_dollars"] >= 0 else "higher"
    ax.text(
        x_reference + 0.015,
        (y_advantaged + y_disadvantaged) / 2,
        "At equal need,\n"
        f"disadvantaged group scored\n${abs(gap['advantaged_minus_disadvantaged_dollars']):,.0f} {direction}",
        fontsize=8,
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.text(
        0.02,
        0.97,
        "Model received neither\ngroup nor true need",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.set_xlabel("Latent true need (evaluation only, 0–1.2 scale)")
    ax.set_ylabel("Learned priority score (predicted annual cost, $)")
    ax.grid(color="#E2E2E2", linewidth=0.5)
    ax.set_axisbelow(True)
    ax_top.tick_params(labelbottom=False, left=False, labelleft=False)
    ax_top.spines["left"].set_visible(False)
    ax_top.set_ylabel("Density", fontsize=7)
    ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.set_xlabel("Density", fontsize=7)
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.11, top=0.97)
    caption = (
        f"Figure 5. At the median latent need ({x_reference:.2f}), the learned policy scores a "
        f"disadvantaged patient ${abs(gap['advantaged_minus_disadvantaged_dollars']):,.0f} "
        f"{direction} on fitted group lines, although the model received neither group membership "
        "nor true need; marginal strips show the generated distributions."
    )
    manifest = save_figure(fig, 5, figures_dir, config, caption)
    serializable_gap = {key: value for key, value in gap.items() if key != "line_coefficients"}
    return manifest, serializable_gap


def figure6_severity_robustness(
    results: pd.DataFrame, figures_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    if "attenuated_severity_robustness" not in set(results["condition"]):
        return None
    moderate = results[np.isclose(results["scarcity"], config["moderate_scarcity"])]
    means = (
        moderate.groupby(["condition", "policy"])["allocation_gap_disadv_minus_adv_pp"]
        .mean()
        .unstack("condition")
        .reindex(POLICY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(POLICY_ORDER))[::-1]
    for position, policy in zip(y, POLICY_ORDER):
        primary = float(means.loc[policy, "primary"])
        robust = float(means.loc[policy, "attenuated_severity_robustness"])
        ax.plot(
            [primary, robust],
            [position, position],
            color=POLICY_COLORS[policy],
            linewidth=1.4,
            alpha=0.9,
        )
        ax.scatter(primary, position, color=POLICY_COLORS[policy], marker="o", s=32, zorder=3)
        ax.scatter(
            robust,
            position,
            facecolor="white",
            edgecolor=POLICY_COLORS[policy],
            marker="^",
            linewidth=1.2,
            s=38,
            zorder=3,
        )
    ax.axvline(0.0, color="#888888", linewidth=0.7)
    ax.set_yticks(y, POLICY_ORDER)
    ax.set_xlabel(
        "Allocation gap: disadvantaged − advantaged\n(percentage points)"
    )
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        1.04,
        "● Clean severity     △ Attenuated severity",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.98))
    sickest_change = float(
        means.loc["Sickest first", "attenuated_severity_robustness"]
        - means.loc["Sickest first", "primary"]
    )
    learned_change = float(
        means.loc["Learned cost model", "attenuated_severity_robustness"]
        - means.loc["Learned cost model", "primary"]
    )
    caption = (
        f"Figure 6. Attenuating the observed severity score changes the disadvantaged-minus-"
        f"advantaged allocation gap by {sickest_change:+.1f} points for sickest-first and "
        f"{learned_change:+.1f} points for the learned policy. Survival probabilities remain "
        "functions of clinical severity, so benefit-based policies are unchanged by construction."
    )
    return save_figure(fig, 6, figures_dir, config, caption)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_coefficient_table(
    coefficient_table: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    current_config_id = config_id(config)
    table = coefficient_table.copy()
    table.insert(0, "config_id", current_config_id)
    table.to_csv(output_dir / "table1_coefficients.csv", index=False, float_format="%.10g")
    rows = []
    for record in coefficient_table.itertuples(index=False):
        correlation = (
            "—"
            if pd.isna(record.correlation_with_group)
            else f"{record.correlation_with_group:+.3f}"
        )
        rows.append(
            (
                str(record.feature).replace("_", " ").title(),
                f"{record.coefficient:,.2f}",
                f"{record.standard_error:,.2f}",
                correlation,
            )
        )
    text = (
        "# Table 1. Fitted cost-model coefficients\n\n"
        f"Config ID: `{current_config_id}`. Coefficients are in raw feature units; standard "
        "errors are conventional OLS standard errors from the fixed diagnostic training "
        "population. Correlations with group use 0 = advantaged and 1 = disadvantaged in "
        "the independent diagnostic score population. Group and latent true need were not "
        "model features.\n\n"
        + markdown_table(
            ["Feature", "Coefficient ($)", "Standard error", "Correlation with group"], rows
        )
        + "\n"
    )
    (output_dir / "table1_coefficients.md").write_text(text, encoding="utf-8")


def _interval_rows(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    group: str = "policy",
) -> pd.DataFrame:
    aggregations: dict[str, Any] = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_lower"] = (
            metric,
            lambda values: np.percentile(values, 2.5),
        )
        aggregations[f"{metric}_upper"] = (
            metric,
            lambda values: np.percentile(values, 97.5),
        )
    return frame.groupby(group, sort=False).agg(**aggregations).reindex(POLICY_ORDER).reset_index()


def write_numbers(
    results: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    diagnostic_gap: Mapping[str, Any],
    sanity: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    current_config_id = config_id(config)
    moderate = _primary_moderate(results, config)
    outcome_summary = _interval_rows(
        moderate,
        ["lives_saved_per_1000", "life_years_saved_per_1000", "realized_survivors_per_1000"],
    )
    group_means = (
        moderate.groupby("policy")
        .agg(
            allocation_advantaged=("allocation_rate_advantaged", "mean"),
            allocation_disadvantaged=("allocation_rate_disadvantaged", "mean"),
            allocation_gap_pp=("allocation_gap_disadv_minus_adv_pp", "mean"),
            survival_advantaged=("survival_rate_advantaged", "mean"),
            survival_disadvantaged=("survival_rate_disadvantaged", "mean"),
            survival_gap_pp=("survival_gap_disadv_minus_adv_pp", "mean"),
            need_priority_gap_pp=("need_priority_gap_pp", "mean"),
        )
        .reindex(POLICY_ORDER)
        .reset_index()
    )
    scarcity_summary = percentile_summary(
        results[(results["condition"] == "primary") & (results["scarcity"] < 1.0)],
        "lives_saved_per_1000",
        ["scarcity", "policy"],
    )
    scarcity_primary = results[
        (results["condition"] == "primary") & (results["scarcity"] < 1.0)
    ]
    paired_scarcity = paired_difference_from_policy(
        scarcity_primary,
        "lives_saved_per_1000",
        reference_policy="Lottery",
        output_column="survivors_above_lottery_per_1000",
    )
    paired_scarcity_summary = percentile_summary(
        paired_scarcity,
        "survivors_above_lottery_per_1000",
        ["scarcity", "policy"],
    )
    learned_sickest_crossovers = sorted(
        [
            crossover
            for crossover in _policy_crossovers(paired_scarcity_summary)
            if "Sickest first" in crossover[2]
            and "Learned cost model" in crossover[2]
        ],
        key=lambda value: value[0],
    )

    sections: list[str] = [
        "# Paper-ready numbers",
        "",
        f"Config ID: `{current_config_id}`. Values below were generated, not transcribed from plots. ",
        "Intervals are 2.5th and 97.5th percentiles across Monte Carlo draws. All outcomes are "
        "illustrative rather than estimates of real populations.",
        "",
        "## Fixed modelling choices",
        "",
        f"- Seed: `{config['base_seed']}`.",
        f"- Draws per condition: {config['n_draws']:,}.",
        f"- Training/scored patients per draw: {config['n_train']:,}/{config['n_score']:,}.",
        f"- Disadvantaged population probability: {config['p_disadvantaged']:.0%}.",
        f"- Structural severity penalty: {config['severity']['structural_penalty']:.1f} points on a 0–24 scale.",
        f"- Latent-need weights: {config['true_need']['severity_weight']:.0%} normalized severity and {config['true_need']['benefit_weight']:.0%} normalized treatment benefit.",
        f"- Access attenuation: {config['access']['attenuation_delta']:.0%} for both prior utilisation and cost.",
        f"- Robustness attenuation of the observed severity score: {config['severity']['robustness_observed_attenuation']:.0%}.",
        f"- Moderate-scarcity headline: resources for {config['moderate_scarcity']:.0%} of arrivals.",
        "",
        "## Primary outcomes at moderate scarcity",
        "",
        markdown_table(
            [
                "Policy",
                "Expected survivors / 1,000 (95% interval)",
                "Expected survivor-years / 1,000 (95% interval)",
                "Realized survivors / 1,000 (95% interval)",
            ],
            (
                (
                    row.policy,
                    f"{row.lives_saved_per_1000_mean:.1f} ({row.lives_saved_per_1000_lower:.1f}–{row.lives_saved_per_1000_upper:.1f})",
                    f"{row.life_years_saved_per_1000_mean:,.0f} ({row.life_years_saved_per_1000_lower:,.0f}–{row.life_years_saved_per_1000_upper:,.0f})",
                    f"{row.realized_survivors_per_1000_mean:.1f} ({row.realized_survivors_per_1000_lower:.1f}–{row.realized_survivors_per_1000_upper:.1f})",
                )
                for row in outcome_summary.itertuples(index=False)
            ),
        ),
        "",
        "## Primary group outcomes at moderate scarcity",
        "",
        "Gaps are disadvantaged minus advantaged except `need priority gap`, where a positive value means the disadvantaged group is ranked lower within matched latent-need strata.",
        "",
        markdown_table(
            [
                "Policy",
                "Allocation A / D (%)",
                "Allocation gap (pp)",
                "Expected survival A / D (%)",
                "Survival gap (pp)",
                "Need-priority gap (pp)",
            ],
            (
                (
                    row.policy,
                    f"{100 * row.allocation_advantaged:.1f} / {100 * row.allocation_disadvantaged:.1f}",
                    f"{row.allocation_gap_pp:+.1f}",
                    f"{100 * row.survival_advantaged:.1f} / {100 * row.survival_disadvantaged:.1f}",
                    f"{row.survival_gap_pp:+.1f}",
                    f"{row.need_priority_gap_pp:+.1f}",
                )
                for row in group_means.itertuples(index=False)
            ),
        ),
        "",
        "## Expected survivors across scarcity levels",
        "",
        markdown_table(
            ["Resources", *POLICY_ORDER],
            (
                (
                    f"{scarcity:.0%}",
                    *(
                        f"{scarcity_summary[(scarcity_summary['scarcity'] == scarcity) & (scarcity_summary['policy'] == policy)]['mean'].iloc[0]:.1f}"
                        for policy in POLICY_ORDER
                    ),
                )
                for scarcity in sorted(scarcity_summary["scarcity"].unique())
            ),
        ),
        "",
        "## Expected survivors above lottery across scarcity levels",
        "",
        "These are paired differences: lottery is subtracted within the same generated population, draw, and scarcity level before means or intervals are calculated.",
        "",
        markdown_table(
            ["Resources", *POLICY_ORDER],
            (
                (
                    f"{scarcity:.0%}",
                    *(
                        f"{paired_scarcity_summary[(paired_scarcity_summary['scarcity'] == scarcity) & (paired_scarcity_summary['policy'] == policy)]['mean'].iloc[0]:+.1f}"
                        for policy in POLICY_ORDER
                    ),
                )
                for scarcity in sorted(paired_scarcity_summary["scarcity"].unique())
            ),
        ),
        "",
        "- Learned-cost-model / sickest-first crossover estimates: "
        + ", ".join(f"{crossover[0]:.1%}" for crossover in learned_sickest_crossovers)
        + " of arrivals supplied.",
        "",
        "## Learned-policy diagnostic",
        "",
        f"- Held-out predicted-cost/cost correlation: {diagnostics['heldout_cost_correlation']:.3f}.",
        f"- Held-out cost-model R²: {diagnostics['heldout_r2']:.3f}.",
        f"- Reference latent need in Figure 5: {diagnostic_gap['need_reference']:.3f}.",
        f"- Fitted priority at that need, advantaged/disadvantaged: ${diagnostic_gap['advantaged_prediction']:,.0f} / ${diagnostic_gap['disadvantaged_prediction']:,.0f}.",
        f"- At equal need, advantaged-minus-disadvantaged predicted priority: ${diagnostic_gap['advantaged_minus_disadvantaged_dollars']:,.0f}.",
        "- The learned model received neither group membership nor latent true need.",
        "",
        "## Coefficient channel",
        "",
        markdown_table(
            ["Feature", "Coefficient", "Standard error", "Correlation with group"],
            (
                (
                    str(row.feature),
                    f"{row.coefficient:,.2f}",
                    f"{row.standard_error:,.2f}",
                    "—" if pd.isna(row.correlation_with_group) else f"{row.correlation_with_group:+.3f}",
                )
                for row in coefficient_table.itertuples(index=False)
            ),
        ),
    ]

    if "attenuated_severity_robustness" in set(results["condition"]):
        robust = results[
            np.isclose(results["scarcity"], config["moderate_scarcity"])
        ]
        robust_means = (
            robust.groupby(["policy", "condition"])["allocation_gap_disadv_minus_adv_pp"]
            .mean()
            .unstack("condition")
            .reindex(POLICY_ORDER)
        )
        sections.extend(
            [
                "",
                "## Attenuated-severity robustness condition",
                "",
                "This condition attenuates only the observed severity feature. Treated and untreated survival probabilities remain functions of clinical severity, so save-the-most-lives and life-years are unchanged by construction; this is a narrow measurement robustness check, not a fully biased outcome-model scenario.",
                "",
                markdown_table(
                    ["Policy", "Clean severity gap (pp)", "Attenuated severity gap (pp)", "Change (pp)"],
                    (
                        (
                            policy,
                            f"{robust_means.loc[policy, 'primary']:+.1f}",
                            f"{robust_means.loc[policy, 'attenuated_severity_robustness']:+.1f}",
                            f"{robust_means.loc[policy, 'attenuated_severity_robustness'] - robust_means.loc[policy, 'primary']:+.1f}",
                        )
                        for policy in POLICY_ORDER
                    ),
                ),
            ]
        )

    sections.extend(
        [
            "",
            "## Sanity checks",
            "",
            f"- Protected/evaluation columns absent from X: passed by assertion before every fit (`{', '.join(FEATURES)}` only).",
            "- Untreated survival below treated survival for every generated patient: passed by assertion.",
            f"- Lottery group-balance check: passed; maximum aggregate absolute selected-share error {sanity['lottery_max_absolute_share_error_pp']:.2f} percentage points.",
            "- All policies identical at full allocation: passed.",
            "- Save-the-most-lives never below lottery: passed.",
            f"- Minimum held-out predicted-cost correlation across fitted draws: {sanity['minimum_heldout_cost_correlation']:.3f} (positive check passed).",
            f"- Expected versus Bernoulli policy-rank Spearman correlation at moderate scarcity: {sanity['expected_vs_bernoulli_rank_spearman']:.3f} (stability check passed).",
            "",
        ]
    )
    (output_dir / "numbers.md").write_text("\n".join(sections), encoding="utf-8")


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_readme(
    output_dir: Path,
    config: Mapping[str, Any],
    runtime_seconds: float,
    sanity: Mapping[str, Any],
) -> None:
    versions = library_versions()
    current_config_id = config_id(config)
    text = f"""# Who Should the Algorithm Save? — allocation simulation

This repository is a seeded, illustrative Monte Carlo experiment. It does not estimate any real population. The learned policy is ordinary unweighted linear regression trained to predict generated cost from age, observed severity, treated and untreated survival probabilities, and prior utilisation. It never receives group membership or latent true need.

## How to run

Python 3.12 is recommended. From the repository root, create a virtual
environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\\Scripts\\Activate.ps1` instead of the `source` command.

Run the complete simulation:

```bash
python simulation.py
```

The command regenerates `results.csv`, `config.json`, `run_metadata.json`,
`numbers.md`, the coefficient tables, this README, and all files under
`figures/`. To keep the checked-in results unchanged, write a run to another
directory:

```bash
python simulation.py --output-dir outputs/my-run
```

For a smaller smoke run, use:

```bash
python simulation.py --output-dir outputs/smoke --draws 2 --train-size 100 --score-size 100 --primary-only
```

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

See all command-line options with `python simulation.py --help`.

The final run used seed `{config['base_seed']}`, config ID `{current_config_id}`, {config['n_draws']} draws per condition, {config['n_train']:,} training patients per draw, and {config['n_score']:,} scored patients per draw. Runtime on the generating machine was {runtime_seconds:.1f} seconds.

## Outputs

- `results.csv`: complete run-level results.
- `numbers.md`: paper-ready source of every reported quantity.
- `config.json`: exact parameters used.
- `table1_coefficients.md` and `.csv`: fitted model table.
- `figures/figure1` through `figure6`: 300 dpi PNG and PDF.
- `figures/grayscale_checks/`: automatic grayscale conversions for print review.
- `figures/captions.md` and `figures/manifest.json`: result-driven captions and config provenance.

All executable sanity checks passed. Expected and Bernoulli policy rankings had Spearman correlation {sanity['expected_vs_bernoulli_rank_spearman']:.3f} at moderate scarcity.

## Scope of the severity robustness condition

The robustness condition attenuates the disadvantaged group's observed severity score only. Survival probabilities continue to use clinical severity, so policies sorting on treatment benefit are unchanged by construction. It isolates the observed-severity measurement channel rather than simulating a fully biased survival model.

## Library versions

{markdown_table(['Library', 'Version'], ((name, version) for name, version in versions.items()))}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_figure_metadata(manifests: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    current_config_id = manifests[0]["config_id"] if manifests else "unknown"
    captions = ["# Figure captions", "", f"Config ID: `{current_config_id}`.", ""]
    for manifest in manifests:
        captions.extend([manifest["caption"], ""])
    (figures_dir / "captions.md").write_text("\n".join(captions), encoding="utf-8")
    (figures_dir / "manifest.json").write_text(
        json.dumps({"config_id": current_config_id, "figures": list(manifests)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolved_config(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(config)
    current_config_id = config_id(config)

    (output_dir / "config.json").write_text(
        json.dumps({"config_id": current_config_id, "parameters": config}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Config {current_config_id}; seed {config['base_seed']}", flush=True)
    start = time.perf_counter()
    results = run_monte_carlo(config)
    sanity = aggregate_sanity_checks(results, config)
    results.to_csv(output_dir / "results.csv", index=False, float_format="%.10g")

    _, diagnostic_score, _, coefficient_table, diagnostics = diagnostic_analysis(config)
    write_coefficient_table(coefficient_table, output_dir, config)

    print("Rendering figures", flush=True)
    manifests: list[dict[str, Any]] = []
    manifests.append(figure1_lives_at_moderate_scarcity(results, figures_dir, config))
    manifests.append(figure2_scarcity_lines(results, figures_dir, config))
    manifests.append(figure3_lives_life_years(results, figures_dir, config))
    manifests.append(figure4_group_outcomes(results, figures_dir, config))
    figure5_manifest, diagnostic_gap = figure5_priority_vs_need(
        diagnostic_score, figures_dir, config
    )
    manifests.append(figure5_manifest)
    figure6_manifest = figure6_severity_robustness(results, figures_dir, config)
    if figure6_manifest is not None:
        manifests.append(figure6_manifest)
    write_figure_metadata(manifests, figures_dir)

    grayscale_dir = figures_dir / "grayscale_checks"
    for manifest in manifests:
        png_path = output_dir / manifest["png"]
        create_grayscale_check(png_path, grayscale_dir / png_path.name)

    write_numbers(
        results,
        coefficient_table,
        diagnostics,
        diagnostic_gap,
        sanity,
        output_dir,
        config,
    )
    runtime_seconds = time.perf_counter() - start
    write_readme(output_dir, config, runtime_seconds, sanity)
    run_metadata = {
        "config_id": current_config_id,
        "runtime_seconds": runtime_seconds,
        "versions": library_versions(),
        "sanity_checks": sanity,
        "rows_in_results_csv": len(results),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Complete: {len(results):,} run-level rows, {len(manifests)} figures, "
        f"{runtime_seconds:.1f} seconds",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
