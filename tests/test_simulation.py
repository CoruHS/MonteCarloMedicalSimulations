from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

import simulation


class AllocationSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(simulation.CONFIG)

    def test_population_is_seeded_and_survival_constraint_holds(self) -> None:
        first = simulation.generate_population(
            800, 12345, self.config, attenuate_observed_severity=False
        )
        second = simulation.generate_population(
            800, 12345, self.config, attenuate_observed_severity=False
        )
        self.assertTrue(first.equals(second))
        self.assertTrue(
            np.all(first["p_survive_untreated"] < first["p_survive_treated"])
        )
        self.assertTrue(np.allclose(first["severity"], first["clinical_severity"]))

    def test_robustness_changes_only_observed_severity_channel(self) -> None:
        primary = simulation.generate_population(
            1000, 222, self.config, attenuate_observed_severity=False
        )
        robust = simulation.generate_population(
            1000, 222, self.config, attenuate_observed_severity=True
        )
        invariant_columns = [
            "group",
            "age",
            "clinical_severity",
            "true_need",
            "cost",
            "prior_utilisation",
            "benefit",
        ]
        for column in invariant_columns:
            self.assertTrue(np.allclose(primary[column], robust[column]))
        disadvantaged = primary["group"].to_numpy() == 1
        self.assertTrue(
            np.all(
                robust.loc[disadvantaged, "severity"]
                <= primary.loc[disadvantaged, "severity"]
            )
        )

    def test_protected_and_evaluation_columns_are_absent_from_model(self) -> None:
        self.assertNotIn("group", simulation.FEATURES)
        self.assertNotIn("true_need", simulation.FEATURES)
        self.assertNotIn("cost", simulation.FEATURES)
        population = simulation.generate_population(
            500, 333, self.config, attenuate_observed_severity=False
        )
        simulation.assert_model_features(population)

    def test_standard_model_learns_heldout_cost_signal(self) -> None:
        train = simulation.generate_population(
            1500, 444, self.config, attenuate_observed_severity=False
        )
        score = simulation.generate_population(
            700, 445, self.config, attenuate_observed_severity=False
        )
        model = simulation.fit_cost_model(train)
        predictions = model.predict(score)
        self.assertGreater(np.corrcoef(predictions, score["cost"])[0, 1], 0.5)

    def test_full_allocation_is_policy_invariant(self) -> None:
        population = simulation.generate_population(
            600, 555, self.config, attenuate_observed_severity=False
        )
        model = simulation.fit_cost_model(population.iloc[:400])
        score = population.iloc[400:].reset_index(drop=True)
        priorities = simulation.policy_priority_scores(score, model.predict(score))
        metrics = []
        for policy in simulation.POLICY_ORDER:
            order = simulation.priority_order(
                priorities[policy], score["tie_breaker"].to_numpy()
            )
            metrics.append(
                simulation.allocation_metrics(
                    score,
                    order,
                    1.0,
                    policy=policy,
                    condition="primary",
                    draw=0,
                    train_seed=1,
                    score_seed=2,
                    model_train_r2=model.train_r2,
                    heldout_cost_correlation=0.5,
                    need_priority_gap_pp=0.0,
                    current_config_id="test",
                )
            )
        lives = [row["lives_saved"] for row in metrics]
        self.assertAlmostEqual(max(lives), min(lives), places=10)

    def test_save_most_lives_weakly_beats_lottery(self) -> None:
        population = simulation.generate_population(
            1000, 666, self.config, attenuate_observed_severity=False
        )
        model = simulation.fit_cost_model(population.iloc[:600])
        score = population.iloc[600:].reset_index(drop=True)
        priorities = simulation.policy_priority_scores(score, model.predict(score))
        expected = {}
        for policy in ("Lottery", "Save the most lives"):
            order = simulation.priority_order(
                priorities[policy], score["tie_breaker"].to_numpy()
            )
            treated = np.zeros(len(score), dtype=bool)
            treated[order[: len(score) // 2]] = True
            expected[policy] = np.where(
                treated,
                score["p_survive_treated"],
                score["p_survive_untreated"],
            ).sum()
        self.assertGreaterEqual(expected["Save the most lives"], expected["Lottery"])

    def test_crossover_count_excludes_equivalent_lottery_fcfs_noise(self) -> None:
        scarcity = [0.1, 0.2, 0.3]
        values = {
            "Lottery": [100.0, 101.0, 100.0],
            "First come, first served": [101.0, 100.0, 101.0],
            "Sickest first": [110.0, 112.0, 114.0],
            "Save the most lives": [120.0, 122.0, 124.0],
            "Youngest first": [90.0, 92.0, 94.0],
            "Life-years": [109.0, 113.0, 115.0],
            "Learned cost model": [115.0, 116.0, 117.0],
        }
        summary = pd.DataFrame(
            [
                {"scarcity": level, "policy": policy, "mean": mean}
                for policy, means in values.items()
                for level, mean in zip(scarcity, means)
            ]
        )
        crossovers = simulation._policy_crossovers(summary)
        labels = [label for _, _, label in crossovers]
        self.assertFalse(
            any(
                "Lottery" in label and "First come, first served" in label
                for label in labels
            )
        )

    def test_policy_differences_are_paired_with_lottery_within_draw(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "condition": "primary",
                    "draw": draw,
                    "scarcity": scarcity,
                    "policy": policy,
                    "outcome": outcome,
                }
                for draw, scarcity, lottery, learned in (
                    (0, 0.1, 100.0, 104.0),
                    (1, 0.1, 300.0, 310.0),
                    (0, 0.2, 500.0, 503.0),
                    (1, 0.2, 700.0, 698.0),
                )
                for policy, outcome in (
                    ("Lottery", lottery),
                    ("Learned cost model", learned),
                )
            ]
        )
        paired = simulation.paired_difference_from_policy(
            frame,
            metric="outcome",
            reference_policy="Lottery",
            output_column="above_lottery",
        )
        lottery = paired[paired["policy"] == "Lottery"]["above_lottery"]
        learned = paired[paired["policy"] == "Learned cost model"].sort_values(
            ["scarcity", "draw"]
        )
        self.assertTrue(np.array_equal(lottery.to_numpy(), np.zeros(len(lottery))))
        np.testing.assert_allclose(
            learned["above_lottery"].to_numpy(),
            np.array([4.0, 10.0, 3.0, -2.0]),
        )


if __name__ == "__main__":
    unittest.main()
