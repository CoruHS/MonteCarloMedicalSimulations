# Paper-ready numbers

Config ID: `d3f82c853096`. Values below were generated, not transcribed from plots. 
Intervals are 2.5th and 97.5th percentiles across Monte Carlo draws. All outcomes are illustrative rather than estimates of real populations.

## Fixed modelling choices

- Seed: `20260805`.
- Draws per condition: 200.
- Training/scored patients per draw: 4,000/1,000.
- Disadvantaged population probability: 40%.
- Structural severity penalty: 2.0 points on a 0–24 scale.
- Latent-need weights: 60% normalized severity and 40% normalized treatment benefit.
- Access attenuation: 30% for both prior utilisation and cost.
- Robustness attenuation of the observed severity score: 15%.
- Moderate-scarcity headline: resources for 50% of arrivals.

## Primary outcomes at moderate scarcity

| Policy | Expected survivors / 1,000 (95% interval) | Expected survivor-years / 1,000 (95% interval) | Realized survivors / 1,000 (95% interval) |
| --- | --- | --- | --- |
| Lottery | 608.9 (599.0–618.2) | 16,016 (15,363–16,731) | 609.3 (583.0–638.0) |
| First come, first served | 608.9 (598.2–618.5) | 16,007 (15,336–16,773) | 609.8 (580.0–640.0) |
| Sickest first | 633.5 (622.6–644.4) | 16,614 (15,907–17,408) | 634.0 (603.0–665.0) |
| Save the most lives | 639.2 (628.0–649.5) | 16,236 (15,572–17,010) | 639.7 (611.0–671.0) |
| Youngest first | 599.1 (589.1–608.2) | 17,497 (16,815–18,195) | 599.2 (567.0–628.0) |
| Life-years | 614.2 (603.8–623.7) | 17,706 (17,005–18,460) | 614.4 (581.0–643.0) |
| Learned cost model | 633.3 (621.9–644.2) | 16,361 (15,694–17,137) | 634.2 (606.0–665.0) |

## Primary group outcomes at moderate scarcity

Gaps are disadvantaged minus advantaged except `need priority gap`, where a positive value means the disadvantaged group is ranked lower within matched latent-need strata.

| Policy | Allocation A / D (%) | Allocation gap (pp) | Expected survival A / D (%) | Survival gap (pp) | Need-priority gap (pp) |
| --- | --- | --- | --- | --- | --- |
| Lottery | 50.0 / 50.0 | +0.0 | 63.4 / 57.1 | -6.3 | -0.0 |
| First come, first served | 50.0 / 50.0 | +0.0 | 63.4 / 57.1 | -6.3 | -0.0 |
| Sickest first | 44.2 / 58.8 | +14.6 | 64.5 / 61.7 | -2.8 | -1.5 |
| Save the most lives | 46.2 / 55.7 | +9.5 | 65.6 / 61.4 | -4.1 | +1.4 |
| Youngest first | 50.0 / 50.0 | -0.0 | 62.3 / 56.3 | -5.9 | -1.1 |
| Life-years | 47.6 / 53.6 | +6.0 | 63.4 / 58.4 | -5.0 | -0.5 |
| Learned cost model | 53.5 / 44.7 | -8.7 | 67.4 / 57.2 | -10.1 | +18.0 |

## Expected survivors across scarcity levels

| Resources | Lottery | First come, first served | Sickest first | Save the most lives | Youngest first | Life-years | Learned cost model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | 496.9 | 496.8 | 499.3 | 504.8 | 491.4 | 501.4 | 502.6 |
| 20% | 524.9 | 524.8 | 533.6 | 540.1 | 516.7 | 531.8 | 536.1 |
| 30% | 552.9 | 552.9 | 568.1 | 574.6 | 543.3 | 560.6 | 569.2 |
| 40% | 580.9 | 580.9 | 601.7 | 607.7 | 570.7 | 588.1 | 601.6 |
| 50% | 608.9 | 608.9 | 633.5 | 639.2 | 599.1 | 614.2 | 633.3 |
| 60% | 637.0 | 636.9 | 663.1 | 668.3 | 628.0 | 639.3 | 663.4 |
| 70% | 665.0 | 665.0 | 690.0 | 694.7 | 657.6 | 663.8 | 691.0 |
| 80% | 692.9 | 693.1 | 713.9 | 717.8 | 687.7 | 689.2 | 715.1 |
| 90% | 721.0 | 721.0 | 734.2 | 736.6 | 718.3 | 718.6 | 735.0 |

## Expected survivors above lottery across scarcity levels

These are paired differences: lottery is subtracted within the same generated population, draw, and scarcity level before means or intervals are calculated.

| Resources | Lottery | First come, first served | Sickest first | Save the most lives | Youngest first | Life-years | Learned cost model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | +0.0 | -0.1 | +2.5 | +7.9 | -5.4 | +4.6 | +5.8 |
| 20% | +0.0 | -0.1 | +8.7 | +15.2 | -8.1 | +6.9 | +11.3 |
| 30% | +0.0 | -0.0 | +15.3 | +21.7 | -9.6 | +7.8 | +16.3 |
| 40% | +0.0 | -0.1 | +20.7 | +26.8 | -10.2 | +7.1 | +20.7 |
| 50% | +0.0 | +0.1 | +24.6 | +30.3 | -9.8 | +5.4 | +24.4 |
| 60% | +0.0 | -0.0 | +26.1 | +31.4 | -8.9 | +2.4 | +26.5 |
| 70% | +0.0 | -0.0 | +25.0 | +29.7 | -7.4 | -1.1 | +26.0 |
| 80% | +0.0 | +0.1 | +21.0 | +24.8 | -5.2 | -3.8 | +22.2 |
| 90% | +0.0 | +0.1 | +13.2 | +15.6 | -2.7 | -2.4 | +14.0 |

- Learned-cost-model / sickest-first crossover estimates: 39.6%, 53.7% of arrivals supplied.

## Learned-policy diagnostic

- Held-out predicted-cost/cost correlation: 0.866.
- Held-out cost-model R²: 0.751.
- Reference latent need in Figure 5: 0.576.
- Fitted priority at that need, advantaged/disadvantaged: $13,038 / $10,655.
- At equal need, advantaged-minus-disadvantaged predicted priority: $2,382.
- The learned model received neither group membership nor latent true need.

## Coefficient channel

| Feature | Coefficient | Standard error | Correlation with group |
| --- | --- | --- | --- |
| Intercept | 577.85 | 1,594.20 | — |
| age | -0.88 | 6.70 | -0.008 |
| severity | 178.53 | 44.69 | +0.167 |
| p_survive_treated | 9,361.81 | 654.48 | -0.137 |
| p_survive_untreated | -9,669.42 | 969.21 | -0.143 |
| prior_utilisation | 1,140.15 | 14.12 | -0.287 |

## Attenuated-severity robustness condition

This condition attenuates only the observed severity feature. Treated and untreated survival probabilities remain functions of clinical severity, so save-the-most-lives and life-years are unchanged by construction; this is a narrow measurement robustness check, not a fully biased outcome-model scenario.

| Policy | Clean severity gap (pp) | Attenuated severity gap (pp) | Change (pp) |
| --- | --- | --- | --- |
| Lottery | +0.0 | +0.0 | +0.0 |
| First come, first served | +0.0 | +0.0 | +0.0 |
| Sickest first | +14.6 | +1.8 | -12.8 |
| Save the most lives | +9.5 | +9.5 | +0.0 |
| Youngest first | -0.0 | -0.0 | +0.0 |
| Life-years | +6.0 | +6.0 | +0.0 |
| Learned cost model | -8.7 | -22.9 | -14.2 |

## Sanity checks

- Protected/evaluation columns absent from X: passed by assertion before every fit (`age, severity, p_survive_treated, p_survive_untreated, prior_utilisation` only).
- Untreated survival below treated survival for every generated patient: passed by assertion.
- Lottery group-balance check: passed; maximum aggregate absolute selected-share error 0.06 percentage points.
- All policies identical at full allocation: passed.
- Save-the-most-lives never below lottery: passed.
- Minimum held-out predicted-cost correlation across fitted draws: 0.836 (positive check passed).
- Expected versus Bernoulli policy-rank Spearman correlation at moderate scarcity: 0.964 (stability check passed).
