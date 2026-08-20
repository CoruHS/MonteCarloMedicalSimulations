# Allocation Simulation: Complete Implementation Explanation

This document explains the complete implementation of the allocation simulation for *Who Should the Algorithm Save?* It covers the patient-generating process, every allocation strategy, the learned model, all outcome measures, the Monte Carlo procedure, worked examples, robustness analysis, sanity checks, and interpretation.

The implementation is in [simulation.py](simulation.py), the exact parameters are in [config.json](config.json), and the final numerical results are in [numbers.md](numbers.md).

## Overview

The simulation follows this sequence:

    Generate patients
        ↓
    Calculate survival probabilities and latent true need
        ↓
    Reduce recorded utilisation and cost for the disadvantaged group
        ↓
    Train an ordinary linear regression to predict cost
        ↓
    Rank patients under seven allocation strategies
        ↓
    Treat the top K patients
        ↓
    Measure survival, survivor-years, allocation disparity, and priority disparity
        ↓
    Repeat across 200 simulated populations and ten scarcity levels

## 1. Patient-level notation

For patient $i$:

| Symbol | Meaning |
|---|---|
| $G_i$ | Group membership |
| $A_i$ | Age |
| $S_i$ | Underlying clinical severity |
| $S_i^{obs}$ | Observed severity available to policies |
| $p_i^T$ | Survival probability if treated |
| $p_i^U$ | Survival probability if untreated |
| $B_i$ | Treatment benefit |
| $L_i$ | Remaining life expectancy after survival |
| $N_i$ | Latent true need |
| $U_i$ | Recorded prior healthcare utilisation |
| $C_i$ | Future healthcare cost |
| $T_i$ | Whether the patient receives treatment |

Group is coded as:

$$
G_i =
\begin{cases}
0, & \text{advantaged group} \\
1, & \text{disadvantaged group}
\end{cases}
$$

The crucial rule is that $N_i$, latent true need, exists only for evaluation. No allocation policy can use it.

Group membership is also excluded from the learned model. It is used only afterward to audit the model's outcomes.

## 2. Generating group membership

Each patient has a 40% probability of belonging to the disadvantaged group:

$$
G_i \sim \mathrm{Bernoulli}(0.40)
$$

For a population of 1,000 patients, we therefore expect approximately:

- 600 advantaged patients;
- 400 disadvantaged patients.

The exact numbers vary between Monte Carlo draws.

## 3. Generating age

Age follows a truncated normal distribution:

$$
A_i \sim \mathcal{N}(57,18^2)
$$

subject to:

$$
18 \leq A_i \leq 95
$$

“Truncated” means that impossible values are rejected and redrawn.

For example:

- an age draw of 62 is accepted;
- an age draw of 102 is discarded;
- an age draw of 15 is also discarded.

This creates a roughly bell-shaped adult population without allowing ages outside the specified range.

## 4. Generating clinical severity

### 4.1 Base severity

Base severity is drawn from a Beta distribution and scaled to a 0–24 range:

$$
X_i \sim \mathrm{Beta}(2.2,3.0)
$$

$$
S_i^{base}=24X_i
$$

The expected base severity is approximately:

$$
E[S_i^{base}]
=
24\left(\frac{2.2}{2.2+3.0}\right)
\approx 10.15
$$

The Beta distribution produces many moderate cases and fewer extremely mild or extremely severe cases.

### 4.2 Structural disadvantage

Clinical severity is:

$$
S_i =
\mathrm{clip}
\left(
S_i^{base}+2G_i+\varepsilon_i^S,\,
0,\,
24
\right)
$$

where:

$$
\varepsilon_i^S \sim \mathcal{N}(0,1.25^2)
$$

The term $2G_i$ is the structural severity penalty.

For the advantaged group:

$$
2G_i=2(0)=0
$$

For the disadvantaged group:

$$
2G_i=2(1)=2
$$

Therefore, disadvantaged patients enter approximately two severity points sicker on average.

The clip operation keeps the final value between 0 and 24.

This two-point difference is a modeling decision, not an estimate of any real racial, ethnic, or socioeconomic population.

## 5. Observed versus clinical severity

The simulation contains two conditions.

### 5.1 Primary condition

Observed severity is accurate:

$$
S_i^{obs}=S_i
$$

The sickest-first rule and the learned model see the true clinical severity score.

### 5.2 Robustness condition

Observed severity is attenuated by 15% for the disadvantaged group:

$$
S_i^{obs}=S_i(1-0.15G_i)
$$

Consequently:

$$
S_i^{obs} =
\begin{cases}
S_i, & G_i=0 \\
0.85S_i, & G_i=1
\end{cases}
$$

For two patients with clinical severity 16:

$$
S_{\mathrm{adv}}^{obs}=16
$$

$$
S_{\mathrm{disadv}}^{obs}=16(0.85)=13.6
$$

The disadvantaged patient is just as sick, but the recorded score is lower.

Only observed severity changes in this robustness condition. Survival probabilities continue to be calculated from clinical severity, so treatment benefit is unchanged by construction. The patient's underlying survival probabilities, life expectancy, true need, utilisation, and cost remain paired with the primary condition. This is a narrow test of the observed-severity measurement channel, not a fully biased survival-model scenario.

## 6. Survival with treatment

The treated-survival logit is:

$$
z_i^T
=
3.2
-0.16S_i
-0.025(A_i-50)
$$

The logit is converted into a probability using the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
p_i^T=\sigma(z_i^T)
$$

Older age and greater severity lower treated survival.

## 7. Survival without treatment

The untreated-survival logit is:

$$
z_i^U
=
2.0
-0.18S_i
-0.027(A_i-50)
$$

Therefore:

$$
p_i^U=\sigma(z_i^U)
$$

### Why treatment is always beneficial

Subtract the untreated logit from the treated logit:

$$
z_i^T-z_i^U
=
1.2+0.02S_i+0.002(A_i-50)
$$

The minimum occurs around $S_i=0$ and $A_i=18$:

$$
1.2+0.02(0)+0.002(18-50)=1.136
$$

Thus:

$$
z_i^T-z_i^U>0
$$

Because the sigmoid function is strictly increasing:

$$
z_i^T>z_i^U
\quad\Longrightarrow\quad
p_i^T>p_i^U
$$

Treatment therefore improves survival for every generated patient.

The code also checks this for every patient and stops with an error if the condition fails.

## 8. Treatment benefit

Treatment benefit is the difference between treated and untreated survival:

$$
B_i=p_i^T-p_i^U
$$

This measures how much treatment changes the patient's probability of survival.

### Example

Consider a 60-year-old patient with severity 12.

Treated logit:

$$
z^T
=
3.2-0.16(12)-0.025(60-50)
$$

$$
z^T=1.03
$$

Treated survival:

$$
p^T=\sigma(1.03)\approx0.737
$$

Untreated logit:

$$
z^U
=
2.0-0.18(12)-0.027(60-50)
$$

$$
z^U=-0.43
$$

Untreated survival:

$$
p^U=\sigma(-0.43)\approx0.394
$$

Treatment benefit:

$$
B=0.737-0.394=0.343
$$

Treatment raises this illustrative patient's probability of survival by approximately 34.3 percentage points.

## 9. Remaining life expectancy

Remaining life expectancy is generated as:

$$
L_i
=
\mathrm{clip}
\left(
84-A_i-0.35S_i+\varepsilon_i^L,\,
1,\,
70
\right)
$$

where:

$$
\varepsilon_i^L\sim\mathcal{N}(0,4^2)
$$

For the 60-year-old patient with severity 12, ignoring random noise:

$$
L=84-60-0.35(12)
$$

$$
L=19.8
$$

The random noise prevents remaining life expectancy from being a completely deterministic transformation of age.

This formula is illustrative and is not intended as a clinical life-expectancy model.

## 10. Latent true need

True need combines clinical severity and treatment benefit:

$$
N_i
=
\mathrm{clip}
\left(
0.60\frac{S_i}{24}
+
0.40\frac{B_i}{0.40}
+
\varepsilon_i^N,\,
0,\,
1.2
\right)
$$

where:

$$
\varepsilon_i^N\sim\mathcal{N}(0,0.05^2)
$$

Severity and benefit are normalized so they contribute on comparable scales.

Because:

$$
0.40\frac{B_i}{0.40}=B_i
$$

the formula can also be written as:

$$
N_i
=
\mathrm{clip}
\left(
0.025S_i+B_i+\varepsilon_i^N,\,
0,\,
1.2
\right)
$$

### Example

For the severity-12 patient with benefit 0.343, ignoring noise:

$$
N
=
0.60\frac{12}{24}
+
0.40\frac{0.343}{0.40}
$$

$$
N=0.300+0.343=0.643
$$

This definition deliberately does not equal any individual policy's ranking variable.

It is not identical to:

- severity;
- treatment benefit;
- age;
- remaining life expectancy;
- cost.

That prevents one allocation policy from becoming morally correct by definition.

## 11. Arrival time

Arrival time is drawn uniformly over a 24-hour window:

$$
H_i\sim\mathrm{Uniform}(0,24)
$$

An arrival at hour 2 precedes an arrival at hour 11.

Arrival time is independent of clinical need in the simulation.

## 12. The access attenuation

The access multiplier is:

$$
D_i=1-0.30G_i
$$

For the advantaged group:

$$
D_i=1-0.30(0)=1
$$

For the disadvantaged group:

$$
D_i=1-0.30(1)=0.70
$$

Thus:

$$
D_i =
\begin{cases}
1.00, & G_i=0 \\
0.70, & G_i=1
\end{cases}
$$

At equal need, recorded utilisation and future cost are 30% lower for the disadvantaged patient.

## 13. Prior healthcare utilisation

Prior utilisation is:

$$
U_i
=
\max
\left(
0,\,
12N_iD_i+\varepsilon_i^U
\right)
$$

where:

$$
\varepsilon_i^U\sim\mathcal{N}(0,1^2)
$$

Consider two patients with equal true need:

$$
N_{\mathrm{adv}}=N_{\mathrm{disadv}}=0.60
$$

Ignoring noise, advantaged utilisation is:

$$
U_{\mathrm{adv}}
=
12(0.60)(1)
=
7.20
$$

Disadvantaged utilisation is:

$$
U_{\mathrm{disadv}}
=
12(0.60)(0.70)
=
5.04
$$

The patients are equally needy, but the disadvantaged patient has less recorded utilisation.

## 14. Future healthcare cost

Future cost is the training label:

$$
C_i
=
\max
\left(
0,\,
24000N_iD_i+\varepsilon_i^C
\right)
$$

where:

$$
\varepsilon_i^C\sim\mathcal{N}(0,1800^2)
$$

At true need 0.60, ignoring noise:

$$
C_{\mathrm{adv}}
=
24000(0.60)(1)
=
\$14{,}400
$$

$$
C_{\mathrm{disadv}}
=
24000(0.60)(0.70)
=
\$10{,}080
$$

Both patients have the same need, but the disadvantaged patient receives a lower cost label.

## 15. Why the proxy feature is necessary

Suppose cost were attenuated by group, but all model features were clean measurements of need and contained no group-correlated information.

Two equal-need patients would then have identical feature vectors:

$$
X_{\mathrm{adv}}=X_{\mathrm{disadv}}
$$

A deterministic model must produce identical predictions for identical inputs:

$$
\widehat C_{\mathrm{adv}}
=
\widehat C_{\mathrm{disadv}}
$$

The model would learn an average cost across groups. It could not systematically distinguish between them.

The simulation avoids that null design by including prior utilisation.

The mechanism is:

    Access barrier
        ↓
    Lower recorded utilisation at equal need
        ↓
    Lower generated cost at equal need
        ↓
    Utilisation legitimately predicts cost
        ↓
    The model assigns lower predicted cost
        ↓
    The patient receives lower allocation priority

Nothing unusual or discriminatory is inserted into the regression algorithm.

## 16. Training the learned model

For every Monte Carlo draw:

- 4,000 patients form the training population;
- a separate 1,000 patients form the scoring population.

The model does not score patients on whom it was trained.

### Model features

The feature vector is:

$$
X_i=
\begin{bmatrix}
A_i \\
S_i^{obs} \\
p_i^T \\
p_i^U \\
U_i
\end{bmatrix}
$$

In words, the features are:

- age;
- observed severity;
- treated survival probability;
- untreated survival probability;
- prior utilisation.

The label is future cost:

$$
y_i=C_i
$$

The following are explicitly excluded:

- group membership;
- latent true need;
- cost as an input feature.

Cost appears only as the target the model is trained to predict.

## 17. Ordinary least-squares regression

The learned prediction is:

$$
\widehat C_i
=
\beta_0
+\beta_1A_i
+\beta_2S_i^{obs}
+\beta_3p_i^T
+\beta_4p_i^U
+\beta_5U_i
$$

The model chooses coefficients that minimize squared prediction errors:

$$
\widehat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\sum_{i=1}^{n}
\left(
C_i-X_i\boldsymbol{\beta}
\right)^2
$$

The closed-form expression is:

$$
\widehat{\boldsymbol{\beta}}
=
(X^\top X)^+X^\top y
$$

The superscript $+$ denotes the matrix pseudoinverse.

There is:

- no group reweighting;
- no fairness constraint;
- no special loss;
- no custom decision rule;
- no neural network.

It is ordinary, unweighted linear regression.

## 18. Coefficient standard errors

Training residuals are:

$$
e_i=C_i-\widehat C_i
$$

Estimated residual variance is:

$$
\widehat{\sigma}^2
=
\frac{\sum_i e_i^2}{n-p}
$$

The coefficient covariance matrix is:

$$
\widehat{\mathrm{Var}}
\left(
\widehat{\boldsymbol{\beta}}
\right)
=
\widehat{\sigma}^2(X^\top X)^+
$$

The standard error for coefficient $j$ is:

$$
SE(\widehat{\beta}_j)
=
\sqrt{
\widehat{\mathrm{Var}}
\left(
\widehat{\boldsymbol{\beta}}
\right)_{jj}
}
$$

## 19. Fitted coefficients

The fixed diagnostic model produced:

| Feature | Coefficient | Standard error | Correlation with group |
|---|---:|---:|---:|
| Intercept | 577.85 | 1,594.20 | — |
| Age | −0.88 | 6.70 | −0.008 |
| Severity | 178.53 | 44.69 | +0.167 |
| Treated survival | 9,361.81 | 654.48 | −0.137 |
| Untreated survival | −9,669.42 | 969.21 | −0.143 |
| Prior utilisation | 1,140.15 | 14.12 | −0.287 |

Prior utilisation has:

- a strong positive coefficient;
- the strongest negative correlation with disadvantaged-group membership.

This is the clearest location of the proxy channel.

The other coefficients should not be interpreted causally because age, severity, treated survival, and untreated survival are mathematically related and strongly correlated.

## 20. Equal-need model example

Return to the 60-year-old patient with severity 12:

$$
p^T=0.737
$$

$$
p^U=0.394
$$

$$
N=0.643
$$

Expected prior utilisation for the advantaged patient is:

$$
U_{\mathrm{adv}}
=
12(0.643)
=
7.71
$$

Expected prior utilisation for the disadvantaged patient is:

$$
U_{\mathrm{disadv}}
=
12(0.643)(0.70)
=
5.40
$$

Using the fitted regression coefficients gives approximately:

$$
\widehat C_{\mathrm{adv}}
=
\$14{,}550
$$

$$
\widehat C_{\mathrm{disadv}}
=
\$11{,}911
$$

The difference is:

$$
\$14{,}550-\$11{,}911
=
\$2{,}638
$$

The group variable never enters the regression. The difference arises from the group-correlated utilisation history.

The full diagnostic analysis found a fitted-line difference of \$2,382 at median latent need.

The model was nevertheless accurate:

$$
\mathrm{Corr}(\widehat C,C)=0.866
$$

$$
R^2=0.751
$$

This is an accurate model of cost. The ethical problem is that cost is not identical to need.

## 21. Lottery strategy

For lottery, every patient receives a random number:

$$
R_i\sim\mathrm{Uniform}(0,1)
$$

Lottery priority is:

$$
P_i^{lottery}=R_i
$$

The largest values are treated.

No patient characteristic affects the probability of allocation.

At scarcity level $s$:

$$
P(T_i=1\mid G_i=0)
\approx s
$$

$$
P(T_i=1\mid G_i=1)
\approx s
$$

At 50% scarcity, approximately 50% of each group receives treatment.

## 22. First come, first served

Arrival time is $H_i$. Earlier arrivals must rank higher, so the implementation uses:

$$
P_i^{FCFS}=-H_i
$$

For example:

$$
H_A=2
\quad\Longrightarrow\quad
P_A^{FCFS}=-2
$$

$$
H_B=8
\quad\Longrightarrow\quad
P_B^{FCFS}=-8
$$

Because $-2>-8$, patient A ranks first.

This policy does not use:

- severity;
- benefit;
- age;
- group;
- true need.

## 23. Sickest-first strategy

Sickest-first priority is observed severity:

$$
P_i^{sickest}=S_i^{obs}
$$

The highest observed severity is treated first.

In the primary condition:

$$
S_i^{obs}=S_i
$$

Because the disadvantaged group has the structural two-point severity increase, sickest-first tends to allocate more resources to that group.

In the robustness condition, disadvantaged observed severity is reduced by 15%, weakening that allocation advantage.

## 24. Save-the-most-lives strategy

Priority is treatment benefit:

$$
P_i^{lives}=B_i
$$

Recall:

$$
B_i=p_i^T-p_i^U
$$

### Why this maximizes expected survivors

Expected survival for patient $i$ under an allocation is:

$$
q_i=T_ip_i^T+(1-T_i)p_i^U
$$

Total expected survivors are:

$$
E[S]=\sum_i q_i
$$

Substitute the first equation:

$$
E[S]
=
\sum_i
\left[
T_ip_i^T+(1-T_i)p_i^U
\right]
$$

Rearrange:

$$
E[S]
=
\sum_i p_i^U
+
\sum_iT_i(p_i^T-p_i^U)
$$

Since $B_i=p_i^T-p_i^U$:

$$
E[S]
=
\sum_i p_i^U
+
\sum_iT_iB_i
$$

The first term is identical for every policy applied to the same population:

$$
\sum_i p_i^U
$$

Therefore, if exactly $K$ patients can be treated, maximizing expected survivors is equivalent to choosing the $K$ largest values of $B_i$.

That is why save-the-most-lives is mathematically guaranteed to produce at least as many expected survivors as lottery.

## 25. Youngest-first strategy

Priority is negative age:

$$
P_i^{youngest}=-A_i
$$

A 25-year-old has priority:

$$
-25
$$

A 70-year-old has priority:

$$
-70
$$

Because $-25>-70$, the younger patient ranks first.

This rule does not calculate treatment benefit or expected survivor-years. It only uses chronological age.

## 26. Life-years strategy

Life-years priority is:

$$
P_i^{LY}=B_iL_i
$$

This is the expected number of survivor-years added by treatment.

Total expected survivor-years are:

$$
E[LY]
=
\sum_i q_iL_i
$$

Substituting the survival equation:

$$
E[LY]
=
\sum_i
\left[
T_ip_i^T+(1-T_i)p_i^U
\right]L_i
$$

Rearranging:

$$
E[LY]
=
\sum_i p_i^UL_i
+
\sum_iT_iB_iL_i
$$

The first term is fixed for the same population. Therefore expected survivor-years are maximized by selecting the largest values of:

$$
B_iL_i
$$

This policy can differ substantially from youngest-first.

A very young patient with almost no treatment benefit may contribute fewer expected survivor-years than a somewhat older patient with a large treatment benefit.

## 27. Learned cost strategy

The learned policy ranks predicted cost:

$$
P_i^{learned}=\widehat C_i
$$

Patients with the largest predicted costs receive treatment first.

The model does not directly maximize:

- survival;
- survivor-years;
- severity;
- true need;
- equality.

Its implicit objective is:

    Prioritize predicted future healthcare spending.

That objective comes from the choice of training label.

## 28. Allocation mechanics

For population size $N$ and resource share $s$, the number of resources is:

$$
K=\mathrm{round}(sN)
$$

With $N=1000$:

| Resource share | Resources |
|---:|---:|
| 10% | 100 |
| 30% | 300 |
| 50% | 500 |
| 90% | 900 |
| 100% | 1,000 |

Each policy orders all patients and treats the first $K$.

There are no dynamic arrival windows, treatment withdrawals, or reallocations.

### Tie-breaking

Every patient receives a seeded random tie-breaker.

Sorting uses:

1. policy score descending;
2. random tie-breaker for equal scores.

### Nested scarcity allocations

The policy ordering is calculated once for each population.

Therefore:

- the top 10% are contained in the top 20%;
- the top 20% are contained in the top 30%;
- and so forth.

The population is not rerandomized at each scarcity level.

## 29. Expected survival metric

For each patient:

$$
q_i =
\begin{cases}
p_i^T, & T_i=1 \\
p_i^U, & T_i=0
\end{cases}
$$

Equivalently:

$$
q_i=T_ip_i^T+(1-T_i)p_i^U
$$

The reported lives-saved value is:

$$
\sum_iq_i
$$

Strictly speaking, this is expected total survivors, including people who would survive untreated.

The incremental number saved by treatment would be:

$$
\sum_{i:T_i=1}B_i
$$

Both measures generate the same policy ranking for a fixed population and scarcity because the untreated baseline is identical across policies.

## 30. Expected survivor-years

Expected survivor-years are:

$$
\sum_iq_iL_i
$$

The incremental survivor-years created by treatment would be:

$$
\sum_{i:T_i=1}B_iL_i
$$

Results are reported per 1,000 arrivals:

$$
M_{1000}=M\frac{1000}{N}
$$

Since the principal scoring population contains 1,000 patients, this normally leaves the value unchanged.

## 31. Allocation rate by group

For group $g$, the allocation rate is:

$$
AR_g
=
\frac{
\sum_iT_iI(G_i=g)
}{
\sum_iI(G_i=g)
}
$$

Here, $I(\cdot)$ is an indicator equal to 1 when the condition is true.

The allocation gap is:

$$
\Delta_{allocation}
=
100(AR_1-AR_0)
$$

A negative gap means the disadvantaged group receives treatment at a lower within-group rate.

The learned policy produced:

$$
AR_0=53.5\%
$$

$$
AR_1=44.7\%
$$

Therefore:

$$
\Delta_{allocation}
=
44.7-53.5
=
-8.7
$$

This is approximately a −8.7 percentage-point gap.

If the population were exactly 600 advantaged patients and 400 disadvantaged patients, this would correspond approximately to:

$$
600(0.535)\approx321
$$

advantaged patients treated, and:

$$
400(0.447)\approx179
$$

disadvantaged patients treated, for roughly 500 total treatments.

## 32. Expected survival rate by group

For group $g$:

$$
SR_g
=
\frac{
\sum_iq_iI(G_i=g)
}{
\sum_iI(G_i=g)
}
$$

The survival gap is:

$$
\Delta_{survival}
=
100(SR_1-SR_0)
$$

Equal allocation does not guarantee equal survival because the groups have different underlying severity distributions.

For example, lottery allocates approximately 50% of each group, but the disadvantaged group still has lower expected survival because it enters sicker on average.

## 33. Need-adjusted priority gap

Raw allocation rates do not answer whether equally needy patients were ranked equally.

### Priority percentile

The highest-ranked patient receives priority percentile 1:

$$
P_i=1
$$

The lowest-ranked patient receives priority percentile 0:

$$
P_i=0
$$

For rank $r_i$, where rank 0 is highest:

$$
P_i
=
1-\frac{r_i}{N-1}
$$

### Matching patients by true need

Patients are divided into 20 pooled true-need quantile bins.

Within bin $b$, calculate:

$$
d_b
=
\overline P_{0b}
-
\overline P_{1b}
$$

Each bin is weighted by the number of patients who can be compared across groups:

$$
w_b=\min(n_{0b},n_{1b})
$$

The final gap is:

$$
\Delta_{need}
=
100
\frac{
\sum_bw_bd_b
}{
\sum_bw_b
}
$$

A positive value means the disadvantaged group is ranked lower among patients with similar latent need.

For the learned model:

$$
\Delta_{need}=18.0
$$

This represents an approximately 18 priority-percentile-point advantage for advantaged patients at matched need.

It does not mean exactly 18% fewer patients are treated. Allocation depends on where the resource cutoff falls.

## 34. Figure 5's dollar-valued gap

Figure 5 uses a related but different measure.

After the model has made its predictions, separate audit lines are fitted for each group:

$$
\widehat C=\alpha_g+\gamma_gN
$$

These lines are used only for evaluation. Group and true need were not used by the decision model.

At median true need:

$$
N=0.576
$$

the fitted predictions were:

$$
\widehat C_{\mathrm{adv}}=\$13{,}038
$$

$$
\widehat C_{\mathrm{disadv}}=\$10{,}655
$$

Therefore:

$$
\Delta_{\$}
=
\$13{,}038-\$10{,}655
=
\$2{,}382
$$

The \$2,382 result is measured in predicted-cost units.

The 18-point need-priority gap is measured in rank-percentile units. They are two descriptions of the same mechanism, not the same statistic.

## 35. Complete five-patient example

Suppose there are five patients and only two treatment slots.

For simplicity, this example omits random noise. Learned scores use the fitted diagnostic coefficients.

| Patient | Group | Age | Severity | Arrival | Benefit | Remaining years | $B_iL_i$ | Learned score | Lottery key |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Advantaged | 20 | 1 | 5 h | 0.045 | 63.6 | 2.87 | \$1,834 | 0.10 |
| B | Disadvantaged | 60 | 12 | 1 h | 0.343 | 19.8 | 6.79 | \$11,911 | 0.90 |
| C | Advantaged | 75 | 16 | 2 h | 0.329 | 3.4 | 1.12 | \$16,378 | 0.40 |
| D | Disadvantaged | 45 | 18 | 3 h | 0.361 | 32.7 | 11.79 | \$14,815 | 0.70 |
| E | Advantaged | 35 | 10 | 4 h | 0.231 | 45.5 | 10.53 | \$10,885 | 0.20 |

Each strategy selects its two highest-priority patients:

| Strategy | Ranking basis | Selected patients |
|---|---|---|
| Lottery | Random key | B and D |
| First come | Earliest arrival | B and C |
| Sickest first | Highest severity | D and C |
| Save most lives | Highest benefit | D and B |
| Youngest first | Lowest age | A and E |
| Life-years | Highest benefit × remaining years | D and E |
| Learned model | Highest predicted cost | C and D |

The resulting expected outcomes are:

| Strategy | Expected survivors | Expected survivor-years |
|---|---:|---:|
| Lottery | 3.100 | 123.91 |
| First come | 3.069 | 113.24 |
| Sickest first | 3.087 | 118.25 |
| Save most lives | 3.100 | 123.91 |
| Youngest first | 2.673 | 118.73 |
| Life-years | 2.989 | 127.65 |
| Learned model | 3.087 | 118.25 |

Lottery happens to select the same patients as save-the-most-lives in this tiny example. That is random luck rather than a general result.

The example shows the distinct objectives:

- save-the-most-lives maximizes expected survivors;
- life-years maximizes expected survivor-years;
- youngest-first is not identical to life-years;
- the learned model follows predicted cost.

Patient C receives the highest learned score even though:

- C does not have the greatest treatment benefit;
- C has only 3.4 remaining expected years;
- C is not first under sickest-first.

C's unattenuated utilisation contributes to a high cost prediction.

## 36. Monte Carlo procedure

For Monte Carlo draw $d$:

$$
seed_{train}
=
20260805+1000+2d
$$

$$
seed_{score}
=
20260805+1001+2d
$$

For draw zero:

$$
seed_{train}=20261805
$$

$$
seed_{score}=20261806
$$

Each draw performs these steps:

1. Generate 4,000 training patients.
2. Generate 1,000 separate scoring patients.
3. Fit the cost-prediction model.
4. Predict cost for the scoring patients.
5. Produce all seven policy rankings.
6. Allocate resources at each scarcity level.
7. Record all metrics.

## 37. Scarcity levels

The resource shares are:

$$
s\in
\{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\}
$$

The 100% level is included as an end-to-end check. If everyone receives treatment, policy order should have no effect.

## 38. Total number of results

The experiment contains:

$$
2
\times
200
\times
10
\times
7
=
28{,}000
$$

The factors are:

- 2 measurement conditions;
- 200 Monte Carlo draws;
- 10 scarcity levels;
- 7 allocation strategies.

Thus [results.csv](results.csv) contains 28,000 run-level rows.

## 39. Monte Carlo intervals

For metric $x$, the reported mean is:

$$
\overline{x}
=
\frac{1}{200}
\sum_{d=1}^{200}x_d
$$

The interval is:

$$
\left[
Q_{0.025}(x),
Q_{0.975}(x)
\right]
$$

These are the empirical 2.5th and 97.5th percentiles across the 200 simulated populations.

They describe Monte Carlo variability, not uncertainty about a real hospital population.

## 40. Bernoulli outcome check

The headline results use expected probabilities because they have lower noise.

The simulation also generates realized outcomes.

Each patient receives:

$$
V_i\sim\mathrm{Uniform}(0,1)
$$

The realized outcome is:

$$
Y_i=I(V_i<q_i)
$$

The same $V_i$ is reused across policies.

Therefore:

- if $V_i<p_i^U$, the patient survives with or without treatment;
- if $p_i^U\leq V_i<p_i^T$, treatment saves the patient;
- if $V_i\geq p_i^T$, the patient dies with or without treatment.

Since:

$$
p_i^T>p_i^U
$$

treatment can never convert a simulated survivor into a death.

Expected and Bernoulli policy rankings had Spearman correlation:

$$
\rho=0.964
$$

The policy ranking is therefore not an artifact of using expected values.

## 41. Main results at 50% scarcity

| Strategy | Expected survivors per 1,000 | Survivor-years per 1,000 | Allocation: advantaged / disadvantaged | Need-priority gap |
|---|---:|---:|---:|---:|
| Lottery | 608.9 | 16,016 | 50.0% / 50.0% | −0.0 pp |
| First come | 608.9 | 16,007 | 50.0% / 50.0% | −0.0 pp |
| Sickest first | 633.5 | 16,614 | 44.2% / 58.8% | −1.5 pp |
| Save most lives | 639.2 | 16,236 | 46.2% / 55.7% | +1.4 pp |
| Youngest first | 599.1 | 17,497 | 50.0% / 50.0% | −1.1 pp |
| Life-years | 614.2 | 17,706 | 47.6% / 53.6% | −0.5 pp |
| Learned model | 633.3 | 16,361 | 53.5% / 44.7% | +18.0 pp |

Figure 1 uses a dot-and-interval display rather than bars. A bar must begin at zero because its length encodes magnitude, but that would compress the entire policy difference into a small part of the axis. A point has no encoded length, so the axis can legitimately focus on the observed interval, making the differences and uncertainty visible without exaggerating them. Each dot is the Monte Carlo mean and each horizontal line is the 95% percentile interval.

Figure 2 plots each policy's expected survivors **above lottery**, paired within the same draw and scarcity level:

$$
D_{pds}=L_{pds}-L_{\mathrm{Lottery},ds}
$$

Here, $p$ is policy, $d$ is Monte Carlo draw, and $s$ is scarcity. The subtraction happens before the mean and percentile interval are calculated. This removes the mechanically rising survivor baseline as more resources become available while preserving every pairwise policy crossing, because subtracting the same lottery value from two policies cannot change which is larger:

$$
\left(L_{Ads}-L_{\mathrm{Lottery},ds}\right)
-
\left(L_{Bds}-L_{\mathrm{Lottery},ds}\right)
=
L_{Ads}-L_{Bds}
$$

Figure 2 deliberately excludes sign flips between lottery and first come, first served. Arrival time is uniform and independent of every patient characteristic, so those policies are statistically identical in this data-generating process. Their tiny mean differences change sign through Monte Carlo noise and are not substantive crossovers. After excluding that pair, four substantive plotted crossover locations remain: life-years crosses sickest-first and the two procedural rules, while the learned policy crosses sickest-first at approximately 39.6% resource availability and crosses back at approximately 53.7%. The life-years crossings with lottery and first come occur at effectively the same plotted location and are visually deduplicated.

Figure 3 draws lottery and first come, first served as one combined, explicitly labelled point. Their mean coordinates are visually coincident because both policies select a random subset with respect to patient outcomes in this data-generating process. Combining the marker prevents one policy from being hidden behind the other and prevents two labels beside one visible point from looking like an error.

Figure 4 includes a light vertical reference at the 50% allocation rate in its allocation panel. At the headline scarcity level, a policy with equal group allocation rates would place both group markers on that line. Figure 5 repeats inside the image that the learned model received neither group membership nor latent true need, and its darker vertical guide fixes the equal-need location used for the annotated score gap. Figure 6 is unchanged.

### Interpretation

Save-the-most-lives produces the most expected survivors:

$$
639.2
$$

The learned model produces:

$$
633.3
$$

The relative difference is approximately:

$$
\frac{639.2-633.3}{633.3}\times100
\approx0.9\%
$$

The learned model is therefore reasonably efficient in total survival terms.

But it allocates treatment at very different group rates:

$$
53.5\%
\quad\mathrm{versus}\quad
44.7\%
$$

and produces an 18-point need-adjusted priority gap.

That combination is central to the argument:

> A model can perform well on aggregate outcomes while ranking equally needy patients unequally.

## 42. Lives versus life-years

The life-years policy produced the greatest expected survivor-years:

$$
17{,}706
$$

Youngest-first also produced many survivor-years:

$$
17{,}497
$$

but fewer expected survivors:

$$
599.1
$$

Save-the-most-lives produced more expected survivors:

$$
639.2
$$

but fewer survivor-years:

$$
16{,}236
$$

Across policies, expected survivors and survivor-years had correlation:

$$
r=-0.34
$$

Thus this simulation shows a tradeoff rather than perfect agreement between the two objectives.

## 43. Severity robustness results

At 50% scarcity:

| Strategy | Clean-severity gap | Attenuated-severity gap | Change |
|---|---:|---:|---:|
| Sickest first | +14.6 pp | +1.8 pp | −12.8 pp |
| Learned model | −8.7 pp | −22.9 pp | −14.2 pp |

Under clean severity, sickest-first treats more disadvantaged patients because that group is sicker on average.

After their observed severity is reduced by 15%, that allocation advantage nearly disappears.

The learned model's gap becomes more adverse because it now receives two downward group-correlated signals:

1. lower prior utilisation;
2. lower observed severity.

The other policies remain unchanged because survival probabilities continue to use clinical severity and these policies do not use observed severity:

- lottery uses randomness;
- first-come uses arrival time;
- save-the-most-lives uses treatment benefit;
- youngest-first uses age;
- life-years uses benefit and remaining life expectancy.

## 44. Sanity checks

All executable sanity checks passed.

### Feature exclusion

The model receives only:

$$
X_i=
[
A_i,\,
S_i^{obs},\,
p_i^T,\,
p_i^U,\,
U_i
]
$$

Group and true need are absent.

### Treatment always helps

For every patient:

$$
p_i^U<p_i^T
$$

### Lottery balance

Lottery's selected group composition differed from the population composition by at most 0.06 percentage points after aggregation.

### Full allocation

At:

$$
s=1
$$

everyone is treated, so every policy produces identical outcomes.

The observed maximum difference was:

$$
0
$$

### Survival optimization

Save-the-most-lives never produced fewer expected survivors than lottery.

### Model validity

The minimum held-out predicted-cost correlation across fitted Monte Carlo models was:

$$
0.836
$$

The fixed diagnostic model's correlation was:

$$
0.866
$$

### Expected versus realized outcomes

The expected and Bernoulli rankings had:

$$
\rho=0.964
$$

## 45. What the simulation demonstrates

The learned model does not secretly contain an instruction to discriminate.

It learns a statistically valid relationship:

    Lower prior utilisation predicts lower future cost.

But recorded utilisation is not a clean measure of health need:

    Access barriers reduce recorded utilisation even when need is equal.

Combining those relationships produces:

    Equal true need
        ↓
    Lower recorded utilisation for the disadvantaged patient
        ↓
    Lower expected cost
        ↓
    Lower learned priority
        ↓
    Lower probability of receiving treatment

The model is accurate at predicting its target. The target is the ethical problem.

The written policies explicitly reveal their values:

- lottery values equal chance;
- first-come values procedural order;
- sickest-first values severity;
- save-the-most-lives values total survival;
- youngest-first values priority to youth;
- life-years values expected survival duration.

The learned policy does not state its value in its mechanism. Its operational value is inherited from its target:

    Prioritize predicted future spending.

That is why the simulation supports the paper's thesis: when allocation is delegated to a learned model, public scrutiny must focus on what the model is trained to predict, not merely on whether its code is accurate or whether it explicitly contains a protected attribute.

## 46. Important interpretation limits

The results are illustrative rather than predictive.

- The age, severity, survival, life-expectancy, access, utilisation, and cost parameters are modeling choices, not estimates of real clinical populations.
- The 60/40 definition of true need is a normative choice. A different definition could change comparisons.
- The simulation models a single simultaneous allocation rather than a dynamic hospital queue.
- Survival probabilities are known by construction, whereas real hospitals must estimate them.
- The model is intentionally simple because the claim concerns target choice, not model complexity.
- The Monte Carlo intervals measure variation across generated populations; they are not clinical confidence intervals.
- The metric named lives saved is technically expected total survivors. Incremental lives saved would subtract the common untreated baseline, without changing rankings for fixed scarcity.

## 47. Output artifacts

The full reproducible evidence base consists of:

- [simulation.py](simulation.py): simulation, model, metrics, checks, and figure generation;
- [config.json](config.json): exact parameters and config identifier;
- [results.csv](results.csv): all 28,000 run-level results;
- [numbers.md](numbers.md): paper-ready numerical registry;
- [table1_coefficients.md](table1_coefficients.md): diagnostic coefficient table;
- [figures/captions.md](figures/captions.md): computed figure captions;
- [README.md](README.md): reproduction instructions and library versions;
- [figures](figures): six figures in 300-dpi PNG and PDF formats;
- [figures/grayscale_checks](figures/grayscale_checks): grayscale print checks.
