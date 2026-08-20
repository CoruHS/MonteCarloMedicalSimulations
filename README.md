# Who Should the Algorithm Save? — allocation simulation

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
`.venv\Scripts\Activate.ps1` instead of the `source` command.

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

The final run used seed `20260805`, config ID `d3f82c853096`, 200 draws per condition, 4,000 training patients per draw, and 1,000 scored patients per draw. Runtime on the generating machine was 11.5 seconds.

## Outputs

- `results.csv`: complete run-level results.
- `numbers.md`: paper-ready source of every reported quantity.
- `config.json`: exact parameters used.
- `table1_coefficients.md` and `.csv`: fitted model table.
- `figures/figure1` through `figure6`: 300 dpi PNG and PDF.
- `figures/grayscale_checks/`: automatic grayscale conversions for print review.
- `figures/captions.md` and `figures/manifest.json`: result-driven captions and config provenance.

All executable sanity checks passed. Expected and Bernoulli policy rankings had Spearman correlation 0.964 at moderate scarcity.

## Scope of the severity robustness condition

The robustness condition attenuates the disadvantaged group's observed severity score only. Survival probabilities continue to use clinical severity, so policies sorting on treatment benefit are unchanged by construction. It isolates the observed-severity measurement channel rather than simulating a fully biased survival model.

## Library versions

| Library | Version |
| --- | --- |
| python | 3.12.2 |
| numpy | 1.26.4 |
| pandas | 2.3.2 |
| scikit-learn | 1.7.2 |
| scipy | 1.13.1 |
| matplotlib | 3.10.3 |
