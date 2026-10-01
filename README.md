# Regularized Linear Decision Rules — Ausgrid Battery Experiment

Seminar experiment for **IND2097 — Special Topics in Operations Research (2026.2)**, based on Nazare & Street (2023), *Solving Multistage Stochastic Linear Programming via Regularized Linear Decision Rules: An Application to Hydrothermal Dispatch Planning*.

## Goal

Adapt the paper's regularized two-stage LDR idea to a residential battery control problem using Ausgrid household electricity data.

The experiment compares:

- LDR without regularization;
- AdaLASSO-regularized LDR;
- no-battery baseline;
- perfect-foresight oracle benchmark.

The chronological data split is:

- **70% training**;
- **15% validation**;
- **15% out-of-sample test**.

## Model

Net load is defined as

[
L_t = GC_t - GG_t,
]

where `GC` is household consumption and `GG` is gross photovoltaic generation.

Battery state evolves as

[
B_t = B_{t-1} + \eta_c c_t - \frac{d_t}{\eta_d}.
]

The LDR defines a state target of the form

[
B_t^{\text{target}} = \theta_{t0} + \sum_{\ell,k}\theta_{t\ell k}z_{t-\ell}^k.
]

AdaLASSO regularizes the LDR coefficients to control policy complexity and evaluate whether a more parsimonious policy improves out-of-sample performance.

## Repository structure

```text
notebooks/
  ausgrid_ldr_adalasso.ipynb   # main Julia notebook
src/
  ausgrid_ldr_adalasso.jl      # code exported from the notebook
presentation/
  main.tex                     # Beamer presentation
  assets/casa.png              # motivation figure
  figures/                     # figures used in the presentation
results/
  csv/                         # validation and OOS tables
  summary/                     # numerical summary
references/
  README.md                    # bibliography / paper link
```

## Requirements

- Julia
- CSV.jl
- DataFrames.jl
- JuMP.jl
- Gurobi.jl
- Plots.jl
- StatsPlots.jl

A valid Gurobi installation/license is required.

## Running

Open `notebooks/ausgrid_ldr_adalasso.ipynb` in VS Code or Jupyter with a Julia kernel and execute cells from top to bottom.

The notebook downloads the public Ausgrid subset automatically and creates organized output folders for plots, CSVs, LaTeX macros, and summaries.

## Reference

Nazare, F., & Street, A. (2023). *Solving multistage stochastic linear programming via regularized linear decision rules: An application to hydrothermal dispatch planning*. European Journal of Operational Research, 309, 345–358. https://doi.org/10.1016/j.ejor.2022.12.039

## Author

Carlos Souza — PUC-Rio, Department of Industrial Engineering.
