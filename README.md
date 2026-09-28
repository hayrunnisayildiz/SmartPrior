# SmartPrior

Predicts a rock property (e.g. copper grade) **and its uncertainty** at any 3D point from sparse drillholes, using a Julia / Lux.jl neural network. Tested against kriging on GTK drillhole data from the Keivitsa deposit, Finland.

**Status:** research prototype. Cu and density are done; susceptibility is not run yet.

![Cross-section: predicted Cu and its uncertainty, kriging vs neural field](docs/figures/section_mu_sigma.png)

*Top: predicted Cu. Bottom: uncertainty (brighter = less certain). Kriging (left) is equally confident everywhere; the neural field (right) is less certain where drilling is sparse.*

## Results

10-fold cross-validation, holding out whole drillholes.

| Target | Skill: kriging | Skill: neural field | 90 % coverage: kriging | 90 % coverage: neural field |
|---|---:|---:|---:|---:|
| Cu (261 holes) | 0.106 | **0.151** | 0.78 | **0.87** |
| Density (263 holes) | 0.058 | **0.065** | 0.77 | **0.90** |

- **Skill:** 0 = no better than the average; higher is better.
- **Coverage:** share of held-out samples inside the 90 % interval; ideal is 0.90.

**In short:** the neural field is at least as accurate as kriging, and its uncertainty is more honest. The accuracy gain is not statistically significant. Details: [technical report](docs/2026-09_keivitsa_technical_report.md).

![Calibration curve](docs/figures/calibration_curve.png)

*The closer to the dashed line, the more honest the uncertainty. The neural field (red) is closer than kriging (blue).*

## How it works

![Method overview](docs/figures/method_overview.svg)

## Quick start

Needs Julia 1.10+. Get the Keivitsa data from GTK's [Hakku](https://hakku.gtk.fi/en) service (the data are not in this repository).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

export KEIVITSA_ROOT=/path/to/gtk          # folder that contains report/3_DRILLINGS/
julia --project=. examples/feasibility_keivitsa.jl
```

Results go to `tmp_feasibility_keivitsa/`. A full run takes a few hours on a laptop. The other scripts in `examples/` build the block model and the figures.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | the package: data loading, desurvey, covariates |
| `examples/` | cross-validation, block model and figure scripts |
| `examples/keivitsa_common.jl` | shared kriging and network code |
| `sites/keivitsa.toml` | site settings |
| `docs/` | technical report, data notes, figures |
| `test/` | tests (the test data are synthetic) |
| `legacy/` | archived older code, not used |

## Data licence

The data are used under the [GTK basic licence](https://www.gtk.fi/en/basic-licence/). It allows academic use but not redistribution, so no data are committed. Figures derived from the data:

> Keivitsa drillhole data (Hakku), edited © Geological Survey of Finland 2026.

## Licence and citation

Code: MIT, see [`LICENSE`](LICENSE).

> Yıldız, H. (2026). *SmartPrior: neural-field priors with uncertainty from sparse drillholes.* https://github.com/hayrunnisayildiz/SmartPrior
