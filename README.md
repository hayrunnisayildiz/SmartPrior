# SmartPrior

Predicts a rock property (e.g. copper grade) **and its uncertainty** at any 3D point from sparse drillholes, using a Julia / Lux.jl neural network. Tested against kriging on GTK drillhole data from the Keivitsa deposit, Finland.

**Status:** research prototype. Cu and density are done; susceptibility is not run yet.

![Cross-section: predicted Cu and its uncertainty, kriging vs neural field](docs/figures/section_mu_sigma.png)

**Kriging gives a patchier grade map; only the neural field shows where it is unsure.**

- One vertical east–west slice through the deposit. Left: kriging. Right: neural field. Both are computed on the same blocks, so the outlines match; only the colours differ.
- **Top row, predicted Cu:** yellow is high grade, blue is low. Dots are real drill samples.
- **Bottom row, uncertainty (σ):** brighter means less certain. Kriging is the same dark colour almost everywhere. The neural field turns orange near the surface and around the lone hole on the left, where there is little data.

## Results

10-fold cross-validation, holding out whole drillholes.

| Target | Skill: kriging | Skill: neural field | 90 % coverage: kriging | 90 % coverage: neural field |
|---|---:|---:|---:|---:|
| Cu (261 holes) | 0.106 | **0.151** | 0.78 | **0.87** |
| Density (263 holes) | 0.058 | **0.065** | 0.77 | **0.90** |

- **Skill:** 0 = no better than the average; higher is better.
- **Coverage:** share of held-out samples inside the 90 % interval; ideal is 0.90.

**In short (Cu):** the neural field is at least as accurate as kriging, and its uncertainty is more honest. The accuracy gain is not statistically significant. Details: [technical report](docs/2026-09_keivitsa_technical_report.md).

![Calibration curve](docs/figures/calibration_curve.png)

**Cu: the neural field's uncertainty is more honest than kriging's.**

- Each point asks: "if a method says it is X % sure, how often is it actually right?"
- On the dashed line, the answer matches exactly. Below the line, the method is overconfident.
- The neural field (red) stays close to the line. Kriging (blue) drops further below it, most clearly at 90 %.

### Density

Data: 30,908 core measurements (kg/m³) from 263 holes. Same 10-fold test as for Cu.

| Method | Skill | 90 % coverage |
|---|---:|---:|
| Kriging | 0.058 | 0.77 |
| Neural field, position only (`nn_xyz`) | 0.059 | 0.89 |
| Neural field, position + depth (`nn_cov`) | 0.065 | **0.90** |

- **Accuracy:** position says little about density. The error drops only from 146.1 to 137.5 kg/m³ compared with the average, and no method reaches the 0.10 skill target.
- **Hole by hole:** the network beats kriging on 153 of 263 holes (mean skill difference +0.055, 95 % CI 0.016 … 0.093).
- **Uncertainty:** the network's intervals hit the target (0.89–0.90); kriging's are too narrow (0.77).

Details: [density section of the report](docs/2026-09_keivitsa_technical_report.md#density).

## How it works

![Method overview](docs/figures/method_overview.svg)

*Read top to bottom: prepare the drillhole data, train the model, then test it on drillholes it has never seen.*

## Quick start

Needs Julia 1.10+. Get the Keivitsa data from GTK's [Hakku](https://hakku.gtk.fi/en) service (the data are not in this repository).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

export KEIVITSA_ROOT=/path/to/gtk          # folder that contains report/3_DRILLINGS/
julia --project=. examples/feasibility_keivitsa.jl
```

Results go to `tmp_feasibility_keivitsa/`. A full run takes a few hours on a laptop. The other scripts in `examples/` build the block model and the figures.

## Data licence

The data are used under the [GTK basic licence](https://www.gtk.fi/en/basic-licence/). It allows academic use but not redistribution, so no data are committed. Figures derived from the data:

> Keivitsa drillhole data (Hakku), edited © Geological Survey of Finland 2026.

## Licence and citation

Code: MIT, see [`LICENSE`](LICENSE).

> Yıldız, H. (2026). *SmartPrior: neural-field priors with uncertainty from sparse drillholes.* https://github.com/hayrunnisayildiz/SmartPrior
