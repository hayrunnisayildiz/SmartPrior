# Keivitsa (GTK) — legacy case study

**Not the active pipeline.** Readers remain in `src/KeivitsaIO.jl` and
`examples/train_keivitsa_prior.jl`. Numbers below are from `tmp_keivitsa_prior*`
(gitignored). Do not mix with Cloncurry district results.

Same 971 + 16,215-point geochemistry input unless noted. Grid: 50×34×N, 25 m
in X/Y, EPSG:2393 (KKJ Finland Zone 3).

## 1) Training budget (epochs)

25 m Z, width 128, depth 4, resistivity included, **no** σ floor.

| | 30 epoch (`tmp_keivitsa_prior`) | 300 epoch, best = 250 (`tmp_keivitsa_prior_e300`) |
|---|---:|---:|
| log10-RMSE (988 grade anchors) | 0.460 | **0.249** |
| High-grade pred/background | 2.65× | **5.81×** |
| Assay high/background (reference) | 6.72× | 6.72× |
| Max predicted Cu (ppm) | 3,831 | 16,751 |
| Cells ≥ 5,000 ppm | 0 | 621 |

More training closed most of the amplitude-compression gap. Loss ticked
back up between epoch 250 and 300 (density NLL 0.191 → 0.314);
`train_prior` keeps the best checkpoint.

## 2) Z-axis grid resolution

Hypothesis: 10 m Z instead of 25 m would connect the fragmented isosurface.

| | 25 m Z, ep. 250 (`e300` checkpoint) | 10 m Z, ep. 250 (`tmp_keivitsa_prior_z10`) |
|---|---:|---:|
| Cells | 23,800 | 57,800 |
| log10-RMSE | 0.249 | 0.291 |
| High-grade pred/bg | 5.81× | 5.86× |
| Fragmented 5,000 ppm shell? | Yes | **Still yes** |

**Hypothesis rejected.** Finer Z did not connect the fragments and did not
improve RMSE.

## 3) Sigma floor

Without a floor, density `σ` collapsed below the group std and NLL went
negative. Current training code floors `σ ≥ 1 ×` group std per property
(`sigma_bounds_from_anchors`).

| | No floor (e300, best ep. 250) | With floor (`tmp_keivitsa_prior_sigmafloor`, best ep. 200) |
|---|---:|---:|
| Total NLL | −5.12 | **+0.66** |
| Anchors sitting at the floor | — | 87–100% across properties |

Fixes the negative-NLL artifact. `σ` then loses most cell-to-cell range.

> Value share is not gradient share. Tables above are loss *values*, not
> which term is driving updates.

## 4) Network capacity

25 m Z, 250 epoch, σ floor on.

| width | depth | log10-RMSE | pred/bg | wall | run dir |
|---:|---:|---:|---:|---:|---|
| 128 | 4 | 0.287 | 5.55× | — | `tmp_keivitsa_prior_sigmafloor` |
| **256** | 4 | **0.232** | 5.78× | 395 s | `tmp_keivitsa_prior_wide` |
| 128 | 6 | 0.281 | 5.72× | 312 s | `tmp_keivitsa_prior_deep` |

Width helped RMSE; it did **not** move high-grade contrast (~5.5–5.8× vs
assay 6.72×).

Figures remain under `docs/assets/` (from `examples/export_keivitsa_blockmodel.jl`).
