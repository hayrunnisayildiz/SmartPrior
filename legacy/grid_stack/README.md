# Archived grid stack

These files were part of `SmartPrior` until they were archived. They hold the grid-based prior: `PriorGrid`, feature
channels, `PriorNet`, the prior loss, the training loop with deep ensembles,
and grid metrics. The current Keivitsa pipeline does not use any of them: its
network is `build_mlp` in `examples/keivitsa_common.jl`.

The code is unchanged apart from comments. It is wrapped in its own module,
`GridStack`, so the main package no longer loads it.

## Layout

| Path | Contents |
|---|---|
| `GridStack.jl` | module wrapper with the original exports |
| `src/Grid.jl`, `src/Features.jl` | grid and feature channels |
| `src/PriorNet.jl`, `src/Losses.jl`, `src/Train.jl` | network, loss, training |
| `src/Metrics.jl` | coverage, calibration, RMSE, … on grids |
| `test/` | the original tests, run through `test/runtests.jl` |

## Running the archived tests

```bash
julia --project=. -e 'using Pkg; Pkg.add("JLD2")'   # removed from the main project
julia --project=. legacy/grid_stack/test/runtests.jl
```

## Known issues at the time of archiving

- `train_prior` keeps `best_params = params` without a copy. `Optimisers.update!`
  mutates the arrays in place, so the returned "best" parameters are the last
  ones. Use `deepcopy(params)` if this code is revived.
- `predict_ensemble` fails for `nproperties > 1` (a matrix is assigned into a
  `Vector{Vector{Float64}}`).
- `save_prior` does not store the network architecture (width, depth,
  bounds, property names).
