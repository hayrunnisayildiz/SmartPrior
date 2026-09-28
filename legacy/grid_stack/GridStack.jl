"""
    GridStack

Archived grid-based prior stack (grid, feature channels, `PriorNet`, loss,
training loop, metrics). It was taken out of `SmartPrior` because the current
Keivitsa pipeline does not use it. The code is kept unchanged so it can be
restored; see `README.md` next to this file.
"""
module GridStack

using LinearAlgebra
using Printf
using Random
using Statistics

using Lux
using Lux: Chain, Dense, gelu, sigmoid
using Random: AbstractRNG, Xoshiro

import JLD2
import Optimisers
import Zygote

include(joinpath("src", "Grid.jl"))
include(joinpath("src", "Features.jl"))
include(joinpath("src", "PriorNet.jl"))
include(joinpath("src", "Losses.jl"))
include(joinpath("src", "Train.jl"))
include(joinpath("src", "Metrics.jl"))

# grid
export PriorGrid
export ncells, cell_volumes, cell_centers, normalized_centers, depth_below_top
export containing_cell
export cu_log10, aggregate_to_cells, map_points_to_cells

# features
export FeatureStack, PointSamples, LabelSamples
export nchannels, build_features, feature_matrix
export standardize, extrude, idw_to_grid, gaussian_smooth_xy, gradient_xy
export topography_channels, depth_channels, coverage_channels
export nearest_sample_channels, geochemistry_channels, lithology_channels
export append_channels

# neural field
export PriorNet, setup_prior, predict, predict_grid
export fourier_encode, encode_features

# training objective
export heteroscedastic_nll, smoothness, reference_penalty, sigma_penalty
export LossWeights, PriorTargets, prior_loss, loss_report
export sigma_bounds_from_anchors

# training
export TrainConfig, TrainResult, PriorEnsemble
export init_params, train_prior, train_ensemble
export predict_ensemble, predict_ensemble_grid
export save_prior, load_prior

# metrics
export coverage, volume_reduction, rmse, mae, anomaly_correlation, calibration

end # module
