"""
    SmartPriorMT

Uncertainty-aware starting models ("smart priors") for magnetotelluric inversion.

The package turns complementary geophysical data (gravity, topography, shallow
structural constraints) into a per-cell nominal resistivity `mu` and a per-cell
confidence `sigma` on the same tensor grid that MTGeophysics.jl inverts on. The
pair defines both a starting model and a per-cell search interval, which narrows
the space a stochastic solver such as VFSA has to explore.

The mapping is learned by a Lux.jl neural field trained against differentiable
physics: a closed-form prism gravity operator and an AD-safe 1D MT recursion.
"""
module SmartPriorMT

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

using MTGeophysics: WS3DModel, load_ws3d_model, write_ws3d_model
using MTGeophysics: RBFMap, build_rbf_map, apply_rbf_map!

include("Grid.jl")
include("Gravity.jl")
include("MT1DAD.jl")
include("Features.jl")
include("PriorNet.jl")
include("Losses.jl")
include("Train.jl")
include("Export.jl")
include("BoundedVFSA.jl")
include("Synthetic.jl")
include("Metrics.jl")
include("Profile2D.jl")
include("RealDataIO.jl")

# grid
export PriorGrid
export ncells, cell_volumes, cell_centers, normalized_centers, depth_below_top

# gravity
export prism_gz, gravity_matrix, forward_gravity

# 1-D MT physics (AD-safe)
export mt1d_impedance, mt1d_apparent, mt1d_column_response
export skin_depth, bostick_depth, bostick_resistivity, niblett_bostick

# observations and features
export GravityObs, MTSites, FeatureStack
export nsites, nchannels, build_features, feature_matrix, nb_baseline
export standardize, extrude, idw_to_grid, gaussian_smooth_xy, gradient_xy
export gravity_channels, topography_channels, depth_channels, coverage_channels

# neural field
export PriorNet, setup_prior, predict, predict_grid
export fourier_encode, encode_features

# training objective
export GravityCoupling, density_from_mu, physical_slope, DENSITY_SCALE
export heteroscedastic_nll, gravity_misfit, mt_column_misfit, smoothness, reference_penalty, sigma_penalty
export LossWeights, PriorTargets, prior_loss, loss_report

# training
export TrainConfig, TrainResult, PriorEnsemble
export init_params, coupling_of, train_prior, train_ensemble
export predict_ensemble, predict_ensemble_grid
export save_prior, load_prior

# export to the inversion's format
export PriorBundle, prior_bounds, apply_air_mask, write_prior, prior_from_ensemble

# synthetic ground truth
export SyntheticModel, truth_halfspace, truth_vector
export add_layer, add_block, add_dipping_slab, add_topography
export synth_gravity, synth_mt_sites

# metrics
export coverage, volume_reduction, rmse, mae, anomaly_correlation, calibration
export prior_report, print_report, compare_starts

# 2-D profile bridge
export grid_from_mt2dmesh, to_mt2d, from_mt2d, profile_sites

# field EDI
export read_musgrave_edi, build_musgrave_datafile2d, musgrave_phase_tensor_skew

end # module
