"""
    SmartPrior

Julia/Lux.jl neural field for 3D block models: predict grade, density,
magnetic susceptibility, and other rock properties with heteroscedastic
uncertainty from **covariates that are known everywhere** in the model box —
normalised coordinates and depth — plus optional semi-synthetic fields for
method tests.

**Not inputs (leakage):** pXRF geochemistry, drillhole lithology, and
distance to the nearest sample. Those are training labels or confidence
masks only; [`load_site`](@ref) rejects them as covariates.

**Not a geophysical inversion:** gravity, magnetics, and MT are neither
inputs nor outputs here.

**Sites:** [`load_site`](@ref) loads a [`SampleTable`](@ref) from TOML under
`sites/` — **Keivitsa** (GTK). Desurvey is in [`Desurvey.jl`](@ref); the site
reader is [`KeivitsaIO.jl`](@ref). The older grid stack is archived under
`legacy/grid_stack/` and is not part of this module.
"""
module SmartPrior

using Random
using Statistics

using Random: AbstractRNG, Xoshiro

using ArchGDAL
using TOML

include("Schema.jl")
include("Desurvey.jl")
include("Covariates.jl")
include("SyntheticFields.jl")
include("KeivitsaIO.jl")
include("Sites.jl")

# observations and covariates
export PropertySpec, SampleTable, nsamples, subset, observed_mask, training_mask
export real_hole_mask
export Covariate, evaluate, channel_names, evaluate_all
export CoordinateCovariate, DepthCovariate, DepthBelowSurface

# desurvey
export DesurveyPath, desurvey, positions

# Keivitsa
export keivitsa_sample_table, KeivitsaReport, KeivitsaExclusion, exclusion_summary

# site files
export load_site

# synthetic fields
export GaussianField, gaussian_field, latent_field, smooth_field, SYNTHETIC_RFF_M

end # module
