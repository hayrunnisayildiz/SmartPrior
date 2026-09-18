# Training objective for the prior field.
#
# The loss has six jobs, and they pull against each other on purpose:
#
#   1. likelihood     -- match the anchors we trust, and say how much we trust
#                        ourselves while doing it (heteroscedastic NLL)
#   2. gravity        -- explain the observed gravity anomaly
#   3. MT             -- stay consistent with the observed sounding curves
#   4. smoothness     -- prefer spatially coherent structure
#   5. sigma          -- keep unconstrained cells at a stated default
#   6. damping        -- charge for departing from the reference model
#
# Without the damping term a residual-mode network can wander `residual_span`
# decades from the baseline at no cost. Data terms then settle well below one
# (fitting noise) and the prior scores worse than the Niblett-Bostick field it
# started from.
#
# The density-resistivity link deserves a note, because it is the weakest joint
# in any joint gravity-MT scheme. Rather than imposing a petrophysical law, the
# slope is a single trainable scalar whose sign is free. Dense rock is conductive
# in a graphitic shale and resistive in a fresh granite, and serpentinised
# ultramafics manage to be dense and conductive at once; no fixed transform can
# be right in all three. A learnable slope lets the data decide, and when gravity
# carries no usable information the slope decays to zero and the prior falls back
# to the Niblett-Bostick background instead of being actively misled.
#
# That slope is dimensionless, and it has to be. Expressed directly in kg/m^3 per
# decade it would need to reach values in the hundreds while every network weight
# lives at order one, and a single learning rate cannot serve both: the slope
# crawls, the gravity misfit stays enormous, and the fastest way for the network
# to reduce it is to drive its own output to the edge of the residual range and
# flatten. The observed symptom is a perfectly uniform mu pinned at
# `reference - residual_span`. Factoring out `DENSITY_SCALE` puts the trainable
# quantity at order one and the failure disappears.

"""
    DENSITY_SCALE

Density contrast in kg/m^3 corresponding to one decade of resistivity at unit
coupling, fixed at 100.

A non-dimensionalisation constant, not a petrophysical claim. It is chosen so a
trained [`GravityCoupling`](@ref) slope comes out at order one for ordinary
crustal contrasts, which are a few hundred kg/m^3 across a decade or two. Read a
learned slope of `-2.5` as "dense rock is conductive, at 250 kg/m^3 per decade";
[`physical_slope`](@ref) does that conversion.
"""
const DENSITY_SCALE = 100.0

"""
    GravityCoupling(slope, offset)

Linear map from a log10 resistivity anomaly to a density contrast.

Both fields are dimensionless multipliers of [`DENSITY_SCALE`](@ref) and both are
trainable. `slope` is *signed*: negative means dense rock is conductive, positive
the opposite.
"""
struct GravityCoupling{T<:Real}
    slope::T
    offset::T
end

GravityCoupling(; slope::Real = 0.0, offset::Real = 0.0) =
    GravityCoupling(promote(float(slope), float(offset))...)

"""
    physical_slope(c::GravityCoupling) -> Real

The coupling slope in kg/m^3 per decade, for reporting and interpretation.
"""
physical_slope(c::GravityCoupling) = c.slope * DENSITY_SCALE

"""
    density_from_mu(coupling, mu, reference) -> Vector

Density contrast in kg/m^3 implied by a log10 resistivity field:

    DENSITY_SCALE * (slope * (mu - reference) + offset)

`reference` is the level the anomaly is measured against, normally the same
Niblett-Bostick baseline the network predicts a residual against, so that only
the *structure* the network adds is asked to explain the gravity field. Feeding
absolute resistivity here instead would make the regional gravity level fight the
absolute resistivity level, two quantities that have no reason to agree.
"""
function density_from_mu(c::GravityCoupling, mu::AbstractVector, reference::AbstractVector)
    length(mu) == length(reference) || throw(DimensionMismatch(
        "density_from_mu: mu has $(length(mu)) entries, reference has $(length(reference))"))
    return DENSITY_SCALE .* (c.slope .* (mu .- reference) .+ c.offset)
end

#---------- individual terms ----------

"""
    heteroscedastic_nll(mu, sigma, target; weight=nothing) -> Real

Negative log-likelihood of `target` under a Gaussian with mean `mu` and standard
deviation `sigma`, dropping the constant term:

    mean(w * (0.5 * ((target - mu) / sigma)^2 + log(sigma)))

The `log(sigma)` term is what makes the uncertainty learnable rather than free:
widening `sigma` buys a smaller squared residual but costs directly, so the
optimum is reached where the predicted spread matches the actual error.

`weight` scales individual anchors, for instance by how much a borehole is
trusted relative to an interpolated constraint.
"""
function heteroscedastic_nll(mu::AbstractVector, sigma::AbstractVector,
                             target::AbstractVector;
                             weight::Union{Nothing,AbstractVector} = nothing)
    n = length(mu)
    (length(sigma) == n && length(target) == n) || throw(DimensionMismatch(
        "heteroscedastic_nll: mu, sigma and target must have equal length"))
    n > 0 || throw(ArgumentError("heteroscedastic_nll: no anchors supplied"))

    terms = @. 0.5 * ((target - mu) / sigma)^2 + log(sigma)
    weight === nothing && return mean(terms)

    length(weight) == n || throw(DimensionMismatch(
        "heteroscedastic_nll: weight has $(length(weight)) entries but $(n) anchors were given"))
    return sum(weight .* terms) / sum(weight)
end

"""
    gravity_misfit(A, density, obs, err; detrend=:mean) -> Real

Chi-squared-per-datum gravity misfit, `mean(((A * density - obs) / err)^2)`.

A value near one means the field is explained to within its uncertainty; pushing
below one is fitting noise.

`detrend = :mean` removes the error-weighted mean of the residual before scoring,
which is almost always what you want. The absolute level of a gravity anomaly is
not a property of the model: it depends on the reference the survey was reduced
to and on mass outside the modelled volume, and no amount of correct structure
will reproduce it. Left in, that constant dominates the misfit and the cheapest
way to reduce it is to invent a uniform density across the whole grid -- which
then distorts the shape the data actually constrains, and can push the coupling
slope to the wrong sign entirely.

Use `:none` only for data you know has been reduced to the same zero level the
forward operator assumes. A regional trend of higher order than a constant should
be removed from the observations before they reach here.
"""
function gravity_misfit(A::AbstractMatrix, density::AbstractVector,
                        obs::AbstractVector, err::AbstractVector;
                        detrend::Symbol = :mean)
    size(A, 2) == length(density) || throw(DimensionMismatch(
        "gravity_misfit: A has $(size(A, 2)) columns but density has $(length(density)) cells"))
    (length(obs) == size(A, 1) && length(err) == size(A, 1)) || throw(DimensionMismatch(
        "gravity_misfit: obs and err must have one entry per station ($(size(A, 1)))"))
    all(>(0), err) || throw(ArgumentError("gravity_misfit: err must be strictly positive"))
    detrend in (:none, :mean) || throw(ArgumentError(
        "gravity_misfit: detrend must be :none or :mean, got :$(detrend)"))

    r = A * density .- obs
    if detrend === :mean
        # the error-weighted constant that minimises the misfit, so a station with
        # a large stated uncertainty does not drag the level it is fitted to
        w = 1 ./ abs2.(err)
        r = r .- (sum(w .* r) / sum(w))
    end
    return mean(abs2.(r ./ err))
end

"""
    mt_column_misfit(mu, grid, sites, site_cells; phase_weight=1.0)
        -> Real

MT misfit computed one column at a time with the 1-D solver.

`site_cells` gives the `(i, j)` grid column under each site. Apparent resistivity
enters in log10 so that a decade of error costs the same wherever it happens, and
phase in degrees.

Apparent resistivity and phase are two separate data types, so the result is the
**weighted mean** of their two misfits with weights `(1, phase_weight)`, averaged
over sites. `phase_weight` therefore re-partitions the term between the two
curves instead of adding to its size: the perfect-fit and one-sigma values do not
move when it changes. Summing the two instead would make the term score
`1 + phase_weight` at a one-sigma fit and hand MT `1 + phase_weight` times
gravity's pull under nominally equal [`LossWeights`](@ref).

When `sites.err_rho_a` and `sites.err_phase` are omitted, the residuals are not
divided by an uncertainty, so the term is not chi-squared per datum and not in
the same units as [`gravity_misfit`](@ref). Supply those errors and each residual
is divided by the uncertainty in the space the residual is scored in, which makes
the term χ²/datum on gravity's scale: a value near one means the column is
explained to within its uncertainty, and pushing below one is fitting noise.

Apparent-resistivity errors arrive in ohm-metres. The residual is `log10 ρ`, so
the error is converted by first-order propagation,
`δ(log10 ρ) ≈ δρ / (ρ ln 10)`, before the division. Phase errors are already in
degrees.

A 1-D response per column is not the 3-D response of the model, so this is a
consistency term rather than a data fit: it stops the network proposing a column
whose own 1-D response contradicts the measured curve, which is a cheap way to
catch the grossly wrong structures without a 3-D forward solve. The proper 3-D
misfit is the inversion's job, and this prior exists to make that job easier.
"""
function mt_column_misfit(mu::AbstractArray{<:Real,3},
                          grid::PriorGrid,
                          sites::MTSites,
                          site_cells::AbstractVector{<:Tuple{Integer,Integer}};
                          phase_weight::Real = 1.0)
    ns = nsites(sites)
    length(site_cells) == ns || throw(DimensionMismatch(
        "mt_column_misfit: expected one grid column per site ($(ns)), got $(length(site_cells))"))
    nx, ny, nz = size(grid)
    size(mu) == (nx, ny, nz) || throw(DimensionMismatch(
        "mt_column_misfit: mu must match the grid $(nx)x$(ny)x$(nz), got $(size(mu))"))
    phase_weight >= 0 || throw(ArgumentError(
        "mt_column_misfit: phase_weight must be non-negative, got $(phase_weight)"))

    have_err_a = sites.err_rho_a !== nothing
    have_err_p = sites.err_phase !== nothing
    if !have_err_a && !have_err_p
        @warn "mt_column_misfit: no error given, term is not chi-squared calibrated and its weight is not comparable to gravity's" maxlog=1
    end
    if have_err_a
        all(>(0), sites.err_rho_a) || throw(ArgumentError(
            "mt_column_misfit: err_rho_a must be strictly positive"))
    end
    if have_err_p
        all(>(0), sites.err_phase) || throw(ArgumentError(
            "mt_column_misfit: err_phase must be strictly positive"))
    end

    f = 1 ./ sites.periods
    total = zero(eltype(mu)) + 0.0
    ln10 = log(10)
    for t in 1:ns
        i, j = site_cells[t]
        (1 <= i <= nx && 1 <= j <= ny) || throw(ArgumentError(
            "mt_column_misfit: site $t maps to column ($i, $j), outside the grid"))

        rho_col = 10 .^ mu[i, j, :]
        pred_a, pred_p = mt1d_column_response(f, rho_col, grid.dz)

        obs_a = view(sites.rho_a, :, t)
        obs_p = view(sites.phase, :, t)

        if have_err_a
            # δ(log10 x) ≈ δx / (x ln 10). Residual is scored in log10, so a
            # linear ohm-metre error has to make that trip before it can
            # normalise; otherwise the term is not χ²/datum and cannot share
            # gravity_misfit's scale.
            err_log10 = view(sites.err_rho_a, :, t) ./ (obs_a .* ln10)
            total += mean(abs2.((log10.(pred_a) .- log10.(obs_a)) ./ err_log10))
        else
            total += mean(abs2.(log10.(pred_a) .- log10.(obs_a)))
        end

        if have_err_p
            total += phase_weight * mean(abs2.((pred_p .- obs_p) ./ view(sites.err_phase, :, t)))
        else
            total += phase_weight * mean(abs2.((pred_p .- obs_p) ./ 45.0))
        end
    end
    # Apparent resistivity and phase are two data types, and the site loop adds
    # one chi-squared-per-datum for each. Dividing by their weights turns that
    # sum into a weighted mean, which is what puts the term on gravity_misfit's
    # scale: a one-sigma residual on every datum scores 1.0, not 1 + phase_weight.
    # Without it `LossWeights(gravity = 1, mt = 1)` gives MT twice the pull.
    return total / (ns * (1 + phase_weight))
end

"""
    smoothness(field, grid; vertical_weight=1.0) -> Real

Mean squared first difference of a `[nx, ny, nz]` field, with each difference
normalised by the distance between the two cell centres so that a graded mesh
does not make the fine layers look rough.

`vertical_weight` below one lets vertical contrast pass more cheaply than
lateral, which suits layered geology: real sections have sharp horizons and
smoother lateral variation.
"""
function smoothness(field::AbstractArray{<:Real,3}, grid::PriorGrid;
                    vertical_weight::Real = 1.0)
    nx, ny, nz = size(grid)
    size(field) == (nx, ny, nz) || throw(DimensionMismatch(
        "smoothness: field must match the grid $(nx)x$(ny)x$(nz), got $(size(field))"))

    total = 0.0
    count = 0

    if nx > 1
        for i in 1:nx-1
            h = grid.cx[i+1] - grid.cx[i]
            d = (field[i+1, :, :] .- field[i, :, :]) ./ h
            total += sum(abs2, d)
            count += length(d)
        end
    end
    if ny > 1
        for j in 1:ny-1
            h = grid.cy[j+1] - grid.cy[j]
            d = (field[:, j+1, :] .- field[:, j, :]) ./ h
            total += sum(abs2, d)
            count += length(d)
        end
    end
    if nz > 1
        for k in 1:nz-1
            h = grid.cz[k+1] - grid.cz[k]
            d = (field[:, :, k+1] .- field[:, :, k]) ./ h
            total += vertical_weight * sum(abs2, d)
            count += length(d)
        end
    end

    count == 0 && return 0.0
    # scale so the number is independent of the survey's absolute size
    return (total / count) * grid.h_median^2
end

"""
    reference_penalty(mu, reference; weight=nothing) -> Real

Mean squared departure of `mu` from `reference`, in decades squared. With
`weight` given, a weighted mean instead, which is where a depth taper belongs:
the Niblett-Bostick baseline is only trustworthy down to the depth the periods
actually penetrate, and below that the reference deserves less authority.

This is the zeroth-order Tikhonov term that residual-mode training otherwise
lacks. None of the other terms charges for the network leaving the baseline, so
without it `mu` can wander `residual_span` decades from `offset` at no cost.
The data terms then settle well below one (fitting noise) with structure in the
wrong place, and the trained prior scores worse than the baseline it started
from. Every decade of structure has a price here, so it only survives if it
buys more misfit reduction than it costs.
"""
function reference_penalty(mu::AbstractVector, reference::AbstractVector;
                           weight::Union{Nothing,AbstractVector} = nothing)
    n = length(mu)
    length(reference) == n || throw(DimensionMismatch(
        "reference_penalty: mu has $(n) entries, reference has $(length(reference))"))
    d2 = abs2.(mu .- reference)
    weight === nothing && return mean(d2)
    length(weight) == n || throw(DimensionMismatch(
        "reference_penalty: weight has $(length(weight)) entries but mu has $(n)"))
    all(>=(0), weight) || throw(ArgumentError("reference_penalty: weight must be non-negative"))
    s = sum(weight)
    s > 0 || throw(ArgumentError("reference_penalty: weights sum to zero"))
    return sum(weight .* d2) / s
end

"""
    sigma_penalty(sigma; target) -> Real

Mean squared deviation of `sigma` from `target`, in decades.

Without it the likelihood term has a degenerate direction: away from any anchor
`sigma` can drift to whichever bound costs least, and since the bounds are hard
the gradient there vanishes and the value sticks. This keeps unconstrained cells
at a stated default rather than at an accident of initialisation.

`target` may be a scalar (applied uniformly) or a vector of the same length as
`sigma` (per-cell targets from [`compute_sigma_targets`](@ref)). The computation
is the same in both cases thanks to broadcasting.
"""
function sigma_penalty(sigma::AbstractVector; target::Union{Real,AbstractVector})
    if target isa AbstractVector
        length(sigma) == length(target) || throw(DimensionMismatch(
            "sigma_penalty: sigma has $(length(sigma)) entries but target has $(length(target))"))
        all(>(0), target) || throw(ArgumentError(
            "sigma_penalty: all per-cell targets must be positive"))
    else
        target > 0 || throw(ArgumentError("sigma_penalty: target must be positive"))
    end
    return mean(abs2.(sigma .- target))
end

#---------- data-driven sigma ----------

"""
    SigmaDriveConfig(; alpha=0.5, beta=0.3, baseline=0.15,
                     sigma_lo=0.05, sigma_hi=0.9)

Configuration for computing per-cell sigma targets from data misfit.

- `alpha`: weight of the MT column residual contribution. A column that fits
  poorly in log10 ρ_a gets a wider search interval.
- `beta`: weight of the gravity insensitivity contribution. Cells the gravity
  operator barely sees (low column norm in the forward matrix) get wider intervals.
- `baseline`: the minimum / default sigma for cells not covered by any data term.
- `sigma_lo`, `sigma_hi`: hard clamp applied after combining the terms. Should
  match or sit inside the network's `sigma_bounds`.
"""
Base.@kwdef struct SigmaDriveConfig
    alpha::Float64 = 0.5
    beta::Float64 = 0.3
    baseline::Float64 = 0.15
    sigma_lo::Float64 = 0.05
    sigma_hi::Float64 = 0.9

    function SigmaDriveConfig(alpha, beta, baseline, sigma_lo, sigma_hi)
        alpha >= 0 || throw(ArgumentError("SigmaDriveConfig: alpha must be non-negative"))
        beta >= 0 || throw(ArgumentError("SigmaDriveConfig: beta must be non-negative"))
        baseline > 0 || throw(ArgumentError("SigmaDriveConfig: baseline must be positive"))
        0 < sigma_lo < sigma_hi ||
            throw(ArgumentError("SigmaDriveConfig: need 0 < sigma_lo < sigma_hi"))
        return new(alpha, beta, baseline, sigma_lo, sigma_hi)
    end
end

"""
    mt_column_residuals(mu3, grid, sites, site_cells) -> Vector{Float64}

Per-site root-mean-square residual in log10 apparent resistivity space.

This is the non-AD, diagnostic version of the column check: it tells how well
the current `mu` field explains each sounding curve through its own 1-D response.
Cells under high-residual columns are poorly constrained and deserve a wider
search interval.
"""
function mt_column_residuals(mu3::AbstractArray{<:Real,3},
                             grid::PriorGrid,
                             sites::MTSites,
                             site_cells::AbstractVector{<:Tuple{Integer,Integer}})
    ns = nsites(sites)
    length(site_cells) == ns || throw(DimensionMismatch(
        "mt_column_residuals: expected one column per site ($(ns)), got $(length(site_cells))"))
    nx, ny, nz = size(grid)
    size(mu3) == (nx, ny, nz) || throw(DimensionMismatch(
        "mt_column_residuals: mu must match the grid $(nx)×$(ny)×$(nz), got $(size(mu3))"))

    f = 1 ./ sites.periods
    residuals = Vector{Float64}(undef, ns)
    for t in 1:ns
        i, j = site_cells[t]
        rho_col = 10 .^ Float64.(mu3[i, j, :])
        pred_a, _ = mt1d_column_response(f, rho_col, grid.dz)
        obs_a = view(sites.rho_a, :, t)
        residuals[t] = sqrt(mean(abs2.(log10.(pred_a) .- log10.(obs_a))))
    end
    return residuals
end

"""
    compute_sigma_targets(mu_vec, grid, targets, coupling;
                          config::SigmaDriveConfig) -> Vector{Float64}

Per-cell sigma targets derived from data misfit.

The idea: where the model already fits well, the search interval can be narrow
(small σ); where it fits poorly or the data have no sensitivity, the interval
should stay wide (large σ). Two signals feed in:

1. **MT column residual** — each site's RMS log10 ρ_a error is assigned to every
   cell in the column beneath it. Cells not under any site get `baseline`.
2. **Gravity sensitivity** — the column norm of the forward operator tells how
   much each cell contributes to the surface anomaly. Cells with low sensitivity
   are gravity-blind and get a wider interval.

This function is called **outside** the AD graph — its output is treated as a
constant target by `sigma_penalty`, so no gradient flows back through it to `mu`.
"""
function compute_sigma_targets(mu_vec::AbstractVector{<:Real},
                               grid::PriorGrid,
                               targets,  # PriorTargets (forward-declared)
                               coupling::GravityCoupling;
                               config::SigmaDriveConfig = SigmaDriveConfig())
    n = ncells(grid)
    length(mu_vec) == n || throw(DimensionMismatch(
        "compute_sigma_targets: mu has $(length(mu_vec)) entries, grid has $(n) cells"))
    nx, ny, nz = size(grid)

    sigma_t = fill(config.baseline, n)

    # ---- MT contribution ----
    if targets.mt !== nothing
        sites, site_cells = targets.mt
        mu3 = reshape(collect(Float64, mu_vec), (nx, ny, nz))
        residuals = mt_column_residuals(mu3, grid, sites, site_cells)
        li = LinearIndices((nx, ny, nz))
        for (t, (ci, cj)) in enumerate(site_cells)
            for k in 1:nz
                sigma_t[li[ci, cj, k]] += config.alpha * residuals[t]
            end
        end
    end

    # ---- gravity sensitivity contribution ----
    if targets.gravity !== nothing
        A, _, _ = targets.gravity
        size(A, 2) == n || throw(DimensionMismatch(
            "compute_sigma_targets: gravity matrix has $(size(A,2)) columns, grid has $(n) cells"))
        col_norms = [norm(view(A, :, c)) for c in 1:n]
        mx = maximum(col_norms)
        if mx > 0
            col_norms ./= mx
            sigma_t .+= config.beta .* (1.0 .- col_norms)
        end
    end

    return clamp.(sigma_t, config.sigma_lo, config.sigma_hi)
end

#---------- assembled objective ----------

"""
    LossWeights(; anchor=1.0, gravity=1.0, mt=1.0, smooth=1.0e-2, sigma=1.0e-2,
                reference=0.0, grade=1.0, density=1.0, susceptibility=1.0,
                resistivity=1.0)

Relative weights of the loss terms.

Defaults put the data terms on equal footing and the spatial regularisers
(`smooth`, `sigma`) two orders of magnitude below, a starting point rather than
a calibration. `reference` is off by default: it multiplies squared decades
against chi-squared-per-datum data terms, so `reference = 0.1` means one decade
of departure from the baseline must buy 0.1 of data misfit to be worth keeping.
Zero means the term is reported but does not enter the total. Any residual-mode
run with a baseline should set it; the value is calibrated per survey by
watching the terms, not guessed.

`grade`, `density`, `susceptibility` and `resistivity` weight the four named
anchor groups used by the Keivitsa line. They do not affect the single-property
`anchor` term.
"""
Base.@kwdef struct LossWeights
    anchor::Float64 = 1.0
    gravity::Float64 = 1.0
    mt::Float64 = 1.0
    smooth::Float64 = 1.0e-2
    sigma::Float64 = 1.0e-2
    reference::Float64 = 0.0
    grade::Float64 = 1.0
    density::Float64 = 1.0
    susceptibility::Float64 = 1.0
    resistivity::Float64 = 1.0
end

const AnchorSet = Tuple{Vector{Int},Vector{Float64},Vector{Float64}}

"""
    PriorTargets(; anchors=nothing, gravity=nothing, mt=nothing, reference=nothing,
                 sigma_target=0.5, sigma_drive=nothing, sigma_target_vec=nothing,
                 phase_weight=1.0, vertical_weight=1.0, reference_weight=nothing,
                 anchors_grade=nothing, anchors_density=nothing,
                 anchors_susceptibility=nothing, anchors_resistivity=nothing,
                 property_names=String[])

Everything the loss needs besides the network output.

- `anchors`: `(cells, values, weights)` where `cells` are linear cell indices,
  `values` log10 resistivities to match and `weights` their relative trust.
  The original single-property group; ignored when the network has several
  outputs.
- `anchors_grade`, `anchors_density`, `anchors_susceptibility`,
  `anchors_resistivity`: the same triple, one independent NLL per Keivitsa
  property. Each uses its own row of the multi-property `(mu, sigma)` and its
  own [`LossWeights`](@ref) field. The NLL is scored in units of that group's
  weighted std and doubled so a one-std residual with `sigma` equal to that
  std is `1.0` (the χ²/datum convention of [`gravity_misfit`](@ref) /
  [`mt_column_misfit`](@ref)). Constant-valued groups keep native units.
- `gravity`: `(A, obs, err)` from [`gravity_matrix`](@ref) and a [`GravityObs`](@ref).
- `mt`: `(sites, site_cells)` for the 1-D consistency term.
- `reference`: serves two terms. It is the level gravity anomalies are measured
  against *and* the model the damping term pulls toward. In residual mode both
  are the Niblett-Bostick baseline, which is also `offset`, so the damping term
  charges directly for the structure the network adds. Required when `gravity`
  is given.
- `reference_weight`: optional per-cell weights for [`reference_penalty`](@ref);
  a depth taper lives here.
- `gravity_detrend`: passed to [`gravity_misfit`](@ref); see there for why the
  default removes the residual's mean.
- `sigma_drive`: when set, [`train_prior`](@ref) computes per-cell sigma targets
  from data misfit at each epoch using [`compute_sigma_targets`](@ref), replacing
  the uniform `sigma_target`.
- `sigma_target_vec`: per-cell sigma targets computed by the training loop. Users
  should not set this directly; it is populated by `train_prior` when
  `sigma_drive` is present.

Any term whose inputs are absent is skipped, so the same objective works on a
survey with gravity but no boreholes, or anchors but no gravity. The damping
term is skipped entirely when `reference` is `nothing`.
"""
Base.@kwdef struct PriorTargets
    anchors::Union{Nothing,AnchorSet} = nothing
    gravity::Union{Nothing,Tuple{Matrix{Float64},Vector{Float64},Vector{Float64}}} = nothing
    mt::Union{Nothing,Tuple{MTSites,Vector{Tuple{Int,Int}}}} = nothing
    reference::Union{Nothing,Vector{Float64}} = nothing
    gravity_detrend::Symbol = :mean
    sigma_target::Float64 = 0.5
    sigma_drive::Union{Nothing,SigmaDriveConfig} = nothing
    sigma_target_vec::Union{Nothing,Vector{Float64}} = nothing
    phase_weight::Float64 = 1.0
    vertical_weight::Float64 = 1.0
    reference_weight::Union{Nothing,Vector{Float64}} = nothing
    anchors_grade::Union{Nothing,AnchorSet} = nothing
    anchors_density::Union{Nothing,AnchorSet} = nothing
    anchors_susceptibility::Union{Nothing,AnchorSet} = nothing
    anchors_resistivity::Union{Nothing,AnchorSet} = nothing
    property_names::Vector{String} = String[]
end

function _has_named_anchors(t::PriorTargets)
    return t.anchors_grade !== nothing || t.anchors_density !== nothing ||
           t.anchors_susceptibility !== nothing || t.anchors_resistivity !== nothing
end

function _has_any_anchors(t::PriorTargets)
    return t.anchors !== nothing || _has_named_anchors(t)
end

"""
    prior_loss(mu, sigma, coupling, grid, targets, weights) -> Real

Total training objective. See [`loss_report`](@ref) for the term-by-term split.

`mu` and `sigma` are flat vectors over cells in `vec` order, or
`[nproperties, ncell]` matrices for a multi-property net.
"""
function prior_loss(mu::AbstractVector, sigma::AbstractVector,
                    coupling::GravityCoupling,
                    grid::PriorGrid,
                    targets::PriorTargets,
                    weights::LossWeights = LossWeights())
    return first(_loss_terms(mu, sigma, coupling, grid, targets, weights))
end

function prior_loss(mus::AbstractMatrix, sigmas::AbstractMatrix,
                    coupling::GravityCoupling,
                    grid::PriorGrid,
                    targets::PriorTargets,
                    weights::LossWeights = LossWeights())
    return first(_loss_terms(mus, sigmas, coupling, grid, targets, weights))
end

"""
    loss_report(mu, sigma, coupling, grid, targets, weights) -> NamedTuple

The total loss alongside each unweighted term, for diagnosing which constraint is
actually driving training. Terms whose inputs were absent come back as `NaN`.
"""
function loss_report(mu::AbstractVector, sigma::AbstractVector,
                     coupling::GravityCoupling,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights = LossWeights())
    total, terms = _loss_terms(mu, sigma, coupling, grid, targets, weights)
    return (total = total, terms...)
end

function loss_report(mus::AbstractMatrix, sigmas::AbstractMatrix,
                     coupling::GravityCoupling,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights = LossWeights())
    total, terms = _loss_terms(mus, sigmas, coupling, grid, targets, weights)
    return (total = total, terms...)
end

function _nll_anchors(mu::AbstractVector, sigma::AbstractVector, anchors, n::Int)
    cells, values, w = anchors
    all(c -> 1 <= c <= n, cells) || throw(ArgumentError(
        "prior_loss: anchor cell indices must lie in 1:$(n)"))
    return heteroscedastic_nll(mu[cells], sigma[cells], values; weight = w)
end

# Weighted std of an anchor group. A constant-valued group has no scale, in
# which case the caller leaves the NLL in native units (`scale = 1`).
function _anchor_scale(values::AbstractVector{<:Real},
                       weight::AbstractVector{<:Real})
    n = length(values)
    n == 0 && return 1.0
    sw = 0.0
    macc = 0.0
    @inbounds for i in 1:n
        wi = Float64(weight[i])
        sw += wi
        macc += wi * Float64(values[i])
    end
    sw <= 0 && return 1.0
    m = macc / sw
    vacc = 0.0
    @inbounds for i in 1:n
        vacc += Float64(weight[i]) * abs2(Float64(values[i]) - m)
    end
    s = sqrt(vacc / sw)
    return s > 1e-12 ? s : 1.0
end

"""
    sigma_bounds_from_anchors(anchors; fraction=1.0, hi=1.2) -> (lo, hi)

Hard squash interval for one property's σ.

The calibrated NLL contains `2 log(σ / s)` where `s` is the group's weighted
std. If σ is allowed below `s`, that term goes negative and looks like a
reward for overconfidence. The lower edge is therefore `fraction * s`
(default `fraction = 1`, so σ cannot undercut the observation scale). A
10–20 % fraction would sit *below* the historical global floor of 0.05 for
a narrow group such as density (`s ≈ 0.1`) and would make the failure worse.

The upper edge is at least `hi` and at least `2 s`, so a wide group
(resistivity, several log10 decades) still has room above its scatter.
"""
function sigma_bounds_from_anchors(anchors; fraction::Real = 1.0, hi::Real = 1.2)
    0 < fraction || throw(ArgumentError(
        "sigma_bounds_from_anchors: fraction must be positive"))
    hi > 0 || throw(ArgumentError("sigma_bounds_from_anchors: hi must be positive"))
    _, values, w = anchors
    s = _anchor_scale(values, w)
    lo = Float64(fraction) * s
    cap = max(Float64(hi), 2 * s)
    lo < cap || throw(ArgumentError(
        "sigma_bounds_from_anchors: lo $(lo) is not below hi $(cap) (s=$(s))"))
    return (lo, cap)
end

"""
    _calibrated_nll(mu, sigma, anchors, n) -> Real

[`heteroscedastic_nll`](@ref) scored in units of the group's own weighted std,
then doubled so a one-std residual with `sigma` equal to that std is `1.0`.

This is the named-anchor analogue of `mt_column_misfit`'s
`total / (ns * (1 + phase_weight))` and of [`gravity_misfit`](@ref) being
χ²/datum: equal [`LossWeights`](@ref) then mean equal pull, instead of density
(narrow g/cm³) cancelling resistivity (several log10 decades) in the total.

`2 * (nll - log(s))` equals `mean(((t-μ)/σ)^2 + 2 log(σ/s))`. `s` is data, not
a network output, so the shift does not leak into the gradient of `μ` or `σ`
beyond the intended rescaling of `log(σ)`.
"""
function _calibrated_nll(mu::AbstractVector, sigma::AbstractVector, anchors, n::Int)
    raw = _nll_anchors(mu, sigma, anchors, n)
    _, values, w = anchors
    s = _anchor_scale(values, w)
    return 2 * (raw - log(s))
end

function _loss_terms(mu::AbstractVector, sigma::AbstractVector,
                     coupling::GravityCoupling,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights)
    n = ncells(grid)
    length(mu) == n || throw(DimensionMismatch(
        "prior_loss: mu has $(length(mu)) entries but the grid has $(n) cells"))
    length(sigma) == n || throw(DimensionMismatch(
        "prior_loss: sigma has $(length(sigma)) entries but the grid has $(n) cells"))
    _has_named_anchors(targets) && throw(ArgumentError(
        "prior_loss: named anchor groups need a multi-property (matrix) mu/sigma"))

    total = zero(eltype(mu)) + 0.0
    anchor_term = NaN
    gravity_term = NaN
    mt_term = NaN
    reference_term = NaN
    grade_term = NaN
    density_term = NaN
    susc_term = NaN
    res_term = NaN

    if targets.reference !== nothing
        length(targets.reference) == n || throw(DimensionMismatch(
            "prior_loss: reference has $(length(targets.reference)) entries but the grid has $(n) cells"))
    end

    if targets.anchors !== nothing
        anchor_term = _nll_anchors(mu, sigma, targets.anchors, n)
        total += weights.anchor * anchor_term
    end

    if targets.gravity !== nothing
        targets.reference === nothing && throw(ArgumentError(
            "prior_loss: a gravity term needs `reference`, the level anomalies are measured against"))
        A, obs, err = targets.gravity
        ρ = density_from_mu(coupling, mu, targets.reference)
        gravity_term = gravity_misfit(A, ρ, obs, err; detrend = targets.gravity_detrend)
        total += weights.gravity * gravity_term
    end

    if targets.mt !== nothing
        sites, site_cells = targets.mt
        mu3 = reshape(mu, size(grid))
        mt_term = mt_column_misfit(mu3, grid, sites, site_cells;
                                   phase_weight = targets.phase_weight)
        total += weights.mt * mt_term
    end

    smooth_term = smoothness(reshape(mu, size(grid)), grid;
                             vertical_weight = targets.vertical_weight)
    total += weights.smooth * smooth_term

    effective_sigma_target = targets.sigma_target_vec !== nothing ?
        targets.sigma_target_vec : targets.sigma_target
    sigma_term = sigma_penalty(sigma; target = effective_sigma_target)
    total += weights.sigma * sigma_term

    if targets.reference !== nothing
        reference_term = reference_penalty(mu, targets.reference;
                                           weight = targets.reference_weight)
        total += weights.reference * reference_term
    end

    return total, (anchor = anchor_term, gravity = gravity_term, mt = mt_term,
                   smooth = smooth_term, sigma = sigma_term, reference = reference_term,
                   grade = grade_term, density = density_term,
                   susceptibility = susc_term, resistivity = res_term)
end

function _property_row(names::Vector{String}, name::AbstractString, nprop::Int)
    k = findfirst(==(name), names)
    k === nothing && throw(ArgumentError(
        "prior_loss: no row named $(name) in property_names $(names)"))
    1 <= k <= nprop || throw(ArgumentError(
        "prior_loss: property $(name) maps to row $k but mu has $nprop rows"))
    return k
end

function _loss_terms(mus::AbstractMatrix, sigmas::AbstractMatrix,
                     coupling::GravityCoupling,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights)
    n = ncells(grid)
    nprop = size(mus, 1)
    size(mus, 2) == n || throw(DimensionMismatch(
        "prior_loss: mu has $(size(mus, 2)) cells but the grid has $(n)"))
    size(sigmas) == size(mus) || throw(DimensionMismatch(
        "prior_loss: sigma size $(size(sigmas)) does not match mu $(size(mus))"))
    targets.gravity === nothing || throw(ArgumentError(
        "prior_loss: gravity is not defined on a multi-property mu"))
    targets.mt === nothing || throw(ArgumentError(
        "prior_loss: the MT term is not defined on a multi-property mu"))

    names = isempty(targets.property_names) ?
        (nprop == 4 ? copy(DEFAULT_MULTI_PROPERTIES) : ["prop$i" for i in 1:nprop]) :
        targets.property_names
    length(names) == nprop || throw(ArgumentError(
        "prior_loss: property_names has $(length(names)) entries, mu has $nprop rows"))

    total = zero(eltype(mus)) + 0.0
    dims = size(grid)

    groups = (
        (targets.anchors_grade, weights.grade, "grade"),
        (targets.anchors_density, weights.density, "density"),
        (targets.anchors_susceptibility, weights.susceptibility, "susceptibility"),
        (targets.anchors_resistivity, weights.resistivity, "resistivity"),
    )
    grade_term = density_term = susc_term = res_term = NaN
    for (anchors, w, pname) in groups
        anchors === nothing && continue
        row = _property_row(names, pname, nprop)
        term = _calibrated_nll(mus[row, :], sigmas[row, :], anchors, n)
        total += w * term
        if pname == "grade"
            grade_term = term
        elseif pname == "density"
            density_term = term
        elseif pname == "susceptibility"
            susc_term = term
        else
            res_term = term
        end
    end

    if targets.anchors !== nothing
        # legacy single group, applied to the first property (resistivity when
        # names follow the Keivitsa order and resistivity is last — so only if
        # the caller really meant property 1). Prefer the named groups.
        throw(ArgumentError(
            "prior_loss: use anchors_grade/density/susceptibility/resistivity " *
            "with a multi-property mu, not the legacy `anchors` field"))
    end

    smooth_acc = zero(eltype(mus)) + 0.0
    for p in 1:nprop
        smooth_acc += smoothness(reshape(mus[p, :], dims), grid;
                                 vertical_weight = targets.vertical_weight)
    end
    smooth_term = smooth_acc / nprop
    total += weights.smooth * smooth_term

    effective_sigma_target = targets.sigma_target_vec !== nothing ?
        targets.sigma_target_vec : targets.sigma_target
    sigma_acc = zero(eltype(sigmas)) + 0.0
    for p in 1:nprop
        sigma_acc += sigma_penalty(sigmas[p, :]; target = effective_sigma_target)
    end
    sigma_term = sigma_acc / nprop
    total += weights.sigma * sigma_term

    reference_term = NaN
    if targets.reference !== nothing
        # damping is a resistivity-prior idea; skip rather than guess a row
    end

    return total, (anchor = NaN, gravity = NaN, mt = NaN,
                   smooth = smooth_term, sigma = sigma_term, reference = reference_term,
                   grade = grade_term, density = density_term,
                   susceptibility = susc_term, resistivity = res_term)
end
