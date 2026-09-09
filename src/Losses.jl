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
phase in degrees scaled by `phase_weight`.

When `sites.err_rho_a` and `sites.err_phase` are omitted, the term is the original
unnormalised mean square — not chi-squared per datum, and not in the same units
as [`gravity_misfit`](@ref). Supply those errors and each residual is divided by
the uncertainty in the space the residual is scored in, which makes the term
χ²/datum. Gravity and MT weights are then comparable: a value near one means
the column is explained to within its uncertainty.

Apparent-resistivity errors arrive in ohm-metres. The residual is `log10 ρ`, so
the error is converted by first-order propagation,
`δ(log10 ρ) ≈ δρ / (ρ ln 10)`, before the division. Phase errors are already in
degrees. `phase_weight` still multiplies the phase term in either mode.

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
    return total / ns
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
"""
function sigma_penalty(sigma::AbstractVector; target::Real)
    target > 0 || throw(ArgumentError("sigma_penalty: target must be positive"))
    return mean(abs2.(sigma .- target))
end

#---------- assembled objective ----------

"""
    LossWeights(; anchor=1.0, gravity=1.0, mt=1.0, smooth=1.0e-2, sigma=1.0e-2,
                reference=0.0)

Relative weights of the loss terms.

Defaults put the data terms on equal footing and the spatial regularisers
(`smooth`, `sigma`) two orders of magnitude below, a starting point rather than
a calibration. `reference` is off by default: it multiplies squared decades
against chi-squared-per-datum data terms, so `reference = 0.1` means one decade
of departure from the baseline must buy 0.1 of data misfit to be worth keeping.
Zero means the term is reported but does not enter the total. Any residual-mode
run with a baseline should set it; the value is calibrated per survey by
watching the terms, not guessed.
"""
Base.@kwdef struct LossWeights
    anchor::Float64 = 1.0
    gravity::Float64 = 1.0
    mt::Float64 = 1.0
    smooth::Float64 = 1.0e-2
    sigma::Float64 = 1.0e-2
    reference::Float64 = 0.0
end

"""
    PriorTargets(; anchors=nothing, gravity=nothing, mt=nothing, reference=nothing,
                 sigma_target=0.5, phase_weight=1.0, vertical_weight=1.0,
                 reference_weight=nothing)

Everything the loss needs besides the network output.

- `anchors`: `(cells, values, weights)` where `cells` are linear cell indices,
  `values` log10 resistivities to match and `weights` their relative trust.
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

Any term whose inputs are absent is skipped, so the same objective works on a
survey with gravity but no boreholes, or anchors but no gravity. The damping
term is skipped entirely when `reference` is `nothing`.
"""
Base.@kwdef struct PriorTargets
    anchors::Union{Nothing,Tuple{Vector{Int},Vector{Float64},Vector{Float64}}} = nothing
    gravity::Union{Nothing,Tuple{Matrix{Float64},Vector{Float64},Vector{Float64}}} = nothing
    mt::Union{Nothing,Tuple{MTSites,Vector{Tuple{Int,Int}}}} = nothing
    reference::Union{Nothing,Vector{Float64}} = nothing
    gravity_detrend::Symbol = :mean
    sigma_target::Float64 = 0.5
    phase_weight::Float64 = 1.0
    vertical_weight::Float64 = 1.0
    reference_weight::Union{Nothing,Vector{Float64}} = nothing
end

"""
    prior_loss(mu, sigma, coupling, grid, targets, weights) -> Real

Total training objective. See [`loss_report`](@ref) for the term-by-term split.

`mu` and `sigma` are flat vectors over cells in `vec` order.
"""
function prior_loss(mu::AbstractVector, sigma::AbstractVector,
                    coupling::GravityCoupling,
                    grid::PriorGrid,
                    targets::PriorTargets,
                    weights::LossWeights = LossWeights())
    return first(_loss_terms(mu, sigma, coupling, grid, targets, weights))
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

    total = zero(eltype(mu)) + 0.0
    anchor_term = NaN
    gravity_term = NaN
    mt_term = NaN
    reference_term = NaN

    if targets.reference !== nothing
        length(targets.reference) == n || throw(DimensionMismatch(
            "prior_loss: reference has $(length(targets.reference)) entries but the grid has $(n) cells"))
    end

    if targets.anchors !== nothing
        cells, values, w = targets.anchors
        all(c -> 1 <= c <= n, cells) || throw(ArgumentError(
            "prior_loss: anchor cell indices must lie in 1:$(n)"))
        anchor_term = heteroscedastic_nll(mu[cells], sigma[cells], values; weight = w)
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

    sigma_term = sigma_penalty(sigma; target = targets.sigma_target)
    total += weights.sigma * sigma_term

    if targets.reference !== nothing
        reference_term = reference_penalty(mu, targets.reference;
                                           weight = targets.reference_weight)
        total += weights.reference * reference_term
    end

    return total, (anchor = anchor_term, gravity = gravity_term, mt = mt_term,
                   smooth = smooth_term, sigma = sigma_term, reference = reference_term)
end
