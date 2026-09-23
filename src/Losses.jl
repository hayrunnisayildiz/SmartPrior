# Training objective for the prior field.
#
# The remaining jobs:
#   1. likelihood     -- match the anchors we trust (heteroscedastic NLL)
#   2. smoothness     -- prefer spatially coherent structure
#   3. sigma          -- keep unconstrained cells at a stated default
#   4. damping        -- optional charge for departing from a reference

"""
    heteroscedastic_nll(mu, sigma, target; weight=nothing) -> Real

Negative log-likelihood of `target` under a Gaussian with mean `mu` and standard
deviation `sigma`, dropping the constant term:

    mean(w * (0.5 * ((target - mu) / sigma)^2 + log(sigma)))

The `log(sigma)` term is what makes the uncertainty learnable rather than free:
widening `sigma` buys a smaller squared residual but costs directly, so the
optimum is reached where the predicted spread matches the actual error.
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
    smoothness(field, grid; vertical_weight=1.0) -> Real

Mean squared first difference of a `[nx, ny, nz]` field, with each difference
normalised by the distance between the two cell centres so that a graded mesh
does not make the fine layers look rough.

`vertical_weight` below one lets vertical contrast pass more cheaply than
lateral, which suits layered geology.
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
    return (total / count) * grid.h_median^2
end

"""
    reference_penalty(mu, reference; weight=nothing) -> Real

Mean squared departure of `mu` from `reference`. With `weight` given, a
weighted mean instead.
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

Mean squared deviation of `sigma` from `target`.

`target` may be a scalar (applied uniformly) or a vector of the same length as
`sigma`.
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

"""
    LossWeights(; anchor=1.0, smooth=1.0e-2, sigma=1.0e-2, reference=0.0,
                grade=1.0, density=1.0, susceptibility=1.0,
                resistivity=1.0, conductivity=1.0)

Relative weights of the loss terms.

`grade`, `density`, `susceptibility`, `resistivity` and `conductivity` weight
the named anchor groups. `conductivity` is the Cloncurry fourth head
(`conductivity_100kHz`). They do not affect the single-property `anchor` term.
"""
Base.@kwdef struct LossWeights
    anchor::Float64 = 1.0
    smooth::Float64 = 1.0e-2
    sigma::Float64 = 1.0e-2
    reference::Float64 = 0.0
    grade::Float64 = 1.0
    density::Float64 = 1.0
    susceptibility::Float64 = 1.0
    resistivity::Float64 = 1.0
    conductivity::Float64 = 1.0
end

const AnchorSet = Tuple{Vector{Int},Vector{Float64},Vector{Float64}}

"""
    PriorTargets(; anchors=nothing, reference=nothing, sigma_target=0.5,
                 sigma_target_vec=nothing, vertical_weight=1.0,
                 reference_weight=nothing, anchors_grade=nothing,
                 anchors_density=nothing, anchors_susceptibility=nothing,
                 anchors_resistivity=nothing, anchors_conductivity=nothing,
                 property_names=String[])

Everything the loss needs besides the network output.

- `anchors`: `(cells, values, weights)` for the single-property path.
- `anchors_grade`, `anchors_density`, `anchors_susceptibility`,
  `anchors_resistivity`, `anchors_conductivity`: one independent NLL per
  named property. `anchors_conductivity` is the Cloncurry fourth head
  (`conductivity_100kHz`). The NLL is scored in units of that group's
  weighted std and doubled so a one-std residual with `sigma` equal to that
  std is `1.0`.
- `reference` / `reference_weight`: optional damping toward a background.
- `sigma_target` / `sigma_target_vec`: default or per-cell σ targets.

Any term whose inputs are absent is skipped.
"""
Base.@kwdef struct PriorTargets
    anchors::Union{Nothing,AnchorSet} = nothing
    reference::Union{Nothing,Vector{Float64}} = nothing
    sigma_target::Float64 = 0.5
    sigma_target_vec::Union{Nothing,Vector{Float64}} = nothing
    vertical_weight::Float64 = 1.0
    reference_weight::Union{Nothing,Vector{Float64}} = nothing
    anchors_grade::Union{Nothing,AnchorSet} = nothing
    anchors_density::Union{Nothing,AnchorSet} = nothing
    anchors_susceptibility::Union{Nothing,AnchorSet} = nothing
    anchors_resistivity::Union{Nothing,AnchorSet} = nothing
    anchors_conductivity::Union{Nothing,AnchorSet} = nothing
    property_names::Vector{String} = String[]
end

function _has_named_anchors(t::PriorTargets)
    return t.anchors_grade !== nothing || t.anchors_density !== nothing ||
           t.anchors_susceptibility !== nothing || t.anchors_resistivity !== nothing ||
           t.anchors_conductivity !== nothing
end

function _has_any_anchors(t::PriorTargets)
    return t.anchors !== nothing || _has_named_anchors(t)
end

"""
    prior_loss(mu, sigma, grid, targets, weights) -> Real

Total training objective. See [`loss_report`](@ref) for the term-by-term split.

`mu` and `sigma` are flat vectors over cells in `vec` order, or
`[nproperties, ncell]` matrices for a multi-property net.
"""
function prior_loss(mu::AbstractVector, sigma::AbstractVector,
                    grid::PriorGrid,
                    targets::PriorTargets,
                    weights::LossWeights = LossWeights())
    return first(_loss_terms(mu, sigma, grid, targets, weights))
end

function prior_loss(mus::AbstractMatrix, sigmas::AbstractMatrix,
                    grid::PriorGrid,
                    targets::PriorTargets,
                    weights::LossWeights = LossWeights())
    return first(_loss_terms(mus, sigmas, grid, targets, weights))
end

"""
    loss_report(mu, sigma, grid, targets, weights) -> NamedTuple

The total loss alongside each unweighted term. Terms whose inputs were absent
come back as `NaN`.
"""
function loss_report(mu::AbstractVector, sigma::AbstractVector,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights = LossWeights())
    total, terms = _loss_terms(mu, sigma, grid, targets, weights)
    return (total = total, terms...)
end

function loss_report(mus::AbstractMatrix, sigmas::AbstractMatrix,
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights = LossWeights())
    total, terms = _loss_terms(mus, sigmas, grid, targets, weights)
    return (total = total, terms...)
end

function _nll_anchors(mu::AbstractVector, sigma::AbstractVector, anchors, n::Int)
    cells, values, w = anchors
    all(c -> 1 <= c <= n, cells) || throw(ArgumentError(
        "prior_loss: anchor cell indices must lie in 1:$(n)"))
    return heteroscedastic_nll(mu[cells], sigma[cells], values; weight = w)
end

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
reward for overconfidence. The lower edge is therefore `fraction * s`.
The upper edge is at least `hi` and at least `2 s`.
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
"""
function _calibrated_nll(mu::AbstractVector, sigma::AbstractVector, anchors, n::Int)
    raw = _nll_anchors(mu, sigma, anchors, n)
    _, values, w = anchors
    s = _anchor_scale(values, w)
    return 2 * (raw - log(s))
end

function _loss_terms(mu::AbstractVector, sigma::AbstractVector,
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
    reference_term = NaN

    if targets.reference !== nothing
        length(targets.reference) == n || throw(DimensionMismatch(
            "prior_loss: reference has $(length(targets.reference)) entries but the grid has $(n) cells"))
    end

    if targets.anchors !== nothing
        anchor_term = _nll_anchors(mu, sigma, targets.anchors, n)
        total += weights.anchor * anchor_term
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

    return total, (anchor = anchor_term, smooth = smooth_term, sigma = sigma_term,
                   reference = reference_term,
                   grade = NaN, density = NaN, susceptibility = NaN,
                   resistivity = NaN, conductivity_100kHz = NaN)
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
                     grid::PriorGrid,
                     targets::PriorTargets,
                     weights::LossWeights)
    n = ncells(grid)
    nprop = size(mus, 1)
    size(mus, 2) == n || throw(DimensionMismatch(
        "prior_loss: mu has $(size(mus, 2)) cells but the grid has $(n)"))
    size(sigmas) == size(mus) || throw(DimensionMismatch(
        "prior_loss: sigma size $(size(sigmas)) does not match mu $(size(mus))"))

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
        (targets.anchors_conductivity, weights.conductivity, "conductivity_100kHz"),
    )
    grade_term = density_term = susc_term = res_term = cond_term = NaN
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
        elseif pname == "conductivity_100kHz"
            cond_term = term
        else
            res_term = term
        end
    end

    if targets.anchors !== nothing
        throw(ArgumentError(
            "prior_loss: use named anchor groups " *
            "(anchors_grade/density/susceptibility/resistivity/conductivity) " *
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

    return total, (anchor = NaN, smooth = smooth_term, sigma = sigma_term,
                   reference = reference_term,
                   grade = grade_term, density = density_term,
                   susceptibility = susc_term, resistivity = res_term,
                   conductivity_100kHz = cond_term)
end
