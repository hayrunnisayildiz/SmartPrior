"""
    BoundedVFSA

EXPERIMENTAL / UNVALIDATED — prepared for a future 3-D driver, never
run in 3-D, not covered by the public API. Not exported.
"""

# Per-cell search bounds for the VFSA inversion.
#
# MTGeophysics' 3D VFSA carries one global interval, `cfg.log_bounds = (lo, hi)`,
# and uses it in exactly two places inside the iteration:
#
#   propose_controls!(delta, T, lo, hi, v0_at_ctrl, nsel, rng; step_scale)
#   v_trial = clamp.(v0_core .+ delta_values, lo, hi)
#
# Everything else -- the RBF parameterisation, the temperature schedule, the
# Metropolis test, the padding decay, the frozen masks -- is independent of how
# the bounds are stored. So the sigma half of a smart prior needs no new inversion
# engine, only cell-indexed versions of those two operations. That is what this
# file provides, and the upstream change it implies is a diff of a few lines.
#
# Replacing the scalar interval buys two distinct things:
#
#   1. The search is confined to what the multi-physics data supports. A cell the
#      prior is confident about cannot wander decades away just because the misfit
#      is flat in that direction, which is where spurious anomalies come from.
#
#   2. The proposal width follows the bounds. Upstream sets the Cauchy step to
#      `(hi - lo) * step_scale`; keeping that formula per cell means confident
#      cells take small steps and uncertain ones take large steps, so the
#      annealing spends its exploration where the prior admits it does not know.
#      That happens for free, without a second tuning parameter.
#
# The rejection rate does not suffer from narrow bounds, which is the objection
# one would expect. Because the step scales with the interval it sits in, the
# acceptance probability of the inner resampling loop is roughly invariant to how
# wide the interval is.

using Random: randperm

"""
    BoundedCore(lo, hi, ix, iy, kz)

Per-cell log10 resistivity bounds restricted to the inversion's perturbable core.

`lo` and `hi` are core-shaped, matching what `build_rbf_map` indexes. Non-finite
entries mark cells with no bounds, which is how air arrives here: the prior
writes `NaN` there, and those cells must be frozen rather than clamped.

Build one with [`core_bounds`](@ref) from the full-grid cubes.
"""
struct BoundedCore
    lo::Array{Float64,3}
    hi::Array{Float64,3}
    ix::UnitRange{Int}
    iy::UnitRange{Int}
    kz::UnitRange{Int}

    function BoundedCore(lo::AbstractArray{<:Real,3}, hi::AbstractArray{<:Real,3},
                         ix::UnitRange{Int}, iy::UnitRange{Int}, kz::UnitRange{Int})
        dims = (length(ix), length(iy), length(kz))
        size(lo) == dims || throw(DimensionMismatch(
            "BoundedCore: lo is $(size(lo)) but the core is $(dims)"))
        size(hi) == dims || throw(DimensionMismatch(
            "BoundedCore: hi is $(size(hi)) but the core is $(dims)"))

        l = Array{Float64,3}(lo)
        h = Array{Float64,3}(hi)

        # a cell is either bounded on both sides or on neither; a half-open
        # interval would let the clamp pass NaN into the model
        @inbounds for idx in eachindex(l)
            fl, fh = isfinite(l[idx]), isfinite(h[idx])
            fl == fh || throw(ArgumentError(
                "BoundedCore: cell $(idx) has one finite bound and one non-finite"))
            fl && (l[idx] < h[idx] || throw(ArgumentError(
                "BoundedCore: cell $(idx) has lo $(l[idx]) not below hi $(h[idx])")))
        end
        return new(l, h, ix, iy, kz)
    end
end

Base.size(bc::BoundedCore) = size(bc.lo)

function Base.show(io::IO, bc::BoundedCore)
    w = filter(isfinite, bc.hi .- bc.lo)
    @printf(io, "BoundedCore(%dx%dx%d, %d frozen, width %.2f-%.2f decades)",
            size(bc)..., count(!isfinite, bc.lo),
            minimum(w; init = NaN), maximum(w; init = NaN))
end

"""
    core_bounds(lo, hi, ix, iy, kz) -> BoundedCore

Slice full-grid bound cubes down to the core index ranges.

Get the ranges the way the inversion does, so the slice lines up with the RBF
map exactly:

    ix, iy = MTGeophysics.core_ranges(m; tol = cfg.pad_tol, expand = cfg.core_expand_cells)
    kz = 1:cfg.z_core_cells
"""
core_bounds(lo::AbstractArray{<:Real,3}, hi::AbstractArray{<:Real,3},
            ix::UnitRange{Int}, iy::UnitRange{Int}, kz::UnitRange{Int}) =
    BoundedCore(lo[ix, iy, kz], hi[ix, iy, kz], ix, iy, kz)

"""
    core_bounds(b::PriorBundle, ix, iy, kz) -> BoundedCore

Take the bounds straight from a trained prior.
"""
function core_bounds(b::PriorBundle, ix::UnitRange{Int}, iy::UnitRange{Int},
                     kz::UnitRange{Int})
    lo, hi = prior_bounds(b)
    return core_bounds(lo, hi, ix, iy, kz)
end

"""
    frozen_from_bounds(bc::BoundedCore) -> BitArray{3}

Cells with no bounds, as a mask to hand to `build_rbf_map(...; exclude = mask)`.

Passing this is not optional. A control point placed in an unbounded cell has no
interval to propose within, and the proposal would fall back to the midpoint of
`NaN`.
"""
frozen_from_bounds(bc::BoundedCore) = BitArray(.!isfinite.(bc.lo))

"""
    control_bounds(rbfmap, bc::BoundedCore) -> (lo_ctrl, hi_ctrl)

Bounds of the cell each control point sits in.

`rbfmap.ci/cj/ck` are core-local indices, the same frame as `bc`, so this is a
direct lookup. Errors if any control landed in an unbounded cell, which means the
mask from [`frozen_from_bounds`](@ref) was not passed to `build_rbf_map`.
"""
function control_bounds(rbfmap, bc::BoundedCore)
    (rbfmap.nx, rbfmap.ny, rbfmap.nz) == size(bc) || throw(DimensionMismatch(
        "control_bounds: rbfmap core is $((rbfmap.nx, rbfmap.ny, rbfmap.nz)) but bounds are $(size(bc))"))

    M = length(rbfmap.ci)
    lo = Vector{Float64}(undef, M)
    hi = Vector{Float64}(undef, M)
    @inbounds for q in 1:M
        i, j, k = rbfmap.ci[q], rbfmap.cj[q], rbfmap.ck[q]
        l, h = bc.lo[i, j, k], bc.hi[i, j, k]
        isfinite(l) || error("control_bounds: control $q sits in an unbounded cell " *
                             "($i,$j,$k); pass frozen_from_bounds(bc) as build_rbf_map's exclude")
        lo[q] = l
        hi[q] = h
    end
    return lo, hi
end

# reproduces MTGeophysics' own vfsa_y: the Cauchy-like VFSA generating function,
# a step in [-1, 1] scaled by the temperature. Kept here rather than reached for
# across the module boundary because it is unexported, and tested against theirs.
@inline function vfsa_step(u::Real, T::Real)
    s = ifelse(u >= 0.5, 1.0, -1.0)
    return s * T * ((1 + 1 / T)^(abs(2u - 1.0)) - 1.0)
end

"""
    propose_controls_bounded!(delta_params, T, lo, hi, v0_at_ctrl, nsel, rng;
                              step_scale=1.0) -> idxs

Per-cell-bound version of `MTGeophysics.propose_controls!`.

Perturbs `nsel` randomly chosen controls by a Cauchy step of width
`(hi[q] - lo[q]) * step_scale`, resampling up to 100 times to land inside that
control's own interval. Returns the indices touched.

With `lo` and `hi` constant this consumes the random stream in the same order as
the upstream scalar version and produces identical results, which is how the two
are kept in step.

A control whose current value sits outside its interval -- possible when the
start model is not the same model the bounds were derived from -- will usually
exhaust the 100 tries and be reset to the middle of its interval. That is the
intended recovery, but it is worth knowing that it happens silently.
"""
function propose_controls_bounded!(delta_params::AbstractVector{Float64},
                                   T::Real,
                                   lo::AbstractVector{<:Real},
                                   hi::AbstractVector{<:Real},
                                   v0_at_ctrl::AbstractVector{<:Real},
                                   nsel::Integer,
                                   rng::AbstractRNG;
                                   step_scale::Real = 1.0)
    M = length(delta_params)
    all(==(M), (length(lo), length(hi), length(v0_at_ctrl))) || throw(DimensionMismatch(
        "propose_controls_bounded!: delta_params, lo, hi and v0_at_ctrl must have equal length"))
    1 <= nsel <= M || throw(ArgumentError(
        "propose_controls_bounded!: nsel must be in 1:$(M), got $(nsel)"))

    idxs = randperm(rng, M)[1:nsel]
    @inbounds for id in idxs
        l, h = lo[id], hi[id]
        dm = (h - l) * step_scale
        base = v0_at_ctrl[id] + delta_params[id]
        cand = 0.5 * (l + h)
        for _ in 1:100
            c = base + vfsa_step(rand(rng), T) * dm
            if l <= c <= h
                cand = c
                break
            end
        end
        delta_params[id] = cand - v0_at_ctrl[id]
    end
    return idxs
end

"""
    clamp_to_bounds(v, bc::BoundedCore) -> Array{Float64,3}

Clamp a core field into its per-cell interval, leaving unbounded cells alone.

Replaces the upstream `clamp.(v0_core .+ delta, lo, hi)`. Unbounded cells pass
through untouched instead of becoming `NaN`, since `clamp(x, NaN, NaN)` would
otherwise poison the model handed to the forward solver.
"""
function clamp_to_bounds(v::AbstractArray{<:Real,3}, bc::BoundedCore)
    size(v) == size(bc) || throw(DimensionMismatch(
        "clamp_to_bounds: field is $(size(v)) but the core is $(size(bc))"))
    out = Array{Float64,3}(v)
    @inbounds for idx in eachindex(out)
        l = bc.lo[idx]
        isfinite(l) || continue
        out[idx] = clamp(out[idx], l, bc.hi[idx])
    end
    return out
end

"""
    bounded_trial!(delta_params, buffer, rbfmap, bc, T, lo_ctrl, hi_ctrl, v0_ctrl,
                   v0_core, nsel, rng; step_scale=1.0, frozen=nothing)
        -> (v_trial, idxs)

One bounded VFSA proposal, start to finish.

This is the block the upstream loop runs per trial, with the scalar bounds
swapped for per-cell ones: perturb the controls, spread the deltas through the
RBF map, add them to the start model, clamp, and restore frozen cells that the
kernels bled into.

`delta_params` and `buffer` are updated in place. Feed the returned `v_trial` to
`MTGeophysics.embed_core!` and continue with the padding decay and forward solve
exactly as before.
"""
function bounded_trial!(delta_params::AbstractVector{Float64},
                        buffer::Array{Float64,3},
                        rbfmap,
                        bc::BoundedCore,
                        T::Real,
                        lo_ctrl::AbstractVector{<:Real},
                        hi_ctrl::AbstractVector{<:Real},
                        v0_ctrl::AbstractVector{<:Real},
                        v0_core::AbstractArray{<:Real,3},
                        nsel::Integer,
                        rng::AbstractRNG;
                        step_scale::Real = 1.0,
                        frozen::Union{Nothing,AbstractArray{Bool,3}} = nothing)
    size(v0_core) == size(bc) || throw(DimensionMismatch(
        "bounded_trial!: v0_core is $(size(v0_core)) but the core is $(size(bc))"))

    idxs = propose_controls_bounded!(delta_params, T, lo_ctrl, hi_ctrl, v0_ctrl,
                                     nsel, rng; step_scale = step_scale)
    apply_rbf_map!(buffer, rbfmap, delta_params)
    v_trial = clamp_to_bounds(v0_core .+ buffer, bc)

    # kernels have compact but not cell-sized support, so a control next to a
    # frozen cell writes into it; upstream restores the start value and so must we
    if frozen !== nothing
        @inbounds for idx in eachindex(v_trial)
            frozen[idx] && (v_trial[idx] = v0_core[idx])
        end
    end
    return v_trial, idxs
end

"""
    bound_report(bc::BoundedCore; reference=nothing) -> NamedTuple

Summary of how much a per-cell prior actually narrows the search.

`reference` is the scalar interval the run would otherwise have used, e.g.
`cfg.log_bounds`. `volume_ratio` is the geometric-mean width relative to it,
which is the honest figure: the search space is a product over cells, so widths
combine multiplicatively and an arithmetic mean would flatter a prior that is
tight almost everywhere and loose in a few cells.
"""
function bound_report(bc::BoundedCore; reference::Union{Nothing,Tuple{Real,Real}} = nothing)
    w = filter(isfinite, vec(bc.hi .- bc.lo))
    isempty(w) && throw(ArgumentError("bound_report: no bounded cells"))

    geo = exp(mean(log.(w)))
    ratio = reference === nothing ? NaN :
            geo / (Float64(reference[2]) - Float64(reference[1]))

    return (nbounded = length(w),
            nfrozen = count(!isfinite, bc.lo),
            width_min = minimum(w),
            width_median = median(w),
            width_max = maximum(w),
            width_geomean = geo,
            volume_ratio = ratio)
end
