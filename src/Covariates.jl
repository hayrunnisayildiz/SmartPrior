# Pointwise covariates, evaluated at whatever `(x, y, z)` columns the caller
# passes.
#
# Not covariates: pXRF elements, drillhole lithology (future outputs), and
# distance to the nearest sample (a confidence mask only).

"""
    Covariate

A spatial field the network may see. [`evaluate`](@ref) returns a `k × N`
matrix at the columns of a `3 × N` coordinate matrix (rows are x, y, z, in
metres). [`channel_names`](@ref) has length `k`, in the same order.
"""
abstract type Covariate end

function _xyz64(xyz::AbstractMatrix{<:Real})
    size(xyz, 1) == 3 || throw(ArgumentError(
        "evaluate: xyz must be a 3×N matrix (rows x, y, z), got $(size(xyz))"))
    return xyz isa Matrix{Float64} ? xyz : Matrix{Float64}(xyz)
end

# maps a coordinate vector onto [-1, 1] using the box extent; a degenerate
# axis collapses to 0 rather than dividing by zero
function _unit_scale(c::AbstractVector{Float64}, lo::Float64, hi::Float64)
    span = hi - lo
    span > 0 || return zeros(length(c))
    return 2 .* (c .- lo) ./ span .- 1
end

# Geometric depth scale: skin depth of a reference half-space. Used only as a
# dimensionless depth channel (`depth_over_skin`), not as an MT solver.
const _MU0 = 4π * 1e-7
_skin_depth(rho::Real, T::Real) = sqrt(2 * rho * T / (2π * _MU0))

"""
    CoordinateCovariate(x_min, x_max, y_min, y_max, z_min, z_max)

Normalised coordinates `x_norm`, `y_norm`, `z_norm` on `[-1, 1]`, mapped from
the box edges. Pass the box (site `[bounds]`), not the min/max of the query
points.
"""
struct CoordinateCovariate <: Covariate
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    z_min::Float64
    z_max::Float64
end

channel_names(::CoordinateCovariate) = ["x_norm", "y_norm", "z_norm"]

function evaluate(c::CoordinateCovariate, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    out = Matrix{Float64}(undef, 3, n)
    n == 0 && return out
    out[1, :] .= _unit_scale(vec(pts[1, :]), c.x_min, c.x_max)
    out[2, :] .= _unit_scale(vec(pts[2, :]), c.y_min, c.y_max)
    out[3, :] .= _unit_scale(vec(pts[3, :]), c.z_min, c.z_max)
    return out
end

"""
    DepthCovariate(z_datum, d0, rho_ref, period_max, log_mean, log_std)
    DepthCovariate(z_datum, z_max, d0, rho_ref, period_max)

`log_depth` and `depth_over_skin`. `depth_over_skin` divides height above
`z_datum` by the skin depth of a `rho_ref` half-space at `period_max` (a
geometric scale, not an MT input).

Warning: `d = z - z_datum` is height above `z_datum`, not depth below a
surface. `z` is elevation, positive up, and `z_datum` is the bottom of the
box. `log_depth` therefore increases upward. Depth positive downward from a
surface is [`DepthBelowSurface`](@ref).

`log_depth` is `log10(max(d, d0) / d0)` shifted by `log_mean` and `log_std`.
Those two numbers are fixed when the covariate is built and are not
recomputed from the points passed to [`evaluate`](@ref).

- From a vertical box `[z_datum, z_max]`: the moments are those of one
  column of cell centres filling the box with nominal thickness `2 * d0`.
- `log_mean` and `log_std` may instead be supplied directly (the site file).
"""
struct DepthCovariate <: Covariate
    z_datum::Float64
    d0::Float64
    rho_ref::Float64
    period_max::Float64
    log_mean::Float64
    log_std::Float64

    function DepthCovariate(z_datum::Real, d0::Real, rho_ref::Real, period_max::Real,
                            log_mean::Real, log_std::Real)
        d0 > 0 || throw(ArgumentError("DepthCovariate: d0 must be positive"))
        rho_ref > 0 || throw(ArgumentError("DepthCovariate: rho_ref must be positive"))
        period_max > 0 || throw(ArgumentError(
            "DepthCovariate: period_max must be positive"))
        isfinite(log_mean) || throw(ArgumentError(
            "DepthCovariate: log_mean must be finite"))
        (isfinite(log_std) && log_std >= 0) || throw(ArgumentError(
            "DepthCovariate: log_std must be finite and ≥ 0"))
        return new(Float64(z_datum), Float64(d0), Float64(rho_ref), Float64(period_max),
                   Float64(log_mean), Float64(log_std))
    end
end

function _box_log_depth_moments(z_datum::Real, z_max::Real, d0::Real)
    span = Float64(z_max - z_datum)
    span > 0 || throw(ArgumentError(
        "DepthCovariate: z_max must be greater than z_datum"))
    d0 > 0 || throw(ArgumentError("DepthCovariate: d0 must be positive"))
    cell_z = 2 * Float64(d0)
    nz = max(1, round(Int, span / cell_z))
    dz = span / nz
    raw = Vector{Float64}(undef, nz)
    for k in 1:nz
        d = (k - 0.5) * dz
        raw[k] = log10(max(d, Float64(d0)) / Float64(d0))
    end
    μ = mean(raw)
    σ = nz == 1 ? 0.0 : std(raw)
    isfinite(σ) && σ > 0 || (σ = 0.0)
    return μ, σ
end

function DepthCovariate(z_datum::Real, z_max::Real, d0::Real,
                        rho_ref::Real, period_max::Real)
    μ, σ = _box_log_depth_moments(z_datum, z_max, d0)
    return DepthCovariate(z_datum, d0, rho_ref, period_max, μ, σ)
end

channel_names(::DepthCovariate) = ["log_depth", "depth_over_skin"]

function evaluate(c::DepthCovariate, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    out = Matrix{Float64}(undef, 2, n)
    n == 0 && return out
    d = vec(pts[3, :]) .- c.z_datum
    raw = log10.(max.(d, c.d0) ./ c.d0)
    if c.log_std > 0
        out[1, :] .= (raw .- c.log_mean) ./ c.log_std
    else
        out[1, :] .= 0.0
    end
    δ = _skin_depth(c.rho_ref, c.period_max)
    out[2, :] .= d ./ δ
    return out
end

# Power 2. At a collar the weight is infinite, so the surface there is that
# collar (the mean, when several collars share the point).
const _SURFACE_IDW_POWER = 2.0

function _idw_elevation(east::Vector{Float64}, north::Vector{Float64},
                        elevation::Vector{Float64}, x::Float64, y::Float64)
    num = 0.0
    den = 0.0
    n_hit = 0
    hit = 0.0
    p = _SURFACE_IDW_POWER
    @inbounds for i in eachindex(east)
        d = hypot(x - east[i], y - north[i])
        if d == 0
            n_hit += 1
            hit += elevation[i]
            continue
        end
        w = inv(d^p)
        num += w * elevation[i]
        den += w
    end
    n_hit == 0 || return hit / n_hit
    return num / den
end

"""
    DepthBelowSurface(east, north, elevation, d0; log_mean, log_std)
    DepthBelowSurface(east, north, elevation, d0; z_min, z_max)

`log_depth` for depth positive downward from a surface:
`d = z_surface(x, y) - z`, then `log10(max(d, d0) / d0)`, shifted by
`log_mean` and `log_std`.

`z_surface` is inverse-distance weighting (power 2) of the collar
elevations passed in. Those stations are copied at construction and are
not read again in [`evaluate`](@ref).

The two moments are fixed at construction, the same rule as
[`DepthCovariate`](@ref). Pass `log_mean` and `log_std`, or a vertical box
`[z_min, z_max]`. The box uses that covariate's cell-centre population
(nominal thickness `2 * d0`). They are not recomputed from the points
passed to [`evaluate`](@ref).
"""
struct DepthBelowSurface <: Covariate
    east::Vector{Float64}
    north::Vector{Float64}
    elevation::Vector{Float64}
    d0::Float64
    log_mean::Float64
    log_std::Float64

    function DepthBelowSurface(east::AbstractVector{<:Real},
                               north::AbstractVector{<:Real},
                               elevation::AbstractVector{<:Real},
                               d0::Real, log_mean::Real, log_std::Real)
        n = length(east)
        (length(north) == n && length(elevation) == n) || throw(ArgumentError(
            "DepthBelowSurface: east, north and elevation must have equal length"))
        n > 0 || throw(ArgumentError(
            "DepthBelowSurface: need at least one collar elevation"))
        d0 > 0 || throw(ArgumentError("DepthBelowSurface: d0 must be positive"))
        isfinite(log_mean) || throw(ArgumentError(
            "DepthBelowSurface: log_mean must be finite"))
        (isfinite(log_std) && log_std >= 0) || throw(ArgumentError(
            "DepthBelowSurface: log_std must be finite and ≥ 0"))
        ex = Vector{Float64}(undef, n)
        ny = Vector{Float64}(undef, n)
        ez = Vector{Float64}(undef, n)
        for i in 1:n
            (isfinite(east[i]) && isfinite(north[i]) && isfinite(elevation[i])) ||
                throw(ArgumentError(
                    "DepthBelowSurface: collar coordinates must be finite"))
            ex[i] = Float64(east[i])
            ny[i] = Float64(north[i])
            ez[i] = Float64(elevation[i])
        end
        return new(ex, ny, ez, Float64(d0), Float64(log_mean), Float64(log_std))
    end
end

function DepthBelowSurface(east::AbstractVector{<:Real},
                           north::AbstractVector{<:Real},
                           elevation::AbstractVector{<:Real},
                           d0::Real;
                           z_min::Union{Nothing,Real} = nothing,
                           z_max::Union{Nothing,Real} = nothing,
                           log_mean::Union{Nothing,Real} = nothing,
                           log_std::Union{Nothing,Real} = nothing)
    if log_mean !== nothing || log_std !== nothing
        (log_mean !== nothing && log_std !== nothing) || throw(ArgumentError(
            "DepthBelowSurface: log_mean and log_std must be set together"))
        (z_min === nothing && z_max === nothing) || throw(ArgumentError(
            "DepthBelowSurface: pass log_mean and log_std, or z_min and z_max, not both"))
        return DepthBelowSurface(east, north, elevation, d0, log_mean, log_std)
    end
    (z_min !== nothing && z_max !== nothing) || throw(ArgumentError(
        "DepthBelowSurface: pass log_mean and log_std, or z_min and z_max"))
    μ, σ = _box_log_depth_moments(z_min, z_max, d0)
    return DepthBelowSurface(east, north, elevation, d0, μ, σ)
end

channel_names(::DepthBelowSurface) = ["log_depth"]

function evaluate(c::DepthBelowSurface, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    out = Matrix{Float64}(undef, 1, n)
    n == 0 && return out
    raw = Vector{Float64}(undef, n)
    @inbounds for t in 1:n
        zs = _idw_elevation(c.east, c.north, c.elevation, pts[1, t], pts[2, t])
        d = zs - pts[3, t]
        raw[t] = log10(max(d, c.d0) / c.d0)
    end
    if c.log_std > 0
        out[1, :] .= (raw .- c.log_mean) ./ c.log_std
    else
        out[1, :] .= 0.0
    end
    return out
end

const _REJECTED_COVARIATES = (
    "sample_distance", "geochemistry", "lithology", "pxrf",
)

"""
    evaluate_all(covs, xyz) -> (Matrix, Vector{String})

Stack [`evaluate`](@ref) down the channel axis. Channel names are concatenated
in the same order and must be unique.
"""
function evaluate_all(covs::AbstractVector{<:Covariate}, xyz::AbstractMatrix{<:Real})
    isempty(covs) && throw(ArgumentError("evaluate_all: no covariates"))
    for c in covs
        c isa Covariate || throw(ArgumentError(
            "evaluate_all: expected Covariate, got $(typeof(c))"))
    end
    parts = Matrix{Float64}[]
    names = String[]
    n = size(_xyz64(xyz), 2)
    for c in covs
        part = evaluate(c, xyz)
        cn = channel_names(c)
        size(part, 1) == length(cn) || throw(ArgumentError(
            "evaluate_all: $(typeof(c)) returned $(size(part, 1)) rows for " *
            "$(length(cn)) names"))
        size(part, 2) == n || throw(DimensionMismatch(
            "evaluate_all: $(typeof(c)) returned $(size(part, 2)) columns, expected $n"))
        push!(parts, part)
        append!(names, cn)
    end
    allunique(names) || throw(ArgumentError(
        "evaluate_all: duplicate channel names: $(names)"))
    return vcat(parts...), names
end

function _reject_non_covariate(name::AbstractString)
    key = lowercase(strip(String(name)))
    if key in _REJECTED_COVARIATES || startswith(key, "geochem") || startswith(key, "lith_")
        throw(ArgumentError(
            "covariate $(repr(name)) is not a covariate: pXRF, drillhole " *
            "lithology, and sample_distance are properties or a confidence mask"))
    end
    return nothing
end
