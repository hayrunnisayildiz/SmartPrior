# Writing a trained prior out in the format the inversion reads.
#
# Four files come out of a run, all WS3D models on the inversion's own grid:
#
#   prior.rho  the nominal model, mu           -> VFSA3DMT/VFSA2DMT start model
#   prior.std  the per-cell confidence, sigma  -> for plotting and reuse
#   prior.lo   mu - k*sigma, clamped           -> per-cell lower search bound
#   prior.hi   mu + k*sigma, clamped           -> per-cell upper search bound
#
# `prior.rho` alone is already useful and needs no change to MTGeophysics.jl:
# VFSA3DMT takes a start-model path, and swapping a half-space for this is the
# whole of the mu half of the idea. The bound files need a solver that accepts
# per-cell limits; MTGeophysics v0.5.0 `log_bounds` is scalar-only, so
# `prior.lo` / `prior.hi` are written for inspection only.
#
# The naming deliberately echoes the `model.mean` / `model.std` files that
# MTGeophysics' own AnalyseEnsemble3D writes, so the two sets of outputs can sit
# in one directory and be read by the same plotting code.

"""
    PriorBundle(grid, mu, sigma; spread=nothing, log_rho_bounds=(0.0, 5.0),
                k=2.0, min_width=0.1)

A trained prior ready to be written out.

- `mu`: nominal log10 resistivity, `[nx, ny, nz]`. `NaN` marks air.
- `sigma`: per-cell standard deviation in decades.
- `spread`: optional ensemble disagreement, the epistemic part of `sigma`.
- `k`: half-width of the search interval in standard deviations. Two covers
  about 95 per cent of a Gaussian, which is the usual compromise between a tight
  interval and one that still contains the answer.
- `min_width`: smallest interval in decades a live cell may be given. Residual
  mode can place `mu` outside `log_rho_bounds`, and clamping both ends would then
  collapse the interval and silently freeze the cell.
"""
struct PriorBundle
    grid::PriorGrid
    mu::Array{Float64,3}
    sigma::Array{Float64,3}
    spread::Union{Nothing,Array{Float64,3}}
    log_rho_bounds::Tuple{Float64,Float64}
    k::Float64
    min_width::Float64

    function PriorBundle(grid::PriorGrid,
                         mu::AbstractArray{<:Real,3},
                         sigma::AbstractArray{<:Real,3};
                         spread::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
                         log_rho_bounds::Tuple{Real,Real} = (0.0, 5.0),
                         k::Real = 2.0,
                         min_width::Real = 0.1)
        dims = size(grid)
        size(mu) == dims || throw(DimensionMismatch(
            "PriorBundle: mu must match the grid $(dims), got $(size(mu))"))
        size(sigma) == dims || throw(DimensionMismatch(
            "PriorBundle: sigma must match the grid $(dims), got $(size(sigma))"))
        spread === nothing || size(spread) == dims || throw(DimensionMismatch(
            "PriorBundle: spread must match the grid $(dims), got $(size(spread))"))

        log_rho_bounds[1] < log_rho_bounds[2] ||
            throw(ArgumentError("PriorBundle: log_rho_bounds must be increasing"))
        k > 0 || throw(ArgumentError("PriorBundle: k must be positive"))
        min_width > 0 || throw(ArgumentError("PriorBundle: min_width must be positive"))
        min_width <= log_rho_bounds[2] - log_rho_bounds[1] || throw(ArgumentError(
            "PriorBundle: min_width $(min_width) exceeds the width of log_rho_bounds"))

        # sigma has to be positive wherever mu is live, or the interval is
        # meaningless; air cells are exempt since they never get perturbed
        live = .!isnan.(mu)
        all(s -> !isfinite(s) || s > 0, sigma[live]) ||
            throw(ArgumentError("PriorBundle: sigma must be positive in live cells"))

        return new(grid, Array{Float64,3}(mu), Array{Float64,3}(sigma),
                   spread === nothing ? nothing : Array{Float64,3}(spread),
                   (Float64(log_rho_bounds[1]), Float64(log_rho_bounds[2])),
                   Float64(k), Float64(min_width))
    end
end

Base.size(b::PriorBundle) = size(b.mu)

function Base.show(io::IO, b::PriorBundle)
    nx, ny, nz = size(b)
    nair = count(isnan, b.mu)
    @printf(io, "PriorBundle(%dx%dx%d, %d air cells, k=%.1f, sigma %.2f-%.2f)",
            nx, ny, nz, nair, b.k,
            minimum(filter(isfinite, b.sigma); init = NaN),
            maximum(filter(isfinite, b.sigma); init = NaN))
end

"""
    prior_bounds(b::PriorBundle) -> (lo, hi)

Per-cell search interval, `mu +- k*sigma` clamped into `log_rho_bounds` and
widened to at least `min_width`.

Air cells stay `NaN` in both, which is how they survive a round trip through the
WS3D format and stay frozen in the inversion.

Where clamping alone would leave an interval narrower than `min_width`, the
interval is shifted rather than centred: pushing it inwards from whichever global
bound it hit is the only way to keep the width without leaving the physical range.
"""
function prior_bounds(b::PriorBundle)
    gl, gh = b.log_rho_bounds
    lo = similar(b.mu)
    hi = similar(b.mu)

    @inbounds for idx in eachindex(b.mu)
        m = b.mu[idx]
        if isnan(m)
            lo[idx] = NaN
            hi[idx] = NaN
            continue
        end
        s = b.sigma[idx]
        half = isfinite(s) ? b.k * s : b.k
        l = clamp(m - half, gl, gh)
        h = clamp(m + half, gl, gh)

        if h - l < b.min_width
            if l <= gl
                l = gl
                h = gl + b.min_width
            elseif h >= gh
                h = gh
                l = gh - b.min_width
            else
                mid = (l + h) / 2
                l = mid - b.min_width / 2
                h = mid + b.min_width / 2
            end
        end
        lo[idx] = l
        hi[idx] = h
    end
    return lo, hi
end

"""
    apply_air_mask(mu, air) -> Array{Float64,3}

Set the cells flagged in `air` to `NaN`.

`MTGeophysics.load_ws3d_model` tags anything above 1e15 ohm-metres as `NaN` on
read, and `write_ws3d_model` maps non-finite entries back to 1e17, so `NaN` is
the round-trip-safe way to carry air through a WS3D file. Use
`MTGeophysics.air_mask_from_model` on the mesh's own start model to get `air`.
"""
function apply_air_mask(mu::AbstractArray{<:Real,3}, air::AbstractArray{Bool,3})
    size(mu) == size(air) || throw(DimensionMismatch(
        "apply_air_mask: mu is $(size(mu)) but the mask is $(size(air))"))
    out = Array{Float64,3}(mu)
    out[air] .= NaN
    return out
end

"""
    write_prior(dir, b::PriorBundle; rotation=0.0, prefix="prior") -> NamedTuple

Write the bundle as four WS3D models in `dir`, returning their paths.

Files go out through `MTGeophysics.write_ws3d_model` rather than being formatted
here. That matters: the WS3D layout writes each depth slice with the `i` index
reversed, and hand-rolling it is the easiest way to produce a model that loads
without error but is mirrored north-south.
"""
function write_prior(dir::AbstractString, b::PriorBundle;
                     rotation::Real = 0.0,
                     prefix::AbstractString = "prior")
    mkpath(dir)
    lo, hi = prior_bounds(b)
    g = b.grid

    paths = Dict{Symbol,String}()
    for (key, field) in ((:rho, b.mu), (:std, b.sigma), (:lo, lo), (:hi, hi))
        path = joinpath(dir, string(prefix, ".", key))
        write_ws3d_model(path, g.dx, g.dy, g.dz, Array{Float64,3}(field), g.origin;
                         rotation = rotation)
        paths[key] = path
    end

    if b.spread !== nothing
        path = joinpath(dir, string(prefix, ".spread"))
        write_ws3d_model(path, g.dx, g.dy, g.dz, b.spread, g.origin; rotation = rotation)
        paths[:spread] = path
    end

    return (rho = paths[:rho], std = paths[:std], lo = paths[:lo], hi = paths[:hi],
            spread = get(paths, :spread, nothing))
end

"""
    prior_from_ensemble(grid, e::PriorEnsemble, X; offset=nothing, air=nothing,
                        k=2.0, min_width=0.1, log_rho_bounds=nothing) -> PriorBundle

Run an ensemble over the grid and package the result.

`log_rho_bounds` defaults to the network's own bounds, which is normally what you
want: they are the physical range the field was trained to respect.
"""
function prior_from_ensemble(grid::PriorGrid, e::PriorEnsemble, X::AbstractMatrix;
                             offset::Union{Nothing,AbstractVector} = nothing,
                             air::Union{Nothing,AbstractArray{Bool,3}} = nothing,
                             k::Real = 2.0,
                             min_width::Real = 0.1,
                             log_rho_bounds::Union{Nothing,Tuple{Real,Real}} = nothing)
    mu, sigma, spread = predict_ensemble_grid(e, X, size(grid); offset = offset)
    bounds = log_rho_bounds === nothing ? e.net.log_rho_bounds : log_rho_bounds

    mu_masked = air === nothing ? Array{Float64,3}(mu) : apply_air_mask(mu, air)

    return PriorBundle(grid, mu_masked, sigma;
                       spread = spread,
                       log_rho_bounds = bounds,
                       k = k,
                       min_width = min_width)
end
