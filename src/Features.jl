# Multi-physics data to feature tensor.
#
# Every channel produced here is dimensionless and standardised within the
# survey. That is a deliberate constraint rather than tidiness: a channel carrying
# absolute mGal or absolute metres ties the trained network to one survey's units
# and extent, which is exactly the coupling that stops a prior generator from
# transferring between sites. Standardising in-survey also removes the regional
# level of a Bouguer field, which no local inversion can constrain anyway.
#
# Feature tensors are [nx, ny, nz, nchannel] so that a channel is a contiguous
# 3-D slice matching the model grid.

"""
    GravityObs(x, y, z, value, err)

Scattered gravity observations in the grid's own frame.

`x`, `y`, `z` are station coordinates in metres (`z` positive down, so stations
above the grid top have negative `z`), `value` the anomaly in mGal and `err` its
uncertainty in mGal.

The anomaly is expected to be a *complete Bouguer* anomaly, i.e. with the
topographic mass effect already removed, since the forward operator in
[`gravity_matrix`](@ref) models density contrast in the model volume only.
"""
struct GravityObs
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    value::Vector{Float64}
    err::Vector{Float64}

    function GravityObs(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                       z::AbstractVector{<:Real}, value::AbstractVector{<:Real},
                       err::AbstractVector{<:Real})
        n = length(x)
        all(==(n), (length(y), length(z), length(value), length(err))) ||
            throw(ArgumentError("GravityObs: all fields must have the same length"))
        n > 0 || throw(ArgumentError("GravityObs: need at least one station"))
        return new(collect(Float64, x), collect(Float64, y), collect(Float64, z),
                   collect(Float64, value), collect(Float64, err))
    end
end

Base.length(o::GravityObs) = length(o.x)

"""
    MTSites(x, y, periods, rho_a, phase; err_rho_a=nothing, err_phase=nothing)

MT sounding curves at scattered sites, in the grid's own frame.

`x`, `y` are site coordinates in metres. `rho_a` and `phase` are
`[nperiod, nsite]` matrices of apparent resistivity (ohm-metres) and impedance
phase (degrees); for tensor data pass a rotationally invariant average such as
the Berdichevsky mean rather than a single off-diagonal component.

`err_rho_a` and `err_phase` are optional keyword-only `[nperiod, nsite]`
uncertainties in the same units as `rho_a` and `phase`. The five-argument
positional constructor is unchanged: omit them and they stay `nothing`, which
leaves [`mt_column_misfit`](@ref) scoring raw residuals rather than χ²/datum, so
its weight is no longer comparable to the gravity term's.
"""
struct MTSites
    x::Vector{Float64}
    y::Vector{Float64}
    periods::Vector{Float64}
    rho_a::Matrix{Float64}
    phase::Matrix{Float64}
    err_rho_a::Union{Nothing,Matrix{Float64}}
    err_phase::Union{Nothing,Matrix{Float64}}

    function MTSites(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                     periods::AbstractVector{<:Real},
                     rho_a::AbstractMatrix{<:Real}, phase::AbstractMatrix{<:Real};
                     err_rho_a::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                     err_phase::Union{Nothing,AbstractMatrix{<:Real}} = nothing)
        ns = length(x)
        length(y) == ns || throw(ArgumentError("MTSites: x and y must have the same length"))
        ns > 0 || throw(ArgumentError("MTSites: need at least one site"))
        np = length(periods)
        size(rho_a) == (np, ns) || throw(ArgumentError(
            "MTSites: rho_a must be (nperiod, nsite) = ($np, $ns), got $(size(rho_a))"))
        size(phase) == (np, ns) || throw(ArgumentError(
            "MTSites: phase must be (nperiod, nsite) = ($np, $ns), got $(size(phase))"))
        err_a = _mtsites_err("err_rho_a", err_rho_a, np, ns)
        err_p = _mtsites_err("err_phase", err_phase, np, ns)
        return new(collect(Float64, x), collect(Float64, y), collect(Float64, periods),
                   Matrix{Float64}(rho_a), Matrix{Float64}(phase), err_a, err_p)
    end
end

function _mtsites_err(::String, ::Nothing, ::Integer, ::Integer)
    return nothing
end

function _mtsites_err(name::String, err::AbstractMatrix{<:Real}, np::Integer, ns::Integer)
    size(err) == (np, ns) || throw(ArgumentError(
        "MTSites: $name must be (nperiod, nsite) = ($np, $ns), got $(size(err))"))
    return Matrix{Float64}(err)
end

nsites(s::MTSites) = length(s.x)

"""
    PointSamples(x, y, values, names; z=nothing)

Scattered numeric samples for nearest-neighbour interpolation onto a prior grid.

`values` is `[nsample, nchannel]` with one column per entry in `names`. `z` is
optional: omit it (or pass `nothing`) for a plan-view interpolant that is then
extruded through depth, the same pattern as [`gravity_channels`](@ref).
"""
struct PointSamples
    x::Vector{Float64}
    y::Vector{Float64}
    z::Union{Nothing,Vector{Float64}}
    values::Matrix{Float64}
    names::Vector{String}

    function PointSamples(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                          values::AbstractMatrix{<:Real}, names::Vector{String};
                          z::Union{Nothing,AbstractVector{<:Real}} = nothing)
        n = length(x)
        length(y) == n || throw(ArgumentError("PointSamples: x and y must have the same length"))
        n > 0 || throw(ArgumentError("PointSamples: need at least one sample"))
        size(values, 1) == n || throw(ArgumentError(
            "PointSamples: values must have one row per sample ($(n)), got $(size(values, 1))"))
        size(values, 2) == length(names) || throw(ArgumentError(
            "PointSamples: $(size(values, 2)) value columns but $(length(names)) names"))
        allunique(names) || throw(ArgumentError("PointSamples: channel names must be unique"))
        zv = if z === nothing
            nothing
        else
            length(z) == n || throw(ArgumentError(
                "PointSamples: z has $(length(z)) entries but x has $(n)"))
            collect(Float64, z)
        end
        return new(collect(Float64, x), collect(Float64, y), zv,
                   Matrix{Float64}(values), copy(names))
    end
end

Base.length(s::PointSamples) = length(s.x)

"""
    LabelSamples(x, y, labels; z=nothing)

Scattered categorical labels, typically lithology at borehole interval midpoints.

`z` follows [`PointSamples`](@ref): omit it for a plan-view nearest neighbour
extruded through depth.
"""
struct LabelSamples
    x::Vector{Float64}
    y::Vector{Float64}
    z::Union{Nothing,Vector{Float64}}
    labels::Vector{String}

    function LabelSamples(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                          labels::AbstractVector{<:AbstractString};
                          z::Union{Nothing,AbstractVector{<:Real}} = nothing)
        n = length(x)
        length(y) == n || throw(ArgumentError("LabelSamples: x and y must have the same length"))
        length(labels) == n || throw(ArgumentError(
            "LabelSamples: labels has $(length(labels)) entries but x has $(n)"))
        n > 0 || throw(ArgumentError("LabelSamples: need at least one sample"))
        zv = if z === nothing
            nothing
        else
            length(z) == n || throw(ArgumentError(
                "LabelSamples: z has $(length(z)) entries but x has $(n)"))
            collect(Float64, z)
        end
        return new(collect(Float64, x), collect(Float64, y), zv, String.(labels))
    end
end

Base.length(s::LabelSamples) = length(s.x)

"""
    mt_apparent_errors_from_impedance(rho_a, z, z_err, frequencies)
        -> (err_rho_a, err_phase)

Propagate absolute impedance uncertainties (the same σ on Re and Im) to linear
apparent-resistivity and phase errors. Used when building [`MTSites`](@ref) from
2D data files whose native errors live on `Z`.
"""
function mt_apparent_errors_from_impedance(rho_a::AbstractMatrix{<:Real},
                                           z::AbstractMatrix{<:Complex},
                                           z_err::AbstractMatrix{<:Real},
                                           frequencies::AbstractVector{<:Real})
    size(rho_a) == size(z) == size(z_err) ||
        throw(DimensionMismatch("mt_apparent_errors_from_impedance: rho_a, z and z_err must match"))
    length(frequencies) == size(rho_a, 1) ||
        throw(DimensionMismatch("mt_apparent_errors_from_impedance: one frequency per period row"))

    nf, ns = size(rho_a)
    err_rho = Matrix{Float64}(undef, nf, ns)
    err_phase = Matrix{Float64}(undef, nf, ns)
    @inbounds for s in 1:ns, f in 1:nf
        zf = z[f, s]
        σ = z_err[f, s]
        ω = 2π * frequencies[f]
        if !isfinite(real(zf)) || !isfinite(imag(zf)) || !isfinite(σ) || σ <= 0 || ω <= 0
            err_rho[f, s] = NaN
            err_phase[f, s] = NaN
            continue
        end
        re, im = real(zf), imag(zf)
        inv_mu0w = 1 / (MU0 * ω)
        err_rho[f, s] = sqrt((2re * inv_mu0w)^2 * σ^2 + (2im * inv_mu0w)^2 * σ^2)
        mag2 = abs2(zf)
        err_phase[f, s] = mag2 > 0 ?
            rad2deg(sqrt(((-im / mag2) * σ)^2 + ((re / mag2) * σ)^2)) : NaN
    end
    return err_rho, err_phase
end

"""
    mt_apparent_errors_from_relative_noise(rho_a, noise) -> (err_rho_a, err_phase)

Uncertainties matching [`synth_mt_sites`](@ref)'s `noise` model: relative σ on
`rho_a` and `(noise/2 * 180/π)` degrees on phase.
"""
function mt_apparent_errors_from_relative_noise(rho_a::AbstractMatrix{<:Real}, noise::Real)
    noise >= 0 || throw(ArgumentError("mt_apparent_errors_from_relative_noise: noise must be non-negative"))
    err_rho = Matrix{Float64}(noise .* rho_a)
    err_phase = fill(noise / 2 * 180 / π, size(rho_a))
    return err_rho, err_phase
end

"""
    FeatureStack(data, names)

Feature tensor of size `[nx, ny, nz, nchannel]` with one name per channel.

Index by channel name with `stack["gravity"]` to get the `[nx, ny, nz]` slice.
"""
struct FeatureStack
    data::Array{Float64,4}
    names::Vector{String}

    function FeatureStack(data::Array{Float64,4}, names::Vector{String})
        size(data, 4) == length(names) || throw(ArgumentError(
            "FeatureStack: $(size(data, 4)) channels but $(length(names)) names"))
        allunique(names) || throw(ArgumentError("FeatureStack: channel names must be unique"))
        return new(data, names)
    end
end

nchannels(s::FeatureStack) = length(s.names)
Base.size(s::FeatureStack) = size(s.data)

function Base.getindex(s::FeatureStack, name::AbstractString)
    k = findfirst(==(String(name)), s.names)
    k === nothing && throw(KeyError(name))
    return view(s.data, :, :, :, k)
end

function Base.show(io::IO, s::FeatureStack)
    nx, ny, nz, nc = size(s.data)
    print(io, "FeatureStack($(nx)x$(ny)x$(nz), $(nc) channels: ", join(s.names, ", "), ")")
end

#---------- grid utilities ----------

"""
    standardize(A) -> Array

Shift and scale to zero mean and unit standard deviation. A constant input maps
to all zeros instead of dividing by zero.
"""
function standardize(A::AbstractArray{<:Real})
    all(isfinite, A) || throw(ArgumentError("standardize: input has non-finite entries"))
    μ = mean(A)
    σ = std(A)
    return σ > 0 ? (A .- μ) ./ σ : zeros(size(A))
end

"""
    extrude(M, nz) -> Array{Float64,3}

Replicate an `[nx, ny]` map down `nz` layers.
"""
function extrude(M::AbstractMatrix{<:Real}, nz::Integer)
    nx, ny = size(M)
    out = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz
        out[:, :, k] .= M
    end
    return out
end

"""
    idw_to_grid(g::PriorGrid, sx, sy, v; power=2.0, smoothing=nothing) -> Matrix

Inverse-distance interpolation of scattered values `v` at `(sx, sy)` onto the
grid's horizontal cell centres, returning an `[nx, ny]` map.

Weights are `1 / (d^power + s^power)`, so the smoothing length `s` (metres, by
default the median horizontal cell size) keeps the interpolant finite at a
station and sets the scale below which the field is flattened. Chosen over a
triangulation because station layouts are often sparse and irregular and an
extrapolating interpolant is more useful here than a hull-limited one.
"""
function idw_to_grid(g::PriorGrid,
                     sx::AbstractVector{<:Real},
                     sy::AbstractVector{<:Real},
                     v::AbstractVector{<:Real};
                     power::Real = 2.0,
                     smoothing::Union{Nothing,Real} = nothing)
    ns = length(sx)
    (length(sy) == ns && length(v) == ns) ||
        throw(ArgumentError("idw_to_grid: sx, sy and v must have equal length"))
    ns > 0 || throw(ArgumentError("idw_to_grid: need at least one station"))
    power > 0 || throw(ArgumentError("idw_to_grid: power must be positive"))

    s = smoothing === nothing ? g.h_median : float(smoothing)
    s > 0 || throw(ArgumentError("idw_to_grid: smoothing must be positive"))
    sp = s^power

    nx, ny, _ = size(g)
    out = Array{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        num = 0.0
        den = 0.0
        for t in 1:ns
            d = hypot(g.cx[i] - sx[t], g.cy[j] - sy[t])
            w = 1.0 / (d^power + sp)
            num += w * v[t]
            den += w
        end
        out[i, j] = num / den
    end
    return out
end

# area-weighted separable Gaussian smoothing on a possibly non-uniform grid;
# weighting by cell width is what keeps the result independent of how the mesh
# grades, which an index-space kernel would not be
function _smooth_axis(M::AbstractMatrix{Float64}, centers::Vector{Float64},
                      widths::Vector{Float64}, σ::Float64, dim::Int)
    n = length(centers)
    out = similar(M)
    inv2σ² = 1.0 / (2σ^2)
    @inbounds for p in 1:n
        wsum = 0.0
        acc = dim == 1 ? zeros(size(M, 2)) : zeros(size(M, 1))
        for q in 1:n
            d = centers[p] - centers[q]
            w = widths[q] * exp(-d * d * inv2σ²)
            w == 0 && continue
            wsum += w
            if dim == 1
                @views acc .+= w .* M[q, :]
            else
                @views acc .+= w .* M[:, q]
            end
        end
        if dim == 1
            @views out[p, :] .= acc ./ wsum
        else
            @views out[:, p] .= acc ./ wsum
        end
    end
    return out
end

"""
    gaussian_smooth_xy(g::PriorGrid, M, sigma_m) -> Matrix

Smooth an `[nx, ny]` map with a Gaussian of standard deviation `sigma_m` metres.

Applied as two area-weighted 1-D passes, so a graded mesh does not bias the
result the way an index-space kernel would.

Weights are renormalised per output cell, which keeps a constant field exactly
constant and avoids the darkening a truncated kernel would cause at the grid
edge. The trade-off is that on a strongly graded mesh a lone spike can have its
smoothed maximum land one cell off the spike, because the local weight sum
differs between cells. Constant preservation matters more here: these channels
are standardised afterwards, so an edge artefact would shift every statistic.
"""
function gaussian_smooth_xy(g::PriorGrid, M::AbstractMatrix{<:Real}, sigma_m::Real)
    nx, ny, _ = size(g)
    size(M) == (nx, ny) || throw(DimensionMismatch(
        "gaussian_smooth_xy: expected a $(nx)x$(ny) map, got $(size(M))"))
    sigma_m > 0 || throw(ArgumentError("gaussian_smooth_xy: sigma_m must be positive"))

    A = Matrix{Float64}(M)
    A = _smooth_axis(A, g.cx, g.dx, float(sigma_m), 1)
    A = _smooth_axis(A, g.cy, g.dy, float(sigma_m), 2)
    return A
end

"""
    gradient_xy(g::PriorGrid, M) -> (dMdx, dMdy)

Central-difference horizontal gradients of an `[nx, ny]` map, per metre, with
one-sided differences at the edges. Degenerate single-cell axes return zeros.
"""
function gradient_xy(g::PriorGrid, M::AbstractMatrix{<:Real})
    nx, ny, _ = size(g)
    size(M) == (nx, ny) || throw(DimensionMismatch(
        "gradient_xy: expected a $(nx)x$(ny) map, got $(size(M))"))

    dMdx = zeros(nx, ny)
    dMdy = zeros(nx, ny)

    if nx > 1
        @inbounds for j in 1:ny
            dMdx[1, j] = (M[2, j] - M[1, j]) / (g.cx[2] - g.cx[1])
            dMdx[nx, j] = (M[nx, j] - M[nx-1, j]) / (g.cx[nx] - g.cx[nx-1])
            for i in 2:nx-1
                dMdx[i, j] = (M[i+1, j] - M[i-1, j]) / (g.cx[i+1] - g.cx[i-1])
            end
        end
    end
    if ny > 1
        @inbounds for i in 1:nx
            dMdy[i, 1] = (M[i, 2] - M[i, 1]) / (g.cy[2] - g.cy[1])
            dMdy[i, ny] = (M[i, ny] - M[i, ny-1]) / (g.cy[ny] - g.cy[ny-1])
            for j in 2:ny-1
                dMdy[i, j] = (M[i, j+1] - M[i, j-1]) / (g.cy[j+1] - g.cy[j-1])
            end
        end
    end
    return dMdx, dMdy
end

# linear interpolation on a sorted abscissa, flat outside the range; duplicate
# abscissa values collapse to the first of the pair
function _interp_linear(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, x::Real)
    n = length(xs)
    n == 0 && throw(ArgumentError("_interp_linear: empty abscissa"))
    n == 1 && return float(ys[1])
    x <= xs[1] && return float(ys[1])
    x >= xs[n] && return float(ys[n])
    k = searchsortedlast(xs, x)
    k = clamp(k, 1, n - 1)
    dx = xs[k+1] - xs[k]
    dx == 0 && return float(ys[k])
    θ = (x - xs[k]) / dx
    return (1 - θ) * ys[k] + θ * ys[k+1]
end

#---------- physics-derived baseline ----------

"""
    nb_baseline(g::PriorGrid, sites::MTSites; power=2.0, smoothing=nothing)
        -> Array{Float64,3}

Niblett-Bostick background model in log10 ohm-metres on the grid.

Each site's sounding curve is transformed to a depth-resistivity profile, sampled
at the grid's cell-centre depths by interpolating in log-depth, and the resulting
columns are blended laterally by inverse-distance weighting.

This is the level the MT data already constrains on its own. It serves two roles:
a strong input channel, and the reference the network predicts a *residual*
against, which is what keeps the learned part free of the absolute resistivity
level of any one survey.
"""
function nb_baseline(g::PriorGrid, sites::MTSites;
                     power::Real = 2.0,
                     smoothing::Union{Nothing,Real} = nothing)
    nx, ny, nz = size(g)
    ns = nsites(sites)

    depth = depth_below_top(g)
    zc = [depth[1, 1, k] for k in 1:nz]
    # log-depth sampling: MT resolution degrades geometrically with depth, so a
    # linear interpolation in depth would over-weight the deep, poorly resolved end
    log_zc = log10.(max.(zc, 1.0))

    cols = Array{Float64}(undef, nz, ns)
    @inbounds for t in 1:ns
        d, ρ = niblett_bostick(sites.periods, view(sites.rho_a, :, t), view(sites.phase, :, t))
        keep = findall(i -> d[i] > 0 && ρ[i] > 0, eachindex(d))
        if isempty(keep)
            cols[:, t] .= 0.0
            continue
        end
        ld = log10.(d[keep])
        lr = log10.(ρ[keep])
        for k in 1:nz
            cols[k, t] = _interp_linear(ld, lr, log_zc[k])
        end
    end

    s = smoothing === nothing ? g.h_median : float(smoothing)
    sp = s^power

    out = Array{Float64}(undef, nx, ny, nz)
    @inbounds for j in 1:ny, i in 1:nx
        wsum = 0.0
        w = Vector{Float64}(undef, ns)
        for t in 1:ns
            dist = hypot(g.cx[i] - sites.x[t], g.cy[j] - sites.y[t])
            w[t] = 1.0 / (dist^power + sp)
            wsum += w[t]
        end
        for k in 1:nz
            acc = 0.0
            for t in 1:ns
                acc += w[t] * cols[k, t]
            end
            out[i, j, k] = acc / wsum
        end
    end
    return out
end

"""
    nb_baseline_lateral_std(baseline) -> Float64

Sample standard deviation of depth-averaged columns of an NB baseline (or any
`nx×ny×nz` log₁₀ρ field). Measures column-to-column lateral spread — the part of
the baseline that varies between soundings — without using any truth model.
"""
function nb_baseline_lateral_std(baseline::AbstractArray{<:Real,3})
    nx, ny, nz = size(baseline)
    n = nx * ny
    n == 0 && return 0.0
    col_means = Vector{Float64}(undef, n)
    t = 0
    @inbounds for j in 1:ny, i in 1:nx
        t += 1
        s = 0.0
        for k in 1:nz
            s += Float64(baseline[i, j, k])
        end
        col_means[t] = s / nz
    end
    n < 2 && return 0.0
    return std(col_means)
end

"""
    residual_span_half_band(log_rho_bounds) -> Float64

Permissive, truth-free residual span: half the physical log₁₀ρ band width.

This is the Musgrave rule of thumb (`log_rho_bounds = (0.5, 4.5)` → span `2.0`):
the network may reach anywhere inside the physical band from a mid-band baseline,
without being told how far the truth sits from that baseline.
"""
function residual_span_half_band(log_rho_bounds::Tuple{Real,Real})
    lo, hi = Float64(log_rho_bounds[1]), Float64(log_rho_bounds[2])
    lo < hi || throw(ArgumentError(
        "residual_span_half_band: log_rho_bounds must be increasing, got $(log_rho_bounds)"))
    return 0.5 * (hi - lo)
end

"""
    residual_span_from_baseline(baseline; k=3.0, floor=1.0, ceil=5.0) -> Float64

Data-driven residual span from lateral NB spread: `clamp(k · σ_lat, floor, ceil)`,
where `σ_lat` is [`nb_baseline_lateral_std`](@ref).

Portable to field surveys (no truth). `k` is an external multiplier, not fit to
truth; `floor` / `ceil` keep the reachable band usable when the baseline is
almost uniform or wildly variable.
"""
function residual_span_from_baseline(baseline::AbstractArray{<:Real,3};
                                     k::Real = 3.0,
                                     floor::Real = 1.0,
                                     ceil::Real = 5.0)
    k > 0 || throw(ArgumentError("residual_span_from_baseline: k must be positive"))
    floor > 0 || throw(ArgumentError("residual_span_from_baseline: floor must be positive"))
    ceil >= floor || throw(ArgumentError(
        "residual_span_from_baseline: ceil must be ≥ floor, got floor=$(floor) ceil=$(ceil)"))
    span = float(k) * nb_baseline_lateral_std(baseline)
    return clamp(span, float(floor), float(ceil))
end

#---------- channel builders ----------

# default multi-scale smoothing lengths, as multiples of the median horizontal
# cell size; separating wavelengths is how a gravity map hints at source depth
const DEFAULT_GRAVITY_SCALES = (2.0, 5.0, 12.0)

"""
    gravity_cell_sensitivity(g::PriorGrid, obs::GravityObs; units=:mgal)
        -> Array{Float64,3}

[`gravity_cell_sensitivity`](@ref) from a [`GravityObs`](@ref) station set.
"""
gravity_cell_sensitivity(g::PriorGrid, obs::GravityObs; kwargs...) =
    gravity_cell_sensitivity(g, obs.x, obs.y, obs.z; kwargs...)

"""
    gravity_sensitivity_channel(g::PriorGrid, obs::GravityObs) -> Array{Float64,3}

Depth-aware gravity feature: prism-kernel sensitivity *per unit volume*,
modulated by the interpolated surface anomaly, then standardised in-survey.

Unlike the extruded maps in [`gravity_channels`](@ref), this field varies with
depth. [`gravity_cell_sensitivity`](@ref) is the column L2 of
[`gravity_matrix`](@ref) — how much that *cell* moves the stations, which grows
with cell volume. Dividing by [`cell_volumes`](@ref) recovers an
upward-continuation-style decay: a given density contrast at depth z contributes
less than the same contrast near the surface, even when deep cells are thicker.
A kernel-only map (no anomaly) would be almost a 1-D depth function on a
well-covered profile (redundant with `log_depth`); the product is what gives
the network a z-dependent gravity *location*.

The kernel is the same operator as [`gravity_matrix`](@ref) / [`synth_gravity`](@ref).
OPERATOR inverse crime exists if this channel is later trained against a loss
that uses that operator; the channel does not remove it. Slab density still
does not pass through [`density_from_mu`](@ref) — no petrophysical inverse crime.
"""
function gravity_sensitivity_channel(g::PriorGrid, obs::GravityObs)
    nx, ny, nz = size(g)
    sens = gravity_cell_sensitivity(g, obs)
    vol = cell_volumes(g)
    anomaly = idw_to_grid(g, obs.x, obs.y, obs.value)
    raw = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        raw[i, j, k] = (sens[i, j, k] / vol[i, j, k]) * anomaly[i, j]
    end
    return standardize(raw)
end

"""
    gravity_channels(g::PriorGrid, obs::GravityObs; scales=DEFAULT_GRAVITY_SCALES,
                     sensitivity=false) -> (Vector{Array{Float64,3}}, Vector{String})

Feature channels derived from a gravity survey: the standardised anomaly, its two
horizontal gradients, and one band-pass channel per entry in `scales`.

`scales` are Gaussian smoothing lengths in multiples of the median horizontal
cell size. Each band-pass channel is the difference between successively smoothed
versions, so short and long wavelengths land in separate channels; a deep source
shows up only in the long ones, which is the closest a single gravity map gets to
depth information. Every one of those maps is extruded through depth — they do
not vary with `z`.

When `sensitivity=true`, appends one extra channel `"gravity_sensitivity"`:
per-volume prism-kernel sensitivity times the interpolated surface anomaly
(see [`gravity_sensitivity_channel`](@ref)). Default `false` keeps the historical
channel count.
"""
function gravity_channels(g::PriorGrid, obs::GravityObs;
                          scales = DEFAULT_GRAVITY_SCALES,
                          sensitivity::Bool = false)
    nx, ny, nz = size(g)
    h = g.h_median

    raw = idw_to_grid(g, obs.x, obs.y, obs.value)
    chans = Array{Float64,3}[]
    names = String[]

    push!(chans, extrude(standardize(raw), nz))
    push!(names, "gravity")

    gx, gy = gradient_xy(g, standardize(raw))
    push!(chans, extrude(standardize(gx), nz)); push!(names, "gravity_dx")
    push!(chans, extrude(standardize(gy), nz)); push!(names, "gravity_dy")

    prev = standardize(raw)
    for (n, m) in enumerate(scales)
        σ = float(m) * h
        smoothed = gaussian_smooth_xy(g, standardize(raw), σ)
        push!(chans, extrude(standardize(prev .- smoothed), nz))
        push!(names, "gravity_band$(n)")
        prev = smoothed
    end
    push!(chans, extrude(standardize(prev), nz))
    push!(names, "gravity_long")

    if sensitivity
        push!(chans, gravity_sensitivity_channel(g, obs))
        push!(names, "gravity_sensitivity")
    end

    return chans, names
end

"""
    topography_channels(g::PriorGrid, surface_z) -> (Vector{Array{Float64,3}}, Vector{String})

Feature channels from the ground surface: the standardised surface height and the
depth of each cell below the surface, normalised by the grid's depth extent.

`surface_z` is an `[nx, ny]` map of the ground surface in the grid's frame
(metres, `z` positive down), as produced by `MTGeophysics.extract_topography`.
"""
function topography_channels(g::PriorGrid, surface_z::AbstractMatrix{<:Real})
    nx, ny, nz = size(g)
    size(surface_z) == (nx, ny) || throw(DimensionMismatch(
        "topography_channels: expected a $(nx)x$(ny) surface map, got $(size(surface_z))"))

    chans = Array{Float64,3}[]
    names = String[]

    push!(chans, extrude(standardize(surface_z), nz))
    push!(names, "surface_z")

    span = max(g.z[end] - g.z[1], eps())
    below = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        below[i, j, k] = (g.cz[k] - surface_z[i, j]) / span
    end
    push!(chans, below)
    push!(names, "depth_below_surface")

    return chans, names
end

"""
    depth_channels(g::PriorGrid; rho_ref=100.0, period_max=1000.0)
        -> (Vector{Array{Float64,3}}, Vector{String})

Depth channels: standardised log-depth, and depth divided by the skin depth of a
`rho_ref` half-space at `period_max`.

The second channel is the important one. It states depth in units of how deep the
data can actually see, so a shallow high-frequency survey and a deep long-period
one present the same numbers to the network for cells that are equally well
resolved.
"""
function depth_channels(g::PriorGrid; rho_ref::Real = 100.0, period_max::Real = 1000.0)
    rho_ref > 0 || throw(ArgumentError("depth_channels: rho_ref must be positive"))
    period_max > 0 || throw(ArgumentError("depth_channels: period_max must be positive"))

    d = depth_below_top(g)
    d0 = g.dz[1] / 2

    chans = Array{Float64,3}[]
    names = String[]

    push!(chans, standardize(log10.(max.(d, d0) ./ d0)))
    push!(names, "log_depth")

    δ = skin_depth(rho_ref, period_max)
    push!(chans, d ./ δ)
    push!(names, "depth_over_skin")

    return chans, names
end

"""
    coverage_channels(g::PriorGrid, sx, sy, sz=nothing; name="site_distance")
        -> (Vector{Array{Float64,3}}, Vector{String})

Distance from each cell to the nearest sample, in units of the median horizontal
cell size, passed through `log1p`.

Tells the network where the data has something to say. Far from every site the
prior should widen rather than invent structure, and this is the channel that
lets `sigma` learn to do that.

With `sz === nothing` the distance is plan-view and the map is extruded through
depth, matching the original MT-site channel. Passing `sz` makes the distance
fully 3-D, so a deep cell can sit far from a shallow sample even when they share
an (x, y).
"""
function coverage_channels(g::PriorGrid,
                           sx::AbstractVector{<:Real},
                           sy::AbstractVector{<:Real},
                           sz::Union{Nothing,AbstractVector{<:Real}} = nothing;
                           name::AbstractString = "site_distance")
    length(sx) == length(sy) ||
        throw(ArgumentError("coverage_channels: sx and sy must have equal length"))
    isempty(sx) && throw(ArgumentError("coverage_channels: need at least one site"))
    if sz !== nothing
        length(sz) == length(sx) || throw(ArgumentError(
            "coverage_channels: sz has $(length(sz)) entries but sx has $(length(sx))"))
    end

    nx, ny, nz = size(g)
    h = g.h_median
    ns = length(sx)

    if sz === nothing
        near = Array{Float64}(undef, nx, ny)
        @inbounds for j in 1:ny, i in 1:nx
            best = Inf
            for t in 1:ns
                best = min(best, hypot(g.cx[i] - sx[t], g.cy[j] - sy[t]))
            end
            near[i, j] = log1p(best / h)
        end
        return Array{Float64,3}[extrude(near, nz)], String[String(name)]
    end

    near = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        best = Inf
        for t in 1:ns
            isfinite(sz[t]) || continue
            best = min(best, sqrt((g.cx[i] - sx[t])^2 + (g.cy[j] - sy[t])^2 +
                                  (g.cz[k] - sz[t])^2))
        end
        near[i, j, k] = log1p(best / h)
    end
    return Array{Float64,3}[near], String[String(name)]
end

# nearest sample index per (i, j) in plan view, or per cell in 3-D. `mask` selects
# which samples compete; an all-false mask returns zeros (no neighbour)
function _nearest_index_map(g::PriorGrid,
                            sx::AbstractVector{<:Real},
                            sy::AbstractVector{<:Real},
                            sz::Union{Nothing,AbstractVector{<:Real}},
                            mask::AbstractVector{Bool})
    nx, ny, nz = size(g)
    ns = length(sx)
    if sz === nothing
        idx = zeros(Int, nx, ny)
        @inbounds for j in 1:ny, i in 1:nx
            best = 0
            bestd = Inf
            for t in 1:ns
                mask[t] || continue
                d = hypot(g.cx[i] - sx[t], g.cy[j] - sy[t])
                if d < bestd
                    bestd = d
                    best = t
                end
            end
            idx[i, j] = best
        end
        return idx
    end
    idx = zeros(Int, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        best = 0
        bestd = Inf
        for t in 1:ns
            mask[t] || continue
            isfinite(sz[t]) || continue
            d2 = (g.cx[i] - sx[t])^2 + (g.cy[j] - sy[t])^2 + (g.cz[k] - sz[t])^2
            if d2 < bestd
                bestd = d2
                best = t
            end
        end
        idx[i, j, k] = best
    end
    return idx
end

function _take_nearest(g::PriorGrid, idx::Array{Int,2}, values::AbstractVector{<:Real})
    nx, ny, nz = size(g)
    out = Array{Float64,3}(undef, nx, ny, nz)
    fill_val = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        t = idx[i, j]
        v = t == 0 ? fill_val : Float64(values[t])
        for k in 1:nz
            out[i, j, k] = v
        end
    end
    return out
end

function _take_nearest(g::PriorGrid, idx::Array{Int,3}, values::AbstractVector{<:Real})
    nx, ny, nz = size(g)
    size(idx) == (nx, ny, nz) || throw(DimensionMismatch(
        "_take_nearest: index map must match the grid $(nx)x$(ny)x$(nz), got $(size(idx))"))
    out = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        t = idx[i, j, k]
        out[i, j, k] = t == 0 ? 0.0 : Float64(values[t])
    end
    return out
end

"""
    nearest_sample_channels(g::PriorGrid, samples::PointSamples; standardize_values=true)
        -> (Vector{Array{Float64,3}}, Vector{String})

Nearest-neighbour interpolation of [`PointSamples`](@ref) onto the grid.

Each channel is interpolated independently: a sample that is missing (non-finite)
in one column does not compete for that column, so a copper-only station does
not paint a nickel channel. Empty channels (no finite samples) are dropped.

When `standardize_values` is true, each interpolated field is then standardised
in-survey, matching [`gravity_channels`](@ref).
"""
function nearest_sample_channels(g::PriorGrid, samples::PointSamples;
                                 standardize_values::Bool = true)
    nx, ny, nz = size(g)
    chans = Array{Float64,3}[]
    names = String[]
    ns = length(samples)
    for (c, name) in enumerate(samples.names)
        mask = Vector{Bool}(undef, ns)
        @inbounds for t in 1:ns
            mask[t] = isfinite(samples.values[t, c])
        end
        any(mask) || continue
        idx = _nearest_index_map(g, samples.x, samples.y, samples.z, mask)
        field = _take_nearest(g, idx, view(samples.values, :, c))
        push!(chans, standardize_values ? standardize(field) : field)
        push!(names, name)
    end
    isempty(chans) && throw(ArgumentError(
        "nearest_sample_channels: no finite samples in any channel"))
    return chans, names
end

"""
    geochemistry_channels(g::PriorGrid, samples::PointSamples; log_transform=true)
        -> (Vector{Array{Float64,3}}, Vector{String})

Geochemistry feature channels: log10 of positive concentrations, nearest-sample
interpolation, then in-survey standardisation.

Non-positive entries are treated as missing so a detection-limit flag of zero
does not become `-Inf` after the log. Channel names are prefixed with
`geochem_` unless they already start with that token.
"""
function geochemistry_channels(g::PriorGrid, samples::PointSamples;
                               log_transform::Bool = true)
    values = copy(samples.values)
    if log_transform
        @inbounds for i in eachindex(values)
            v = values[i]
            values[i] = isfinite(v) && v > 0 ? log10(v) : NaN
        end
    end
    names = [startswith(n, "geochem_") ? n : "geochem_" * n for n in samples.names]
    logged = PointSamples(samples.x, samples.y, values, names; z = samples.z)
    return nearest_sample_channels(g, logged; standardize_values = true)
end

function _lith_token(label::AbstractString)
    raw = String(label)
    if !isvalid(raw)
        # Latin-1 bytes from dBase (e.g. Ä = 0xC4) survive as U+00xx
        raw = String(Char.(codeunits(raw)))
    end
    token = uppercase(strip(raw))
    token = replace(token, r"[^A-Z0-9]+" => "_")
    token = strip(token, '_')
    return isempty(token) ? "OTHER" : token
end

"""
    lithology_channels(g::PriorGrid, samples::LabelSamples; min_frequency=0.01)
        -> (Vector{Array{Float64,3}}, Vector{String})

One-hot lithology channels from nearest-sample labels.

Classes whose share of the sample set is below `min_frequency` collapse to
`OTHER`, so a 200-way rock-type vocabulary does not explode the feature width.
The resulting 0/1 fields are *not* standardised: a one-hot bit is already on a
fixed scale, and shifting it by its class frequency would mix prevalence into
every cell.
"""
function lithology_channels(g::PriorGrid, samples::LabelSamples;
                            min_frequency::Real = 0.01)
    0 <= min_frequency < 1 || throw(ArgumentError(
        "lithology_channels: min_frequency must lie in [0, 1), got $(min_frequency)"))

    n = length(samples)
    tokens = [_lith_token(s) for s in samples.labels]
    counts = Dict{String,Int}()
    for t in tokens
        counts[t] = get(counts, t, 0) + 1
    end
    keep = Set{String}()
    for (lab, c) in counts
        lab == "OTHER" && continue
        (c / n) >= min_frequency && push!(keep, lab)
    end
    collapsed = [lab in keep ? lab : "OTHER" for lab in tokens]
    classes = sort!(collect(keep))
    any(==("OTHER"), collapsed) && push!(classes, "OTHER")
    isempty(classes) && throw(ArgumentError("lithology_channels: no labels left to encode"))

    class_index = Dict(lab => i for (i, lab) in enumerate(classes))
    onehot = zeros(n, length(classes))
    @inbounds for t in 1:n
        onehot[t, class_index[collapsed[t]]] = 1.0
    end
    names = ["lith_" * c for c in classes]
    pts = PointSamples(samples.x, samples.y, onehot, names; z = samples.z)
    return nearest_sample_channels(g, pts; standardize_values = false)
end

"""
    build_features(g::PriorGrid; gravity=nothing, surface_z=nothing, sites=nothing,
                   baseline=nothing, coordinates=true, rho_ref=100.0,
                   gravity_scales=DEFAULT_GRAVITY_SCALES,
                   gravity_sensitivity=false, geochemistry=nothing,
                   lithology=nothing, coverage_points=nothing,
                   lithology_min_frequency=0.01) -> FeatureStack

Assemble the available data into a feature tensor.

Every input is optional so the same call works on a survey with gravity but no
topography, or sites but no gravity; only the channels that can be built are
included. `coordinates = true` appends the normalised cell-centre coordinates,
which anchor a per-survey neural field but must be dropped when training a model
intended to transfer between surveys.

`baseline` is a `[nx, ny, nz]` log10 resistivity background, normally the output
of [`nb_baseline`](@ref); when `sites` is given and `baseline` is not, it is
computed automatically.

`gravity_sensitivity` defaults to `false`, which keeps the historical gravity
channel count (extruded surface maps only). Set `true` to append the depth-aware
`"gravity_sensitivity"` channel; ignored when `gravity` is not supplied.

`geochemistry` and `lithology` are the non-geophysical inputs used by the
Keivitsa training line: nearest-sample interpolation of assay/till chemistry and
one-hot rock type. `coverage_points` is a named tuple `(x, y)` or `(x, y, z)` of
those same samples, producing a `"sample_distance"` channel analogous to the MT
`"site_distance"` map.
"""
function build_features(g::PriorGrid;
                        gravity::Union{Nothing,GravityObs} = nothing,
                        surface_z::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                        sites::Union{Nothing,MTSites} = nothing,
                        baseline::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
                        coordinates::Bool = true,
                        rho_ref::Real = 100.0,
                        gravity_scales = DEFAULT_GRAVITY_SCALES,
                        gravity_sensitivity::Bool = false,
                        geochemistry::Union{Nothing,PointSamples} = nothing,
                        lithology::Union{Nothing,LabelSamples} = nothing,
                        coverage_points::Union{Nothing,NamedTuple} = nothing,
                        lithology_min_frequency::Real = 0.01)
    nx, ny, nz = size(g)
    chans = Array{Float64,3}[]
    names = String[]

    if coordinates
        Xn, Yn, Zn = normalized_centers(g)
        append!(chans, (Xn, Yn, Zn))
        append!(names, ("x_norm", "y_norm", "z_norm"))
    end

    period_max = sites === nothing ? 1000.0 : maximum(sites.periods)
    ref = rho_ref
    if sites !== nothing && all(isfinite, sites.rho_a) && !isempty(sites.rho_a)
        ref = median(sites.rho_a)
    end
    c, n = depth_channels(g; rho_ref = ref, period_max = period_max)
    append!(chans, c); append!(names, n)

    if gravity !== nothing
        c, n = gravity_channels(g, gravity; scales = gravity_scales,
                                sensitivity = gravity_sensitivity)
        append!(chans, c); append!(names, n)
    end

    if surface_z !== nothing
        c, n = topography_channels(g, surface_z)
        append!(chans, c); append!(names, n)
    end

    if sites !== nothing
        c, n = coverage_channels(g, sites.x, sites.y)
        append!(chans, c); append!(names, n)
    end

    if geochemistry !== nothing
        c, n = geochemistry_channels(g, geochemistry)
        append!(chans, c); append!(names, n)
    end

    if lithology !== nothing
        c, n = lithology_channels(g, lithology; min_frequency = lithology_min_frequency)
        append!(chans, c); append!(names, n)
    end

    if coverage_points !== nothing
        hasproperty(coverage_points, :x) && hasproperty(coverage_points, :y) ||
            throw(ArgumentError("build_features: coverage_points needs :x and :y"))
        cz = hasproperty(coverage_points, :z) ? coverage_points.z : nothing
        c, n = coverage_channels(g, coverage_points.x, coverage_points.y, cz;
                                 name = "sample_distance")
        append!(chans, c); append!(names, n)
    end

    base = baseline
    if base === nothing && sites !== nothing
        base = nb_baseline(g, sites)
    end
    if base !== nothing
        size(base) == (nx, ny, nz) || throw(DimensionMismatch(
            "build_features: baseline must be $(nx)x$(ny)x$(nz), got $(size(base))"))
        push!(chans, standardize(base))
        push!(names, "baseline")
    end

    isempty(chans) && throw(ArgumentError(
        "build_features: no data supplied, nothing to build"))

    data = Array{Float64,4}(undef, nx, ny, nz, length(chans))
    @inbounds for (k, ch) in enumerate(chans)
        data[:, :, :, k] .= ch
    end
    return FeatureStack(data, names)
end

"""
    feature_matrix(s::FeatureStack) -> Matrix{Float64}

Reshape a stack to `[nchannel, ncell]`, the layout Lux dense layers expect.

Cell ordering matches `vec` of a `[nx, ny, nz]` array, so a network output can be
reshaped straight back onto the grid.
"""
function feature_matrix(s::FeatureStack)
    nx, ny, nz, nc = size(s.data)
    return permutedims(reshape(s.data, nx * ny * nz, nc), (2, 1))
end
