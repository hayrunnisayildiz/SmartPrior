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
keeps [`mt_column_misfit`](@ref) on its original unnormalised formula.
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

#---------- channel builders ----------

# default multi-scale smoothing lengths, as multiples of the median horizontal
# cell size; separating wavelengths is how a gravity map hints at source depth
const DEFAULT_GRAVITY_SCALES = (2.0, 5.0, 12.0)

"""
    gravity_channels(g::PriorGrid, obs::GravityObs; scales=DEFAULT_GRAVITY_SCALES)
        -> (Vector{Array{Float64,3}}, Vector{String})

Feature channels derived from a gravity survey: the standardised anomaly, its two
horizontal gradients, and one band-pass channel per entry in `scales`.

`scales` are Gaussian smoothing lengths in multiples of the median horizontal
cell size. Each band-pass channel is the difference between successively smoothed
versions, so short and long wavelengths land in separate channels; a deep source
shows up only in the long ones, which is the closest a single gravity map gets to
depth information.
"""
function gravity_channels(g::PriorGrid, obs::GravityObs;
                          scales = DEFAULT_GRAVITY_SCALES)
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
    coverage_channels(g::PriorGrid, sx, sy) -> (Vector{Array{Float64,3}}, Vector{String})

Distance from each cell to the nearest MT site, in units of the median horizontal
cell size, passed through `log1p`.

Tells the network where the data has something to say. Far from every site the
prior should widen rather than invent structure, and this is the channel that
lets `sigma` learn to do that.
"""
function coverage_channels(g::PriorGrid,
                           sx::AbstractVector{<:Real},
                           sy::AbstractVector{<:Real})
    length(sx) == length(sy) ||
        throw(ArgumentError("coverage_channels: sx and sy must have equal length"))
    isempty(sx) && throw(ArgumentError("coverage_channels: need at least one site"))

    nx, ny, nz = size(g)
    h = g.h_median

    near = Array{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        best = Inf
        for t in eachindex(sx)
            best = min(best, hypot(g.cx[i] - sx[t], g.cy[j] - sy[t]))
        end
        near[i, j] = log1p(best / h)
    end

    return Array{Float64,3}[extrude(near, nz)], String["site_distance"]
end

"""
    build_features(g::PriorGrid; gravity=nothing, surface_z=nothing, sites=nothing,
                   baseline=nothing, coordinates=true, rho_ref=100.0,
                   gravity_scales=DEFAULT_GRAVITY_SCALES) -> FeatureStack

Assemble the available data into a feature tensor.

Every input is optional so the same call works on a survey with gravity but no
topography, or sites but no gravity; only the channels that can be built are
included. `coordinates = true` appends the normalised cell-centre coordinates,
which anchor a per-survey neural field but must be dropped when training a model
intended to transfer between surveys.

`baseline` is a `[nx, ny, nz]` log10 resistivity background, normally the output
of [`nb_baseline`](@ref); when `sites` is given and `baseline` is not, it is
computed automatically.
"""
function build_features(g::PriorGrid;
                        gravity::Union{Nothing,GravityObs} = nothing,
                        surface_z::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                        sites::Union{Nothing,MTSites} = nothing,
                        baseline::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
                        coordinates::Bool = true,
                        rho_ref::Real = 100.0,
                        gravity_scales = DEFAULT_GRAVITY_SCALES)
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
        c, n = gravity_channels(g, gravity; scales = gravity_scales)
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
