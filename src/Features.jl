# Scattered samples to a feature tensor.
#
# Every channel produced here is dimensionless and standardised within the
# survey, except one-hot lithology (already on a fixed 0/1 scale). Feature
# tensors are [nx, ny, nz, nchannel] so that a channel is a contiguous 3-D
# slice matching the model grid.

"""
    PointSamples(x, y, values, names; z=nothing)

Scattered numeric samples for nearest-neighbour interpolation onto a prior grid.

`values` is `[nsample, nchannel]` with one column per entry in `names`. `z` is
optional: omit it (or pass `nothing`) for a plan-view interpolant that is then
extruded through depth.
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
    FeatureStack(data, names)

Feature tensor of size `[nx, ny, nz, nchannel]` with one name per channel.

Index by channel name with `stack["log_depth"]` to get the `[nx, ny, nz]` slice.
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
station and sets the scale below which the field is flattened.
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

# area-weighted separable Gaussian smoothing on a possibly non-uniform grid
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
edge.
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

#---------- channel builders ----------

# Geometric depth scale: skin depth of a reference half-space. Used only as a
# dimensionless depth channel (`depth_over_skin`), not as an MT solver.
const _MU0 = 4π * 1e-7
_skin_depth(rho::Real, T::Real) = sqrt(2 * rho * T / (2π * _MU0))

"""
    topography_channels(g::PriorGrid, surface_z) -> (Vector{Array{Float64,3}}, Vector{String})

Feature channels from the ground surface: the standardised surface height and the
depth of each cell below the surface, normalised by the grid's depth extent.

`surface_z` is an `[nx, ny]` map of the ground surface in the grid's frame.
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
`rho_ref` half-space at `period_max`. The second channel is a geometric scale
so surveys of different size present comparable numbers; it is not an MT input.
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

    δ = _skin_depth(rho_ref, period_max)
    push!(chans, d ./ δ)
    push!(names, "depth_over_skin")

    return chans, names
end

"""
    coverage_channels(g::PriorGrid, sx, sy, sz=nothing; name="sample_distance")
        -> (Vector{Array{Float64,3}}, Vector{String})

Distance from each cell to the nearest sample, in units of the median horizontal
cell size, passed through `log1p`.

With `sz === nothing` the distance is plan-view and the map is extruded through
depth. Passing `sz` makes the distance fully 3-D.
"""
function coverage_channels(g::PriorGrid,
                           sx::AbstractVector{<:Real},
                           sy::AbstractVector{<:Real},
                           sz::Union{Nothing,AbstractVector{<:Real}} = nothing;
                           name::AbstractString = "sample_distance")
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
in one column does not compete for that column. Empty channels (no finite
samples) are dropped.
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
`OTHER`. The resulting 0/1 fields are *not* standardised.
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
    build_features(g::PriorGrid; surface_z=nothing, coordinates=true,
                   rho_ref=100.0, geochemistry=nothing, lithology=nothing,
                   coverage_points=nothing, lithology_min_frequency=0.01)
        -> FeatureStack

Assemble the available data into a feature tensor.

Every input is optional; only the channels that can be built are included.
`coordinates = true` appends the normalised cell-centre coordinates.

`geochemistry` and `lithology` are nearest-sample interpolations.
`coverage_points` is a named tuple `(x, y)` or `(x, y, z)` producing a
`"sample_distance"` channel.
"""
function build_features(g::PriorGrid;
                        surface_z::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                        coordinates::Bool = true,
                        rho_ref::Real = 100.0,
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

    c, n = depth_channels(g; rho_ref = rho_ref)
    append!(chans, c); append!(names, n)

    if surface_z !== nothing
        c, n = topography_channels(g, surface_z)
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

    isempty(chans) && throw(ArgumentError(
        "build_features: no data supplied, nothing to build"))

    data = Array{Float64,4}(undef, nx, ny, nz, length(chans))
    @inbounds for (k, ch) in enumerate(chans)
        data[:, :, :, k] .= ch
    end
    return FeatureStack(data, names)
end

"""
    append_channels(s::FeatureStack, chans, names) -> FeatureStack

Concatenate extra `[nx,ny,nz]` channels onto an existing stack. Used by the
Cloncurry line to add structural-geology maps without threading them through
[`build_features`](@ref).
"""
function append_channels(s::FeatureStack,
                         chans::AbstractVector{<:AbstractArray{<:Real,3}},
                         names::AbstractVector{<:AbstractString})
    length(chans) == length(names) || throw(ArgumentError(
        "append_channels: $(length(chans)) channels but $(length(names)) names"))
    isempty(chans) && return s
    nx, ny, nz, nc0 = size(s.data)
    for (ch, name) in zip(chans, names)
        size(ch) == (nx, ny, nz) || throw(DimensionMismatch(
            "append_channels: channel $(repr(name)) is $(size(ch)), " *
            "expected ($nx, $ny, $nz)"))
        name in s.names && throw(ArgumentError(
            "append_channels: duplicate channel name $(repr(name))"))
    end
    nc = nc0 + length(chans)
    data = Array{Float64,4}(undef, nx, ny, nz, nc)
    @inbounds data[:, :, :, 1:nc0] .= s.data
    @inbounds for (k, ch) in enumerate(chans)
        data[:, :, :, nc0 + k] .= ch
    end
    return FeatureStack(data, vcat(s.names, String.(names)))
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
