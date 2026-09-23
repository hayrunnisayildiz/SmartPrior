# Tensor-product 3-D grid for the neural-field block model.
# Arrays are indexed [i, j, k]. Axis meaning is the caller's: Cloncurry uses
# x = easting, y = northing, z = elevation (positive up), metres. The origin
# is the (x, y, z) corner of the first cell.

"""
    PriorGrid(dx, dy, dz; origin=[0.0, 0.0, 0.0])

Tensor-product 3-D grid on which a prior model lives.

`dx`, `dy`, `dz` are cell widths in metres. Cell edges and centres are derived
once at construction.
"""
struct PriorGrid
    dx::Vector{Float64}
    dy::Vector{Float64}
    dz::Vector{Float64}
    origin::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    cx::Vector{Float64}
    cy::Vector{Float64}
    cz::Vector{Float64}
    # median horizontal cell size, the natural length unit for this grid. Cached
    # rather than recomputed because several callers want it per invocation, and
    # because `median` sorts its argument in place, which Zygote reports as
    # forbidden mutation when it happens inside a differentiated function.
    h_median::Float64
end

function PriorGrid(dx::AbstractVector{<:Real},
                   dy::AbstractVector{<:Real},
                   dz::AbstractVector{<:Real};
                   origin::AbstractVector{<:Real} = [0.0, 0.0, 0.0])
    dxv = collect(Float64, dx)
    dyv = collect(Float64, dy)
    dzv = collect(Float64, dz)

    (isempty(dxv) || isempty(dyv) || isempty(dzv)) &&
        throw(ArgumentError("PriorGrid: cell width vectors must be non-empty"))
    all(>(0), dxv) || throw(ArgumentError("PriorGrid: dx must be strictly positive"))
    all(>(0), dyv) || throw(ArgumentError("PriorGrid: dy must be strictly positive"))
    all(>(0), dzv) || throw(ArgumentError("PriorGrid: dz must be strictly positive"))
    length(origin) == 3 || throw(ArgumentError("PriorGrid: origin must have 3 entries"))

    o = collect(Float64, origin)
    x = vcat(0.0, cumsum(dxv)) .+ o[1]
    y = vcat(0.0, cumsum(dyv)) .+ o[2]
    z = vcat(0.0, cumsum(dzv)) .+ o[3]

    cx = (x[1:end-1] .+ x[2:end]) ./ 2
    cy = (y[1:end-1] .+ y[2:end]) ./ 2
    cz = (z[1:end-1] .+ z[2:end]) ./ 2

    h_median = median(vcat(dxv, dyv))

    return PriorGrid(dxv, dyv, dzv, o, x, y, z, cx, cy, cz, h_median)
end

Base.size(g::PriorGrid) = (length(g.dx), length(g.dy), length(g.dz))
Base.size(g::PriorGrid, d::Integer) = size(g)[d]

"""
    ncells(g::PriorGrid) -> Int

Total number of cells.
"""
ncells(g::PriorGrid) = prod(size(g))

"""
    cell_volumes(g::PriorGrid) -> Array{Float64,3}

Cell volumes in cubic metres.
"""
function cell_volumes(g::PriorGrid)
    nx, ny, nz = size(g)
    V = Array{Float64,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        V[i, j, k] = g.dx[i] * g.dy[j] * g.dz[k]
    end
    return V
end

"""
    cell_centers(g::PriorGrid) -> (X, Y, Z)

Three `[nx, ny, nz]` arrays of cell-centre coordinates in metres, in the grid's
own frame (the origin is included).
"""
function cell_centers(g::PriorGrid)
    nx, ny, nz = size(g)
    X = Array{Float64,3}(undef, nx, ny, nz)
    Y = similar(X)
    Z = similar(X)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        X[i, j, k] = g.cx[i]
        Y[i, j, k] = g.cy[j]
        Z[i, j, k] = g.cz[k]
    end
    return X, Y, Z
end

"""
    depth_below_top(g::PriorGrid) -> Array{Float64,3}

Cell-centre depth measured from the top edge of the grid, in metres. Unlike
`cz` this drops the origin offset, so it stays a true depth on grids whose
`origin[3]` places the datum above the ground surface (air layers).
"""
function depth_below_top(g::PriorGrid)
    nx, ny, nz = size(g)
    d = Array{Float64,3}(undef, nx, ny, nz)
    z_top = g.z[1]
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        d[i, j, k] = g.cz[k] - z_top
    end
    return d
end

# maps a coordinate vector onto [-1, 1] using its own extent; a degenerate
# single-cell axis collapses to 0 rather than dividing by zero
function _unit_scale(c::AbstractVector{Float64}, lo::Float64, hi::Float64)
    span = hi - lo
    span > 0 || return zeros(length(c))
    return 2 .* (c .- lo) ./ span .- 1
end

"""
    normalized_centers(g::PriorGrid) -> (Xn, Yn, Zn)

Cell-centre coordinates rescaled to `[-1, 1]` along each axis, as three
`[nx, ny, nz]` arrays.

Neural-field inputs have to be scale-free: feeding metres directly makes the
network's effective frequency depend on survey size, so the same architecture
would behave differently on a 5 km and a 500 km grid.
"""
function normalized_centers(g::PriorGrid)
    nx, ny, nz = size(g)
    xn = _unit_scale(g.cx, g.x[1], g.x[end])
    yn = _unit_scale(g.cy, g.y[1], g.y[end])
    zn = _unit_scale(g.cz, g.z[1], g.z[end])

    Xn = Array{Float64,3}(undef, nx, ny, nz)
    Yn = similar(Xn)
    Zn = similar(Xn)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        Xn[i, j, k] = xn[i]
        Yn[i, j, k] = yn[j]
        Zn[i, j, k] = zn[k]
    end
    return Xn, Yn, Zn
end

"""
    containing_cell(g::PriorGrid, x, y, z) -> Int

Linear index of the cell that contains `(x, y, z)`, or `0` if the point lies
outside the grid box (including the far faces, which belong to no cell).

Used to map borehole samples onto the prior mesh: a point on a cell's positive
face falls into the next cell, matching the half-open interval `[edge[i], edge[i+1])`
except that the last cell includes its far edge so a sample sitting exactly on
`grid.x[end]` is not dropped.
"""
function containing_cell(g::PriorGrid, x::Real, y::Real, z::Real)
    nx, ny, nz = size(g)
    (isfinite(x) && isfinite(y) && isfinite(z)) || return 0
    if x < g.x[1] || x > g.x[end] || y < g.y[1] || y > g.y[end] ||
       z < g.z[1] || z > g.z[end]
        return 0
    end

    i = searchsortedlast(g.x, x)
    j = searchsortedlast(g.y, y)
    k = searchsortedlast(g.z, z)
    i = i > nx ? nx : i
    j = j > ny ? ny : j
    k = k > nz ? nz : k
    (i < 1 || j < 1 || k < 1) && return 0
    return LinearIndices((nx, ny, nz))[i, j, k]
end

function Base.show(io::IO, g::PriorGrid)
    nx, ny, nz = size(g)
    @printf(io, "PriorGrid(%d x %d x %d, extent %.1f x %.1f x %.1f km)",
            nx, ny, nz,
            (g.x[end] - g.x[1]) / 1000,
            (g.y[end] - g.y[1]) / 1000,
            (g.z[end] - g.z[1]) / 1000)
end

"""
    cu_log10(cu_ppm; detection_limit=1.0) -> Float64

`log10` copper, with negatives and values below the detection limit lifted to
`detection_limit`.
"""
function cu_log10(cu_ppm::Real; detection_limit::Real = 1.0)
    isfinite(cu_ppm) || return NaN
    v = cu_ppm < 0 ? detection_limit : Float64(cu_ppm)
    return log10(max(v, detection_limit))
end

"""
    aggregate_to_cells(cells, values, weights) -> (cells, values, weights)

Weighted mean of `values` per unique cell index. Index `0` (outside the grid)
is dropped.
"""
function aggregate_to_cells(cells::Vector{Int}, values::Vector{Float64},
                            weights::Vector{Float64})
    acc_v = Dict{Int,Float64}()
    acc_w = Dict{Int,Float64}()
    @inbounds for i in eachindex(cells)
        c = cells[i]
        c == 0 && continue
        v = values[i]
        w = weights[i]
        (isfinite(v) && isfinite(w) && w > 0) || continue
        acc_v[c] = get(acc_v, c, 0.0) + w * v
        acc_w[c] = get(acc_w, c, 0.0) + w
    end
    ks = sort!(collect(keys(acc_w)))
    out_c = Vector{Int}(undef, length(ks))
    out_v = Vector{Float64}(undef, length(ks))
    out_w = Vector{Float64}(undef, length(ks))
    for (i, c) in enumerate(ks)
        out_c[i] = c
        out_w[i] = acc_w[c]
        out_v[i] = acc_v[c] / acc_w[c]
    end
    return out_c, out_v, out_w
end

"""
    map_points_to_cells(g, x, y, z, values; weights=nothing)
        -> (cells, values, weights)

Map scattered samples onto the grid and return the weighted mean per cell.
"""
function map_points_to_cells(g::PriorGrid, x, y, z, values;
                             weights = nothing)
    n = length(x)
    w = weights === nothing ? ones(n) : collect(Float64, weights)
    cells = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        cells[i] = containing_cell(g, x[i], y[i], z[i])
    end
    return aggregate_to_cells(cells, collect(Float64, values), w)
end
