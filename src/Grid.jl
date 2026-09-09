# Tensor-product grid shared by the prior model and the MT inversion mesh.
# Conventions follow the WS3D model format used by MTGeophysics.jl: arrays are
# indexed [i, j, k] with x north, y east, z down, all lengths in metres, and the
# origin placed at the (x, y, z) corner of the first cell.

"""
    PriorGrid(dx, dy, dz; origin=[0.0, 0.0, 0.0])
    PriorGrid(m::WS3DModel)

Tensor-product 3-D grid on which a prior model lives.

`dx`, `dy`, `dz` are cell widths in metres along north, east and down. Cell edges
and centres are derived once at construction. Building from a [`WS3DModel`](@ref)
guarantees the prior lands on exactly the mesh the inversion will use, which is
the whole point of the type: a prior on a different mesh is unusable.
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

PriorGrid(m::WS3DModel) = PriorGrid(m.dx, m.dy, m.dz; origin = m.origin)

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

function Base.show(io::IO, g::PriorGrid)
    nx, ny, nz = size(g)
    @printf(io, "PriorGrid(%d x %d x %d, extent %.1f x %.1f x %.1f km)",
            nx, ny, nz,
            (g.x[end] - g.x[1]) / 1000,
            (g.y[end] - g.y[1]) / 1000,
            (g.z[end] - g.z[1]) / 1000)
end
