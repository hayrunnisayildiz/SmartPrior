# Closed-form gravity forward operator for rectangular prisms.
#
# Gravity is linear in density, so the whole forward problem collapses to a
# sensitivity matrix built once per (grid, station layout) pair. That matters for
# training: the physics term costs one matrix-vector product per step, and its
# gradient with respect to density is the transpose, so it is differentiable by
# construction with no AD machinery involved.

const G_GRAV = 6.67430e-11      # m^3 kg^-1 s^-2 (CODATA 2018)
const M_S2_TO_MGAL = 1e5        # 1 m/s^2 = 1e5 mGal

# floor for the log arguments, which vanish when a station sits exactly on the
# extension of a prism face. Applied as a max rather than an offset so that
# non-degenerate geometries are left bit-for-bit alone and the operator stays
# exactly symmetric.
const _GEOM_EPS = 1e-9

"""
    prism_gz(x1, x2, y1, y2, z1, z2) -> Float64

Vertical gravity attraction at the coordinate origin of a homogeneous
rectangular prism of unit density, in m/s^2 per kg/m^3.

Corner coordinates are given *relative to the observation point*, in metres,
with `z` positive down; a prism with `z1 > 0` therefore lies below the station.
The result is positive for a positive density contrast below the station.

Uses the closed-form solution of Plouff (1976); see also Blakely (1996), ch. 4.
"""
function prism_gz(x1::Real, x2::Real, y1::Real, y2::Real, z1::Real, z2::Real)
    xs = (float(x1), float(x2))
    ys = (float(y1), float(y2))
    zs = (float(z1), float(z2))

    s = 0.0
    @inbounds for i in 1:2, j in 1:2, k in 1:2
        x = xs[i]
        y = ys[j]
        z = zs[k]
        r = sqrt(x * x + y * y + z * z)
        sgn = iseven(i + j + k) ? 1.0 : -1.0
        # r >= |y| and r >= |x|, so both log arguments are non-negative and only
        # reach zero in the degenerate case the floor covers. The atan needs no
        # guard: two-argument atan is defined at zero, and where it would be
        # ambiguous the z factor in front is itself zero.
        s += sgn * (z * atan(x * y, z * r)
                    - x * log(max(y + r, _GEOM_EPS))
                    - y * log(max(x + r, _GEOM_EPS)))
    end
    return G_GRAV * s
end

"""
    gravity_matrix(g::PriorGrid, sx, sy, sz; units=:mgal) -> Matrix{Float64}

Sensitivity matrix `A` of vertical gravity with respect to cell density.

`sx`, `sy`, `sz` are station coordinates in the grid's own frame (metres, `z`
positive down, so a station above the grid top has `sz < g.z[1]`). The returned
matrix has one row per station and one column per cell, ordered to match
`vec(density)` for a `[nx, ny, nz]` density array. With `units = :mgal` the
product `A * vec(rho)` is in mGal for densities in kg/m^3.

Cost is `O(nstations * ncells)`; for a 40 x 40 x 45 grid and a few hundred
stations this is a few hundred MB, so build it once and reuse it.
"""
function gravity_matrix(g::PriorGrid,
                        sx::AbstractVector{<:Real},
                        sy::AbstractVector{<:Real},
                        sz::AbstractVector{<:Real};
                        units::Symbol = :mgal)
    ns = length(sx)
    (length(sy) == ns && length(sz) == ns) ||
        throw(ArgumentError("gravity_matrix: station coordinate vectors must have equal length"))
    units in (:mgal, :si) ||
        throw(ArgumentError("gravity_matrix: units must be :mgal or :si, got $units"))

    scale = units === :mgal ? M_S2_TO_MGAL : 1.0
    nx, ny, nz = size(g)
    A = Array{Float64}(undef, ns, nx * ny * nz)

    @inbounds for s in 1:ns
        ox = float(sx[s])
        oy = float(sy[s])
        oz = float(sz[s])
        col = 1
        for k in 1:nz
            z1 = g.z[k]     - oz
            z2 = g.z[k+1]   - oz
            for j in 1:ny
                y1 = g.y[j]   - oy
                y2 = g.y[j+1] - oy
                for i in 1:nx
                    x1 = g.x[i]   - ox
                    x2 = g.x[i+1] - ox
                    A[s, col] = scale * prism_gz(x1, x2, y1, y2, z1, z2)
                    col += 1
                end
            end
        end
    end
    return A
end

"""
    forward_gravity(A, density) -> Vector

Predicted gravity at the stations `A` was built for.

`density` is a `[nx, ny, nz]` array (or the matching flat vector) of density
*contrast* in kg/m^3 relative to the background the observed anomaly was reduced
to. Kept deliberately trivial so it stays a plain matrix-vector product and
differentiates without any AD rules.
"""
function forward_gravity(A::AbstractMatrix, density::AbstractArray)
    d = vec(density)
    size(A, 2) == length(d) ||
        throw(DimensionMismatch("forward_gravity: A has $(size(A, 2)) columns but density has $(length(d)) cells"))
    return A * d
end
