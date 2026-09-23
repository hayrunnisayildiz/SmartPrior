# Minimum-curvature drillhole desurvey.
#
# Azimuth is clockwise from the positive northing axis (compass). Dip is an
# inclination from horizontal. Output is easting, northing, elevation (z up),
# whatever order the source file used for its coordinates. The caller passes
# the collar already in that output frame.

"""
    DesurveyPath

Station coordinates of one hole, plus the collar and the unit direction at
each station. Measured depth 0 is the collar. Construct with [`desurvey`](@ref)
and evaluate other depths with [`positions`](@ref).
"""
struct DesurveyPath
    collar::NTuple{3,Float64}
    depth::Vector{Float64}
    east::Vector{Float64}
    north::Vector{Float64}
    z::Vector{Float64}
    de::Vector{Float64}
    dn::Vector{Float64}
    dz::Vector{Float64}
end

function Base.show(io::IO, path::DesurveyPath)
    print(io, "DesurveyPath($(length(path.depth)) stations, along-hole ",
          path.depth[1], "–", path.depth[end], " m)")
end

function _enz(collar_xyz)
    e, n, z = if collar_xyz isa NTuple{3,<:Real}
        collar_xyz
    elseif collar_xyz isa AbstractVector{<:Real} && length(collar_xyz) == 3
        (collar_xyz[1], collar_xyz[2], collar_xyz[3])
    else
        throw(ArgumentError(
            "desurvey: collar_xyz must be (easting, northing, elevation), " *
            "got $(typeof(collar_xyz))"))
    end
    (isfinite(e) && isfinite(n) && isfinite(z)) || throw(ArgumentError(
        "desurvey: collar_xyz must be finite, got $(collar_xyz)"))
    return (Float64(e), Float64(n), Float64(z))
end

function _degrees(angle::Real, angle_unit::Symbol)
    if angle_unit === :degree
        return Float64(angle)
    elseif angle_unit === :gon
        # 400 gon = 360°.
        return Float64(angle) * 0.9
    else
        throw(ArgumentError(
            "desurvey: angle_unit must be :degree or :gon, got $(repr(angle_unit))"))
    end
end

# Unit direction (east, north, up). `dip_deg` is from horizontal, positive down,
# after the dip_down_negative flip has been applied.
function _direction(azimuth_deg::Float64, dip_down_deg::Float64)
    ar = deg2rad(azimuth_deg)
    α = deg2rad(dip_down_deg)
    horizontal = cos(α)
    return (sin(ar) * horizontal, cos(ar) * horizontal, -sin(α))
end

# Minimum-curvature step. The ratio factor is tan(β/2) / (β/2), where β is
# the angle between the two unit directions. β = 0 is a straight segment
# (factor 1); the formula is not evaluated, so identical directions do not
# produce a NaN. A dogleg of 180° is rejected: the factor diverges.
function _displace(u1::NTuple{3,Float64}, u2::NTuple{3,Float64}, distance::Float64)
    distance == 0 && return (0.0, 0.0, 0.0)
    cosβ = clamp(u1[1] * u2[1] + u1[2] * u2[2] + u1[3] * u2[3], -1.0, 1.0)
    half = acos(cosβ) / 2
    if half == 0
        return (distance * u1[1], distance * u1[2], distance * u1[3])
    end
    if half >= π / 2 - 1e-8
        throw(ArgumentError(
            "desurvey: dogleg is $(rad2deg(2 * half))°, too close to 180° " *
            "for minimum curvature"))
    end
    scale = 0.5 * distance * tan(half) / half
    return (
        scale * (u1[1] + u2[1]),
        scale * (u1[2] + u2[2]),
        scale * (u1[3] + u2[3]),
    )
end

function _dir_at(u1::NTuple{3,Float64}, u2::NTuple{3,Float64}, β::Float64, t::Float64)
    β == 0 && return u1
    φ = t * β
    sβ = sin(β)
    w1 = sin(β - φ) / sβ
    w2 = sin(φ) / sβ
    return (
        w1 * u1[1] + w2 * u2[1],
        w1 * u1[2] + w2 * u2[2],
        w1 * u1[3] + w2 * u2[3],
    )
end

function _dir(path::DesurveyPath, i::Int)
    return (path.de[i], path.dn[i], path.dz[i])
end

function _station(path::DesurveyPath, i::Int)
    return (path.east[i], path.north[i], path.z[i])
end

function _add(p::NTuple{3,Float64}, d::NTuple{3,Float64})
    return (p[1] + d[1], p[2] + d[2], p[3] + d[3])
end

"""
    desurvey(collar_xyz, survey_depths, azimuths, dips;
             angle_unit=:degree, dip_down_negative=false) -> DesurveyPath

Minimum-curvature trajectory of one hole.

`collar_xyz` is `(easting, northing, elevation)` in metres, elevation up.
It is the point at along-hole depth 0. `survey_depths` are along-hole lengths
from that collar, in metres, and must be non-decreasing. `azimuths` are
clockwise from the positive northing axis. `dips` are inclinations from
horizontal, in `angle_unit` (`:degree` or `:gon`, with 400 gon = 360°).

If `dip_down_negative` is false, a positive dip points downward and +90° is
vertical down. If true, a negative dip points downward and −90° is vertical down.

Between two stations the hole is a circular arc, the unique curve of constant
curvature that meets both directions. Where the directions are the same the
arc is a straight line. Two stations at the same depth must share a direction;
the step length is then zero. A hole with one station is a straight line along
that station's direction.

From the collar to the first station the first direction is held, so a survey
that does not start at depth 0 is still tied to the collar. Past the last
station the last direction is held.
"""
function desurvey(collar_xyz, survey_depths::AbstractVector{<:Real},
                  azimuths::AbstractVector{<:Real}, dips::AbstractVector{<:Real};
                  angle_unit::Symbol = :degree, dip_down_negative::Bool = false)
    collar = _enz(collar_xyz)
    n = length(survey_depths)
    n > 0 || throw(ArgumentError("desurvey: survey is empty"))
    (length(azimuths) == n && length(dips) == n) || throw(ArgumentError(
        "desurvey: survey_depths has length $n, azimuths $(length(azimuths)), " *
        "dips $(length(dips)); the three must match"))
    # Validate the unit before converting, so an unknown unit is not reported
    # as a non-finite angle.
    angle_unit === :degree || angle_unit === :gon || throw(ArgumentError(
        "desurvey: angle_unit must be :degree or :gon, got $(repr(angle_unit))"))

    depth = Vector{Float64}(undef, n)
    de = Vector{Float64}(undef, n)
    dn = Vector{Float64}(undef, n)
    dz = Vector{Float64}(undef, n)
    dirs = Vector{NTuple{3,Float64}}(undef, n)
    for i in 1:n
        md = Float64(survey_depths[i])
        az = Float64(azimuths[i])
        dip = Float64(dips[i])
        (isfinite(md) && isfinite(az) && isfinite(dip)) || throw(ArgumentError(
            "desurvey: station $i has a non-finite depth, azimuth, or dip"))
        md >= 0 || throw(ArgumentError(
            "desurvey: station $i depth must be ≥ 0, got $md"))
        if i > 1 && md < depth[i - 1]
            throw(ArgumentError(
                "desurvey: survey depth decreased at station $i " *
                "($md m after $(depth[i - 1]) m)"))
        end
        depth[i] = md
        dip_down = dip_down_negative ? -_degrees(dip, angle_unit) : _degrees(dip, angle_unit)
        dirs[i] = _direction(_degrees(az, angle_unit), dip_down)
        de[i], dn[i], dz[i] = dirs[i]
    end
    for i in 2:n
        depth[i] == depth[i - 1] || continue
        # A positive dogleg at zero length is not a curve we can place.
        # Azimuths of 0° and 360° are the same direction up to rounding.
        cosβ = clamp(dirs[i - 1][1] * dirs[i][1] + dirs[i - 1][2] * dirs[i][2] +
                     dirs[i - 1][3] * dirs[i][3], -1.0, 1.0)
        if acos(cosβ) > 1e-8
            throw(ArgumentError(
                "desurvey: stations $(i - 1) and $i are at the same depth " *
                "($(depth[i]) m) but their directions differ"))
        end
    end

    east = Vector{Float64}(undef, n)
    north = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    east[1] = collar[1] + depth[1] * de[1]
    north[1] = collar[2] + depth[1] * dn[1]
    z[1] = collar[3] + depth[1] * dz[1]
    for i in 2:n
        step = _displace(dirs[i - 1], dirs[i], depth[i] - depth[i - 1])
        all(isfinite, step) || throw(ArgumentError(
            "desurvey: station $i displacement is not finite"))
        east[i] = east[i - 1] + step[1]
        north[i] = north[i - 1] + step[2]
        z[i] = z[i - 1] + step[3]
    end
    return DesurveyPath(collar, depth, east, north, z, de, dn, dz)
end

"""
    positions(path, depth) -> (easting, northing, elevation)
    positions(path, depths) -> (eastings, northings, elevations)

Coordinates at along-hole `depth` (metres from the collar), or at each entry
of `depths`. Use this for interval midpoints.

Before the first survey station the first direction is held, as a straight
line from the collar. Between stations the position lies on the
minimum-curvature arc. Past the last station the last direction is held.
A negative depth is rejected.
"""
function positions(path::DesurveyPath, depth::Real)
    md = Float64(depth)
    isfinite(md) || throw(ArgumentError(
        "positions: along-hole depth must be finite, got $depth"))
    md >= 0 || throw(ArgumentError(
        "positions: along-hole depth must be ≥ 0, got $md"))
    d = path.depth
    if md < d[1]
        u = _dir(path, 1)
        return (
            path.collar[1] + md * u[1],
            path.collar[2] + md * u[2],
            path.collar[3] + md * u[3],
        )
    end
    if md >= d[end]
        extra = md - d[end]
        u = _dir(path, length(d))
        p = _station(path, length(d))
        return (p[1] + extra * u[1], p[2] + extra * u[2], p[3] + extra * u[3])
    end
    i = searchsortedlast(d, md)
    span = d[i + 1] - d[i]
    if span == 0
        return _station(path, i)
    end
    u1 = _dir(path, i)
    u2 = _dir(path, i + 1)
    travelled = md - d[i]
    cosβ = clamp(u1[1] * u2[1] + u1[2] * u2[2] + u1[3] * u2[3], -1.0, 1.0)
    β = acos(cosβ)
    if β == 0
        return _add(_station(path, i), (travelled * u1[1], travelled * u1[2], travelled * u1[3]))
    end
    um = _dir_at(u1, u2, β, travelled / span)
    return _add(_station(path, i), _displace(u1, um, travelled))
end

function positions(path::DesurveyPath, depths::AbstractVector{<:Real})
    n = length(depths)
    east = Vector{Float64}(undef, n)
    north = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    for i in 1:n
        east[i], north[i], z[i] = positions(path, depths[i])
    end
    return east, north, z
end
