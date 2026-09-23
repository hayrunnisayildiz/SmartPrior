# Pointwise covariates.
#
# The same channels `build_features`, `structure_distance_channels`, and
# `surface_geology_channels` build on cell centres, evaluated at whatever
# `(x, y, z)` columns the caller passes. The grid builders are left in place
# so the current training path does not move.
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

"""
    CoordinateCovariate(x_min, x_max, y_min, y_max, z_min, z_max)

Normalised coordinates `x_norm`, `y_norm`, `z_norm` on `[-1, 1]`, using the
same edge mapping as [`normalized_centers`](@ref). Pass the grid edges, not
the min/max of the query points: cell centres sit half a cell inside the box.
"""
struct CoordinateCovariate <: Covariate
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    z_min::Float64
    z_max::Float64
end

function CoordinateCovariate(g::PriorGrid)
    return CoordinateCovariate(g.x[1], g.x[end], g.y[1], g.y[end], g.z[1], g.z[end])
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
    DepthCovariate(z_datum, d0, rho_ref, period_max)

`log_depth` and `depth_over_skin`, matching [`depth_channels`](@ref).

`z_datum` is the grid's first z edge (`g.z[1]` in [`depth_below_top`](@ref)),
and `d0` is half the first cell thickness (`g.dz[1] / 2`). `log_depth` is
standardised over the points passed to [`evaluate`](@ref), so it matches the
grid channel only when those points are the cell centres. `depth_over_skin`
uses the skin depth of a `rho_ref` half-space at `period_max` as a geometric
scale; it is not an MT input.
"""
struct DepthCovariate <: Covariate
    z_datum::Float64
    d0::Float64
    rho_ref::Float64
    period_max::Float64

    function DepthCovariate(z_datum::Real, d0::Real, rho_ref::Real, period_max::Real)
        d0 > 0 || throw(ArgumentError("DepthCovariate: d0 must be positive"))
        rho_ref > 0 || throw(ArgumentError("DepthCovariate: rho_ref must be positive"))
        period_max > 0 || throw(ArgumentError(
            "DepthCovariate: period_max must be positive"))
        return new(Float64(z_datum), Float64(d0), Float64(rho_ref), Float64(period_max))
    end
end

function DepthCovariate(g::PriorGrid; rho_ref::Real = 100.0, period_max::Real = 1000.0)
    return DepthCovariate(g.z[1], g.dz[1] / 2, rho_ref, period_max)
end

channel_names(::DepthCovariate) = ["log_depth", "depth_over_skin"]

function evaluate(c::DepthCovariate, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    out = Matrix{Float64}(undef, 2, n)
    n == 0 && return out
    # Same expression as depth_channels: height above the first z edge, floored
    # at d0 before the log, then standardised across these points.
    d = vec(pts[3, :]) .- c.z_datum
    out[1, :] .= standardize(log10.(max.(d, c.d0) ./ c.d0))
    δ = _skin_depth(c.rho_ref, c.period_max)
    out[2, :] .= d ./ δ
    return out
end

const _REJECTED_COVARIATES = (
    "sample_distance", "geochemistry", "lithology", "pxrf",
)

function _structure_lines(source)
    if source isa AbstractString
        path = isdir(source) ? cloncurry_structures_path(source) : String(source)
        return load_cloncurry_structures(path)
    end
    return source
end

"""
    StructureDistance(source, length_scale)

Plan-view `struct_fault_distance` and `struct_line_distance`: `log1p(d / length_scale)`
in metres, matching [`structure_distance_channels`](@ref). `z` is ignored, as
in the extruded grid channels. `length_scale` is the grid's `h_median` when
the comparison is against a [`PriorGrid`](@ref).

`source` is a dataset root, a GeoJSON path, or lines from
[`load_cloncurry_structures`](@ref).
"""
struct StructureDistance <: Covariate
    lines::Vector{NamedTuple{(:type, :x, :y),Tuple{String,Vector{Float64},Vector{Float64}}}}
    length_scale::Float64

    function StructureDistance(source, length_scale::Real)
        length_scale > 0 || throw(ArgumentError(
            "StructureDistance: length_scale must be positive"))
        lines = _structure_lines(source)
        isempty(lines) && throw(ArgumentError("StructureDistance: no structural lines"))
        any(l -> l.type == "Fault", lines) || throw(ArgumentError(
            "StructureDistance: no Fault lines in the collection"))
        return new(lines, Float64(length_scale))
    end
end

StructureDistance(g::PriorGrid, source) = StructureDistance(source, g.h_median)

channel_names(::StructureDistance) = ["struct_fault_distance", "struct_line_distance"]

function _line_distance_channel(px::Float64, py::Float64, lines, mask, h::Float64)
    d2 = _min_line_dist2(px, py, lines, mask)
    return isfinite(d2) ? log1p(sqrt(d2) / h) : 0.0
end

function evaluate(c::StructureDistance, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    out = Matrix{Float64}(undef, 2, n)
    h = c.length_scale
    @inbounds for t in 1:n
        px = pts[1, t]
        py = pts[2, t]
        out[1, t] = _line_distance_channel(px, py, c.lines, t -> t == "Fault", h)
        out[2, t] = _line_distance_channel(px, py, c.lines, nothing, h)
    end
    return out
end

function _surface_polygons(source)
    if source isa AbstractString
        path = isdir(source) ? cloncurry_surface_geology_path(source) : String(source)
        return load_cloncurry_surface_geology(path)
    end
    return source
end

function _surface_classes(polys)
    dom_set = Set{String}()
    rock_set = Set{String}()
    for p in polys
        isempty(p.dom_rock) || push!(dom_set, _lith_token(p.dom_rock))
        isempty(p.rock_type) || push!(rock_set, _lith_token(p.rock_type))
    end
    dom_classes = sort!(collect(dom_set))
    rock_classes = sort!(collect(rock_set))
    push!(dom_classes, "OTHER")
    push!(rock_classes, "OTHER")
    return dom_classes, rock_classes
end

"""
    SurfaceGeology(source)

Point-in-polygon one-hot channels `surf_dom_*` and `surf_rock_*`, matching
[`surface_geology_channels`](@ref). The first containing polygon wins. Points
outside every polygon get the `*_OTHER` bit. `z` is ignored.

`source` is a dataset root, a GeoJSON path, or polygons from
[`load_cloncurry_surface_geology`](@ref).
"""
struct SurfaceGeology <: Covariate
    polys::Vector{NamedTuple{(:dom_rock, :rock_type, :rings),
        Tuple{String,String,Vector{Tuple{Vector{Float64},Vector{Float64}}}}}}
    dom_classes::Vector{String}
    rock_classes::Vector{String}
end

function SurfaceGeology(source)
    polys = _surface_polygons(source)
    isempty(polys) && throw(ArgumentError("SurfaceGeology: no surface-geology polygons"))
    dom_classes, rock_classes = _surface_classes(polys)
    return SurfaceGeology(polys, dom_classes, rock_classes)
end

function channel_names(c::SurfaceGeology)
    names = String[]
    for lab in c.dom_classes
        push!(names, "surf_dom_" * lab)
    end
    for lab in c.rock_classes
        push!(names, "surf_rock_" * lab)
    end
    return names
end

function evaluate(c::SurfaceGeology, xyz::AbstractMatrix{<:Real})
    pts = _xyz64(xyz)
    n = size(pts, 2)
    nd = length(c.dom_classes)
    nr = length(c.rock_classes)
    out = zeros(nd + nr, n)
    n == 0 && return out
    dom_index = Dict(lab => i for (i, lab) in enumerate(c.dom_classes))
    rock_index = Dict(lab => i for (i, lab) in enumerate(c.rock_classes))
    @inbounds for t in 1:n
        px = pts[1, t]
        py = pts[2, t]
        hit = 0
        for (k, p) in enumerate(c.polys)
            _point_in_polygon(px, py, p.rings) || continue
            hit = k
            break
        end
        if hit == 0
            out[dom_index["OTHER"], t] = 1.0
            out[nd + rock_index["OTHER"], t] = 1.0
        else
            p = c.polys[hit]
            dtok = isempty(p.dom_rock) ? "OTHER" : _lith_token(p.dom_rock)
            rtok = isempty(p.rock_type) ? "OTHER" : _lith_token(p.rock_type)
            haskey(dom_index, dtok) || (dtok = "OTHER")
            haskey(rock_index, rtok) || (rtok = "OTHER")
            out[dom_index[dtok], t] = 1.0
            out[nd + rock_index[rtok], t] = 1.0
        end
    end
    return out
end

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
