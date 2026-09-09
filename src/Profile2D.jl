# Bridge to MTGeophysics' 2D profile representation.
#
# The 2D engine is the only end-to-end validation available without ModEM: it is
# pure Julia, it solves the induction equation on a finite-volume mesh, and its
# VFSA driver has the same structure as the 3D one. Running the whole pipeline
# through it -- truth, forward, prior, inversion -- tests the idea rather than the
# code, because the 2D solver's physics is not the physics any part of this
# package was trained on.
#
# Three conventions differ between the two representations and each is a silent
# way to invalidate a comparison:
#
#   Axis order. MTGeophysics stores 2D resistivity as [nz, ny], depth first.
#   A PriorGrid is [nx, ny, nz]. Getting this backwards produces a model that
#   loads without complaint and is transposed.
#
#   Air. The 2D mesh puts air layers at the top of its own z grid, at 1e9 ohm-m
#   and with negative node coordinates. A PriorGrid covers the ground only, so
#   the two are offset by `n_air_cells` and the offset has to be applied on every
#   conversion.
#
#   Units. The 2D model file stores natural log resistivity; its in-memory arrays
#   are linear ohm-m; this package works in log10 throughout. Only the linear
#   in-memory form is touched here, and the conversion to log10 is explicit.
#
# A 2D profile has no third dimension, which matters for gravity: a body in a 2D
# section is implicitly infinite along strike. `grid_from_mt2dmesh` gives the
# grid a single very wide cell in x to represent that, so the prism operator
# reproduces the 2D gravity field of an elongated body rather than the much
# weaker field of a compact one.

using MTGeophysics: MT2DMesh

"""
    grid_from_mt2dmesh(mesh; strike_extent=1.0e5) -> PriorGrid

A `PriorGrid` covering the ground cells of a 2D mesh.

The profile direction becomes the grid's y axis and depth its z axis, matching the
mesh cell for cell. The x axis gets one cell of width `strike_extent` standing in
for the infinite strike length a 2D section assumes; the default 100 km is far
larger than any survey-scale anomaly, which is what makes the prism gravity
operator return the 2D field.

Air layers are excluded. Use [`to_mt2d`](@ref) to put a field back on the mesh's
own grid, which re-adds them.
"""
function grid_from_mt2dmesh(mesh::MT2DMesh; strike_extent::Real = 1.0e5)
    strike_extent > 0 ||
        throw(ArgumentError("grid_from_mt2dmesh: strike_extent must be positive"))
    na = mesh.n_air_cells
    na < length(mesh.z_cell_sizes) || throw(ArgumentError(
        "grid_from_mt2dmesh: the mesh is all air ($na of $(length(mesh.z_cell_sizes)) layers)"))

    dz = mesh.z_cell_sizes[(na+1):end]
    z_top = mesh.z_nodes[na+1]

    return PriorGrid([Float64(strike_extent)], mesh.y_cell_sizes, dz;
                     origin = [-strike_extent / 2, mesh.y_nodes[1], z_top])
end

"""
    to_mt2d(log_rho, mesh; air_resistivity=1.0e9) -> Matrix{Float64}

Convert a `[1, ny, nz]` log10 resistivity field to the mesh's `[nz_full, ny]`
linear resistivity array, re-adding the air layers.

Non-finite cells become `air_resistivity`, so a prior carrying `NaN` topography
lands on the value the 2D solver treats as air.
"""
function to_mt2d(log_rho::AbstractArray{<:Real,3}, mesh::MT2DMesh;
                 air_resistivity::Real = 1.0e9)
    na = mesh.n_air_cells
    ny = length(mesh.y_cell_sizes)
    nz_full = length(mesh.z_cell_sizes)
    nz = nz_full - na

    size(log_rho) == (1, ny, nz) || throw(DimensionMismatch(
        "to_mt2d: expected a (1, $ny, $nz) field for this mesh, got $(size(log_rho))"))

    out = fill(Float64(air_resistivity), nz_full, ny)
    @inbounds for k in 1:nz, j in 1:ny
        v = log_rho[1, j, k]
        out[na+k, j] = isfinite(v) ? 10.0^v : Float64(air_resistivity)
    end
    return out
end

to_mt2d(m::SyntheticModel, mesh::MT2DMesh; kwargs...) =
    to_mt2d(m.log_rho, mesh; kwargs...)

"""
    from_mt2d(resistivity, mesh) -> Array{Float64,3}

Inverse of [`to_mt2d`](@ref): take the mesh's `[nz_full, ny]` linear resistivity
array to a `[1, ny, nz]` log10 field over the ground cells only.

This is how an inversion result is brought back for scoring against a truth or a
prior on the same `PriorGrid`.
"""
function from_mt2d(resistivity::AbstractMatrix{<:Real}, mesh::MT2DMesh)
    na = mesh.n_air_cells
    ny = length(mesh.y_cell_sizes)
    nz_full = length(mesh.z_cell_sizes)
    nz = nz_full - na

    size(resistivity) == (nz_full, ny) || throw(DimensionMismatch(
        "from_mt2d: expected a ($nz_full, $ny) array for this mesh, got $(size(resistivity))"))

    out = Array{Float64,3}(undef, 1, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny
        r = resistivity[na+k, j]
        out[1, j, k] = r > 0 ? log10(r) : NaN
    end
    return out
end

"""
    profile_sites(mesh, periods_or_response) -> MTSites

An [`MTSites`](@ref) at the mesh's receiver positions, for the feature pipeline.

Pass an `MT2DResponse` or a loaded 2D data file to carry apparent resistivity
and phase across, or a vector of periods to get a placeholder with `NaN` data --
useful when only the site geometry matters, as it does for the coverage features.
When the input carries impedance errors (`z_xy_error`, as in a loaded
`DataFile2D`), `err_rho_a` and `err_phase` are filled by propagating those
uncertainties so [`mt_column_misfit`](@ref) can score χ²/datum.

The TE mode (`rho_xy`) is used. On a 2D profile the TE apparent resistivity is
the one whose Niblett-Bostick transform behaves like a sounding under the site;
TM is dominated by the galvanic response of lateral contrasts and transforms
poorly.
"""
function profile_sites(mesh::MT2DMesh, response)
    x = zeros(length(mesh.receiver_positions))
    y = collect(Float64, mesh.receiver_positions)
    periods = collect(Float64, response.periods)
    rho_a = Matrix{Float64}(response.rho_xy)
    phase = hasproperty(response, :phase_xy) ?
        Matrix{Float64}(response.phase_xy) :
        throw(ArgumentError("profile_sites: response must provide phase_xy"))

    err_rho = nothing
    err_phase = nothing
    if hasproperty(response, :err_rho_a) && response.err_rho_a !== nothing
        err_rho = Matrix{Float64}(response.err_rho_a)
        err_phase = response.err_phase === nothing ? nothing : Matrix{Float64}(response.err_phase)
    elseif hasproperty(response, :z_xy) && hasproperty(response, :z_xy_error) &&
           hasproperty(response, :frequencies)
        err_rho, err_phase = mt_apparent_errors_from_impedance(
            rho_a, response.z_xy, response.z_xy_error, response.frequencies)
    end

    return MTSites(x, y, periods, rho_a, phase; err_rho_a = err_rho, err_phase = err_phase)
end

function profile_sites(mesh::MT2DMesh, periods::AbstractVector{<:Real})
    ns = length(mesh.receiver_positions)
    np = length(periods)
    return MTSites(zeros(ns), collect(Float64, mesh.receiver_positions),
                   collect(Float64, periods),
                   fill(NaN, np, ns), fill(NaN, np, ns))
end
