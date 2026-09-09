# Synthetic ground truth, and observations derived from it.
#
# The point of a synthetic test here is not to show that the prior can be fitted
# -- it can, it is a neural field with enough capacity to fit anything -- but to
# check that a prior built from gravity and topography alone lands near a truth
# it was never shown, and that its sigma is honest about where it did not.
#
# Two circularity traps make that easy to get wrong, and both are avoided by
# construction here:
#
#   Resistivity and density are specified independently per body. If the truth
#   were built by applying the same linear density-resistivity law that
#   Losses.jl learns, the gravity term would be solving a problem it was handed
#   the answer to. Real rocks correlate the two loosely and with exceptions, so
#   the generators let you set a density contrast that does not follow from the
#   resistivity, and the example uses one that deliberately does not.
#
#   The MT observations from `synth_mt_sites` come from the same 1D column
#   recursion that `mt_column_misfit` uses, so any run that feeds them back as
#   an MT target is testing self-consistency, not physics. They exist to stand
#   in for a Niblett-Bostick baseline and to exercise the pipeline. The real
#   check is examples/compare_prior_2d.jl, which generates data with
#   MTGeophysics' own 2D finite-difference solver -- a different discretisation,
#   different physics, and 2D induction the 1D operator cannot represent.

"""
    SyntheticModel(grid, log_rho, density)

A ground-truth model: log10 resistivity and density contrast on the same grid.

`density` is a contrast in kg/m^3 against the background, matching what
[`forward_gravity`](@ref) expects, so a homogeneous model produces no anomaly.
"""
struct SyntheticModel
    grid::PriorGrid
    log_rho::Array{Float64,3}
    density::Array{Float64,3}

    function SyntheticModel(grid::PriorGrid,
                            log_rho::AbstractArray{<:Real,3},
                            density::AbstractArray{<:Real,3})
        dims = size(grid)
        size(log_rho) == dims || throw(DimensionMismatch(
            "SyntheticModel: log_rho is $(size(log_rho)) but the grid is $(dims)"))
        size(density) == dims || throw(DimensionMismatch(
            "SyntheticModel: density is $(size(density)) but the grid is $(dims)"))
        return new(grid, Array{Float64,3}(log_rho), Array{Float64,3}(density))
    end
end

Base.size(m::SyntheticModel) = size(m.log_rho)

function Base.show(io::IO, m::SyntheticModel)
    @printf(io, "SyntheticModel(%dx%dx%d, log10 rho %.2f-%.2f, density %.0f-%.0f kg/m3)",
            size(m)..., minimum(m.log_rho), maximum(m.log_rho),
            minimum(m.density), maximum(m.density))
end

"""
    truth_halfspace(grid; log_rho=2.0) -> SyntheticModel

Uniform half-space with no density contrast. The starting point for the builders.
"""
truth_halfspace(grid::PriorGrid; log_rho::Real = 2.0) =
    SyntheticModel(grid, fill(Float64(log_rho), size(grid)), zeros(size(grid)))

"""
    add_layer(m; ztop, zbot, log_rho, density=0.0) -> SyntheticModel

Overwrite every cell whose centre depth falls in `[ztop, zbot)`.

Depths are measured in the grid's own vertical coordinate, so they include
`origin[3]`; see [`depth_below_top`](@ref) if you want depth below the datum.
"""
function add_layer(m::SyntheticModel;
                   ztop::Real, zbot::Real,
                   log_rho::Real, density::Real = 0.0)
    ztop < zbot || throw(ArgumentError("add_layer: need ztop < zbot, got $ztop and $zbot"))
    lr = copy(m.log_rho)
    de = copy(m.density)
    cz = m.grid.cz
    @inbounds for k in eachindex(cz)
        (ztop <= cz[k] < zbot) || continue
        lr[:, :, k] .= log_rho
        de[:, :, k] .= density
    end
    return SyntheticModel(m.grid, lr, de)
end

"""
    add_block(m; x, y, z, log_rho, density=0.0) -> SyntheticModel

Overwrite a rectangular body. Each of `x`, `y`, `z` is a `(min, max)` pair of
coordinates; cells whose centre falls inside all three are set.

Resistivity and density are independent on purpose: a body may be conductive and
dense (a sulphide), conductive and light (a brine-filled sediment), or resistive
and dense (an intrusion). A prior that only works when the two agree has learned
nothing worth having.
"""
function add_block(m::SyntheticModel;
                   x::Tuple{Real,Real}, y::Tuple{Real,Real}, z::Tuple{Real,Real},
                   log_rho::Real, density::Real = 0.0)
    for (name, r) in (("x", x), ("y", y), ("z", z))
        r[1] < r[2] || throw(ArgumentError("add_block: $name range must be increasing, got $r"))
    end
    g = m.grid
    lr = copy(m.log_rho)
    de = copy(m.density)
    @inbounds for k in eachindex(g.cz), j in eachindex(g.cy), i in eachindex(g.cx)
        (x[1] <= g.cx[i] <= x[2]) || continue
        (y[1] <= g.cy[j] <= y[2]) || continue
        (z[1] <= g.cz[k] <= z[2]) || continue
        lr[i, j, k] = log_rho
        de[i, j, k] = density
    end
    return SyntheticModel(g, lr, de)
end

"""
    add_dipping_slab(m; x0, z0, dip_deg, thickness, log_rho, density=0.0,
                     strike=:y) -> SyntheticModel

A planar slab dipping in the x-z plane, extended along `strike`.

Dipping structure is the case a half-space start model handles worst: the
inversion tends to break it into a staircase of blobs, because no single depth is
right across the profile. It is the most informative synthetic target for the
question this package is about.

`dip_deg` is measured from horizontal; `x0`, `z0` fix a point on the slab's
mid-plane. `thickness` is the perpendicular thickness.

`strike` names the axis the slab is *invariant* along, so `:y` dips across x and
`:x` dips across y. On a 2D profile grid from [`grid_from_mt2dmesh`](@ref), where
x holds a single wide strike cell and the profile runs along y, the value you want
is `:x`; `:y` there would make the slab uniform across the one x cell and produce
a flat layer instead, a 1D model that looks nothing like the intended target.
That mistake is silent in the resulting array, so it is rejected here.
"""
function add_dipping_slab(m::SyntheticModel;
                          x0::Real, z0::Real, dip_deg::Real, thickness::Real,
                          log_rho::Real, density::Real = 0.0,
                          strike::Symbol = :y)
    strike in (:x, :y) || throw(ArgumentError("add_dipping_slab: strike must be :x or :y"))
    thickness > 0 || throw(ArgumentError("add_dipping_slab: thickness must be positive"))
    0 < dip_deg < 180 || throw(ArgumentError(
        "add_dipping_slab: dip_deg must be in (0, 180), got $dip_deg"))

    g = m.grid
    ndip = strike === :y ? size(g, 1) : size(g, 2)
    ndip > 1 || throw(ArgumentError(
        "add_dipping_slab: strike = :$(strike) dips across the " *
        "$(strike === :y ? "x" : "y") axis, which has only one cell; the result " *
        "would be a flat layer. Use strike = :$(strike === :y ? "x" : "y")."))
    lr = copy(m.log_rho)
    de = copy(m.density)

    # unit normal to a plane dipping by theta in the along-profile / depth plane
    theta = deg2rad(dip_deg)
    nh, nv = -sin(theta), cos(theta)
    half = thickness / 2

    @inbounds for k in eachindex(g.cz), j in eachindex(g.cy), i in eachindex(g.cx)
        h = strike === :y ? g.cx[i] : g.cy[j]
        dist = nh * (h - x0) + nv * (g.cz[k] - z0)
        if abs(dist) <= half
            lr[i, j, k] = log_rho
            de[i, j, k] = density
        end
    end
    return SyntheticModel(g, lr, de)
end

"""
    add_topography(m; elevation, air_log_rho=NaN) -> SyntheticModel

Mark cells above a surface as air.

`elevation` is an `[nx, ny]` array in the grid's vertical coordinate, positive
down like the rest of the grid, so a value of `-200` means ground 200 m above the
datum. Cells whose centre lies above it get `air_log_rho`, which defaults to
`NaN` -- the value MTGeophysics uses for air and the one the export path carries
through a WS3D file.
"""
function add_topography(m::SyntheticModel;
                        elevation::AbstractMatrix{<:Real},
                        air_log_rho::Real = NaN)
    g = m.grid
    nx, ny, _ = size(g)
    size(elevation) == (nx, ny) || throw(DimensionMismatch(
        "add_topography: elevation is $(size(elevation)) but the grid is $((nx, ny))"))

    lr = copy(m.log_rho)
    de = copy(m.density)
    @inbounds for k in eachindex(g.cz), j in 1:ny, i in 1:nx
        if g.cz[k] < elevation[i, j]
            lr[i, j, k] = air_log_rho
            de[i, j, k] = 0.0
        end
    end
    return SyntheticModel(g, lr, de)
end

"""
    synth_gravity(m, sx, sy, sz=nothing; noise=0.0, err=nothing, rng=Xoshiro(0),
                  units=:mgal) -> GravityObs

Forward-model the truth's density onto stations and add Gaussian noise.

`noise` is the standard deviation in mGal. `err` is what goes into the returned
observation as the assumed uncertainty; it defaults to `noise` so a run is
weighted by the noise it actually has, and to 1 per cent of the anomaly range
when `noise` is zero, since a zero uncertainty would make the misfit infinite.

`sz` defaults to the grid's top edge, that is, stations on the datum.
"""
function synth_gravity(m::SyntheticModel,
                       sx::AbstractVector{<:Real},
                       sy::AbstractVector{<:Real},
                       sz::Union{Nothing,AbstractVector{<:Real}} = nothing;
                       noise::Real = 0.0,
                       err::Union{Nothing,Real} = nothing,
                       rng::AbstractRNG = Xoshiro(0),
                       units::Symbol = :mgal)
    noise >= 0 || throw(ArgumentError("synth_gravity: noise must be non-negative"))
    z = sz === nothing ? fill(m.grid.z[1], length(sx)) : sz

    A = gravity_matrix(m.grid, sx, sy, z; units = units)
    clean = forward_gravity(A, m.density)
    value = noise > 0 ? clean .+ noise .* randn(rng, length(clean)) : clean

    sigma = if err !== nothing
        Float64(err)
    elseif noise > 0
        Float64(noise)
    else
        span = maximum(clean) - minimum(clean)
        max(0.01 * span, 1e-6)
    end

    return GravityObs(sx, sy, z, value, fill(sigma, length(value)))
end

"""
    synth_mt_sites(m, sx, sy, periods; noise=0.0, rng=Xoshiro(0)) -> MTSites

Apparent resistivity and phase at each site from the 1D column beneath it.

Takes periods in seconds, the MT convention and what [`MTSites`](@ref) stores, so
the result feeds [`build_features`](@ref) and [`nb_baseline`](@ref) directly.
`noise` is a relative standard deviation applied to apparent resistivity and,
halved and converted to degrees, to phase -- the usual pairing where a 5 per cent
error in rho corresponds to about 1.4 degrees in phase. When `noise > 0`, the
returned sites also carry `err_rho_a` and `err_phase` at that level so
[`mt_column_misfit`](@ref) scores χ²/datum.

This is a 1D approximation taken column by column, which means it misses the
lateral induction that makes 3D MT hard. Do not read a good fit to these numbers
as evidence about 3D performance, and do not feed them to `mt_column_misfit` and
call the result a validation: that operator is the same code.
"""
function synth_mt_sites(m::SyntheticModel,
                        sx::AbstractVector{<:Real},
                        sy::AbstractVector{<:Real},
                        periods::AbstractVector{<:Real};
                        noise::Real = 0.0,
                        rng::AbstractRNG = Xoshiro(0))
    ns = length(sx)
    length(sy) == ns ||
        throw(ArgumentError("synth_mt_sites: sx and sy must have equal length"))
    noise >= 0 || throw(ArgumentError("synth_mt_sites: noise must be non-negative"))
    all(>(0), periods) ||
        throw(ArgumentError("synth_mt_sites: periods must be positive"))

    g = m.grid
    freqs = 1 ./ periods
    nf = length(periods)
    rho_app = Array{Float64}(undef, nf, ns)
    phase = Array{Float64}(undef, nf, ns)

    for s in 1:ns
        i = clamp(searchsortedlast(g.x, sx[s]), 1, length(g.cx))
        j = clamp(searchsortedlast(g.y, sy[s]), 1, length(g.cy))

        # air cells carry no resistivity for the recursion; drop them and start
        # the column at the ground, which is what a site on topography measures
        col = m.log_rho[i, j, :]
        live = findall(isfinite, col)
        isempty(live) && error("synth_mt_sites: column under site $s is entirely air")

        rho = (10.0) .^ col[live]
        t = g.dz[live[1:end-1]]
        r, p = mt1d_apparent(freqs, rho, t)

        rho_app[:, s] = r
        phase[:, s] = p
    end

    if noise > 0
        rho_app .*= (1 .+ noise .* randn(rng, nf, ns))
        phase .+= (noise / 2 * 180 / pi) .* randn(rng, nf, ns)
        err_rho, err_phase = mt_apparent_errors_from_relative_noise(rho_app, noise)
        return MTSites(sx, sy, periods, rho_app, phase;
                       err_rho_a = err_rho, err_phase = err_phase)
    end

    return MTSites(sx, sy, periods, rho_app, phase)
end

"""
    truth_vector(m) -> Vector{Float64}

The truth flattened to match the cell ordering of a feature matrix, so it lines
up with `mu`, `sigma` and the bound cubes.
"""
truth_vector(m::SyntheticModel) = vec(m.log_rho)
