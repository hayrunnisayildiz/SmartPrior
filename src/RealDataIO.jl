# SEG-EDI readers for field soundings. Processed AusLAMP dumps (>=MTSECT,
# Zxx/Zxy/Zyx/Zyy R/I/.VAR, FREQ). Apparent resistivity uses the same identity
# as [`mt1d_apparent`](@ref): ρ_a = |Z|² / (ω μ₀), after converting the file's
# practical (mV/km/nT) impedance to SI ohm.

# mV/km/nT → Ohm. Equates 0.2*T*|Z_prac|² with |Z_SI|²/(ω μ₀).
# Applies to this 11-station profile, not to every AusLAMP file (see SA227).
const PRACTICAL_Z_TO_SI = 4π * 1e-4

# Musgrave 2-D profile, west to east. y is metres east of WA55 on a WGS84 local
# tangent plane. SA326N, SA346, and SA345 are omitted (the last two sit outside
# the DEM coverage).
const MUSGRAVE_PROFILE_STATIONS = (
    "WA55"  => 0.0,
    "WA44"  => 49413.4,
    "WA28"  => 98172.2,
    "WA12"  => 153080.1,
    "SA351" => 252424.5,
    "SA350" => 298395.1,
    "SA349" => 348436.0,
    "SA348" => 403014.2,
    "SA347" => 444727.4,
)

# Local tangent-plane projection shared with the MT profile (WA55 origin).
const MUSGRAVE_LON0 = 127.0135
const MUSGRAVE_EARTH_R = 6_371_000.0
const MUSGRAVE_PROFILE_LATS = (
    -25.9948, -26.0043, -25.9924, -25.9614,
    -26.0370, -26.0020, -25.9948, -26.0353, -26.0180,
)
const MUSGRAVE_LAT0 = sum(MUSGRAVE_PROFILE_LATS) / length(MUSGRAVE_PROFILE_LATS)

export build_musgrave_profile, build_musgrave_datafile2d, build_musgrave_gravity,
       build_musgrave_mesh, build_musgrave_surface_z, musgrave_phase_tensor_skew

using MTGeophysics: BuildMesh2D, MT2DMesh, DataFile2D
using ArchGDAL

"""
    read_musgrave_edi(path) -> NamedTuple

Read one processed Musgrave / AusLAMP SEG-EDI file.

Returns `(lat, lon, elev, frequencies, z_xx, z_xx_error, z_xy, z_xy_error,
z_yx, z_yx_error, z_yy, z_yy_error, rho_xy, phase_xy)`.
`frequencies` is Hz. Impedances are SI ohm (converted from mV/km/nT by
`4π × 10⁻⁴`); each `z_*_error` is that factor times `sqrt(Z**.VAR)`.
`rho_xy` is ohm-metres and `phase_xy` degrees, still from the TE (`Zxy`)
component only. Arrays are ordered by increasing period, matching
[`MTSites`](@ref) / [`mt_column_misfit`](@ref).

Musgrave/AusLAMP koleksiyonundaki Z birimi dosyaya göre değişebilir (bkz.
SA227 vs bu profildeki 11 dosya) — bu fonksiyon mV/km/nT varsayıyor, YENİ bir
istasyon eklerken önce elle doğrulanmalı (bkz. yöntem: T=8s'de rho_a'nın
1-10^4 Ω·m aralığında çıkıp çıkmadığına bak).
"""
function read_musgrave_edi(path::AbstractString)
    lines = readlines(path)
    lat, lon, elev = _edi_head_llh(lines)
    frequencies = _edi_block_floats(lines, "FREQ")
    n = length(frequencies)
    n > 0 || throw(ArgumentError("read_musgrave_edi: empty FREQ block in $path"))

    z_xx, z_xx_error = _edi_impedance(lines, "ZXX", n, path)
    z_xy, z_xy_error = _edi_impedance(lines, "ZXY", n, path)
    z_yx, z_yx_error = _edi_impedance(lines, "ZYX", n, path)
    z_yy, z_yy_error = _edi_impedance(lines, "ZYY", n, path)

    # EDI FREQ is typically decreasing (shortest period first). The rest of the
    # package stores increasing period, so reverse when that is not already so.
    if !issorted(frequencies; rev = true)
        reverse!(frequencies)
        reverse!(z_xx); reverse!(z_xx_error)
        reverse!(z_xy); reverse!(z_xy_error)
        reverse!(z_yx); reverse!(z_yx_error)
        reverse!(z_yy); reverse!(z_yy_error)
    end

    ω = 2π .* frequencies
    rho_xy = abs2.(z_xy) ./ (ω .* MU0)
    phase_xy = rad2deg.(atan.(imag.(z_xy), real.(z_xy)))

    return (lat = lat, lon = lon, elev = elev, frequencies = frequencies,
            z_xx = z_xx, z_xx_error = z_xx_error,
            z_xy = z_xy, z_xy_error = z_xy_error,
            z_yx = z_yx, z_yx_error = z_yx_error,
            z_yy = z_yy, z_yy_error = z_yy_error,
            rho_xy = rho_xy, phase_xy = phase_xy)
end

"""
    musgrave_phase_tensor_skew() -> Matrix{Float64}

Caldwell / Bibby / Brown phase-tensor skew β in degrees for the WA55–SA347
profile. Size `(nperiod, nstation)` = `(23, 9)`, same layout as `MTSites`
ρ_a: increasing period down the rows, west-to-east stations across columns.

Formula (Caldwell, T.G., Bibby, H.M. & Brown, C., 2004. The magnetotelluric
phase tensor. *Geophys. J. Int.* **158**, 457–469):

- eq. (13): `Φ = X⁻¹ Y` with `X = Re(Z)`, `Y = Im(Z)`,
  `Z = [Zxx Zxy; Zyx Zyy]` at each period
- eq. (19): `β = ½ tan⁻¹[(Φ₁₂ − Φ₂₁) / (Φ₁₁ + Φ₂₂)]`

The same invariants appear in Bibby, H.M., Caldwell, T.G. & Brown, C.,
2005. *Geophys. J. Int.* **163**, 915–930. Implemented with the two-argument
arctangent `atan(Φ₁₂ − Φ₂₁, Φ₁₁ + Φ₂₂)`, which equals the ratio form of
eq. (19) whenever `tr(Φ) > 0` (the usual MT quadrant; 1-D `Φ = tan(φ) I`)
and does not fold β into ±45° if the trace is negative (Booker, J.R.,
2014. The magnetotelluric phase tensor: a critical review).

Checked independently of any field file: a 1-D antidiagonal `Z` and a 2-D
strike-aligned antidiagonal `Z` both give `β = 0`; rotating the 2-D tensor
leaves `β = 0` (coordinate invariant); a constructed `Φ` with known
components recovers eq. (19) to machine precision. See `test/TestRealDataIO.jl`.

Does not write TM into [`build_musgrave_datafile2d`](@ref). This is a
dimensionality diagnostic only.
"""
function musgrave_phase_tensor_skew()
    ids = first.(MUSGRAVE_PROFILE_STATIONS)
    records = [read_musgrave_edi(_find_musgrave_edi(id)) for id in ids]

    ref_f = records[1].frequencies
    mismatched = Int[]
    for (s, rec) in enumerate(records)
        rec.frequencies == ref_f || push!(mismatched, s)
    end
    if !isempty(mismatched)
        lines = ["  $(ids[s]): $(length(records[s].frequencies)) frequencies"
                 for s in 1:length(records)]
        throw(ArgumentError(
            "musgrave_phase_tensor_skew: frequencies are not identical " *
            "(differ at $(join(ids[mismatched], ", ")))\n" * join(lines, '\n')))
    end

    ns = length(records)
    np = length(ref_f)
    β = Matrix{Float64}(undef, np, ns)
    for s in 1:ns
        rec = records[s]
        for p in 1:np
            β[p, s] = _phase_tensor_beta(rec.z_xx[p], rec.z_xy[p],
                                        rec.z_yx[p], rec.z_yy[p])
        end
    end
    return β
end

# Caldwell et al. (2004) eq. (19), β in degrees. NaN if X is singular.
function _phase_tensor_beta(zxx::Complex, zxy::Complex,
                            zyx::Complex, zyy::Complex)
    X = [real(zxx) real(zxy); real(zyx) real(zyy)]
    Y = [imag(zxx) imag(zxy); imag(zyx) imag(zyy)]
    (all(isfinite, X) && all(isfinite, Y)) || return NaN
    abs(det(X)) > 0 || return NaN
    Φ = X \ Y
    return rad2deg(0.5 * atan(Φ[1, 2] - Φ[2, 1], Φ[1, 1] + Φ[2, 2]))
end

"""
    build_musgrave_profile() -> MTSites

Assemble the 9-station Musgrave profile (WA55–SA347) into one [`MTSites`](@ref).

Each station is read with [`read_musgrave_edi`](@ref). Frequencies must be
identical across stations (no interpolation or cropping). Impedance errors
become `err_rho_a` / `err_phase` via [`mt_apparent_errors_from_impedance`](@ref).
EDI files are found by basename under `musgrave_edi/` if present, otherwise
`data_aust/`.
"""
function build_musgrave_profile()
    ids = first.(MUSGRAVE_PROFILE_STATIONS)
    y = collect(Float64, last.(MUSGRAVE_PROFILE_STATIONS))
    records = [read_musgrave_edi(_find_musgrave_edi(id)) for id in ids]

    ref_f = records[1].frequencies
    mismatched = Int[]
    for (s, rec) in enumerate(records)
        rec.frequencies == ref_f || push!(mismatched, s)
    end
    if !isempty(mismatched)
        lines = ["  $(ids[s]): $(length(records[s].frequencies)) frequencies"
                 for s in 1:length(records)]
        throw(ArgumentError(
            "build_musgrave_profile: frequencies are not identical " *
            "(differ at $(join(ids[mismatched], ", ")))\n" * join(lines, '\n')))
    end

    ns = length(records)
    np = length(ref_f)
    rho_a = Matrix{Float64}(undef, np, ns)
    phase = Matrix{Float64}(undef, np, ns)
    z_xy = Matrix{ComplexF64}(undef, np, ns)
    z_err = Matrix{Float64}(undef, np, ns)
    for s in 1:ns
        rho_a[:, s] = records[s].rho_xy
        phase[:, s] = records[s].phase_xy
        z_xy[:, s] = records[s].z_xy
        z_err[:, s] = records[s].z_xy_error
    end
    err_rho_a, err_phase = mt_apparent_errors_from_impedance(rho_a, z_xy, z_err, ref_f)

    return MTSites(zeros(ns), y, 1 ./ ref_f, rho_a, phase;
                   err_rho_a = err_rho_a, err_phase = err_phase)
end

"""
    build_musgrave_datafile2d(; model_err_frac=0.4) -> DataFile2D

The WA55–SA347 profile as an MTGeophysics `DataFile2D`, ready for `write_data2d`.

Unlike [`build_musgrave_profile`](@ref), this keeps the native TE impedance
(`z_xy`) and the EDI `ELEV` values. TM (`z_yx`) is filled with `NaN` so
`chi2_rms2d` / VFSA skip those samples rather than fitting a zero tensor.
The EDI TM tensor was read and a residual diagnostic run (prior forward vs
observation); station-to-station misfit is too heterogeneous for a single
`model_err_frac`, so TM is not in v0.1.0. See README, "Açık Sorular / v0.2
Adayları". Diagonal components follow the synthetic template (zero
impedance, error floor 0.05) and are unused.

`z_xy_error` is not the EDI variance alone.
`err = sqrt.(z_xy_error.^2 .+ (model_err_frac .* abs.(z_xy)).^2)`.
`model_err_frac` is a fixed fraction of `|Z_obs|` that captures the
representational error of a 2-D TE model on real field data — something
EDI `z_xy_error` (measurement precision; median ~0.004 decade) does not.
It was estimated once from a diagnostic run (prior forward vs observation:
mean 0.512 decade, median 0.392 decade) and then frozen; it is not
updated during training or VFSA. Default 0.4, overridable by hand.
`z_xy` itself is not modified.

`z_positions` are the EDI `HEAD` elevations as stored (orthometric metres,
positive up, ~500–800 m on this profile). The 2D mesh itself is positive-down
with the air–earth interface at `z = 0`, and `run_mt2d_forward` places
receivers on that surface — it does not read `DataFile2D.z_positions`.
Synthetic `.obs` files therefore write zeros. Negating `ELEV` into mesh depth
would put the sites in the air of a flat mesh; leaving them as EDI heights
keeps the number that is actually in the files.
"""
function build_musgrave_datafile2d(; model_err_frac::Real = 0.4)
    me = Float64(model_err_frac)
    me >= 0 || throw(ArgumentError(
        "build_musgrave_datafile2d: model_err_frac must be non-negative, got $me"))

    ids = first.(MUSGRAVE_PROFILE_STATIONS)
    y = collect(Float64, last.(MUSGRAVE_PROFILE_STATIONS))
    records = [read_musgrave_edi(_find_musgrave_edi(id)) for id in ids]

    ref_f = records[1].frequencies
    mismatched = Int[]
    for (s, rec) in enumerate(records)
        rec.frequencies == ref_f || push!(mismatched, s)
    end
    if !isempty(mismatched)
        lines = ["  $(ids[s]): $(length(records[s].frequencies)) frequencies"
                 for s in 1:length(records)]
        throw(ArgumentError(
            "build_musgrave_datafile2d: frequencies are not identical " *
            "(differ at $(join(ids[mismatched], ", ")))\n" * join(lines, '\n')))
    end

    ns = length(records)
    np = length(ref_f)
    z_xy = Matrix{ComplexF64}(undef, np, ns)
    z_xy_error = Matrix{Float64}(undef, np, ns)
    rho_xy = Matrix{Float64}(undef, np, ns)
    phase_xy = Matrix{Float64}(undef, np, ns)
    z_positions = Vector{Float64}(undef, ns)
    for s in 1:ns
        z_xy[:, s] = records[s].z_xy
        z_xy_error[:, s] = records[s].z_xy_error
        rho_xy[:, s] = records[s].rho_xy
        phase_xy[:, s] = records[s].phase_xy
        z_positions[s] = records[s].elev
    end
    z_xy_error = sqrt.(z_xy_error .^ 2 .+ (me .* abs.(z_xy)) .^ 2)

    nanZ = fill(ComplexF64(NaN, NaN), np, ns)
    return DataFile2D(
        title = "Musgrave AusLAMP profile, WA55-SA347, TE-only (TM incelendi, istasyon-heterojen artık nedeniyle v0.1.0'a dahil edilmedi -- bkz. README)",
        periods = 1.0 ./ Float64.(ref_f),
        frequencies = Float64.(ref_f),
        site_names = collect(String, ids),
        receivers = y,
        x_positions = zeros(ns),
        z_positions = z_positions,
        z_xy = z_xy,
        z_xy_error = z_xy_error,
        z_yx = nanZ,
        z_yx_error = fill(NaN, np, ns),
        z_xx = zeros(ComplexF64, np, ns),
        z_xx_error = fill(0.05, np, ns),
        z_yy = zeros(ComplexF64, np, ns),
        z_yy_error = fill(0.05, np, ns),
        rho_xy = rho_xy,
        phase_xy = phase_xy,
        rho_yx = fill(NaN, np, ns),
        phase_yx = fill(NaN, np, ns),
        path = "",
    )
end

# 1 μm/s² = 0.1 mGal (1 mGal = 10 μm/s²).
const UM_S2_TO_MGAL = 0.1

"""
    build_musgrave_gravity(csv_path; model_err_mgal=15.0) -> GravityObs

Load the Musgrave corridor Bouguer CSV into a [`GravityObs`](@ref).

Columns used: `local_x_perp_m` → `x`, `local_y_along_m` → `y`, `elev_m` → `z`,
`bouguer_mGal` → `value`, `acc_raw` → measurement part of `err`. Coordinates
are already in the WA55 local frame; no further projection is applied.

`acc_raw` is Geoscience Australia's combined Bouguer uncertainty, in μm/s²,
converted to mGal (`acc_err = acc_raw × 0.1`). Stations with missing or
non-positive `acc_raw` fall back to 0.1 mGal; the count is warned, not silent.

`err = sqrt.(acc_err.^2 .+ model_err_mgal^2)`. `model_err_mgal` is the
representational error of a single-slope 2-D prism model on a real
multi-lithology profile — something `acc_raw` does not measure. It was
estimated once from a diagnostic run (detrended residual ~16.6 mGal) and
then frozen; it is not updated during training. Default 15.0, overridable
by hand.
"""
function build_musgrave_gravity(csv_path::AbstractString; model_err_mgal::Real = 15.0)
    me = Float64(model_err_mgal)
    me >= 0 || throw(ArgumentError(
        "build_musgrave_gravity: model_err_mgal must be non-negative, got $me"))

    lines = readlines(csv_path)
    isempty(lines) && throw(ArgumentError("build_musgrave_gravity: empty file $csv_path"))

    header = split(strip(lines[1]), ',')
    col = Dict{String,Int}(strip(name) => i for (i, name) in enumerate(header))
    needed = ("local_x_perp_m", "local_y_along_m", "elev_m", "bouguer_mGal", "acc_raw")
    for name in needed
        haskey(col, name) || throw(ArgumentError(
            "build_musgrave_gravity: missing column $name in $csv_path"))
    end
    ix, iy, iz, iv, ie = col["local_x_perp_m"], col["local_y_along_m"],
                         col["elev_m"], col["bouguer_mGal"], col["acc_raw"]

    x = Float64[]
    y = Float64[]
    z = Float64[]
    value = Float64[]
    acc_err = Float64[]
    n_fallback = 0
    for line in @view lines[2:end]
        s = strip(line)
        isempty(s) && continue
        parts = split(s, ',')
        push!(x, parse(Float64, parts[ix]))
        push!(y, parse(Float64, parts[iy]))
        push!(z, parse(Float64, parts[iz]))
        push!(value, parse(Float64, parts[iv]))
        acc = _parse_acc_raw(parts[ie])
        if acc === nothing || acc <= 0
            n_fallback += 1
            push!(acc_err, 0.1)
        else
            push!(acc_err, acc * UM_S2_TO_MGAL)
        end
    end
    isempty(x) && throw(ArgumentError("build_musgrave_gravity: no data rows in $csv_path"))
    if n_fallback > 0
        @warn "build_musgrave_gravity: $n_fallback stations missing or non-positive acc_raw; using 0.1 mGal fallback"
    end
    err = sqrt.(acc_err .^ 2 .+ me^2)
    return GravityObs(x, y, z, value, err)
end

function _parse_acc_raw(tok::AbstractString)
    s = strip(tok)
    isempty(s) && return nothing
    return tryparse(Float64, s)
end

"""
    build_musgrave_mesh() -> MT2DMesh

2-D MT mesh for the WA55–SA347 profile. Frequencies follow
[`build_musgrave_profile`](@ref) (increasing Hz, as in `BuildMesh2D`'s default).
Core and padding extents match the gravity corridor; layer thicknesses are a
first-cut skin-depth estimate, not a field-validated discretisation.
"""
function build_musgrave_mesh()
    sites = build_musgrave_profile()
    frequencies = collect(1 ./ sites.periods)
    issorted(frequencies) || reverse!(frequencies)
    return BuildMesh2D(;
        frequencies = frequencies,
        y_core_range = (-50000.0, 494727.4),
        y_core_cell = 5000.0,
        y_padding = 100000.0,
        air_cells = 6,
        ground_layers = vcat(fill(200.0, 10), fill(1000.0, 10),
                             fill(5000.0, 10), fill(20000.0, 10)),
        receiver_positions = sites.y,
    )
end

"""
    build_musgrave_surface_z(mesh_or_grid, dem_path) -> Matrix{Float64}

Sample the AusLAMP-corridor DEM onto the 2-D profile's y-cell centres.

Returns a `(1, ny)` elevation map in metres (orthometric, matching EDI `ELEV`).
Longitude is recovered from local y with the WA55 tangent-plane formula at
fixed `lat = MUSGRAVE_LAT0` (the grid has one x-cell, so latitude variation
cannot be represented). Cells whose lon/lat fall outside the GeoTIFF are
assigned the nearest DEM-edge pixel; the count is logged, not silent.

Uses ArchGDAL and reads only the row/column window covering the sampled
pixels, not the full ~18000×18000 raster.
"""
function build_musgrave_surface_z(mesh_or_grid, dem_path::AbstractString)
    cy = _musgrave_y_centers(mesh_or_grid)
    ny = length(cy)
    lon_per_m = (180 / π) / (MUSGRAVE_EARTH_R * cos(deg2rad(MUSGRAVE_LAT0)))
    lons = MUSGRAVE_LON0 .+ cy .* lon_per_m

    return ArchGDAL.read(dem_path) do ds
        gt = ArchGDAL.getgeotransform(ds)
        igt = ArchGDAL.invgeotransform(gt)
        width = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)
        @info "DEM geotransform (cross-check: tie ≈ 126.9998611, -23.9998611, scale ≈ 0.0002777778)" gt[1] gt[4] gt[2] gt[6] width height

        col0 = Vector{Int}(undef, ny)
        row0 = Vector{Int}(undef, ny)
        n_extrap = 0
        for j in 1:ny
            px = igt[1] + lons[j] * igt[2] + MUSGRAVE_LAT0 * igt[3]
            py = igt[4] + lons[j] * igt[5] + MUSGRAVE_LAT0 * igt[6]
            # containing pixel, 0-based GDAL
            c = floor(Int, px)
            r = floor(Int, py)
            if c < 0 || c >= width || r < 0 || r >= height
                n_extrap += 1
            end
            col0[j] = clamp(c, 0, width - 1)
            row0[j] = clamp(r, 0, height - 1)
        end
        @info "DEM edge extrapolation" n_extrap ny

        # 1-based window covering every sampled (clamped) pixel
        c1, c2 = extrema(col0) .+ 1
        r1, r2 = extrema(row0) .+ 1
        band = ArchGDAL.getband(ds, 1)
        window = ArchGDAL.read(band, r1:r2, c1:c2)

        sz = Matrix{Float64}(undef, 1, ny)
        for j in 1:ny
            sz[1, j] = Float64(window[(col0[j] + 1) - c1 + 1, (row0[j] + 1) - r1 + 1])
        end
        return sz
    end
end

_musgrave_y_centers(mesh::MT2DMesh) =
    (mesh.y_nodes[1:end-1] .+ mesh.y_nodes[2:end]) ./ 2
_musgrave_y_centers(grid::PriorGrid) = grid.cy

function _find_musgrave_edi(station_id::AbstractString)
    pkg = dirname(@__DIR__)
    name = station_id * ".edi"
    roots = String[]
    mus = joinpath(pkg, "musgrave_edi")
    isdir(mus) && push!(roots, mus)
    data = joinpath(pkg, "data_aust")
    isdir(data) && push!(roots, data)
    isempty(roots) && push!(roots, pkg)

    hits = String[]
    for root in roots
        for (dir, dirs, files) in walkdir(root)
            filter!(d -> d != ".git" && !startswith(d, '.'), dirs)
            name in files && push!(hits, joinpath(dir, name))
        end
        isempty(hits) || break
    end
    length(hits) == 1 && return hits[1]
    isempty(hits) && throw(ArgumentError(
        "build_musgrave_profile: no $name under $(join(roots, ", "))"))
    throw(ArgumentError(
        "build_musgrave_profile: multiple $name files: $(join(hits, ", "))"))
end

function _edi_strip(line::AbstractString)
    return strip(rstrip(line, '\r'))
end

function _edi_is_tag(line::AbstractString, tag::AbstractString)
    s = _edi_strip(line)
    prefix = ">" * tag
    startswith(s, prefix) || return false
    n = ncodeunits(prefix)
    ncodeunits(s) == n && return true
    c = s[n + 1]
    return c == ' ' || c == '\t' || c == '/'
end

function _edi_head_llh(lines::Vector{<:AbstractString})
    i = findfirst(l -> _edi_is_tag(l, "HEAD"), lines)
    i === nothing && throw(ArgumentError("read_musgrave_edi: missing >HEAD block"))
    parts = String[]
    for j in (i + 1):length(lines)
        s = _edi_strip(lines[j])
        startswith(s, '>') && break
        push!(parts, s)
    end
    head = join(parts, '\n')
    return (_edi_head_float(head, "LAT"),
            _edi_head_float(head, "LONG"),
            _edi_head_float(head, "ELEV"))
end

function _edi_head_float(head::AbstractString, key::AbstractString)
    m = match(Regex("^\\s*" * key * "\\s*=\\s*(\\S+)", "m"), head)
    m === nothing && throw(ArgumentError("read_musgrave_edi: HEAD missing $key="))
    return parse(Float64, m.captures[1])
end

function _edi_block_floats(lines::Vector{<:AbstractString}, tag::AbstractString)
    i = findfirst(l -> _edi_is_tag(l, tag), lines)
    i === nothing && throw(ArgumentError("read_musgrave_edi: missing >$tag block"))
    vals = Float64[]
    for j in (i + 1):length(lines)
        s = _edi_strip(lines[j])
        isempty(s) && continue
        startswith(s, '>') && break
        for tok in split(s)
            push!(vals, parse(Float64, tok))
        end
    end
    return vals
end

function _edi_impedance(lines::Vector{<:AbstractString}, tag::AbstractString,
                        n::Int, path::AbstractString)
    zr = _edi_block_floats(lines, tag * "R")
    zi = _edi_block_floats(lines, tag * "I")
    zv = _edi_block_floats(lines, tag * ".VAR")
    (length(zr) == n && length(zi) == n && length(zv) == n) || throw(ArgumentError(
        "read_musgrave_edi: FREQ/$(tag)R/$(tag)I/$(tag).VAR lengths differ in $path: " *
        "$n, $(length(zr)), $(length(zi)), $(length(zv))"))
    return ComplexF64.(zr, zi) .* PRACTICAL_Z_TO_SI, sqrt.(zv) .* PRACTICAL_Z_TO_SI
end
