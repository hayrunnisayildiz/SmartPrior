# Cloncurry–Ernest Henry (METAL / GDA94 MGA zone 54) readers.
# Gravity, magnetics and MT are not opened here even though they sit in the
# same data package; the network sees pXRF geochemistry and lithology as
# features, and Cu plus petrophysics as anchors.
#
# Coordinate convention: x = easting, y = northing, z = elevation (positive
# up, metres ASL). EPSG:28354.

# pXRF *_Concentration columns from Mg to U. Cu is the grade *target* and is
# excluded from the feature list.
# LE_Concentration is a light-element composite, not an element, and is dropped.
const CLONCURRY_GEOCHEM_ELEMENTS = (
    "Mg", "Al", "Si", "P", "S", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Zn", "As", "Se", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd",
    "Sn", "Sb", "W", "Hg", "Pb", "Bi", "Th", "U",
)
const CLONCURRY_SULFIDE_INDEX_ELEMENTS = ("S", "Fe", "Zn", "Pb")
# Pathfinders. Ni/Cu and Pd/Ni are not available: Cu is held out, Pd is
# absent from the pXRF suite. Fe/S tracks oxide vs sulfide iron; As/S and
# Zn/Pb are district base-metal pathfinders.
const CLONCURRY_GEOCHEM_RATIOS = (("As", "S"), ("Zn", "Pb"), ("Fe", "S"))
const CLONCURRY_CU_DL_PPM = 1.0
# KT-20 100 kHz specimen conductivity. 711 / 1250 finite values are exact 0;
# the smallest positive in the 2026-09-17 dump is 0.02055 S/m. Zeros are lifted
# to this floor so they stay anchors rather than going to -Inf under log10.
const CLONCURRY_COND_FLOOR_S_M = 0.01
const CLONCURRY_PROPERTY_NAMES = ["grade", "density", "susceptibility",
                                  "conductivity_100kHz"]

# Work area requested for the first Cloncurry grid (geographic), converted to
# GDA94 / MGA zone 54. z is elevation ASL, padded around the 457 in-box samples
# (−1512.5 to +224 m).
const CLONCURRY_WORK_BOUNDS = (
    x_min = 447693.6, x_max = 489585.1,
    y_min = 7713186.7, y_max = 7789622.2,
    z_min = -1600.0, z_max = 250.0,
)

const CLONCURRY_CONDUCTIVITY_STATUS =
    "KT-20, 100 kHz, rock-specimen scale (conductivity_mean_S_m_100kHz). " *
    "Not equivalent to MT bulk low-frequency conductivity; do not call this " *
    "resistivity. Austin et al. (2024) §5.2.4."

"""
    CloncurrySamples

Row-aligned METAL petrophysics + pXRF geochemistry. One row per specimen
(1,590 in the 2026-09-17 dump, including a duplicated `E1-284` kept twice).
`cu_ppm` is the grade target (NaN if the cell was empty, detection-limit if
`<LOD`). Geochemistry columns never include Cu. `drillhole` is the collar
id used as the hold-out group; a blank id is treated as its own group.
"""
struct CloncurrySamples
    sample::Vector{String}
    deposit::Vector{String}
    drillhole::Vector{String}
    east::Vector{Float64}
    north::Vector{Float64}
    elev::Vector{Float64}
    lithology::Vector{String}
    cu_ppm::Vector{Float64}
    density_g_cm3::Vector{Float64}
    susceptibility_SI::Vector{Float64}
    conductivity_S_m_100kHz::Vector{Float64}
    geochem::Matrix{Float64}
    geochem_names::Vector{String}
end

Base.length(s::CloncurrySamples) = length(s.sample)

#---------- csv / gzip ----------

function _parse_float(s::AbstractString)
    t = strip(s)
    isempty(t) && return NaN
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

function _split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quote = false
    for c in line
        if c == '"'
            in_quote = !in_quote
        elseif c == ',' && !in_quote
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

function _with_text(f, path::AbstractString)
    isfile(path) || throw(ArgumentError("CloncurryIO: no such file: $path"))
    if endswith(lowercase(path), ".gz")
        return open(path, "r") do raw
            f(GzipDecompressorStream(raw))
        end
    end
    return open(f, path, "r")
end

function _read_csv_selected(path::AbstractString, want::Vector{String};
                            extra_suffix::Union{Nothing,AbstractString} = nothing,
                            optional::Vector{String} = String[])
    return _with_text(path) do io
        header_line = readline(io)
        header = _split_csv_line(header_line)
        idx = Dict{String,Int}()
        for (j, h) in enumerate(header)
            idx[h] = j
        end
        extra = String[]
        extra_idx = Int[]
        if extra_suffix !== nothing
            for (j, h) in enumerate(header)
                endswith(h, extra_suffix) || continue
                push!(extra, h)
                push!(extra_idx, j)
            end
        end
        names = copy(want)
        need = Int[]
        for name in want
            haskey(idx, name) || throw(ArgumentError(
                "CloncurryIO: missing column $name in $path"))
            push!(need, idx[name])
        end
        optional_idx = Int[]
        for name in optional
            haskey(idx, name) || continue
            push!(names, name)
            push!(optional_idx, idx[name])
        end
        ncol = length(names) + length(extra)
        cols = [String[] for _ in 1:ncol]
        nfixed = length(want)
        nopt = length(optional_idx)
        for line in eachline(io)
            isempty(strip(line)) && continue
            f = _split_csv_line(line)
            length(f) < length(header) && continue
            for (k, j) in enumerate(need)
                push!(cols[k], j <= length(f) ? f[j] : "")
            end
            for (k, j) in enumerate(optional_idx)
                push!(cols[nfixed + k], j <= length(f) ? f[j] : "")
            end
            for (k, j) in enumerate(extra_idx)
                push!(cols[nfixed + nopt + k], j <= length(f) ? f[j] : "")
            end
        end
        store = Dict{String,Vector{String}}()
        for (k, name) in enumerate(names)
            store[name] = cols[k]
        end
        for (k, name) in enumerate(extra)
            store[name] = cols[nfixed + nopt + k]
        end
        return store, extra
    end
end

function _is_lod(s::AbstractString)
    t = uppercase(strip(s))
    return t == "<LOD" || t == "LOD" || t == "BDL" || t == "< DL" || t == "<DL"
end

function _parse_concentration_feature(s::AbstractString)
    t = strip(s)
    (isempty(t) || _is_lod(t)) && return NaN
    return _parse_float(t)
end

function _parse_cu_target(s::AbstractString; detection_limit::Real = CLONCURRY_CU_DL_PPM)
    t = strip(s)
    isempty(t) && return NaN
    _is_lod(t) && return Float64(detection_limit)
    v = _parse_float(t)
    isfinite(v) || return NaN
    return v < 0 ? Float64(detection_limit) : v
end

#---------- dataset root ----------

function cloncurry_derived_dir(dataset_root::AbstractString)
    d = joinpath(dataset_root, "derived")
    isdir(d) || throw(ArgumentError(
        "cloncurry_derived_dir: no derived/ under $dataset_root"))
    return d
end

function cloncurry_petrophysics_path(dataset_root::AbstractString)
    p = joinpath(cloncurry_derived_dir(dataset_root), "petrophysics_samples.csv")
    isfile(p) || throw(ArgumentError(
        "cloncurry_petrophysics_path: no such file: $p"))
    return p
end

function cloncurry_metals_path(dataset_root::AbstractString)
    d = cloncurry_derived_dir(dataset_root)
    gz = joinpath(d, "metal_all_fields.csv.gz")
    csv = joinpath(d, "metal_all_fields.csv")
    isfile(gz) && return gz
    isfile(csv) && return csv
    throw(ArgumentError(
        "cloncurry_metals_path: no metal_all_fields.csv.gz (or .csv) in $d"))
end

"""
    cloncurry_work_bounds() -> NamedTuple

GDA94 / MGA zone 54 box for 140.5–140.9°E, 20.68–19.99°S, with elevation
padded around the in-box METAL samples. This is a *proposal*, not a claim
that 25 m cells are feasible (they are not — see the training script).
"""
cloncurry_work_bounds() = CLONCURRY_WORK_BOUNDS

"""
    cloncurry_deposit_bounds(s, deposit; pad_xy=50, pad_z=25) -> NamedTuple

Axis-aligned MGA54 box around every finite-xyz sample whose `deposit`
label matches `deposit` exactly (Ernest Henry option D). Pads by half a
100 m / 50 m cell so hull samples are not sitting on a face.
"""
function cloncurry_deposit_bounds(s::CloncurrySamples, deposit::AbstractString;
                                  pad_xy::Real = 50.0, pad_z::Real = 25.0)
    pad_xy >= 0 || throw(ArgumentError("cloncurry_deposit_bounds: pad_xy must be ≥ 0"))
    pad_z >= 0 || throw(ArgumentError("cloncurry_deposit_bounds: pad_z must be ≥ 0"))
    keep = (s.deposit .== deposit) .& isfinite.(s.east) .&
           isfinite.(s.north) .& isfinite.(s.elev)
    n = count(keep)
    n == 0 && throw(ArgumentError(
        "cloncurry_deposit_bounds: no finite-xyz samples labelled $(repr(deposit))"))
    xs, ys, zs = s.east[keep], s.north[keep], s.elev[keep]
    return (x_min = minimum(xs) - pad_xy, x_max = maximum(xs) + pad_xy,
            y_min = minimum(ys) - pad_xy, y_max = maximum(ys) + pad_xy,
            z_min = minimum(zs) - pad_z,  z_max = maximum(zs) + pad_z,
            deposit = String(deposit), n_samples = n)
end

"""
    cloncurry_sample_bounds(s; pad_xy=0, pad_z=0) -> NamedTuple

Axis-aligned MGA54 box around every finite-xyz sample, any deposit.
Tighter than [`cloncurry_work_bounds`](@ref) *only if* the samples sit
inside that proposal box; on the 2026-09-17 dump they do not (AABB
~118 × 219 km vs the 42 × 76 km work box, which drops ~1,100 samples).
A Cartesian prior grid cannot be a convex hull — this AABB is the
smallest axis-aligned box and therefore the fewest empty cells among
rectangles that still hold every sample.
"""
function cloncurry_sample_bounds(s::CloncurrySamples;
                                 pad_xy::Real = 0.0, pad_z::Real = 0.0)
    pad_xy >= 0 || throw(ArgumentError("cloncurry_sample_bounds: pad_xy must be ≥ 0"))
    pad_z >= 0 || throw(ArgumentError("cloncurry_sample_bounds: pad_z must be ≥ 0"))
    keep = _finite_xyz(s)
    n = count(keep)
    n == 0 && throw(ArgumentError("cloncurry_sample_bounds: no finite (x, y, z)"))
    xs, ys, zs = s.east[keep], s.north[keep], s.elev[keep]
    return (x_min = minimum(xs) - pad_xy, x_max = maximum(xs) + pad_xy,
            y_min = minimum(ys) - pad_xy, y_max = maximum(ys) + pad_xy,
            z_min = minimum(zs) - pad_z,  z_max = maximum(zs) + pad_z,
            n_samples = n)
end

"""
    cloncurry_district_spacing(bounds; target=50000, cell_z=100) -> NamedTuple

XY/Z spacing for the district sample AABB. Horizontal cell size follows
`target` with a fixed reference of 10 Z-layers so XY stays near the
historical ~2300 m mesh when `target=50000`. Vertical spacing defaults to
**100 m** (finer than the earlier ~200 m / 10-layer choice; grade vertical
range is ~300 m). Returns metres, rounded to 100 m (XY) / 50 m (Z).
"""
function cloncurry_district_spacing(bounds; target::Integer = 50_000,
                                    cell_z::Real = 100.0)
    target > 0 || throw(ArgumentError("cloncurry_district_spacing: target must be positive"))
    sx = Float64(bounds.x_max - bounds.x_min)
    sy = Float64(bounds.y_max - bounds.y_min)
    sz = Float64(bounds.z_max - bounds.z_min)
    (sx > 0 && sy > 0 && sz > 0) || throw(ArgumentError(
        "cloncurry_district_spacing: bounds must have positive span"))
    cell_z = max(50.0, round(Float64(cell_z) / 50) * 50)
    # Pin XY to the old nz_ref=10 formula so refining Z does not coarsen E/N.
    nz_xy_ref = 10
    cell = sqrt(sx * sy * nz_xy_ref / target)
    cell = max(100.0, round(cell / 100) * 100)
    nx = max(1, round(Int, sx / cell))
    ny = max(1, round(Int, sy / cell))
    nz = max(1, round(Int, sz / cell_z))
    return (cell = cell, cell_z = cell_z, nx = nx, ny = ny, nz = nz,
            ncells = nx * ny * nz, target = Int(target))
end

"""
    cloncurry_inside_mask(g, s) -> BitVector

Finite-xyz samples whose location falls in `g`.
"""
function cloncurry_inside_mask(g::PriorGrid, s::CloncurrySamples)
    n = length(s)
    m = falses(n)
    @inbounds for i in 1:n
        isfinite(s.east[i]) && isfinite(s.north[i]) && isfinite(s.elev[i]) || continue
        containing_cell(g, s.east[i], s.north[i], s.elev[i]) == 0 && continue
        m[i] = true
    end
    return m
end

"""
    cloncurry_grid(bounds; cell=1000, cell_z=100) -> PriorGrid

Cartesian grid on a Cloncurry MGA54 box. `x` easting, `y` northing, `z`
elevation positive up. Defaults are coarse on purpose: the work area is
~42 × 76 km and a 25 m mesh would be hundreds of millions of cells.
"""
function cloncurry_grid(bounds; cell::Real = 1000.0, cell_z::Real = 100.0)
    cell > 0 || throw(ArgumentError("cloncurry_grid: cell must be positive"))
    cell_z > 0 || throw(ArgumentError("cloncurry_grid: cell_z must be positive"))
    nx = max(1, round(Int, (bounds.x_max - bounds.x_min) / cell))
    ny = max(1, round(Int, (bounds.y_max - bounds.y_min) / cell))
    nz = max(1, round(Int, (bounds.z_max - bounds.z_min) / cell_z))
    dx = fill((bounds.x_max - bounds.x_min) / nx, nx)
    dy = fill((bounds.y_max - bounds.y_min) / ny, ny)
    dz = fill((bounds.z_max - bounds.z_min) / nz, nz)
    return PriorGrid(dx, dy, dz; origin = [bounds.x_min, bounds.y_min, bounds.z_min])
end

#---------- loaders ----------

function load_cloncurry_samples(dataset_root::AbstractString;
                                petrophysics::AbstractString = cloncurry_petrophysics_path(dataset_root),
                                metals::AbstractString = cloncurry_metals_path(dataset_root),
                                cu_detection_limit::Real = CLONCURRY_CU_DL_PPM)
    return load_cloncurry_samples(petrophysics, metals;
                                  cu_detection_limit = cu_detection_limit)
end

function load_cloncurry_samples(petrophysics_path::AbstractString,
                                metals_path::AbstractString;
                                cu_detection_limit::Real = CLONCURRY_CU_DL_PPM)
    petro_need = ["sample", "deposit",
                  "easting_gda94_mga54_m", "northing_gda94_mga54_m",
                  "sample_elevation_asl_m", "lithology_code",
                  "density_mean_g_cm3", "susceptibility_mean_SI",
                  "conductivity_mean_S_m_100kHz"]
    petro, _ = _read_csv_selected(petrophysics_path, petro_need;
                                  optional = ["density_mean_kg_m3", "drillhole"])
    n = length(petro["sample"])
    n > 0 || throw(ArgumentError("load_cloncurry_samples: no petrophysics rows"))
    drillhole = haskey(petro, "drillhole") ?
        [strip(s) for s in petro["drillhole"]] : fill("", n)

    dens = [_parse_float(s) for s in petro["density_mean_g_cm3"]]
    # Optional kg/m³ column: when present it must be 1000 × g/cm³ (the derived
    # dump's conversion). Used as a unit check, never as the training target.
    if haskey(petro, "density_mean_kg_m3")
        n_mismatch = 0
        @inbounds for i in 1:n
            kg = _parse_float(petro["density_mean_kg_m3"][i])
            g = dens[i]
            (isfinite(kg) && isfinite(g)) || continue
            abs(kg - 1000 * g) > 1.0 && (n_mismatch += 1)
        end
        n_mismatch == 0 || throw(ArgumentError(
            "load_cloncurry_samples: density_mean_kg_m3 is not 1000 × g/cm³ " *
            "on $n_mismatch rows; refusing to guess the unit"))
    end

    metal_need = ["Sample"]
    metal, conc_cols = _read_csv_selected(metals_path, metal_need;
                                          extra_suffix = "_Concentration")
    length(metal["Sample"]) == n || throw(ArgumentError(
        "load_cloncurry_samples: petrophysics has $n rows, metals has " *
        "$(length(metal["Sample"])); files must be row-aligned"))
    @inbounds for i in 1:n
        petro["sample"][i] == metal["Sample"][i] || throw(ArgumentError(
            "load_cloncurry_samples: row $i id mismatch " *
            "$(petro["sample"][i]) vs $(metal["Sample"][i])"))
    end

    geochem_names = String[]
    geochem_cols = Vector{Vector{Float64}}()
    for el in CLONCURRY_GEOCHEM_ELEMENTS
        col = el * "_Concentration"
        col in conc_cols || continue
        vals = [_parse_concentration_feature(s) for s in metal[col]]
        any(x -> isfinite(x) && x > 0, vals) || continue
        push!(geochem_names, el)
        push!(geochem_cols, vals)
    end
    "Cu" in geochem_names && throw(ArgumentError(
        "load_cloncurry_samples: Cu leaked into geochemistry features"))
    isempty(geochem_cols) && throw(ArgumentError(
        "load_cloncurry_samples: no usable *_Concentration columns (Cu excluded)"))

    _append_derived_geochem!(geochem_names, geochem_cols)

    cu_col = "Cu_Concentration"
    cu_col in conc_cols || throw(ArgumentError(
        "load_cloncurry_samples: metals file has no Cu_Concentration"))
    cu = [_parse_cu_target(s; detection_limit = cu_detection_limit) for s in metal[cu_col]]

    return CloncurrySamples(
        petro["sample"],
        petro["deposit"],
        drillhole,
        [_parse_float(s) for s in petro["easting_gda94_mga54_m"]],
        [_parse_float(s) for s in petro["northing_gda94_mga54_m"]],
        [_parse_float(s) for s in petro["sample_elevation_asl_m"]],
        [strip(s) for s in petro["lithology_code"]],
        cu,
        dens,
        [_parse_float(s) for s in petro["susceptibility_mean_SI"]],
        [_parse_float(s) for s in petro["conductivity_mean_S_m_100kHz"]],
        reduce(hcat, geochem_cols),
        geochem_names,
    )
end

#---------- derived geochemistry (beside raw elements, not instead) ----------

function _geochem_col(names::Vector{String}, cols::Vector{Vector{Float64}}, el::AbstractString)
    i = findfirst(==(el), names)
    return i === nothing ? nothing : cols[i]
end

"""
Geometric mean of positive finite members among `members`. Needs at least
`min_members` so a lone Fe assay cannot masquerade as a sulfide index.
Stored in concentration units so [`geochemistry_channels`](@ref) log10 yields
the mean of log-concentrations (equal-weight multi-element score).
"""
function _sulfide_index(names::Vector{String}, cols::Vector{Vector{Float64}};
                        members = CLONCURRY_SULFIDE_INDEX_ELEMENTS,
                        min_members::Int = 2)
    vecs = Vector{Vector{Float64}}()
    for el in members
        v = _geochem_col(names, cols, el)
        v === nothing && continue
        push!(vecs, v)
    end
    length(vecs) < min_members && return nothing
    n = length(vecs[1])
    out = fill(NaN, n)
    @inbounds for i in 1:n
        s = 0.0
        k = 0
        for v in vecs
            x = v[i]
            if isfinite(x) && x > 0
                s += log10(x)
                k += 1
            end
        end
        k >= min_members || continue
        out[i] = exp10(s / k)
    end
    any(isfinite, out) || return nothing
    return out
end

function _element_ratio(names::Vector{String}, cols::Vector{Vector{Float64}},
                        num::AbstractString, den::AbstractString)
    a = _geochem_col(names, cols, num)
    b = _geochem_col(names, cols, den)
    (a === nothing || b === nothing) && return nothing
    n = length(a)
    out = fill(NaN, n)
    @inbounds for i in 1:n
        if isfinite(a[i]) && isfinite(b[i]) && a[i] > 0 && b[i] > 0
            out[i] = a[i] / b[i]
        end
    end
    any(isfinite, out) || return nothing
    return out
end

"""
Fe / (Fe + Mg + Ca). Magnetite–hematite iron-oxide vs carbonate–mafic host;
relevant to dense oxide vs sulfide petrophysics in Cloncurry IOCG.
"""
function _fe_oxide_ratio(names::Vector{String}, cols::Vector{Vector{Float64}})
    fe = _geochem_col(names, cols, "Fe")
    mg = _geochem_col(names, cols, "Mg")
    ca = _geochem_col(names, cols, "Ca")
    (fe === nothing || mg === nothing || ca === nothing) && return nothing
    n = length(fe)
    out = fill(NaN, n)
    @inbounds for i in 1:n
        f, m, c = fe[i], mg[i], ca[i]
        (isfinite(f) && isfinite(m) && isfinite(c) && f >= 0 && m >= 0 && c >= 0) || continue
        denom = f + m + c
        denom > 0 || continue
        out[i] = f / denom
    end
    any(isfinite, out) || return nothing
    return out
end

"""
(K + Mg) / Ca. Potassic (± magnesian) alteration vs calcium host. Na is absent
from the Cloncurry pXRF suite, so the Ishikawa-style (K+Mg)/(Na+Ca) denominator
collapses to Ca only.
"""
function _alteration_index(names::Vector{String}, cols::Vector{Vector{Float64}})
    k = _geochem_col(names, cols, "K")
    mg = _geochem_col(names, cols, "Mg")
    ca = _geochem_col(names, cols, "Ca")
    (k === nothing || mg === nothing || ca === nothing) && return nothing
    n = length(k)
    out = fill(NaN, n)
    @inbounds for i in 1:n
        kk, m, c = k[i], mg[i], ca[i]
        (isfinite(kk) && isfinite(m) && isfinite(c) && c > 0 && kk >= 0 && m >= 0) || continue
        out[i] = (kk + m) / c
    end
    any(isfinite, out) || return nothing
    return out
end

function _append_derived_geochem!(names::Vector{String}, cols::Vector{Vector{Float64}})
    function push_derived!(name::AbstractString, vals)
        vals === nothing && return
        name in names && throw(ArgumentError(
            "_append_derived_geochem!: duplicate channel $name"))
        occursin("Cu", name) && throw(ArgumentError(
            "_append_derived_geochem!: Cu leaked into derived name $name"))
        push!(names, String(name))
        push!(cols, vals)
    end

    push_derived!("sulfide_index", _sulfide_index(names, cols))
    push_derived!("fe_oxide_ratio", _fe_oxide_ratio(names, cols))
    push_derived!("alteration_index", _alteration_index(names, cols))
    for (num, den) in CLONCURRY_GEOCHEM_RATIOS
        push_derived!(lowercase(num) * "_over_" * lowercase(den),
                      _element_ratio(names, cols, num, den))
    end
    return nothing
end

function _finite_xyz(s::CloncurrySamples)
    keep = isfinite.(s.east) .& isfinite.(s.north) .& isfinite.(s.elev)
    return keep
end

"""
    cloncurry_geochemistry(s) -> PointSamples

pXRF concentrations (ppm) with Cu held out, plus literature-derived composites
(`sulfide_index`, `fe_oxide_ratio`, `alteration_index`, and pathfinder ratios
from [`CLONCURRY_GEOCHEM_RATIOS`](@ref)). Empty / `<LOD` entries are NaN so
each channel interpolates independently.
"""
function cloncurry_geochemistry(s::CloncurrySamples)
    keep = _finite_xyz(s)
    any(keep) || throw(ArgumentError("cloncurry_geochemistry: no finite (x, y, z)"))
    return PointSamples(s.east[keep], s.north[keep], s.geochem[keep, :],
                        copy(s.geochem_names); z = s.elev[keep])
end

"""
    cloncurry_lithology(s) -> LabelSamples

`lithology_code` one-hot source. Empty codes stay empty and collapse to
`OTHER` in [`lithology_channels`](@ref).
"""
function cloncurry_lithology(s::CloncurrySamples)
    keep = _finite_xyz(s)
    any(keep) || throw(ArgumentError("cloncurry_lithology: no finite (x, y, z)"))
    labels = [isempty(lab) ? "OTHER" : lab for lab in s.lithology[keep]]
    return LabelSamples(s.east[keep], s.north[keep], labels; z = s.elev[keep])
end

function cloncurry_coverage_points(s::CloncurrySamples)
    keep = _finite_xyz(s)
    return (x = s.east[keep], y = s.north[keep], z = s.elev[keep])
end

function _log10_positive(v::Real)
    isfinite(v) && v > 0 || return NaN
    return log10(Float64(v))
end

function _log10_conductivity(v::Real; floor::Real = CLONCURRY_COND_FLOOR_S_M)
    isfinite(v) || return NaN
    return log10(max(Float64(v), Float64(floor)))
end

function cloncurry_property_coverage(s::CloncurrySamples)
    n = length(s)
    frac(mask) = count(mask) / n
    xyz = _finite_xyz(s)
    dens = xyz .& isfinite.(s.density_g_cm3)
    susc = xyz .& isfinite.(s.susceptibility_SI) .& (s.susceptibility_SI .> 0)
    cond = xyz .& isfinite.(s.conductivity_S_m_100kHz)
    grade = xyz .& isfinite.(s.cu_ppm)
    return (n = n,
            n_xyz = count(xyz),
            grade = frac(grade),
            density = frac(dens),
            susceptibility = frac(susc),
            conductivity_100kHz = frac(cond),
            n_grade = count(grade),
            n_density = count(dens),
            n_susceptibility = count(susc),
            n_conductivity_100kHz = count(cond),
            n_conductivity_zero = count(xyz .& isfinite.(s.conductivity_S_m_100kHz) .&
                                        (s.conductivity_S_m_100kHz .== 0)))
end

"""
    cloncurry_grade_eligible(g, s) -> BitVector

Cu samples that fall inside `g`. Used as the eligible mask for
[`spatial_holdout`](@ref) and [`group_holdout`](@ref).
"""
function cloncurry_grade_eligible(g::PriorGrid, s::CloncurrySamples)
    n = length(s)
    mask = falses(n)
    @inbounds for i in 1:n
        isfinite(s.cu_ppm[i]) || continue
        isfinite(s.east[i]) && isfinite(s.north[i]) && isfinite(s.elev[i]) || continue
        containing_cell(g, s.east[i], s.north[i], s.elev[i]) == 0 && continue
        mask[i] = true
    end
    return mask
end

"""
    spatial_holdout(x, y, z, eligible; fraction=0.2, buffer=300, rng)

Split eligible sample indices into `train`, `test`, and `buffer`.

Test points are accepted in shuffled order only if they sit at least `buffer`
metres (3-D Euclidean) from every test point already chosen, so the hold-out
is not one spatial cluster. Any remaining eligible point within `buffer` of a
test point is dropped — it is not used for training or scoring. That is the
spatial buffer: a train assay cannot sit a few cells away from a hidden one.
"""
function spatial_holdout(x::AbstractVector, y::AbstractVector, z::AbstractVector,
                         eligible::AbstractVector{Bool};
                         fraction::Real = 0.2,
                         buffer::Real = 300.0,
                         rng::AbstractRNG = Random.default_rng())
    n = length(x)
    (length(y) == n && length(z) == n && length(eligible) == n) ||
        throw(ArgumentError("spatial_holdout: x, y, z, eligible must match length"))
    (0 < fraction < 1) || throw(ArgumentError(
        "spatial_holdout: fraction must be in (0, 1), got $fraction"))
    buffer >= 0 || throw(ArgumentError("spatial_holdout: buffer must be ≥ 0"))
    cand = findall(eligible)
    isempty(cand) && throw(ArgumentError("spatial_holdout: no eligible samples"))
    order = copy(cand)
    shuffle!(rng, order)
    n_test_target = max(1, round(Int, fraction * length(cand)))
    test = Int[]
    sizehint!(test, n_test_target)
    for i in order
        ok = true
        @inbounds for j in test
            if hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) < buffer
                ok = false
                break
            end
        end
        ok || continue
        push!(test, i)
        length(test) >= n_test_target && break
    end
    isempty(test) && throw(ArgumentError(
        "spatial_holdout: buffer $(buffer) m blocked every test candidate"))
    test_set = Set(test)
    train = Int[]
    buffer_zone = Int[]
    for i in cand
        i in test_set && continue
        near = false
        @inbounds for j in test
            if hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) < buffer
                near = true
                break
            end
        end
        if near
            push!(buffer_zone, i)
        else
            push!(train, i)
        end
    end
    isempty(train) && throw(ArgumentError(
        "spatial_holdout: buffer $(buffer) m left no training samples " *
        "(eligible=$(length(cand)), test=$(length(test)))"))
    return (train = sort!(train),
            test = sort!(test),
            buffer = sort!(buffer_zone),
            n_eligible = length(cand),
            n_test_target = n_test_target,
            buffer_m = Float64(buffer),
            fraction = Float64(fraction))
end

"""
    cloncurry_group_keys(drillhole) -> Vector{String}

Hold-out group id per sample. A blank `drillhole` does not join a shared
"empty" bucket — each such row is `__ungrouped_<row>`.
"""
function cloncurry_group_keys(drillhole::AbstractVector{<:AbstractString})
    n = length(drillhole)
    keys = Vector{String}(undef, n)
    @inbounds for i in 1:n
        g = strip(String(drillhole[i]))
        keys[i] = isempty(g) ? "__ungrouped_$i" : g
    end
    return keys
end

function _group_count_targets(n_g::Int, fractions::NTuple{3,Real})
    t = [round(Int, n_g * f) for f in fractions]
    t[1] = max(1, t[1])
    if n_g >= 3
        t[2] = max(1, t[2])
        t[3] = max(1, t[3])
    end
    return _scale_split_targets(n_g, (t[1], t[2], t[3]))
end

function _scale_split_targets(n::Int, targets)
    t = [Int(x) for x in targets]
    length(t) == 3 || throw(ArgumentError(
        "group_holdout: targets must be (train, val, test)"))
    s = sum(t)
    s > 0 || throw(ArgumentError("group_holdout: targets must sum to a positive count"))
    if s != n
        t = [round(Int, n * x / s) for x in t]
        t[1] += n - sum(t)
    end
    for i in 1:3
        t[i] < 0 && (t[i] = 0)
    end
    t[1] += n - sum(t)
    return t
end

function _greedy_partition(sizes::Vector{Int}, targets::Vector{Int})
    n = length(sizes)
    assign = ones(Int, n)
    counts = zeros(Int, 3)
    for i in sortperm(sizes; rev = true)
        remaining = targets .- counts
        s = argmax(remaining)
        assign[i] = s
        counts[s] += sizes[i]
    end
    err = abs(counts[1] - targets[1]) + abs(counts[2] - targets[2]) +
          abs(counts[3] - targets[3])
    return assign, err, counts
end

# Assign whole groups to train/val/test. For ≤12 groups, search all 3^n
# assignments and keep the lexicographically smallest (test names, then val,
# then train) among minimum L1 distance to `targets`. That is how the
# 191/45/19 Cu draft is realized as two test holes, not 191 holes.
function _partition_groups(names::Vector{String}, sizes::Vector{Int},
                           targets::Vector{Int})
    n = length(names)
    n == length(sizes) || throw(ArgumentError(
        "_partition_groups: names and sizes must match"))
    length(targets) == 3 || throw(ArgumentError(
        "_partition_groups: need 3 targets"))
    n == 0 && throw(ArgumentError("_partition_groups: no groups"))
    n > 12 && return _greedy_partition(sizes, targets)

    best_err = typemax(Int)
    best_nonempty = false
    best_assign = ones(Int, n)
    best_counts = zeros(Int, 3)
    best_key = nothing
    assign = Vector{Int}(undef, n)
    counts = zeros(Int, 3)
    nmax = 3^n
    for code in 0:(nmax - 1)
        fill!(counts, 0)
        c = code
        @inbounds for i in 1:n
            s = (c % 3) + 1
            c = div(c, 3)
            assign[i] = s
            counts[s] += sizes[i]
        end
        err = abs(counts[1] - targets[1]) + abs(counts[2] - targets[2]) +
              abs(counts[3] - targets[3])
        nonempty = true
        @inbounds for s in 1:3
            if targets[s] > 0 && counts[s] == 0
                nonempty = false
                break
            end
        end
        key = (
            String[names[i] for i in 1:n if assign[i] == 3],
            String[names[i] for i in 1:n if assign[i] == 2],
            String[names[i] for i in 1:n if assign[i] == 1],
        )
        better = false
        if nonempty && !best_nonempty
            better = true
        elseif nonempty == best_nonempty
            if err < best_err
                better = true
            elseif err == best_err && (best_key === nothing || key < best_key)
                better = true
            end
        end
        if better
            best_err = err
            best_nonempty = nonempty
            best_assign = copy(assign)
            best_counts = copy(counts)
            best_key = key
        end
    end
    return best_assign, best_err, best_counts
end

"""
    group_holdout(groups, eligible; unit=:samples, targets=nothing,
                  fractions=(0.70, 0.15, 0.15), rng)

Split eligible sample indices into `train`, `val`, and `test` by group id.

Every sample that shares a group (a drillhole id) goes to the same split.
Empty group ids are unique per row via [`cloncurry_group_keys`](@ref).

`unit = :samples` (default) matches **sample counts** to `targets` or
`fractions` (the Ernest Henry 191/45/19 draft is this mode with
`targets = (191, 45, 19)`). `unit = :groups` matches **hole counts**
(~70/15/15 of 121 collars → ~85/18/18). With more than 12 groups,
`:samples` is a greedy fill; `:groups` shuffles collars then cuts.
"""
function group_holdout(groups::AbstractVector{<:AbstractString},
                       eligible::AbstractVector{Bool};
                       unit::Symbol = :samples,
                       targets::Union{Nothing,NTuple{3,Integer}} = nothing,
                       fractions::NTuple{3,Real} = (0.70, 0.15, 0.15),
                       rng::AbstractRNG = Random.default_rng())
    rng isa AbstractRNG || throw(ArgumentError(
        "group_holdout: rng must be an AbstractRNG"))
    (unit === :samples || unit === :groups) || throw(ArgumentError(
        "group_holdout: unit must be :samples or :groups, got $(repr(unit))"))
    n = length(groups)
    length(eligible) == n || throw(ArgumentError(
        "group_holdout: groups and eligible must match length"))
    cand = findall(eligible)
    isempty(cand) && throw(ArgumentError("group_holdout: no eligible samples"))
    gkeys = cloncurry_group_keys(groups)
    grp_to_idx = Dict{String,Vector{Int}}()
    for i in cand
        push!(get!(grp_to_idx, gkeys[i], Int[]), i)
    end
    names = sort!(collect(Base.keys(grp_to_idx)))
    if unit === :groups
        n_g = length(names)
        t = targets === nothing ? _group_count_targets(n_g, fractions) :
            _scale_split_targets(n_g, targets)
        order = copy(names)
        shuffle!(rng, order)
        n_test = t[3]
        n_val = t[2]
        test_names = order[1:n_test]
        val_names = order[(n_test + 1):(n_test + n_val)]
        train_names = order[(n_test + n_val + 1):end]
        role = Dict{String,Int}()
        for h in train_names
            role[h] = 1
        end
        for h in val_names
            role[h] = 2
        end
        for h in test_names
            role[h] = 3
        end
        buckets = [Int[], Int[], Int[]]
        role_names = [String[], String[], String[]]
        counts = zeros(Int, 3)
        for name in names
            s = role[name]
            append!(buckets[s], grp_to_idx[name])
            push!(role_names[s], name)
            counts[s] += length(grp_to_idx[name])
        end
        err = abs(length(role_names[1]) - t[1]) + abs(length(role_names[2]) - t[2]) +
              abs(length(role_names[3]) - t[3])
        return (train = sort!(buckets[1]),
                val = sort!(buckets[2]),
                test = sort!(buckets[3]),
                buffer = Int[],
                n_eligible = length(cand),
                n_groups = n_g,
                n_train_groups = length(role_names[1]),
                n_val_groups = length(role_names[2]),
                n_test_groups = length(role_names[3]),
                groups_train = role_names[1],
                groups_val = role_names[2],
                groups_test = role_names[3],
                targets = t,
                counts = counts,
                assignment_err = err,
                unit = :groups)
    end
    if length(names) > 12
        shuffle!(rng, names)
    end
    sizes = [length(grp_to_idx[name]) for name in names]
    t = targets === nothing ?
        begin
            tf = [round(Int, length(cand) * f) for f in fractions]
            tf[1] += length(cand) - sum(tf)
            tf
        end : _scale_split_targets(length(cand), targets)
    assign, err, counts = _partition_groups(names, sizes, t)
    buckets = [Int[], Int[], Int[]]
    role_names = [String[], String[], String[]]
    for (name, s) in zip(names, assign)
        append!(buckets[s], grp_to_idx[name])
        push!(role_names[s], name)
    end
    return (train = sort!(buckets[1]),
            val = sort!(buckets[2]),
            test = sort!(buckets[3]),
            buffer = Int[],
            n_eligible = length(cand),
            n_groups = length(names),
            n_train_groups = length(role_names[1]),
            n_val_groups = length(role_names[2]),
            n_test_groups = length(role_names[3]),
            groups_train = role_names[1],
            groups_val = role_names[2],
            groups_test = role_names[3],
            targets = t,
            counts = counts,
            assignment_err = err,
            unit = :samples)
end

function group_mask(keys::AbstractVector{<:AbstractString},
                    groups::AbstractVector{<:AbstractString})
    allowed = Set(String.(groups))
    n = length(keys)
    mask = falses(n)
    @inbounds for i in 1:n
        mask[i] = keys[i] in allowed
    end
    return mask
end

"""
    load_cloncurry_anchors(g, s; grade_keep=nothing, keep=nothing) -> NamedTuple

Per-property anchors. A sample is an anchor for a property only when that
property is finite (density / susceptibility / conductivity_100kHz) or when
Cu is present (`<LOD` lifted to the detection limit). Cells with no mapped
samples are dropped by [`map_points_to_cells`](@ref).

`keep` hides every property on rows that are false (drillhole hold-out).
`grade_keep` hides Cu only, as in the older spatial hold-out. Both may be
set; a row is a grade anchor only when both masks allow it.
"""
function load_cloncurry_anchors(g::PriorGrid, s::CloncurrySamples;
                                cu_detection_limit::Real = CLONCURRY_CU_DL_PPM,
                                conductivity_floor::Real = CLONCURRY_COND_FLOOR_S_M,
                                grade_keep::Union{Nothing,AbstractVector{Bool}} = nothing,
                                keep::Union{Nothing,AbstractVector{Bool}} = nothing)
    n = length(s)
    if grade_keep !== nothing
        length(grade_keep) == n || throw(ArgumentError(
            "load_cloncurry_anchors: grade_keep length $(length(grade_keep)) ≠ $n"))
    end
    if keep !== nothing
        length(keep) == n || throw(ArgumentError(
            "load_cloncurry_anchors: keep length $(length(keep)) ≠ $n"))
    end
    grade = Vector{Float64}(undef, n)
    dens = Vector{Float64}(undef, n)
    susc = Vector{Float64}(undef, n)
    cond = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        allowed = keep === nothing || keep[i]
        gval = cu_log10(s.cu_ppm[i]; detection_limit = cu_detection_limit)
        if !allowed || (grade_keep !== nothing && !grade_keep[i])
            gval = NaN
        end
        grade[i] = gval
        if !allowed
            dens[i] = NaN
            susc[i] = NaN
            cond[i] = NaN
        else
            d = s.density_g_cm3[i]
            dens[i] = isfinite(d) && d > 0 ? d : NaN
            susc[i] = _log10_positive(s.susceptibility_SI[i])
            cond[i] = _log10_conductivity(s.conductivity_S_m_100kHz[i];
                                          floor = conductivity_floor)
        end
    end
    grade_a = map_points_to_cells(g, s.east, s.north, s.elev, grade)
    dens_a = map_points_to_cells(g, s.east, s.north, s.elev, dens)
    susc_a = map_points_to_cells(g, s.east, s.north, s.elev, susc)
    cond_a = map_points_to_cells(g, s.east, s.north, s.elev, cond)
    cov = cloncurry_property_coverage(s)
    n_mapped(vals) = count(i -> containing_cell(g, s.east[i], s.north[i], s.elev[i]) != 0 &&
                                isfinite(vals[i]), 1:n)
    stats = merge(cov, (
        n_grade_cells = length(grade_a[1]),
        n_density_cells = length(dens_a[1]),
        n_susceptibility_cells = length(susc_a[1]),
        n_conductivity_100kHz_cells = length(cond_a[1]),
        n_grade_mapped = n_mapped(grade),
        n_density_mapped = n_mapped(dens),
        n_susceptibility_mapped = n_mapped(susc),
        n_conductivity_100kHz_mapped = n_mapped(cond),
    ))
    return (grade = grade_a, density = dens_a, susceptibility = susc_a,
            conductivity_100kHz = cond_a, stats = stats)
end

#---------- structural geology (GeoJSON → MGA54 channels) ----------
# geology/structures.geojson and geology/surface_geology.geojson ship as
# WGS84 lon/lat (download outSR=4326). Distances and PIP run in EPSG:28354
# metres so they match petrophysics easting/northing. Fold / Layering /
# MiscLineFeature are not separate channels — they only feed the all-lines
# distance. Surface map one-hots sit beside drillhole lith_* channels.

function cloncurry_geology_dir(dataset_root::AbstractString)
    d = joinpath(dataset_root, "geology")
    isdir(d) || throw(ArgumentError(
        "cloncurry_geology_dir: no geology/ under $dataset_root"))
    return d
end

function cloncurry_structures_path(dataset_root::AbstractString)
    p = joinpath(cloncurry_geology_dir(dataset_root), "structures.geojson")
    isfile(p) || throw(ArgumentError(
        "cloncurry_structures_path: no such file: $p"))
    return p
end

function cloncurry_surface_geology_path(dataset_root::AbstractString)
    p = joinpath(cloncurry_geology_dir(dataset_root), "surface_geology.geojson")
    isfile(p) || throw(ArgumentError(
        "cloncurry_surface_geology_path: no such file: $p"))
    return p
end

function _cloncurry_mga54_transform(f)
    src = ArchGDAL.importEPSG(4326)
    ArchGDAL.maybesetaxisorder!(src, :trad)
    dst = ArchGDAL.importEPSG(28354)
    return ArchGDAL.createcoordtrans(src, dst) do ct
        f(ct)
    end
end

function _geom_xy_mga54(geom, ct)
    g = ArchGDAL.clone(geom)
    ArchGDAL.transform!(g, ct)
    n = ArchGDAL.ngeom(g)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    @inbounds for i in 0:(n - 1)
        x, y, _ = ArchGDAL.getpoint(g, i)
        xs[i + 1] = x
        ys[i + 1] = y
    end
    return xs, ys
end

"""
    load_cloncurry_structures(path) -> Vector{NamedTuple}

LineString structural features in GDA94 / MGA zone 54 metres. Each entry is
`(type, x, y)` with `type` from the GeoJSON `type` property (Contact, Fault,
Fold, MiscLineFeature, Layering).
"""
function load_cloncurry_structures(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "load_cloncurry_structures: no such file: $path"))
    lines = NamedTuple{(:type, :x, :y),Tuple{String,Vector{Float64},Vector{Float64}}}[]
    ArchGDAL.read(path) do ds
        layer = ArchGDAL.getlayer(ds, 0)
        _cloncurry_mga54_transform() do ct
            for feat in layer
                raw = ArchGDAL.getfield(feat, "type")
                typ = raw === nothing ? "" : strip(String(raw))
                isempty(typ) && continue
                geom = ArchGDAL.getgeom(feat)
                geom === nothing && continue
                xs, ys = _geom_xy_mga54(geom, ct)
                length(xs) < 2 && continue
                push!(lines, (type = typ, x = xs, y = ys))
            end
        end
    end
    isempty(lines) && throw(ArgumentError(
        "load_cloncurry_structures: no LineString features in $path"))
    return lines
end

"""
    load_cloncurry_surface_geology(path) -> Vector{NamedTuple}

Polygon surface-geology units in MGA54 metres. Each entry is
`(dom_rock, rock_type, rings)` where `rings` is a vector of `(x, y)` rings:
index 1 is the exterior, the rest are holes.
"""
function load_cloncurry_surface_geology(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "load_cloncurry_surface_geology: no such file: $path"))
    polys = NamedTuple{(:dom_rock, :rock_type, :rings),
                       Tuple{String,String,Vector{Tuple{Vector{Float64},Vector{Float64}}}}}[]
    ArchGDAL.read(path) do ds
        layer = ArchGDAL.getlayer(ds, 0)
        _cloncurry_mga54_transform() do ct
            for feat in layer
                dom = ArchGDAL.getfield(feat, "dom_rock")
                rock = ArchGDAL.getfield(feat, "rock_type")
                dom_s = dom === nothing ? "" : strip(String(dom))
                rock_s = rock === nothing ? "" : strip(String(rock))
                (isempty(dom_s) && isempty(rock_s)) && continue
                geom = ArchGDAL.getgeom(feat)
                geom === nothing && continue
                nring = ArchGDAL.ngeom(geom)
                nring == 0 && continue
                rings = Tuple{Vector{Float64},Vector{Float64}}[]
                for r in 0:(nring - 1)
                    ring = ArchGDAL.getgeom(geom, r)
                    xs, ys = _geom_xy_mga54(ring, ct)
                    length(xs) < 3 && continue
                    push!(rings, (xs, ys))
                end
                isempty(rings) && continue
                push!(polys, (dom_rock = dom_s, rock_type = rock_s, rings = rings))
            end
        end
    end
    isempty(polys) && throw(ArgumentError(
        "load_cloncurry_surface_geology: no Polygon features in $path"))
    return polys
end

function _point_segment_dist2(px::Float64, py::Float64,
                              ax::Float64, ay::Float64,
                              bx::Float64, by::Float64)
    dx = bx - ax
    dy = by - ay
    len2 = dx * dx + dy * dy
    if len2 == 0
        ex = px - ax
        ey = py - ay
        return ex * ex + ey * ey
    end
    t = ((px - ax) * dx + (py - ay) * dy) / len2
    t = clamp(t, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    ex = px - qx
    ey = py - qy
    return ex * ex + ey * ey
end

function _min_line_dist2(px::Float64, py::Float64,
                         lines::AbstractVector{<:NamedTuple},
                         mask::Union{Nothing,Function} = nothing)
    best = Inf
    @inbounds for line in lines
        mask !== nothing && !mask(line.type) && continue
        xs = line.x
        ys = line.y
        for i in 1:(length(xs) - 1)
            d2 = _point_segment_dist2(px, py, xs[i], ys[i], xs[i + 1], ys[i + 1])
            d2 < best && (best = d2)
        end
    end
    return best
end

# Even-odd ray cast. Rings may be closed (first == last); that is fine.
function _point_in_ring(px::Float64, py::Float64,
                        xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    n = length(xs)
    n == length(ys) || throw(ArgumentError("_point_in_ring: x/y length mismatch"))
    n < 3 && return false
    inside = false
    j = n
    @inbounds for i in 1:n
        yi = Float64(ys[i]); yj = Float64(ys[j])
        xi = Float64(xs[i]); xj = Float64(xs[j])
        if (yi > py) != (yj > py)
            xcross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < xcross
                inside = !inside
            end
        end
        j = i
    end
    return inside
end

function _point_in_polygon(px::Float64, py::Float64,
                           rings::AbstractVector{<:Tuple})
    isempty(rings) && return false
    xs0, ys0 = rings[1]
    _point_in_ring(px, py, xs0, ys0) || return false
    @inbounds for r in 2:length(rings)
        xs, ys = rings[r]
        _point_in_ring(px, py, xs, ys) && return false
    end
    return true
end

function _distance_map(g::PriorGrid, lines, mask)
    nx, ny = size(g, 1), size(g, 2)
    near = Array{Float64}(undef, nx, ny)
    h = g.h_median
    @inbounds for j in 1:ny, i in 1:nx
        d2 = _min_line_dist2(g.cx[i], g.cy[j], lines, mask)
        near[i, j] = isfinite(d2) ? log1p(sqrt(d2) / h) : 0.0
    end
    return near
end

"""
    structure_distance_channels(g, source) -> (chans, names)

Plan-view distance channels (extruded through depth), matching
[`coverage_channels`](@ref): `log1p(d / h_median)` in metres under GDA94/MGA54.

- `struct_fault_distance` — nearest Fault line (IOCG structural control)
- `struct_line_distance` — nearest structural line of any type (Contact, Fault,
  Fold, MiscLineFeature, Layering)

`source` is a dataset root (uses `geology/structures.geojson`), a GeoJSON path,
or a pre-loaded vector from [`load_cloncurry_structures`](@ref).
"""
function structure_distance_channels(g::PriorGrid, source)
    lines = if source isa AbstractString
        path = isdir(source) ? cloncurry_structures_path(source) : String(source)
        load_cloncurry_structures(path)
    else
        source
    end
    isempty(lines) && throw(ArgumentError(
        "structure_distance_channels: no structural lines"))
    any(l -> l.type == "Fault", lines) || throw(ArgumentError(
        "structure_distance_channels: no Fault lines in the collection"))
    nz = size(g, 3)
    fault = _distance_map(g, lines, t -> t == "Fault")
    any_line = _distance_map(g, lines, nothing)
    return Array{Float64,3}[extrude(fault, nz), extrude(any_line, nz)],
           String["struct_fault_distance", "struct_line_distance"]
end

"""
    surface_geology_channels(g, source) -> (chans, names)

Point-in-polygon one-hot channels from the 1:500k surface geology map.
`surf_dom_*` encodes `dom_rock` and `surf_rock_*` encodes `rock_type`. Cells
outside every polygon get the matching `*_OTHER` bit. These are *added*
alongside drillhole [`lithology_channels`](@ref), not a replacement.

`source` is a dataset root, a GeoJSON path, or a pre-loaded vector from
[`load_cloncurry_surface_geology`](@ref).
"""
function surface_geology_channels(g::PriorGrid, source)
    polys = if source isa AbstractString
        path = isdir(source) ? cloncurry_surface_geology_path(source) : String(source)
        load_cloncurry_surface_geology(path)
    else
        source
    end
    isempty(polys) && throw(ArgumentError(
        "surface_geology_channels: no surface-geology polygons"))

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
    dom_index = Dict(c => i for (i, c) in enumerate(dom_classes))
    rock_index = Dict(c => i for (i, c) in enumerate(rock_classes))

    nx, ny, nz = size(g)
    nd = length(dom_classes)
    nr = length(rock_classes)
    dom_maps = [zeros(nx, ny) for _ in 1:nd]
    rock_maps = [zeros(nx, ny) for _ in 1:nr]
    @inbounds for j in 1:ny, i in 1:nx
        px = g.cx[i]; py = g.cy[j]
        hit = 0
        for (k, p) in enumerate(polys)
            _point_in_polygon(px, py, p.rings) || continue
            hit = k
            break
        end
        if hit == 0
            dom_maps[dom_index["OTHER"]][i, j] = 1.0
            rock_maps[rock_index["OTHER"]][i, j] = 1.0
        else
            p = polys[hit]
            dtok = isempty(p.dom_rock) ? "OTHER" : _lith_token(p.dom_rock)
            rtok = isempty(p.rock_type) ? "OTHER" : _lith_token(p.rock_type)
            haskey(dom_index, dtok) || (dtok = "OTHER")
            haskey(rock_index, rtok) || (rtok = "OTHER")
            dom_maps[dom_index[dtok]][i, j] = 1.0
            rock_maps[rock_index[rtok]][i, j] = 1.0
        end
    end

    chans = Array{Float64,3}[]
    names = String[]
    for (c, m) in zip(dom_classes, dom_maps)
        push!(chans, extrude(m, nz))
        push!(names, "surf_dom_" * c)
    end
    for (c, m) in zip(rock_classes, rock_maps)
        push!(chans, extrude(m, nz))
        push!(names, "surf_rock_" * c)
    end
    return chans, names
end

#---------- SampleTable adapter ----------
# The training path above still lifts Cu "<LOD" to 1 ppm and, at log10 time,
# conductivity zeros to 0.01 S/m. This adapter does not. A "<LOD" token, and
# a non-positive number under a data-driven limit policy, is an upper bound.
# The bound is the policy's limit (a percentile of the positive measurements,
# their minimum, or a configured constant), chosen on the whole column.

const _SAMPLE_PROPERTY_ORDER = ("cu", "density", "susceptibility", "conductivity", "lithology")

function _ordered_property_names(props::AbstractDict)
    names = String[]
    for n in _SAMPLE_PROPERTY_ORDER
        haskey(props, n) && push!(names, n)
    end
    extras = sort!(String[string(k) for k in keys(props) if !(string(k) in names)])
    append!(names, extras)
    return names
end

function _cfg_data_path(default::Function, root::AbstractString, cfg, key::AbstractString)
    if cfg isa AbstractDict && haskey(cfg, key)
        p = strip(String(cfg[key]))
        if !isempty(p)
            return isabspath(p) ? p : normpath(joinpath(root, p))
        end
    end
    return String(default())
end

function _filled_label(raw, i::Int, blank_prefix::AbstractString)
    t = strip(String(raw))
    if isempty(t) || lowercase(t) == "missing"
        return String(blank_prefix) * string(i)
    end
    return t
end

function _spec_from_property(name::AbstractString, p::AbstractDict)
    kind = Symbol(String(get(p, "kind", "continuous")))
    transform = Symbol(String(get(p, "transform", "identity")))
    unit = String(get(p, "unit", ""))
    return PropertySpec(Symbol(name), kind, transform, unit)
end

# Percentile is in percent: 1 means the 1st percentile (quantile 0.01).
# Statistics.quantile is Hyndman–Fan type 7.
const _DEFAULT_LOD_PERCENTILE = 1.0

function _policy_limit(positive::Vector{Float64}, policy::AbstractString,
                       p::AbstractDict, prop::Symbol)
    isempty(positive) && throw(ArgumentError(
        "cloncurry_sample_table: $(prop) has censored rows but no positive " *
        "measurement to set the limit"))
    if policy == "min_positive" || (policy == "" && prop === :susceptibility)
        lod = minimum(positive)
        @info "detection limit is the smallest positive measured value" property=prop lod n_positive=length(positive) column=String(get(p, "column", ""))
        return lod
    elseif policy == "percentile"
        pct = haskey(p, "percentile") ? Float64(p["percentile"]) : _DEFAULT_LOD_PERCENTILE
        (0 < pct < 100) || throw(ArgumentError(
            "cloncurry_sample_table: percentile for $(prop) must be in (0, 100), got $pct"))
        lod = quantile(positive, pct / 100)
        @info "detection limit is a percentile of the positive measurements" property=prop percentile=pct lod min_positive=minimum(positive) n_positive=length(positive) column=String(get(p, "column", ""))
        return lod
    elseif policy == "fixed"
        haskey(p, "lod") || throw(ArgumentError(
            "cloncurry_sample_table: lod_policy \"fixed\" needs a lod value for $(prop)"))
        lod = Float64(p["lod"])
        lod > 0 || throw(ArgumentError(
            "cloncurry_sample_table: lod for $(prop) must be positive, got $lod"))
        @info "detection limit taken from the site file" property=prop lod column=String(get(p, "column", ""))
        return lod
    elseif policy == "instrument_floor"
        floor = haskey(p, "floor") ? Float64(p["floor"]) : CLONCURRY_COND_FLOOR_S_M
        floor > 0 || throw(ArgumentError(
            "cloncurry_sample_table: instrument floor for $(prop) must be positive, got $floor"))
        @info "exact zeros stored at the instrument floor and marked censored" property=prop floor column=String(get(p, "column", ""))
        return floor
    else
        throw(ArgumentError(
            "cloncurry_sample_table: unknown lod_policy $(repr(policy)) for $(prop). " *
            "Use percentile, min_positive, fixed, or instrument_floor"))
    end
end

function _continuous_column(raw::Vector{String}, prop::Symbol, p::AbstractDict)
    policy = haskey(p, "lod_policy") ? strip(String(p["lod_policy"])) : ""
    if prop === :susceptibility && !isempty(policy) && policy != "min_positive"
        throw(ArgumentError(
            "cloncurry_sample_table: non-positive susceptibility is censored at " *
            "the smallest positive value; lod_policy must be \"min_positive\" or omitted"))
    end
    n = length(raw)
    values = fill(NaN, n)
    cens = falses(n)
    positive = Float64[]
    n_nonpositive = 0
    data_limit = policy in ("min_positive", "percentile", "instrument_floor") ||
                 prop === :susceptibility
    for i in 1:n
        t = strip(raw[i])
        if isempty(t)
            continue
        elseif _is_lod(t)
            if !data_limit && policy != "fixed"
                throw(ArgumentError(
                    "cloncurry_sample_table: $(prop) row $i is a detection-limit " *
                    "token but the property has no lod_policy"))
            end
            cens[i] = true
        else
            v = _parse_float(t)
            isfinite(v) || continue
            # Susceptibility: every finite ≤ 0 is an upper bound.
            # Other columns: ≤ 0 is an upper bound under a data-driven policy;
            # instrument_floor censors exact zeros only.
            nonpos = if prop === :susceptibility
                v <= 0
            elseif policy == "instrument_floor"
                v == 0
            elseif policy in ("min_positive", "percentile")
                v <= 0
            else
                false
            end
            if nonpos
                cens[i] = true
                n_nonpositive += 1
            else
                values[i] = v
                v > 0 && push!(positive, v)
            end
        end
    end
    if prop === :susceptibility && n_nonpositive > 0
        @info "non-positive susceptibility marked censored" property=prop n_nonpositive n_positive=length(positive) column=String(get(p, "column", ""))
    end
    any(cens) || return values, cens
    lod = _policy_limit(positive, isempty(policy) ? "min_positive" : policy, p, prop)
    values[cens] .= lod
    return values, cens
end

"""
    cloncurry_sample_table(root, cfg) -> SampleTable

Cloncurry petrophysics + pXRF rows as a [`SampleTable`](@ref).

`cfg` is a parsed site file. Continuous numbers stay in the file's unit
(`PropertySpec.transform` is not applied). Cu `"<LOD"` and a conductivity of
exactly 0 are upper bounds (`censored = true`). The bound is the column's
`lod_policy` on the unfiltered file: `"percentile"` (default 1, the 1st
percentile of positive values), `"min_positive"`, `"fixed"` (`lod`), or
`"instrument_floor"` (`floor`). Finite susceptibility ≤ 0 is an upper bound
at the smallest positive susceptibility. The historical 1 ppm / 0.01 S/m
substitutions are not used unless the site file asks for them.

Blank drillhole ids become `__ungrouped_<row>` and blank deposits
`__ungrouped_deposit_<row>`, using the source-file row number, so `hole`
and `group` are filled before an optional `deposit` filter. That filter
keeps rows whose deposit label matches; the detection limit is chosen on
the unfiltered column.
"""
function cloncurry_sample_table(root::AbstractString, cfg::AbstractDict)
    petro_path = _cfg_data_path(root, cfg, "petrophysics") do
        cloncurry_petrophysics_path(root)
    end
    metals_path = _cfg_data_path(root, cfg, "metals") do
        cloncurry_metals_path(root)
    end
    # Alignment, unit, and geochem checks stay with the existing loader.
    # Its Cu substitution is ignored; the columns below are reread as text.
    samples = load_cloncurry_samples(petro_path, metals_path)
    n = length(samples)
    haskey(cfg, "properties") || throw(ArgumentError(
        "cloncurry_sample_table: site file has no [properties]"))
    props = cfg["properties"]
    props isa AbstractDict || throw(ArgumentError(
        "cloncurry_sample_table: [properties] must be a table"))
    names = _ordered_property_names(props)
    isempty(names) && throw(ArgumentError(
        "cloncurry_sample_table: no properties in the site file"))

    petro_cols = String[]
    metal_cols = String[]
    column_of = Dict{String,String}()
    parsed = Dict{String,Any}()
    for name in names
        p = props[name]
        p isa AbstractDict || throw(ArgumentError(
            "cloncurry_sample_table: properties.$name must be a table"))
        haskey(p, "column") || throw(ArgumentError(
            "cloncurry_sample_table: properties.$name has no column"))
        parsed[name] = p
        col = String(p["column"])
        column_of[name] = col
        if endswith(col, "_Concentration")
            push!(metal_cols, col)
        else
            push!(petro_cols, col)
        end
    end
    unique!(petro_cols)
    unique!(metal_cols)

    petro_raw, _ = _read_csv_selected(petro_path, unique!(vcat(["sample"], petro_cols)))
    petro_raw["sample"] == samples.sample || throw(ArgumentError(
        "cloncurry_sample_table: petrophysics reread is not row-aligned with the loader"))
    metal_raw = Dict{String,Vector{String}}()
    if !isempty(metal_cols)
        metal_raw, _ = _read_csv_selected(metals_path, unique!(vcat(["Sample"], metal_cols)))
        metal_raw["Sample"] == samples.sample || throw(ArgumentError(
            "cloncurry_sample_table: metals reread is not row-aligned with the loader"))
    end

    specs = PropertySpec[]
    values = Dict{Symbol,Vector{Float64}}()
    censored = Dict{Symbol,BitVector}()
    classes = Dict{Symbol,Vector{String}}()
    for name in names
        p = parsed[name]
        spec = _spec_from_property(name, p)
        push!(specs, spec)
        col = column_of[name]
        raw = endswith(col, "_Concentration") ? metal_raw[col] : petro_raw[col]
        length(raw) == n || throw(ArgumentError(
            "cloncurry_sample_table: column $(col) has $(length(raw)) rows, expected $n"))
        if spec.kind === :categorical
            haskey(p, "lod_policy") && throw(ArgumentError(
                "cloncurry_sample_table: categorical $(name) cannot have an lod_policy"))
            classes[spec.name] = [strip(s) for s in raw]
        elseif spec.kind === :continuous
            vals, bits = _continuous_column(raw, spec.name, p)
            values[spec.name] = vals
            censored[spec.name] = bits
        else
            throw(ArgumentError(
                "cloncurry_sample_table: unsupported kind $(repr(spec.kind))"))
        end
    end

    holes = [_filled_label(samples.drillhole[i], i, "__ungrouped_") for i in 1:n]
    groups = [_filled_label(samples.deposit[i], i, "__ungrouped_deposit_") for i in 1:n]
    table = SampleTable(samples.east, samples.north, samples.elev,
                         holes, groups, values, censored, classes, specs)
    if haskey(cfg, "deposit") && !isempty(strip(String(cfg["deposit"])))
        dep = strip(String(cfg["deposit"]))
        table = subset(table, table.group .== dep)
        nsamples(table) > 0 || throw(ArgumentError(
            "cloncurry_sample_table: no rows with deposit $(repr(dep))"))
    end
    return table
end
