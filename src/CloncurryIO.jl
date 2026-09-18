# Cloncurry–Ernest Henry (METAL / GDA94 MGA zone 54) readers for the
# non-geophysical prior line. Gravity, magnetics and MT are not opened here
# even though they sit in the same package; the network sees pXRF geochemistry
# and lithology as features, and Cu plus petrophysics as anchors.
#
# Coordinate convention matches KeivitsaIO: x = easting, y = northing,
# z = elevation (positive up, metres ASL). EPSG:28354.

# pXRF *_Concentration columns from Mg to U. Cu is the grade *target* and is
# excluded from the feature list (same leakage rule as Keivitsa drill Cu).
# LE_Concentration is a light-element composite, not an element, and is dropped.
const CLONCURRY_GEOCHEM_ELEMENTS = (
    "Mg", "Al", "Si", "P", "S", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Zn", "As", "Se", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd",
    "Sn", "Sb", "W", "Hg", "Pb", "Bi", "Th", "U",
)
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
`<LOD`). Geochemistry columns never include Cu.
"""
struct CloncurrySamples
    sample::Vector{String}
    deposit::Vector{String}
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
                                  optional = ["density_mean_kg_m3"])
    n = length(petro["sample"])
    n > 0 || throw(ArgumentError("load_cloncurry_samples: no petrophysics rows"))

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

    cu_col = "Cu_Concentration"
    cu_col in conc_cols || throw(ArgumentError(
        "load_cloncurry_samples: metals file has no Cu_Concentration"))
    cu = [_parse_cu_target(s; detection_limit = cu_detection_limit) for s in metal[cu_col]]

    return CloncurrySamples(
        petro["sample"],
        petro["deposit"],
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

function _finite_xyz(s::CloncurrySamples)
    keep = isfinite.(s.east) .& isfinite.(s.north) .& isfinite.(s.elev)
    return keep
end

"""
    cloncurry_geochemistry(s) -> PointSamples

pXRF concentrations (ppm) with Cu held out. Empty / `<LOD` entries are NaN
so each channel interpolates independently.
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
[`spatial_holdout`](@ref).
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
    load_cloncurry_anchors(g, s; grade_keep=nothing) -> NamedTuple

Per-property anchors. A sample is an anchor for a property only when that
property is finite (density / susceptibility / conductivity_100kHz) or when
Cu is present (`<LOD` lifted to the detection limit). Cells with no mapped
samples are dropped by [`map_points_to_cells`](@ref).

`grade_keep` hides Cu labels for a spatial hold-out: those rows become NaN
on the grade head only. Density / susceptibility / conductivity_100kHz are
unchanged.
"""
function load_cloncurry_anchors(g::PriorGrid, s::CloncurrySamples;
                                cu_detection_limit::Real = CLONCURRY_CU_DL_PPM,
                                conductivity_floor::Real = CLONCURRY_COND_FLOOR_S_M,
                                grade_keep::Union{Nothing,AbstractVector{Bool}} = nothing)
    n = length(s)
    if grade_keep !== nothing
        length(grade_keep) == n || throw(ArgumentError(
            "load_cloncurry_anchors: grade_keep length $(length(grade_keep)) ≠ $n"))
    end
    grade = Vector{Float64}(undef, n)
    dens = Vector{Float64}(undef, n)
    susc = Vector{Float64}(undef, n)
    cond = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        gval = cu_log10(s.cu_ppm[i]; detection_limit = cu_detection_limit)
        if grade_keep !== nothing && !grade_keep[i]
            gval = NaN
        end
        grade[i] = gval
        d = s.density_g_cm3[i]
        dens[i] = isfinite(d) && d > 0 ? d : NaN
        susc[i] = _log10_positive(s.susceptibility_SI[i])
        cond[i] = _log10_conductivity(s.conductivity_S_m_100kHz[i];
                                      floor = conductivity_floor)
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
