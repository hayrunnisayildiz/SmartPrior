# GTK Keivitsa (Finland Zone 3 / EPSG:2393) readers for the non-geophysical
# prior line. Gravity and MT are not used here; the network sees geochemistry
# and lithology as features, and drill assays plus petrophysics as anchors.
#
# Coordinate convention in this file matches `config.yaml` `train.grid_bounds`,
# not the WS3D "x north, y east, z down" frame:
#   x = KKJ easting, y = KKJ northing, z = elevation (positive up).

const KEIVITSA_GEOCHEM_PRIORITY = ("S", "FE", "CU", "NI", "CO", "CR", "PD", "PT", "AU")
const KEIVITSA_GEOCHEM_RATIOS = (("NI", "CU"), ("PD", "NI"), ("CO", "NI"))
const KEIVITSA_CU_DL_PPM = 1.0

# Column names PTR_D/J/K, DSR_D/J/K, LUO_R were searched across every
# 6_DESCRIPTION/*.doc (textutil) and report/**/*.txt, including
# 1_DRILLINGS/Liite_MRVK_3-O2_kairaus_GTK.doc and 3_GROUND_GEOPHYSICS/*.
# Hits: the names themselves appear only as petro.txt / petroph.dbf headers.
# Finnish terms: "tiheysmittaus" and "suskeptiivisuus"/"ominaisvastus" in the
# Liite are borehole-method lists, not a column dictionary; ground-geophysics
# "ominaisvastus" is VLF/IP apparent resistivity, not LUO_R; "tiheys" in Q22
# is magnetic flux density.
#
# Density (PTR_D/DSR_D) and susceptibility (PTR_J/DSR_J) are a high-confidence
# inference from GTK's standard combined density–susceptibility–remanence
# measurement (Puranen's method; national database ~130 000 samples):
# https://www.gtk.fi/tutkimusmenetelmat/yhdistetty-tiheys-magneettinen-suskeptibiliteetti-remanentti-magnetoituma-mittaus
# PTR_K/DSR_K are the unused remanence candidate (possible 5th anchor group).
# LUO_R stays unverified: no supporting source was found.
const KEIVITSA_PETRO_STATUS = (
    PTR_D = "high-confidence inference: density, kg/m³ (range ~2.8e3–3.6e3). GTK combined density–susceptibility–remanence is the national petrophysics baseline (Puranen; ~130000 samples; gtk.fi tutkimusmenetelmat). No Keivitsa column dictionary in report/; units from range.",
    DSR_D = "high-confidence inference: density, kg/m³, merged with PTR_D (same GTK density–susceptibility–remanence set)",
    PTR_J = "high-confidence inference: magnetic susceptibility (GTK J in the density–susceptibility–remanence set; Puranen national database). Liite lists suskeptiivisuus as a borehole method, not this column.",
    DSR_J = "high-confidence inference: susceptibility, merged with PTR_J",
    LUO_R = "unverified: inferred resistivity, Ω·m (range ~0.11–1.2e6); Liite lists ominaisvastus as a borehole method, not this column. No supporting source found for the LUO_R mapping.",
    PTR_K = "unused: possible remanent magnetization (third component of GTK's standard density/susceptibility/remanence set). Not yet in the model; candidate for a 5th anchor group.",
    DSR_K = "unused: possible remanent magnetization, merged-pair of PTR_K. Not yet in the model; candidate for a 5th anchor group.",
)

# Drill-core assays fill the surface-shapefile gaps (S, Fe, Cr, Pt). Overlap
# with till/bedrock (Cu, Ni, Co, Pd, Au) keeps the surface channel: till and
# core are different media, and drill Cu is already the grade target.
const KEIVITSA_DRILL_GEOCHEM = ("S", "FE", "CR", "PT")

#---------- small parsers ----------

function _parse_float(s::AbstractString)
    t = strip(s)
    isempty(t) && return NaN
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

function _parse_int_field(s::AbstractString)
    t = strip(s)
    isempty(t) && return 0
    v = tryparse(Int, t)
    return v === nothing ? 0 : v
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

"""
    cu_log10(cu_ppm; detection_limit=1.0) -> Float64

`log10` copper, with negatives and values below the detection limit lifted to
`detection_limit` (Keivitsa config `cu_detection_limit_ppm = 1.0`).
"""
function cu_log10(cu_ppm::Real; detection_limit::Real = KEIVITSA_CU_DL_PPM)
    isfinite(cu_ppm) || return NaN
    v = cu_ppm < 0 ? detection_limit : Float64(cu_ppm)
    return log10(max(v, detection_limit))
end

"""
    merge_petro_pair(a, b) -> Float64

Average when both are finite, otherwise whichever exists, otherwise `NaN`.
"""
function merge_petro_pair(a::Real, b::Real)
    fa, fb = isfinite(a), isfinite(b)
    fa && fb && return (Float64(a) + Float64(b)) / 2
    fa && return Float64(a)
    fb && return Float64(b)
    return NaN
end

# petro.txt stores YKJ-style X = northing (~7.5e6) and Y = easting (~3.5e6).
# Geochemistry shapefiles store KKJ_X = easting. Detect by magnitude.
function _kkj_east_north(a::Real, b::Real)
    if a > 5.0e6 && b < 5.0e6
        return Float64(b), Float64(a)   # a northing, b easting
    end
    return Float64(a), Float64(b)       # a easting, b northing
end

#---------- DBF ----------

# GTK shapefile DBFs are Latin-1 / CP1252 (Finnish Ä = 0xC4), not UTF-8.
# `String(bytes)` would leave an invalid Char that later `uppercase` throws on.
function _decode_dbf_text(bytes::AbstractVector{UInt8})
    n = findlast(!iszero, bytes)
    n === nothing && return ""
    raw = Vector{UInt8}(undef, n)
    copyto!(raw, 1, bytes, 1, n)
    isvalid(String, raw) && return String(raw)
    return String(Char.(raw))
end

function read_dbf(path::AbstractString; columns::Union{Nothing,Vector{String}} = nothing)
    isfile(path) || throw(ArgumentError("read_dbf: no such file: $path"))
    return open(path, "r") do io
        skip(io, 4)
        nrec = Int(ltoh(read(io, UInt32)))
        hlen = Int(ltoh(read(io, UInt16)))
        rec_len = Int(ltoh(read(io, UInt16)))
        seek(io, 32)
        nfields = (hlen - 33) ÷ 32
        fields = NamedTuple{(:name, :typ, :length),Tuple{String,Char,Int}}[]
        for _ in 1:nfields
            raw = read(io, 32)
            name = strip(_decode_dbf_text(@view raw[1:11]))
            typ = Char(raw[12])
            length = Int(raw[17])
            push!(fields, (name = name, typ = typ, length = length))
        end
        seek(io, hlen)
        want = columns === nothing ? nothing : Set(columns)
        col_ok = columns === nothing ? trues(nfields) :
            [f.name in want for f in fields]
        store = Dict{String,Vector{String}}()
        for (k, f) in enumerate(fields)
            col_ok[k] && (store[f.name] = Vector{String}(undef, nrec))
        end
        kept = 0
        for _ in 1:nrec
            rec = read(io, rec_len)
            length(rec) < rec_len && break
            rec[1] == UInt8('*') && continue
            kept += 1
            pos = 2
            for (k, f) in enumerate(fields)
                slice = rec[pos:pos + f.length - 1]
                pos += f.length
                col_ok[k] || continue
                store[f.name][kept] = strip(_decode_dbf_text(slice))
            end
        end
        for v in Base.values(store)
            resize!(v, kept)
        end
        return store
    end
end

function _dbf_float(table::Dict{String,Vector{String}}, name::String)
    haskey(table, name) || return fill(NaN, length(first(Base.values(table))))
    return [_parse_float(s) for s in table[name]]
end

function _dbf_string(table::Dict{String,Vector{String}}, name::String)
    haskey(table, name) || return fill("", length(first(Base.values(table))))
    return table[name]
end

function _element_column(names, symbol::AbstractString)
    u = uppercase(symbol)
    for n in names
        nu = uppercase(n)
        nu == u && return n
    end
    for n in names
        nu = uppercase(n)
        startswith(nu, u * "_") || continue
        endswith(nu, "_L") && continue
        return n
    end
    return nothing
end

#---------- desurvey (stepwise, matching nisai.preprocessing.desurvey) ----------

function _collar_plunge_rad(dip_deg::Real)
    d = Float64(dip_deg)
    plunge_deg = d < 0 ? -d : d
    return deg2rad(plunge_deg)
end

function _interp_at(x::Real, xs::Vector{Float64}, ys::Vector{Float64})
    isempty(xs) && return isempty(ys) ? 0.0 : ys[1]
    return Float64(_interp_linear(xs, ys, x))
end

"""
    desurvey_depths(collar_e, collar_n, collar_z, depths;
                    azimuth=0, dip=90, survey_depth=[], survey_az=[], survey_dip=[])
        -> (easting, northing, elevation)

Stepwise straight-segment integration along the hole. `dip` is degrees from
horizontal (positive down), `azimuth` degrees from north clockwise. Survey
stations, when given, are linearly interpolated in depth.
"""
function desurvey_depths(collar_e::Real, collar_n::Real, collar_z::Real,
                         depths::AbstractVector{<:Real};
                         azimuth::Real = 0.0, dip::Real = 90.0,
                         survey_depth::AbstractVector{<:Real} = Float64[],
                         survey_az::AbstractVector{<:Real} = Float64[],
                         survey_dip::AbstractVector{<:Real} = Float64[],
                         step_m::Real = 2.0)
    n = length(depths)
    east = Vector{Float64}(undef, n)
    north = Vector{Float64}(undef, n)
    elev = Vector{Float64}(undef, n)
    n == 0 && return east, north, elev

    sd = collect(Float64, survey_depth)
    saz = isempty(survey_az) ? Float64[] : deg2rad.(Float64.(survey_az))
    sdi = isempty(survey_dip) ? Float64[] : [_collar_plunge_rad(d) for d in survey_dip]
    have = !isempty(sd) && length(sd) == length(saz) == length(sdi)
    az0 = deg2rad(Float64(azimuth))
    di0 = _collar_plunge_rad(dip)

    max_d = maximum(d -> isfinite(d) ? Float64(d) : 0.0, depths; init = 0.0)
    if max_d <= 0
        fill!(east, Float64(collar_e))
        fill!(north, Float64(collar_n))
        fill!(elev, Float64(collar_z))
        return east, north, elev
    end

    n_steps = max(10, Int(ceil(max_d / max(step_m, 0.1))))
    path_d = range(0.0, max_d; length = n_steps + 1)
    path_e = Vector{Float64}(undef, length(path_d))
    path_n = similar(path_e)
    path_z = similar(path_e)
    path_e[1] = Float64(collar_e)
    path_n[1] = Float64(collar_n)
    path_z[1] = Float64(collar_z)
    for i in 2:length(path_d)
        d = path_d[i]
        az = have ? _interp_at(d, sd, saz) : az0
        di = have ? _interp_at(d, sd, sdi) : di0
        ds = path_d[i] - path_d[i - 1]
        path_e[i] = path_e[i - 1] + sin(az) * cos(di) * ds
        path_n[i] = path_n[i - 1] + cos(az) * cos(di) * ds
        path_z[i] = path_z[i - 1] - sin(di) * ds
    end
    pd = collect(path_d)
    for t in 1:n
        d = Float64(depths[t])
        if !isfinite(d) || d <= 0
            east[t] = Float64(collar_e)
            north[t] = Float64(collar_n)
            elev[t] = Float64(collar_z)
        else
            east[t] = _interp_linear(pd, path_e, d)
            north[t] = _interp_linear(pd, path_n, d)
            elev[t] = _interp_linear(pd, path_z, d)
        end
    end
    return east, north, elev
end

#---------- grid ----------

"""
    read_keivitsa_grid_bounds(config_path) -> NamedTuple

Read `train.grid_bounds` from the Keivitsa `config.yaml`.
"""
function read_keivitsa_grid_bounds(config_path::AbstractString)
    text = read(config_path, String)
    m = match(r"grid_bounds:\s*\n((?:[ \t]+\w+:[ \t]*[-+0-9.eE]+\s*\n)+)", text)
    m === nothing && throw(ArgumentError(
        "read_keivitsa_grid_bounds: no train.grid_bounds block in $config_path"))
    block = m.captures[1]
    grab(key) = begin
        mm = match(Regex(key * ":\\s*([-+0-9.eE]+)"), block)
        mm === nothing && throw(ArgumentError(
            "read_keivitsa_grid_bounds: missing $key"))
        parse(Float64, mm.captures[1])
    end
    return (x_min = grab("x_min"), x_max = grab("x_max"),
            y_min = grab("y_min"), y_max = grab("y_max"),
            z_min = grab("z_min"), z_max = grab("z_max"))
end

"""
    keivitsa_grid(bounds; cell=25.0, cell_z=cell) -> PriorGrid

Cartesian grid on the Keivitsa `train.grid_bounds` box.
`x` is easting, `y` northing, `z` elevation positive up. `cell` is the
horizontal (x/y) spacing in metres; `cell_z` the vertical spacing (defaults
to `cell`, so a cubic mesh). Bounds are unchanged: layer counts are
`round((max-min)/spacing)`, then the span is divided evenly.
"""
function keivitsa_grid(bounds; cell::Real = 25.0, cell_z::Real = cell)
    cell > 0 || throw(ArgumentError("keivitsa_grid: cell must be positive"))
    cell_z > 0 || throw(ArgumentError("keivitsa_grid: cell_z must be positive"))
    nx = max(1, round(Int, (bounds.x_max - bounds.x_min) / cell))
    ny = max(1, round(Int, (bounds.y_max - bounds.y_min) / cell))
    nz = max(1, round(Int, (bounds.z_max - bounds.z_min) / cell_z))
    dx = fill((bounds.x_max - bounds.x_min) / nx, nx)
    dy = fill((bounds.y_max - bounds.y_min) / ny, ny)
    dz = fill((bounds.z_max - bounds.z_min) / nz, nz)
    return PriorGrid(dx, dy, dz; origin = [bounds.x_min, bounds.y_min, bounds.z_min])
end

function keivitsa_report_root(dataset_root::AbstractString)
    report = joinpath(dataset_root, "source", "gtk", "report")
    isdir(report) || throw(ArgumentError(
        "keivitsa_report_root: no source/gtk/report under $dataset_root"))
    return report
end

#---------- collar / survey ----------

function load_keivitsa_collars(report_root::AbstractString)
    path = joinpath(report_root, "3_DRILLINGS", "Logs", "Shape_files", "collar.dbf")
    t = read_dbf(path)
    n = length(t["HOLE_ID"])
    lookup = Dict{String,NamedTuple}()
    for i in 1:n
        hid = strip(t["HOLE_ID"][i])
        isempty(hid) && continue
        lookup[hid] = (
            east = _parse_float(t["KKJ_EAST"][i]),
            north = _parse_float(t["KKJ_NORTH"][i]),
            elev = _parse_float(t["Z"][i]),
            azimuth = _parse_float(get(t, "STARTAZIM", fill("", n))[i]),
            dip = begin
                d = _parse_float(get(t, "STARTDIP", fill("", n))[i])
                isfinite(d) ? d : 90.0
            end,
        )
    end
    return lookup
end

function load_keivitsa_surveys(report_root::AbstractString)
    path = joinpath(report_root, "3_DRILLINGS", "Logs", "Shape_files", "survey.dbf")
    t = read_dbf(path)
    n = length(t["HOLE_ID"])
    by_hole = Dict{String,NamedTuple{(:depth, :az, :dip),
        Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}}()
    for i in 1:n
        hid = strip(t["HOLE_ID"][i])
        d = _parse_float(t["DEPTH"][i])
        (isempty(hid) || !isfinite(d)) && continue
        rec = get!(by_hole, hid) do
            (depth = Float64[], az = Float64[], dip = Float64[])
        end
        push!(rec.depth, d)
        push!(rec.az, _parse_float(t["AZIMUTH"][i]))
        push!(rec.dip, _parse_float(t["DIP"][i]))
    end
    for (hid, rec) in by_hole
        perm = sortperm(rec.depth)
        by_hole[hid] = (depth = rec.depth[perm], az = rec.az[perm], dip = rec.dip[perm])
    end
    return by_hole
end

function _desurvey_hole(collars, surveys, hole_id, depths)
    info = get(collars, hole_id, nothing)
    info === nothing && return fill(NaN, length(depths)), fill(NaN, length(depths)),
        fill(NaN, length(depths))
    surv = get(surveys, hole_id, nothing)
    return desurvey_depths(info.east, info.north, info.elev, depths;
                           azimuth = isfinite(info.azimuth) ? info.azimuth : 0.0,
                           dip = info.dip,
                           survey_depth = surv === nothing ? Float64[] : surv.depth,
                           survey_az = surv === nothing ? Float64[] : surv.az,
                           survey_dip = surv === nothing ? Float64[] : surv.dip)
end

#---------- geochemistry / lithology ----------

function load_keivitsa_geochemistry(report_root::AbstractString)
    dir = joinpath(report_root, "4_GEOCHEMISTRY")
    shps = sort(filter(p -> endswith(p, ".dbf"), readdir(dir; join = true)))
    isempty(shps) && throw(ArgumentError("load_keivitsa_geochemistry: no .dbf in $dir"))

    east = Float64[]
    north = Float64[]
    depth = Float64[]
    cols = Dict{String,Vector{Float64}}()
    for el in KEIVITSA_GEOCHEM_PRIORITY
        cols[el] = Float64[]
    end

    for path in shps
        t = read_dbf(path)
        n = length(first(Base.values(t)))
        names = collect(keys(t))
        e = haskey(t, "KKJ_X") ? _dbf_float(t, "KKJ_X") :
            haskey(t, "X_COORD") ? _dbf_float(t, "X_COORD") : fill(NaN, n)
        nn = haskey(t, "KKJ_Y") ? _dbf_float(t, "KKJ_Y") :
            haskey(t, "Y_COORD") ? _dbf_float(t, "Y_COORD") : fill(NaN, n)
        d = haskey(t, "DEPTH") ? _dbf_float(t, "DEPTH") : fill(NaN, n)
        # geochem KKJ_X is easting (~3.5e6); still run the swap helper
        for i in 1:n
            ee, nnth = _kkj_east_north(e[i], nn[i])
            push!(east, ee)
            push!(north, nnth)
            push!(depth, d[i])
        end
        for el in KEIVITSA_GEOCHEM_PRIORITY
            src = _element_column(names, el)
            vals = src === nothing ? fill(NaN, n) : _dbf_float(t, src)
            append!(cols[el], vals)
        end
    end

    present = String[]
    data_cols = Vector{Vector{Float64}}()
    for el in KEIVITSA_GEOCHEM_PRIORITY
        v = cols[el]
        any(x -> isfinite(x) && x > 0, v) || continue
        push!(present, el)
        push!(data_cols, v)
    end

    n = length(east)
    for (num, den) in KEIVITSA_GEOCHEM_RATIOS
        inum = findfirst(==(num), present)
        iden = findfirst(==(den), present)
        (inum === nothing || iden === nothing) && continue
        ratio = fill(NaN, n)
        a = data_cols[inum]
        b = data_cols[iden]
        @inbounds for i in 1:n
            if isfinite(a[i]) && isfinite(b[i]) && a[i] > 0 && b[i] > 0
                ratio[i] = a[i] / b[i]
            end
        end
        any(isfinite, ratio) || continue
        push!(present, lowercase(num) * "_over_" * lowercase(den))
        push!(data_cols, ratio)
    end

    isempty(data_cols) && throw(ArgumentError(
        "load_keivitsa_geochemistry: none of the priority elements were present"))
    value_mat = reduce(hcat, data_cols)
    missing_priority = [el for el in KEIVITSA_GEOCHEM_PRIORITY if !(el in present)]
    return PointSamples(east, north, value_mat, present), missing_priority
end

function keivitsa_cleaned_intervals_path(dataset_root::AbstractString)
    p = joinpath(dataset_root, "processed", "keivitsa_cleaned_intervals.csv")
    isfile(p) || throw(ArgumentError(
        "keivitsa_cleaned_intervals_path: no such file: $p"))
    return p
end

"""
    load_keivitsa_drill_geochemistry(csv_path) -> PointSamples

Interval-level drill assays from `processed/keivitsa_cleaned_intervals.csv`.
Only the surface-shapefile gaps are returned (`S`, `FE`, `CR`, `PT`); Cu/Ni/Co/Pd/Au
stay on the till/bedrock channels (see [`combine_geochemistry`](@ref)).
Coordinates are already desurveyed (easting, northing, elevation, EPSG:2393).
"""
function load_keivitsa_drill_geochemistry(csv_path::AbstractString)
    isfile(csv_path) || throw(ArgumentError(
        "load_keivitsa_drill_geochemistry: no such file: $csv_path"))

    east = Float64[]
    north = Float64[]
    elev = Float64[]
    cols = Dict(el => Float64[] for el in KEIVITSA_DRILL_GEOCHEM)
    header = String[]
    idx = Dict{String,Int}()
    open(csv_path, "r") do io
        header_line = readline(io)
        header = uppercase.(strip.(_split_csv_line(header_line)))
        for (j, h) in enumerate(header)
            idx[h] = j
        end
        need = ("X", "Y", "Z", KEIVITSA_DRILL_GEOCHEM...)
        for name in need
            haskey(idx, name) || throw(ArgumentError(
                "load_keivitsa_drill_geochemistry: missing column $name in $csv_path"))
        end
        ix, iy, iz = idx["X"], idx["Y"], idx["Z"]
        iel = Dict(el => idx[el] for el in KEIVITSA_DRILL_GEOCHEM)
        for line in eachline(io)
            isempty(strip(line)) && continue
            f = _split_csv_line(line)
            length(f) < length(header) && continue
            x = _parse_float(f[ix])
            y = _parse_float(f[iy])
            z = _parse_float(f[iz])
            (isfinite(x) && isfinite(y) && isfinite(z)) || continue
            ee, nn = _kkj_east_north(x, y)
            push!(east, ee)
            push!(north, nn)
            push!(elev, z)
            for el in KEIVITSA_DRILL_GEOCHEM
                push!(cols[el], _parse_float(f[iel[el]]))
            end
        end
    end
    isempty(east) && throw(ArgumentError(
        "load_keivitsa_drill_geochemistry: no finite (x, y, z) rows"))
    present = String[]
    data_cols = Vector{Vector{Float64}}()
    for el in KEIVITSA_DRILL_GEOCHEM
        v = cols[el]
        any(x -> isfinite(x) && x > 0, v) || continue
        push!(present, el)
        push!(data_cols, v)
    end
    isempty(data_cols) && throw(ArgumentError(
        "load_keivitsa_drill_geochemistry: none of $(KEIVITSA_DRILL_GEOCHEM) were present"))
    return PointSamples(east, north, reduce(hcat, data_cols), present; z = elev)
end

"""
    combine_geochemistry(first, second) -> (PointSamples, skipped)

Stack two geochemistry tables. Channels whose name already exists in `first`
(case-insensitive) are dropped from `second` — surface till/bedrock wins over
drill core for Cu, Ni, Co, Pd, Au. Returned `skipped` lists those dropped names.

Each channel is interpolated independently later, so a drill-only element (S)
does not borrow a till sample's location.
"""
function combine_geochemistry(first::PointSamples, second::PointSamples)
    n1 = length(first)
    n2 = length(second)
    names1 = copy(first.names)
    have = Set(uppercase.(names1))
    keep2 = Int[]
    names2 = String[]
    skipped = String[]
    for (j, n) in enumerate(second.names)
        if uppercase(n) in have
            push!(skipped, n)
        else
            push!(keep2, j)
            push!(names2, n)
            push!(have, uppercase(n))
        end
    end
    nchan = length(names1) + length(names2)
    values = fill(NaN, n1 + n2, nchan)
    values[1:n1, 1:length(names1)] .= first.values
    if !isempty(keep2)
        values[n1+1:end, length(names1)+1:end] .= second.values[:, keep2]
    end
    z1 = first.z === nothing ? fill(NaN, n1) : first.z
    z2 = second.z === nothing ? fill(NaN, n2) : second.z
    combined = PointSamples(vcat(first.x, second.x), vcat(first.y, second.y),
                            values, vcat(names1, names2); z = vcat(z1, z2))
    return combined, skipped
end

function load_keivitsa_lithology(report_root::AbstractString, collars, surveys)
    rt = joinpath(report_root, "3_DRILLINGS", "Logs", "Shape_files", "rocktype.dbf")
    ld = joinpath(report_root, "3_DRILLINGS", "Logs", "Shape_files", "lithdesc.dbf")
    t = read_dbf(rt)
    n = length(t["HOLE_ID"])
    isfile(ld) || @warn "lithdesc.dbf missing; one-hot uses rocktype.shp only" ld
    if isfile(ld)
        td = read_dbf(ld; columns = ["HOLE_ID", "FROM", "TO"])
        length(td["HOLE_ID"]) == n || @warn "lithdesc row count differs from rocktype" rocktype=n lithdesc=length(td["HOLE_ID"])
    end

    labels = t["ROCKTYPE"]
    from = _dbf_float(t, "FROM")
    to_ = _dbf_float(t, "TO")
    hid = strip.(t["HOLE_ID"])
    mids = 0.5 .* (from .+ to_)

    east = fill(NaN, n)
    north = fill(NaN, n)
    elev = fill(NaN, n)
    by_hole = Dict{String,Vector{Int}}()
    for i in 1:n
        push!(get!(Vector{Int}, by_hole, hid[i]), i)
    end
    for (hole, idxs) in by_hole
        e, nn, z = _desurvey_hole(collars, surveys, hole, mids[idxs])
        east[idxs] .= e
        north[idxs] .= nn
        elev[idxs] .= z
    end
    keep = isfinite.(east) .& isfinite.(north) .& isfinite.(elev)
    return LabelSamples(east[keep], north[keep], labels[keep]; z = elev[keep])
end

#---------- anchors ----------

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

function load_keivitsa_grade_anchors(g::PriorGrid, report_root::AbstractString,
                                    collars, surveys;
                                    detection_limit::Real = KEIVITSA_CU_DL_PPM)
    dir = joinpath(report_root, "3_DRILLINGS", "Assays", "Shape_files")
    files = sort(filter(p -> endswith(p, ".dbf"), readdir(dir; join = true)))
    east = Float64[]
    north = Float64[]
    elev = Float64[]
    clog = Float64[]
    wts = Float64[]
    seen = Set{String}()
    n_raw = 0
    n_cu = 0
    holes = Set{String}()

    for path in files
        t = read_dbf(path)
        names = collect(keys(t))
        cucol = _element_column(names, "CU")
        cucol === nothing && continue
        n = length(t["HOLE_ID"])
        cu = _dbf_float(t, cucol)
        hid = strip.(t["HOLE_ID"])
        from = _dbf_float(t, "FROM")
        to_ = _dbf_float(t, "TO")
        n_raw += n
        by_hole = Dict{String,Vector{Int}}()
        for i in 1:n
            push!(get!(Vector{Int}, by_hole, hid[i]), i)
        end
        for (hole, idxs) in by_hole
            mids = 0.5 .* (from[idxs] .+ to_[idxs])
            e, nn, z = _desurvey_hole(collars, surveys, hole, mids)
            for (k, i) in enumerate(idxs)
                v = cu[i]
                isfinite(v) || continue
                n_cu += 1
                key = hole * "|" * string(from[i]) * "|" * string(to_[i])
                key in seen && continue
                push!(seen, key)
                push!(holes, hole)
                push!(east, e[k])
                push!(north, nn[k])
                push!(elev, z[k])
                push!(clog, cu_log10(v; detection_limit = detection_limit))
                len = to_[i] - from[i]
                push!(wts, isfinite(len) && len > 0 ? len : 1.0)
            end
        end
    end

    cells, values, weights = map_points_to_cells(g, east, north, elev, clog; weights = wts)
    stats = (n_assay_rows = n_raw, n_cu_rows = n_cu, n_unique_intervals = length(clog),
             n_holes = length(holes), n_cells = length(cells),
             n_mapped_samples = count(!=(0),
                 [containing_cell(g, east[i], north[i], elev[i]) for i in eachindex(east)]))
    return (cells, values, weights), stats
end

"""
    read_petro_txt(path) -> NamedTuple

Read GTK `petro.txt`: caret-separated, four header rows, data from line 5.
`PTR_K` and `DSR_K` are parsed and returned but must not be used as anchors.
They are a remanence candidate (third component of GTK's standard
density/susceptibility/remanence set) and are not yet a fifth property group.
"""
function read_petro_txt(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 5 || throw(ArgumentError("read_petro_txt: $path has no data rows"))
    header = split(replace(lines[1], '\r' => ""), '^')
    names = [strip(h) for h in header]
    idx = Dict(names[i] => i for i in eachindex(names))
    need = ("Tunnus", "Syvyys", "ReiKkj_X", "ReiKkj_y", "Rei_Z",
            "PTR_D", "PTR_J", "PTR_K", "LUO_R", "DSR_D", "DSR_J", "DSR_K")
    for n in need
        haskey(idx, n) || throw(ArgumentError("read_petro_txt: missing column $n"))
    end

    n = length(lines) - 4
    hole = Vector{String}(undef, n)
    depth = Vector{Float64}(undef, n)
    kkj_x = Vector{Float64}(undef, n)
    kkj_y = Vector{Float64}(undef, n)
    rei_z = Vector{Float64}(undef, n)
    ptr_d = Vector{Float64}(undef, n)
    ptr_j = Vector{Float64}(undef, n)
    ptr_k = Vector{Float64}(undef, n)
    luo_r = Vector{Float64}(undef, n)
    dsr_d = Vector{Float64}(undef, n)
    dsr_j = Vector{Float64}(undef, n)
    dsr_k = Vector{Float64}(undef, n)

    for (t, line) in enumerate(view(lines, 5:length(lines)))
        parts = split(replace(line, '\r' => ""), '^')
        getf(col) = idx[col] <= length(parts) ? _parse_float(parts[idx[col]]) : NaN
        gets(col) = idx[col] <= length(parts) ? strip(parts[idx[col]]) : ""
        hole[t] = gets("Tunnus")
        depth[t] = getf("Syvyys")
        kkj_x[t] = getf("ReiKkj_X")
        kkj_y[t] = getf("ReiKkj_y")
        rei_z[t] = getf("Rei_Z")
        ptr_d[t] = getf("PTR_D")
        ptr_j[t] = getf("PTR_J")
        ptr_k[t] = getf("PTR_K")
        luo_r[t] = getf("LUO_R")
        dsr_d[t] = getf("DSR_D")
        dsr_j[t] = getf("DSR_J")
        dsr_k[t] = getf("DSR_K")
    end
    return (hole = hole, depth = depth, kkj_x = kkj_x, kkj_y = kkj_y, rei_z = rei_z,
            ptr_d = ptr_d, ptr_j = ptr_j, ptr_k = ptr_k, luo_r = luo_r,
            dsr_d = dsr_d, dsr_j = dsr_j, dsr_k = dsr_k)
end

function petro_coverage(petro)
    n = length(petro.hole)
    frac(v) = count(isfinite, v) / n
    return (n = n,
            ptr_d = frac(petro.ptr_d), ptr_j = frac(petro.ptr_j), ptr_k = frac(petro.ptr_k),
            dsr_d = frac(petro.dsr_d), dsr_j = frac(petro.dsr_j), dsr_k = frac(petro.dsr_k),
            luo_r = frac(petro.luo_r),
            density = count(i -> isfinite(merge_petro_pair(petro.ptr_d[i], petro.dsr_d[i])), 1:n) / n,
            susceptibility = count(i -> isfinite(merge_petro_pair(petro.ptr_j[i], petro.dsr_j[i])), 1:n) / n)
end

function load_keivitsa_petrophysics_anchors(g::PriorGrid, petro, collars, surveys)
    n = length(petro.hole)
    dens = Vector{Float64}(undef, n)
    susc = Vector{Float64}(undef, n)
    res = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        d = merge_petro_pair(petro.ptr_d[i], petro.dsr_d[i])
        j = merge_petro_pair(petro.ptr_j[i], petro.dsr_j[i])
        dens[i] = isfinite(d) && d > 0 ? d / 1000 : NaN          # g/cm³
        susc[i] = isfinite(j) && j > 0 ? log10(j) : NaN
        r = petro.luo_r[i]
        res[i] = isfinite(r) && r > 0 ? log10(r) : NaN
    end

    east = fill(NaN, n)
    north = fill(NaN, n)
    elev = fill(NaN, n)
    by_hole = Dict{String,Vector{Int}}()
    for i in 1:n
        push!(get!(Vector{Int}, by_hole, petro.hole[i]), i)
    end
    for (hole, idxs) in by_hole
        e, nn, z = _desurvey_hole(collars, surveys, hole, petro.depth[idxs])
        if all(!isfinite, e)
            # collar miss: fall back to petro.txt KKJ + vertical depth
            for i in idxs
                ee, nnth = _kkj_east_north(petro.kkj_x[i], petro.kkj_y[i])
                east[i] = ee
                north[i] = nnth
                elev[i] = petro.rei_z[i] - petro.depth[i]
            end
        else
            east[idxs] .= e
            north[idxs] .= nn
            elev[idxs] .= z
        end
    end

    dens_a = map_points_to_cells(g, east, north, elev, dens)
    susc_a = map_points_to_cells(g, east, north, elev, susc)
    res_a = map_points_to_cells(g, east, north, elev, res)
    return (density = dens_a, susceptibility = susc_a, resistivity = res_a)
end
