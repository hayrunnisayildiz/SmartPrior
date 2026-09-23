# Keivitsa (GTK) drillholes → SampleTable.
#
# Site-specific only at the edges: file layout, KKJ axis order, and the
# degree / dip-down conventions confirmed for this survey. The trajectory
# itself is [`desurvey`](@ref), which still throws on a bad hole. This
# adapter catches that error for one hole, records it, and continues.
# [`desurvey`](@ref) is called with angle_unit=:degree and
# dip_down_negative=false: azimuth clockwise from north, +90° vertical down.

"""
    KeivitsaExclusion

One hole-level drop. `message` is the [`desurvey`](@ref) / [`positions`](@ref)
error when the reason is a geometry failure, and a short explanation otherwise.
`n_cu` and `n_petro` are the samples removed with the hole (or, for
`outside_bounds` and `other_method`, the samples removed while the rest of
the hole may remain).
"""
struct KeivitsaExclusion
    hole::String
    reason::String
    message::String
    n_cu::Int
    n_petro::Int
end

"""
    KeivitsaReport

Counts from one [`keivitsa_sample_table`](@ref) call.

`cu_samples_read` / `cu_holes_read` are method 511P before collar, geometry,
and bounds filters. `cu_censored` is the left-censored count on that same
set (the fixed limit is applied before the spatial filter). `cu_outside_bounds`
is 511P samples removed by `[bounds]` after the other filters.
`susc_censored` is kept rows whose susceptibility was ≤ 0. Those rows store
the smallest positive susceptibility. The raw readings are written to the log.
"""
struct KeivitsaReport
    exclusions::Vector{KeivitsaExclusion}
    cu_samples_read::Int
    cu_holes_read::Int
    cu_censored::Int
    cu_samples_kept::Int
    cu_holes_kept::Int
    cu_censored_kept::Int
    cu_outside_bounds::Int
    ptr_samples_kept::Int
    ptr_holes_kept::Int
    dsr_samples_kept::Int
    dsr_holes_kept::Int
    susc_censored::Int
    petro_rows_ignored::Int
end

# Geometry failures are per-hole data problems. Anything above this fraction
# of the 511P holes is treated as a parser or convention bug.
const _GEOMETRY_DROP_LIMIT = 0.05
const _GEOMETRY_REASONS = (
    "no_survey",
    "duplicate_depth",
    "unsorted_stations",
    "negative_depth",
    "desurvey_error",
)

const _DEFAULT_COLLAR = "report/3_DRILLINGS/Logs/Shape_files/collar.shp"
const _DEFAULT_SURVEY = "report/3_DRILLINGS/Logs/Shape_files/kalte.txt"
const _DEFAULT_ASSAYS = "report/3_DRILLINGS/Assays/Shape_files/511P.txt"
const _DEFAULT_PETRO = "report/3_DRILLINGS/Downhole_soundings_and_core_measurements/petro.txt"

function _geometry_reason(msg::AbstractString)
    if occursin("same depth", msg)
        return "duplicate_depth"
    elseif occursin("decreased", msg)
        return "unsorted_stations"
    elseif occursin("≥ 0", msg) || occursin(">= 0", msg)
        return "negative_depth"
    else
        return "desurvey_error"
    end
end

function _hole_id(raw)
    id = uppercase(strip(String(raw)))
    return id
end

function _read_latin1(path::AbstractString)
    isfile(path) || throw(ArgumentError("keivitsa_sample_table: no such file: $path"))
    return String(Char.(read(path)))
end

function _caret_rows(path::AbstractString)
    text = replace(_read_latin1(path), "\r\n" => "\n", "\r" => "\n")
    lines = String[]
    for line in split(text, "\n")
        isempty(strip(line)) && continue
        push!(lines, String(line))
    end
    length(lines) >= 4 || throw(ArgumentError(
        "keivitsa_sample_table: $path needs a header and three metadata rows"))
    header = String[strip(h) for h in split(lines[1], '^')]
    types = String[strip(h) for h in split(lines[2], '^')]
    any(t -> t in ("int", "char", "real", "float"), types) || throw(ArgumentError(
        "keivitsa_sample_table: $path line 2 is not a GTK type row"))
    rows = Vector{Dict{String,String}}()
    for line in lines[5:end]
        parts = split(line, '^')
        row = Dict{String,String}()
        for (j, name) in enumerate(header)
            row[name] = j <= length(parts) ? strip(String(parts[j])) : ""
        end
        push!(rows, row)
    end
    return header, rows
end

function _require_columns(header, names, what)
    for name in names
        name in header || throw(ArgumentError(
            "keivitsa_sample_table: $what is missing column $name"))
    end
    return nothing
end

function _gtk_float(raw::AbstractString)
    t = strip(raw)
    (isempty(t) || all(==('*'), t)) && return nothing
    v = tryparse(Float64, t)
    v === nothing && throw(ArgumentError(
        "keivitsa_sample_table: cannot parse number $(repr(t))"))
    return v
end

function _as_float(v)
    v === nothing && return nothing
    v isa Real && return isfinite(Float64(v)) ? Float64(v) : nothing
    return _gtk_float(string(v))
end

function _as_string(v)
    v === nothing && return ""
    return strip(string(v))
end

function _cfg_path(root, cfg, key, default)
    if cfg isa AbstractDict && haskey(cfg, key)
        p = strip(string(cfg[key]))
        if !isempty(p)
            return isabspath(p) ? p : normpath(joinpath(root, p))
        end
    end
    return normpath(joinpath(root, default))
end

function _props(cfg)
    haskey(cfg, "properties") || throw(ArgumentError(
        "keivitsa_sample_table: site file has no [properties]"))
    props = cfg["properties"]
    props isa AbstractDict || throw(ArgumentError(
        "keivitsa_sample_table: [properties] must be a table"))
    return props
end

function _one_prop(props, name)
    haskey(props, name) || throw(ArgumentError(
        "keivitsa_sample_table: missing properties.$name"))
    p = props[name]
    p isa AbstractDict || throw(ArgumentError(
        "keivitsa_sample_table: properties.$name must be a table"))
    return p
end

function _spec(p::AbstractDict, name::Symbol, default_kind, default_transform, default_unit)
    kind = Symbol(strip(string(get(p, "kind", default_kind))))
    transform = Symbol(strip(string(get(p, "transform", default_transform))))
    unit = string(get(p, "unit", default_unit))
    return PropertySpec(name, kind, transform, unit)
end

function _cu_lod(props)
    p = _one_prop(props, "cu")
    policy = strip(string(get(p, "lod_policy", "")))
    policy == "fixed" || throw(ArgumentError(
        "keivitsa_sample_table: properties.cu lod_policy must be \"fixed\", " *
        "got $(repr(policy))"))
    haskey(p, "lod") || throw(ArgumentError(
        "keivitsa_sample_table: properties.cu lod_policy \"fixed\" needs lod"))
    lod = Float64(p["lod"])
    lod > 0 || throw(ArgumentError(
        "keivitsa_sample_table: properties.cu lod must be positive, got $lod"))
    return lod
end

function _reject_lod(p, name)
    if haskey(p, "lod_policy") && !isempty(strip(string(p["lod_policy"])))
        throw(ArgumentError(
            "keivitsa_sample_table: properties.$name keeps measured values; " *
            "lod_policy is only for Cu and for susceptibility"))
    end
    return nothing
end

function _susc_policy(p)
    policy = strip(string(get(p, "lod_policy", "")))
    if !(policy in ("", "min_positive"))
        throw(ArgumentError(
            "keivitsa_sample_table: non-positive susceptibility is censored at " *
            "the smallest positive value; properties.susceptibility lod_policy " *
            "must be \"min_positive\" or omitted, got $(repr(policy))"))
    end
    return policy
end

# Collar elevations for DepthBelowSurface. Z = 0 is the missing-elevation
# sentinel, the same rule as the sample filter. Other holes are kept: a
# collar with a surveyed elevation is a surface station even when its
# samples later fall outside the box.
function _surface_collars(root::AbstractString, cfg)
    path = _cfg_path(root, cfg, "collar", _DEFAULT_COLLAR)
    collars = _read_collars(path)
    east = Float64[]
    north = Float64[]
    elev = Float64[]
    for id in sort!(collect(keys(collars)))
        c = collars[id]
        (isfinite(c.east) && isfinite(c.north) && isfinite(c.z)) || continue
        c.z == 0 && continue
        push!(east, c.east)
        push!(north, c.north)
        push!(elev, c.z)
    end
    isempty(east) && throw(ArgumentError(
        "depth_below_surface: no collar with a finite elevation other than 0"))
    return east, north, elev
end

function _censor_susceptibility!(susc::Vector{Float64}, cens::BitVector, holes)
    positive = Float64[]
    raw = Float64[]
    raw_holes = String[]
    for i in eachindex(susc)
        v = susc[i]
        isfinite(v) || continue
        if v > 0
            push!(positive, v)
        else
            push!(raw, v)
            push!(raw_holes, holes[i])
        end
    end
    isempty(raw) && return 0
    isempty(positive) && throw(ArgumentError(
        "keivitsa_sample_table: susceptibility has non-positive values but no " *
        "positive measurement to set the limit"))
    lod = minimum(positive)
    for i in eachindex(susc)
        v = susc[i]
        (isfinite(v) && v <= 0) || continue
        cens[i] = true
        susc[i] = lod
    end
    @info "non-positive susceptibility marked censored" n=length(raw) lod raw hole=raw_holes
    return length(raw)
end

struct _Box
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    zmin::Float64
    zmax::Float64
end

function _box_of(cfg)
    haskey(cfg, "bounds") || throw(ArgumentError(
        "keivitsa_sample_table: missing [bounds]"))
    b = cfg["bounds"]
    b isa AbstractDict || throw(ArgumentError(
        "keivitsa_sample_table: [bounds] must be a table"))
    function pair(axis)
        haskey(b, axis) || throw(ArgumentError(
            "keivitsa_sample_table: bounds.$axis is missing"))
        v = b[axis]
        length(v) == 2 || throw(ArgumentError(
            "keivitsa_sample_table: bounds.$axis must be [min, max]"))
        lo = Float64(v[1])
        hi = Float64(v[2])
        hi > lo || throw(ArgumentError(
            "keivitsa_sample_table: bounds.$axis max ($hi) must be greater than min ($lo)"))
        return lo, hi
    end
    xlo, xhi = pair("x")
    ylo, yhi = pair("y")
    zlo, zhi = pair("z")
    return _Box(xlo, xhi, ylo, yhi, zlo, zhi)
end

function _inside(box::_Box, x, y, z)
    return box.xmin <= x <= box.xmax && box.ymin <= y <= box.ymax &&
           box.zmin <= z <= box.zmax
end

struct _Collar
    east::Float64
    north::Float64
    z::Float64
end

function _store_collar!(collars, hole, east, north, z)
    id = _hole_id(hole)
    isempty(id) && throw(ArgumentError(
        "keivitsa_sample_table: a collar row has an empty hole id"))
    haskey(collars, id) && throw(ArgumentError(
        "keivitsa_sample_table: duplicate collar for $id"))
    (east === nothing || north === nothing || z === nothing) && throw(ArgumentError(
        "keivitsa_sample_table: collar $id has a non-numeric coordinate"))
    collars[id] = _Collar(east, north, z)
    return nothing
end

function _collars_caret(path)
    header, rows = _caret_rows(path)
    _require_columns(header, ("HOLE_ID", "KKJ_NORTH", "KKJ_EAST", "Z"), path)
    collars = Dict{String,_Collar}()
    for row in rows
        _store_collar!(collars, row["HOLE_ID"],
                       _gtk_float(row["KKJ_EAST"]),
                       _gtk_float(row["KKJ_NORTH"]),
                       _gtk_float(row["Z"]))
    end
    return collars
end

function _collars_ogr(path)
    collars = Dict{String,_Collar}()
    ArchGDAL.read(path) do ds
        layer = ArchGDAL.getlayer(ds, 0)
        for feat in layer
            _store_collar!(collars,
                           _as_string(ArchGDAL.getfield(feat, "HOLE_ID")),
                           _as_float(ArchGDAL.getfield(feat, "KKJ_EAST")),
                           _as_float(ArchGDAL.getfield(feat, "KKJ_NORTH")),
                           _as_float(ArchGDAL.getfield(feat, "Z")))
        end
    end
    return collars
end

function _read_collars(path)
    ext = lowercase(splitext(path)[2])
    if ext == ".txt"
        return _collars_caret(path)
    elseif ext in (".shp", ".dbf")
        return _collars_ogr(path)
    else
        throw(ArgumentError(
            "keivitsa_sample_table: collar file must be .shp, .dbf, or caret .txt, got $path"))
    end
end

struct _Cu
    fr::Float64
    to::Float64
    value::Float64
    censored::Bool
end

struct _Petro
    depth::Float64
    density::Float64
    susc::Float64
    source::String
end

struct _Placed
    x::Float64
    y::Float64
    z::Float64
    cu::Float64
    cu_cens::Bool
    density::Float64
    susc::Float64
    source::String
end

function _method_column(header)
    for name in ("Method", "method", "Menetelma")
        name in header && return name
    end
    return nothing
end

function _parse_cu_row(row, lod)
    flag = strip(get(row, "Cu_L", ""))
    raw = _gtk_float(get(row, "Cu", ""))
    if flag == "" && raw === nothing
        return :missing
    end
    if !(flag in ("", "<", "!"))
        throw(ArgumentError(
            "keivitsa_sample_table: Cu flag $(repr(flag)) is not \"\", \"<\", or \"!\""))
    end
    fr = _gtk_float(row["Ylasyvyys"])
    to = _gtk_float(row["Alasyvyys"])
    (fr === nothing || to === nothing) && throw(ArgumentError(
        "keivitsa_sample_table: a Cu interval has a non-numeric depth"))
    cens = flag in ("<", "!") || (raw !== nothing && raw <= 0)
    value = cens ? lod : raw
    return _Cu(fr, to, value, cens)
end

function _interval_message(rows::Vector{_Cu})
    isempty(rows) && return nothing
    iv = sort!([(r.fr, r.to) for r in rows])
    for i in eachindex(iv)
        a, b = iv[i]
        if b < a
            return "interval from $a m is deeper than to $b m"
        end
        i == 1 && continue
        pa, pb = iv[i - 1]
        if a == pa && b == pb
            return "duplicate interval $(a) to $(b) m"
        elseif a < pb
            return "intervals overlap ($(pa) to $(pb) m and $(a) to $(b) m)"
        end
    end
    return nothing
end

function _petro_conflict(rows::Vector{_Petro})
    seen = Dict{Float64,Tuple{Float64,Float64,String}}()
    for r in rows
        prev = get(seen, r.depth, nothing)
        if prev === nothing
            seen[r.depth] = (r.density, r.susc, r.source)
            continue
        end
        same = prev[3] == r.source && isequal(prev[1], r.density) && isequal(prev[2], r.susc)
        if !same
            return "duplicate petro depth $(r.depth) m with different readings"
        end
    end
    return nothing
end

function _read_survey(path)
    header, rows = _caret_rows(path)
    _require_columns(header, ("Tunnus", "Syvyys", "Kaltevuus", "Suunta"), path)
    by = Dict{String,Vector{NTuple{3,Float64}}}()
    for row in rows
        id = _hole_id(row["Tunnus"])
        isempty(id) && throw(ArgumentError(
            "keivitsa_sample_table: a survey row has an empty hole id"))
        md = _gtk_float(row["Syvyys"])
        az = _gtk_float(row["Suunta"])
        dip = _gtk_float(row["Kaltevuus"])
        (md === nothing || az === nothing || dip === nothing) && throw(ArgumentError(
            "keivitsa_sample_table: survey $id has a non-numeric station"))
        push!(get!(() -> NTuple{3,Float64}[], by, id), (md, az, dip))
    end
    return by
end

function _read_assays(path, lod)
    header, rows = _caret_rows(path)
    _require_columns(header, ("Tunnus", "Ylasyvyys", "Alasyvyys", "Cu", "Cu_L"), path)
    method_col = _method_column(header)
    by = Dict{String,Vector{_Cu}}()
    other = Dict{String,Int}()
    missing = Dict{String,Int}()
    n_read = 0
    n_cens = 0
    holes = Set{String}()
    for row in rows
        id = _hole_id(row["Tunnus"])
        isempty(id) && throw(ArgumentError(
            "keivitsa_sample_table: an assay row has an empty hole id"))
        if method_col !== nothing
            method = uppercase(strip(row[method_col]))
            if method != "511P"
                other[id] = get(other, id, 0) + 1
                continue
            end
        end
        push!(holes, id)
        n_read += 1
        parsed = _parse_cu_row(row, lod)
        if parsed === :missing
            missing[id] = get(missing, id, 0) + 1
            continue
        end
        parsed.censored && (n_cens += 1)
        push!(get!(() -> _Cu[], by, id), parsed)
    end
    return by, other, missing, n_read, n_cens, holes
end

function _read_petro(path)
    header, rows = _caret_rows(path)
    _require_columns(header, ("Tunnus", "Syvyys", "PTR_D", "PTR_K", "DSR_D", "DSR_K"), path)
    by = Dict{String,Vector{_Petro}}()
    ignored = 0
    for row in rows
        id = _hole_id(row["Tunnus"])
        isempty(id) && throw(ArgumentError(
            "keivitsa_sample_table: a petro row has an empty hole id"))
        ptr_d = _gtk_float(row["PTR_D"])
        ptr_k = _gtk_float(row["PTR_K"])
        dsr_d = _gtk_float(row["DSR_D"])
        dsr_k = _gtk_float(row["DSR_K"])
        has_ptr = ptr_d !== nothing || ptr_k !== nothing
        has_dsr = dsr_d !== nothing || dsr_k !== nothing
        if has_ptr && has_dsr
            throw(ArgumentError(
                "keivitsa_sample_table: $id has PTR and DSR on one row; " *
                "they are not averaged"))
        end
        if !has_ptr && !has_dsr
            ignored += 1
            continue
        end
        md = _gtk_float(row["Syvyys"])
        md === nothing && throw(ArgumentError(
            "keivitsa_sample_table: petro $id has a non-numeric depth"))
        if has_ptr
            push!(get!(() -> _Petro[], by, id),
                  _Petro(md,
                         ptr_d === nothing ? NaN : ptr_d,
                         ptr_k === nothing ? NaN : ptr_k,
                         "PTR"))
        else
            push!(get!(() -> _Petro[], by, id),
                  _Petro(md,
                         dsr_d === nothing ? NaN : dsr_d,
                         dsr_k === nothing ? NaN : dsr_k,
                         "DSR"))
        end
    end
    return by, ignored
end

function _place_hole(collar::_Collar, stations, cu::Vector{_Cu}, petro::Vector{_Petro})
    depths = [s[1] for s in stations]
    azs = [s[2] for s in stations]
    dips = [s[3] for s in stations]
    path = desurvey((collar.east, collar.north, collar.z), depths, azs, dips;
                    angle_unit = :degree, dip_down_negative = false)
    placed = _Placed[]
    for r in cu
        e, n, z = positions(path, (r.fr + r.to) / 2)
        push!(placed, _Placed(e, n, z, r.value, r.censored, NaN, NaN, ""))
    end
    for r in petro
        e, n, z = positions(path, r.depth)
        push!(placed, _Placed(e, n, z, NaN, false, r.density, r.susc, r.source))
    end
    return placed
end

function _drop!(out, hole, reason, message, n_cu, n_petro)
    (n_cu == 0 && n_petro == 0) && return nothing
    push!(out, KeivitsaExclusion(hole, reason, message, n_cu, n_petro))
    return nothing
end

function _guard_geometry!(exclusions, cu_holes::Set{String})
    n = length(cu_holes)
    n == 0 && throw(ArgumentError(
        "keivitsa_sample_table: no method-511P samples"))
    geom = KeivitsaExclusion[]
    seen = Set{String}()
    for e in exclusions
        e.reason in _GEOMETRY_REASONS || continue
        e.hole in cu_holes || continue
        e.hole in seen && continue
        push!(seen, e.hole)
        push!(geom, e)
    end
    n_geom = length(geom)
    # Strictly more than 5%. 1 of 20 holes is the boundary and is kept.
    n_geom * 20 > n || return nothing
    counts = Dict{String,Int}()
    for e in geom
        counts[e.reason] = get(counts, e.reason, 0) + 1
    end
    parts = String["$reason ($(counts[reason]) holes)" for reason in sort!(collect(keys(counts)))]
    ids = join((e.hole for e in geom), ", ")
    pct = 100 * n_geom / n
    throw(ArgumentError(
        "keivitsa_sample_table: geometry errors dropped $n_geom of $n " *
        "holes with 511P Cu ($(round(pct; digits=2))%), more than 5%. " *
        "Reasons: " * join(parts, ", ") * ". Holes: " * ids))
end

function _log_report(report::KeivitsaReport)
    rate = report.cu_samples_read == 0 ? NaN : report.cu_censored / report.cu_samples_read
    @info "Keivitsa 511P" samples_read=report.cu_samples_read holes_read=report.cu_holes_read censored=report.cu_censored censored_rate=rate samples_kept=report.cu_samples_kept holes_kept=report.cu_holes_kept outside_bounds=report.cu_outside_bounds
    @info "Keivitsa petrophysics kept" PTR_samples=report.ptr_samples_kept PTR_holes=report.ptr_holes_kept DSR_samples=report.dsr_samples_kept DSR_holes=report.dsr_holes_kept susceptibility_censored=report.susc_censored petro_rows_ignored=report.petro_rows_ignored
    for row in exclusion_summary(report)
        @info "Keivitsa exclusion" reason=row.reason holes=row.holes cu_samples=row.cu_samples petro_samples=row.petro_samples
    end
    return nothing
end

"""
    exclusion_summary(report) -> Vector

One row per exclusion reason: `reason`, `holes`, `cu_samples`, `petro_samples`.
"""
function exclusion_summary(report::KeivitsaReport)
    reasons = sort!(unique(e.reason for e in report.exclusions))
    rows = NamedTuple{(:reason, :holes, :cu_samples, :petro_samples),
                      Tuple{String,Int,Int,Int}}[]
    for reason in reasons
        es = [e for e in report.exclusions if e.reason == reason]
        push!(rows, (
            reason = reason,
            holes = length(es),
            cu_samples = sum(e.n_cu for e in es),
            petro_samples = sum(e.n_petro for e in es),
        ))
    end
    return rows
end

function Base.show(io::IO, report::KeivitsaReport)
    rate = report.cu_samples_read == 0 ? NaN : 100 * report.cu_censored / report.cu_samples_read
    println(io, "KeivitsaReport")
    println(io, "  511P read: ", report.cu_samples_read, " samples / ",
            report.cu_holes_read, " holes")
    println(io, "  511P censored before spatial filter: ", report.cu_censored,
            " (", round(rate; digits=3), "%)")
    println(io, "  511P kept: ", report.cu_samples_kept, " samples / ",
            report.cu_holes_kept, " holes; censored kept ", report.cu_censored_kept)
    println(io, "  511P outside bounds: ", report.cu_outside_bounds, " samples")
    println(io, "  PTR kept: ", report.ptr_samples_kept, " samples / ",
            report.ptr_holes_kept, " holes")
    println(io, "  DSR kept: ", report.dsr_samples_kept, " samples / ",
            report.dsr_holes_kept, " holes")
    println(io, "  susceptibility ≤ 0 censored at the smallest positive value: ",
            report.susc_censored)
    println(io, "  petro rows with no density or susceptibility: ",
            report.petro_rows_ignored)
    println(io, "  exclusions:")
    for row in exclusion_summary(report)
        println(io, "    ", rpad(row.reason, 22), " holes ", lpad(row.holes, 4),
                "  cu ", lpad(row.cu_samples, 6),
                "  petro ", lpad(row.petro_samples, 6))
    end
end

"""
    keivitsa_sample_table(root, cfg) -> (table::SampleTable, report::KeivitsaReport)

GTK drillholes as a [`SampleTable`](@ref).

Assay coordinates in the file are ignored. Each 511P interval is placed at
its along-hole midpoint by [`desurvey`](@ref); each PTR or DSR sample is
placed at its along-hole depth. `table.x` is easting (`KKJ_EAST` / `Kkj_y`)
and `table.y` is northing (`KKJ_NORTH` / `Kkj_X`). `group` is `"Keivitsa"`.

Cu censoring uses `properties.cu` `lod_policy = "fixed"` on every 511P row,
before the bounds filter. Flags `"<"` and `"!"`, and a numeric Cu ≤ 0, are
upper bounds at `lod`. Density is kg/m³. Susceptibility is 10⁻⁶ SI. A finite
value ≤ 0 is an upper bound at the smallest positive susceptibility
(`lod_policy = "min_positive"`, or omitted). The raw reading is written to
the log; the table stores the limit. `petro_source` is `"PTR"` or `"DSR"`,
a categorical field, not a covariate. Rows with neither density nor
susceptibility (the `LUO_R` soundings) are not samples.

A hole is dropped, not fatal to the load, when its collar elevation is 0,
it has no survey, its assay intervals overlap or repeat, or [`desurvey`](@ref)
/ [`positions`](@ref) throws. The exception is stored on the exclusion.
If those geometry failures remove more than 5% of the holes that have 511P
Cu, this function throws [`ArgumentError`](@ref) and lists the reasons.
Samples outside `[bounds]` are removed after that check.
"""
function keivitsa_sample_table(root::AbstractString, cfg::AbstractDict)
    props = _props(cfg)
    lod = _cu_lod(props)
    density_spec = _spec(_one_prop(props, "density"), :density, "continuous", "identity", "kg/m3")
    susc_spec = _spec(_one_prop(props, "susceptibility"), :susceptibility, "continuous", "log10", "1e-6 SI")
    _reject_lod(_one_prop(props, "density"), "density")
    _susc_policy(_one_prop(props, "susceptibility"))
    cu_spec = _spec(_one_prop(props, "cu"), :cu, "continuous", "log10", "ppm")
    source_spec = if haskey(props, "petro_source")
        _spec(_one_prop(props, "petro_source"), :petro_source, "categorical", "identity", "")
    else
        PropertySpec(:petro_source, :categorical, :identity, "")
    end
    source_spec.kind === :categorical || throw(ArgumentError(
        "keivitsa_sample_table: petro_source must be categorical"))
    box = _box_of(cfg)

    collar_path = _cfg_path(root, cfg, "collar", _DEFAULT_COLLAR)
    survey_path = _cfg_path(root, cfg, "survey", _DEFAULT_SURVEY)
    assay_path = _cfg_path(root, cfg, "assays", _DEFAULT_ASSAYS)
    petro_path = _cfg_path(root, cfg, "petrophysics", _DEFAULT_PETRO)

    collars = _read_collars(collar_path)
    survey = _read_survey(survey_path)
    cu_by, other_method, missing_cu, n_read, n_cens, cu_holes = _read_assays(assay_path, lod)
    petro_by, petro_ignored = _read_petro(petro_path)

    exclusions = KeivitsaExclusion[]
    for (hole, n) in sort!(collect(pairs(other_method)))
        _drop!(exclusions, hole, "other_method",
               "assay method is not 511P", n, 0)
    end
    for (hole, n) in sort!(collect(pairs(missing_cu)))
        _drop!(exclusions, hole, "missing_cu",
               "511P row has no Cu value and no flag", n, 0)
    end

    placed_by = Dict{String,Vector{_Placed}}()
    hole_ids = sort!(collect(union(keys(cu_by), keys(petro_by))))
    for hole in hole_ids
        cu = get(cu_by, hole, _Cu[])
        petro = get(petro_by, hole, _Petro[])
        n_cu = length(cu)
        n_petro = length(petro)
        collar = get(collars, hole, nothing)
        if collar === nothing
            _drop!(exclusions, hole, "no_collar", "no collar record", n_cu, n_petro)
            continue
        end
        if !isfinite(collar.z) || collar.z == 0
            _drop!(exclusions, hole, "missing_elevation",
                   "collar elevation is 0 or non-finite", n_cu, n_petro)
            continue
        end
        stations = get(survey, hole, nothing)
        if stations === nothing || isempty(stations)
            _drop!(exclusions, hole, "no_survey", "no survey stations", n_cu, n_petro)
            continue
        end
        interval = _interval_message(cu)
        if interval !== nothing
            _drop!(exclusions, hole, "overlapping_intervals", interval, n_cu, n_petro)
            continue
        end
        conflict = _petro_conflict(petro)
        if conflict !== nothing
            _drop!(exclusions, hole, "duplicate_petro_depth", conflict, n_cu, n_petro)
            continue
        end
        # Identical petro rows at one depth are not averaged and not doubled.
        if length(petro) != length(unique(r.depth for r in petro))
            kept = _Petro[]
            seen = Set{Float64}()
            n_extra = 0
            for r in petro
                if r.depth in seen
                    n_extra += 1
                    continue
                end
                push!(seen, r.depth)
                push!(kept, r)
            end
            _drop!(exclusions, hole, "duplicate_petro_depth",
                   "identical petro rows at one depth; one kept", 0, n_extra)
            petro = kept
            n_petro = length(petro)
        end
        local placed
        try
            placed = _place_hole(collar, stations, cu, petro)
        catch err
            err isa ArgumentError || rethrow()
            msg = sprint(showerror, err)
            _drop!(exclusions, hole, _geometry_reason(err.msg), msg, n_cu, n_petro)
            continue
        end
        placed_by[hole] = placed
    end

    _guard_geometry!(exclusions, cu_holes)

    kept = Dict{String,Vector{_Placed}}()
    for (hole, rows) in placed_by
        inside = _Placed[]
        n_cu_out = 0
        n_petro_out = 0
        for r in rows
            if _inside(box, r.x, r.y, r.z)
                push!(inside, r)
            elseif isfinite(r.cu)
                n_cu_out += 1
            else
                n_petro_out += 1
            end
        end
        _drop!(exclusions, hole, "outside_bounds",
               "samples outside the site bounds", n_cu_out, n_petro_out)
        isempty(inside) || (kept[hole] = inside)
    end

    holes = sort!(collect(keys(kept)))
    n = sum(length(kept[h]) for h in holes)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    hole_col = Vector{String}(undef, n)
    group = fill("Keivitsa", n)
    sample_type = fill("drillhole", n)
    cu = fill(NaN, n)
    cu_cens = falses(n)
    density = fill(NaN, n)
    susc = fill(NaN, n)
    source = fill("", n)
    i = 1
    for hole in holes
        for r in kept[hole]
            x[i] = r.x
            y[i] = r.y
            z[i] = r.z
            hole_col[i] = hole
            cu[i] = r.cu
            cu_cens[i] = r.cu_cens
            density[i] = r.density
            susc[i] = r.susc
            source[i] = r.source
            i += 1
        end
    end
    susc_cens = falses(n)
    n_susc_cens = _censor_susceptibility!(susc, susc_cens, hole_col)
    specs = PropertySpec[cu_spec, density_spec, susc_spec, source_spec]
    table = SampleTable(
        x, y, z, hole_col, group, sample_type,
        Dict(:cu => cu, :density => density, :susceptibility => susc),
        Dict(:cu => cu_cens, :density => falses(n), :susceptibility => susc_cens),
        Dict(:petro_source => source),
        specs)
    cu_mask = observed_mask(table, :cu)
    ptr = table.classes[:petro_source] .== "PTR"
    dsr = table.classes[:petro_source] .== "DSR"
    outside = 0
    for e in exclusions
        e.reason == "outside_bounds" || continue
        outside += e.n_cu
    end
    report = KeivitsaReport(
        exclusions,
        n_read,
        length(cu_holes),
        n_cens,
        count(cu_mask),
        length(unique(table.hole[cu_mask])),
        count(cu_cens .& cu_mask),
        outside,
        count(ptr),
        length(unique(table.hole[ptr])),
        count(dsr),
        length(unique(table.hole[dsr])),
        n_susc_cens,
        petro_ignored)
    _log_report(report)
    return table, report
end
