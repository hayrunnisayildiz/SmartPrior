# Keivitsa adapter sanity checks (GTK data via KEIVITSA_ROOT only).
#
#   julia --project=. examples/keivitsa_inspect.jl
#
# Writes under tmp_keivitsa_inspect/ (gitignored). No GTK files are committed.

using SmartPrior
using GLMakie
using ArchGDAL
using Printf
using Random
using Statistics
using TOML

const ROOT_DIR = dirname(@__DIR__)
const SITE_TOML = joinpath(ROOT_DIR, "sites", "keivitsa.toml")
const WORK = joinpath(ROOT_DIR, "tmp_keivitsa_inspect")
const SECTION_NORTH = 7_512_250.0
const SECTION_HALF_WIDTH = 40.0
const AZIMUTH_PAIR_RADIUS = 20.0
const HOLE_LENGTH_TOL = 0.1
const RNG = Xoshiro(2026)

function keivitsa_root()
    cfg = TOML.parsefile(SITE_TOML)
    if haskey(cfg, "root") && !isempty(strip(String(cfg["root"])))
        r = String(cfg["root"])
        return isabspath(r) ? r : normpath(joinpath(dirname(SITE_TOML), r))
    end
    env = get(ENV, "KEIVITSA_ROOT", "")
    !isempty(env) || error("set KEIVITSA_ROOT or root in sites/keivitsa.toml")
    return env
end

function _read_latin1(path::AbstractString)
    isfile(path) || error("missing file: $path")
    return String(Char.(read(path)))
end

function caret_rows(path::AbstractString)
    text = replace(_read_latin1(path), "\r\n" => "\n", "\r" => "\n")
    lines = String[]
    for line in split(text, "\n")
        isempty(strip(line)) && continue
        push!(lines, String(line))
    end
    length(lines) >= 4 || error("$path: expected header + 3 metadata rows")
    header = String[strip(h) for h in split(lines[1], '^')]
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

function gtk_float(s)
    t = strip(String(s))
    (isempty(t) || t == "*************") && return nothing
    x = tryparse(Float64, replace(t, ',' => '.'))
    x === nothing && error("non-numeric GTK value: $(repr(t))")
    return x
end

function hole_id(raw)
    return uppercase(strip(String(raw)))
end

function resolve_gtk(root, cfg, key, default)
    rel = get(cfg, key, default)
    p = String(rel)
    return isabspath(p) ? p : normpath(joinpath(root, p))
end

function read_collars(path)
    collars = Dict{String,NamedTuple{(:east, :north, :z, :length),NTuple{4,Float64}}}()
    ArchGDAL.read(path) do ds
        layer = ArchGDAL.getlayer(ds, 0)
        for feat in layer
            id = hole_id(ArchGDAL.getfield(feat, "HOLE_ID"))
            east = Float64(ArchGDAL.getfield(feat, "KKJ_EAST"))
            north = Float64(ArchGDAL.getfield(feat, "KKJ_NORTH"))
            z = Float64(ArchGDAL.getfield(feat, "Z"))
            len = Float64(ArchGDAL.getfield(feat, "LENGTH"))
            collars[id] = (east, north, z, len)
        end
    end
    return collars
end

function read_survey(path)
    _, rows = caret_rows(path)
    by = Dict{String,Vector{NTuple{3,Float64}}}()
    for row in rows
        id = hole_id(row["Tunnus"])
        md = gtk_float(row["Syvyys"])
        az = gtk_float(row["Suunta"])
        dip = gtk_float(row["Kaltevuus"])
        (md === nothing || az === nothing || dip === nothing) && continue
        push!(get!(() -> NTuple{3,Float64}[], by, id), (md, az, dip))
    end
    for (_, st) in by
        sort!(st; by = s -> s[1])
    end
    return by
end

function read_cu511p(path, lod)
    header, rows = caret_rows(path)
    method_col = nothing
    for name in ("Method", "method", "Menetelma")
        name in header && (method_col = name)
    end
    by = Dict{String,Vector{NamedTuple{(:fr, :to, :cu, :censored),NTuple{4,Any}}}}()
    for row in rows
        if method_col !== nothing
            uppercase(strip(row[method_col])) != "511P" && continue
        end
        id = hole_id(row["Tunnus"])
        fr = gtk_float(row["Ylasyvyys"])
        to = gtk_float(row["Alasyvyys"])
        flag = strip(get(row, "Cu_L", ""))
        raw = gtk_float(get(row, "Cu", ""))
        (fr === nothing || to === nothing) && continue
        if flag == "" && raw === nothing
            continue
        end
        cens = flag in ("<", "!") || (raw !== nothing && raw <= 0)
        val = cens ? lod : raw
        push!(get!(() -> NamedTuple[], by, id),
              (fr = fr, to = to, cu = val, censored = cens))
    end
    return by
end

function desurvey_hole(collar, stations; mirror_azimuth = false)
    depths = [s[1] for s in stations]
    azs = Float64[s[2] for s in stations]
    dips = [s[3] for s in stations]
    if mirror_azimuth
        azs = [mod(360.0 - az, 360.0) for az in azs]
    end
    return desurvey((collar.east, collar.north, collar.z), depths, azs, dips;
                    angle_unit = :degree, dip_down_negative = false)
end

function path_length_md(path::DesurveyPath)
    return path.depth[end]
end

const _AZ180_TOL = 1e-6

function azimuth_multiple_of_180(az::Real)
    a = mod(Float64(az), 360.0)
    rem180 = mod(a, 180.0)
    return rem180 <= _AZ180_TOL || rem180 >= 180.0 - _AZ180_TOL
end

function affected_hole(stations)
    any(s -> s[3] < 85.0 && !azimuth_multiple_of_180(s[2]), stations)
end

function affected_holes(survey)
    Set(h for (h, st) in survey if affected_hole(st))
end

function log10_cu(v)
    return log10(Float64(v))
end

function _idw_z(cov::DepthBelowSurface, x::Float64, y::Float64)
    num = 0.0
    den = 0.0
    n_hit = 0
    hit = 0.0
    @inbounds for i in eachindex(cov.east)
        d = hypot(x - cov.east[i], y - cov.north[i])
        if d == 0
            n_hit += 1
            hit += cov.elevation[i]
            continue
        end
        w = inv(d^2)
        num += w * cov.elevation[i]
        den += w
    end
    n_hit == 0 || return hit / n_hit
    return num / den
end

function physical_depth_below_surface(cov::DepthBelowSurface, x, y, z)
    out = similar(z, Float64)
    for i in eachindex(z)
        out[i] = _idw_z(cov, Float64(x[i]), Float64(y[i])) - Float64(z[i])
    end
    return out
end

function _inside_bounds(cfg, x, y, z)
    b = cfg["bounds"]
    return b["x"][1] <= x <= b["x"][2] && b["y"][1] <= y <= b["y"][2] &&
           b["z"][1] <= z <= b["z"][2]
end

function _match_interval_md(collar, stations, cu_rows, tx, ty, tz)
    path = desurvey_hole(collar, stations)
    best = Inf
    best_md = NaN
    for r in cu_rows
        md = (r.fr + r.to) / 2
        x, y, z = positions(path, md)
        d = hypot(x - tx, y - ty, z - tz)
        d < best && (best = d; best_md = md)
    end
    return best_md
end

function azimuth_cu_pool(table, cu_by, collars, survey)
    pool = typeof((hole = "", cu = 0.0, logcu = 0.0, md = 0.0,
                    tx = 0.0, ty = 0.0, tz = 0.0))[]
    cu_cens = table.censored[:cu]
    cu_val = table.values[:cu]
    cu_obs = observed_mask(table, :cu)
    for i in eachindex(table.hole)
        cu_obs[i] || continue
        cu_cens[i] && continue
        cu_val[i] <= 1.0 && continue
        hole = table.hole[i]
        collar = get(collars, hole, nothing)
        collar === nothing && continue
        st = get(survey, hole, nothing)
        st === nothing && continue
        haskey(cu_by, hole) || continue
        md = _match_interval_md(collar, st, cu_by[hole], table.x[i], table.y[i], table.z[i])
        push!(pool, (
            hole = hole,
            cu = cu_val[i],
            logcu = log10_cu(cu_val[i]),
            md = md,
            tx = table.x[i],
            ty = table.y[i],
            tz = table.z[i],
        ))
    end
    return pool
end

function crosshole_azimuth_metrics(pool, source_holes, collars, survey; mirror = false)
    hole_index = Dict{String,Int}()
    pts = Vector{NTuple{6,Float64}}() # x,y,z,logcu,hole_idx,source_flag
    for s in pool
        collar = collars[s.hole]
        st = survey[s.hole]
        if mirror
            path = desurvey_hole(collar, st; mirror_azimuth = true)
            x, y, z = positions(path, s.md)
        else
            x, y, z = s.tx, s.ty, s.tz
        end
        if !haskey(hole_index, s.hole)
            hole_index[s.hole] = length(hole_index) + 1
        end
        src = s.hole in source_holes ? 1.0 : 0.0
        push!(pts, (x, y, z, s.logcu, hole_index[s.hole], src))
    end
    n = length(pts)
    n < 2 && return (corr = NaN, mean_abs_delta = NaN, n_pairs = 0)
    matched = NTuple{2,Float64}[]
    for i in 1:n
        pts[i][6] == 0 && continue
        xi, yi, zi, ci, hi = pts[i][1], pts[i][2], pts[i][3], pts[i][4], pts[i][5]
        best = Inf
        best_j = 0
        for j in 1:n
            i == j && continue
            hi == pts[j][5] && continue
            d = hypot(xi - pts[j][1], yi - pts[j][2], zi - pts[j][3])
            d <= AZIMUTH_PAIR_RADIUS || continue
            d < best && (best = d; best_j = j)
        end
        best_j == 0 && continue
        push!(matched, (ci, pts[best_j][4]))
    end
    isempty(matched) && return (corr = NaN, mean_abs_delta = NaN, n_pairs = 0)
    a = [p[1] for p in matched]
    b = [p[2] for p in matched]
    μa, μb = mean(a), mean(b)
    num = sum((a[i] - μa) * (b[i] - μb) for i in eachindex(a))
    den = sqrt(sum((a[i] - μa)^2 for i in eachindex(a)) *
               sum((b[i] - μb)^2 for i in eachindex(b)))
    corr = den == 0 ? NaN : num / den
    mad = mean(abs.(a .- b))
    return (corr = corr, mean_abs_delta = mad, n_pairs = length(matched))
end

function median_nn_collars(collars)
    ids = String[]
    east = Float64[]
    north = Float64[]
    for (id, c) in collars
        isfinite(c.z) && c.z != 0 || continue
        push!(ids, id)
        push!(east, c.east)
        push!(north, c.north)
    end
    n = length(ids)
    n <= 1 && return NaN
    dists = Float64[]
    for i in 1:n
        best = Inf
        for j in 1:n
            i == j && continue
            d = hypot(east[i] - east[j], north[i] - north[j])
            d < best && (best = d)
        end
        push!(dists, best)
    end
    return median(dists)
end

function read_kivit_lithology(path)
    _, rows = caret_rows(path)
    by = Dict{String,Vector{Tuple{Float64,Float64,String}}}()
    for row in rows
        id = hole_id(row["Tunnus"])
        fr = gtk_float(row["Ylasyvyys"])
        to = gtk_float(row["Alasyvyys"])
        rock = strip(row["Kivilaji"])
        (fr === nothing || to === nothing) && continue
        isempty(rock) && continue
        push!(get!(() -> Tuple{Float64,Float64,String}[], by, id), (fr, to, rock))
    end
    for v in values(by)
        sort!(v; by = iv -> iv[1])
    end
    return by
end

function lithology_at(litho_by, hole, depth)
    rows = get(litho_by, hole, nothing)
    rows === nothing && return nothing
    for (fr, to, rock) in rows
        fr <= depth < to && return rock
    end
    return nothing
end

function read_petro_with_lithology(petro_path, litho_by)
    _, rows = caret_rows(petro_path)
    ptr = Dict{String,Vector{NTuple{2,Float64}}}()
    dsr = Dict{String,Vector{NTuple{2,Float64}}}()
    n_no_rock = 0
    n_total = 0
    for row in rows
        id = hole_id(row["Tunnus"])
        md = gtk_float(row["Syvyys"])
        md === nothing && continue
        pd = gtk_float(row["PTR_D"])
        pk = gtk_float(row["PTR_K"])
        dd = gtk_float(row["DSR_D"])
        dk = gtk_float(row["DSR_K"])
        has_ptr = pd !== nothing && pk !== nothing
        has_dsr = dd !== nothing && dk !== nothing
        (has_ptr || has_dsr) || continue
        n_total += 1
        rock = lithology_at(litho_by, id, md)
        rock === nothing && (n_no_rock += 1; continue)
        if has_ptr
            push!(get!(() -> NTuple{2,Float64}[], ptr, rock), (pd, pk))
        else
            push!(get!(() -> NTuple{2,Float64}[], dsr, rock), (dd, dk))
        end
    end
    return ptr, dsr, n_no_rock, n_total
end


function write_counts(report, table, cfg, surf, out_dir)
    cu_m = observed_mask(table, :cu)
    dens_m = observed_mask(table, :density)
    susc_m = observed_mask(table, :susceptibility)
    cu_cens = count(table.censored[:cu] .& cu_m)
    susc_cens = count(table.censored[:susceptibility] .& susc_m)
    lod = Float64(cfg["properties"]["cu"]["lod"])
    open(joinpath(out_dir, "counts.tsv"), "w") do io
        println(io, "metric\tvalue")
        for (label, val) in (
            ("cu_samples", count(cu_m)),
            ("cu_holes", length(unique(table.hole[cu_m]))),
            ("cu_censored", cu_cens),
            ("cu_lod_ppm", lod),
            ("density_samples", count(dens_m)),
            ("density_holes", length(unique(table.hole[dens_m]))),
            ("susceptibility_samples", count(susc_m)),
            ("susceptibility_holes", length(unique(table.hole[susc_m]))),
            ("susceptibility_censored", susc_cens),
            ("ptr_samples", report.ptr_samples_kept),
            ("ptr_holes", report.ptr_holes_kept),
            ("dsr_samples", report.dsr_samples_kept),
            ("dsr_holes", report.dsr_holes_kept),
        )
            println(io, label, '\t', val)
        end
        b = cfg["bounds"]
        for axis in ("x", "y", "z")
            v = b[axis]
            println(io, "bounds_$(axis)_min\t", v[1])
            println(io, "bounds_$(axis)_max\t", v[2])
        end
    end
end

function plot_traces(cu_by, collars, survey, table, out_dir)
    cu_mask = observed_mask(table, :cu)
    cu_holes = Set(table.hole[cu_mask])
    traces = Vector{NTuple{3,Vector{Float64}}}()
    cu_x = Float64[]
    cu_y = Float64[]
    cu_z = Float64[]
    cu_c = Float64[]
    for hole in cu_holes
        collar = get(collars, hole, nothing)
        collar === nothing && continue
        stations = get(survey, hole, nothing)
        stations === nothing && continue
        path = desurvey_hole(collar, stations)
        push!(traces, (copy(path.east), copy(path.north), copy(path.z)))
    end
    for i in eachindex(table.hole)
        cu_mask[i] || continue
        v = table.values[:cu][i]
        isfinite(v) || continue
        push!(cu_x, table.x[i])
        push!(cu_y, table.y[i])
        push!(cu_z, table.z[i])
        push!(cu_c, log10(max(v, 1.0)))
    end
    GLMakie.activate!(; visible = false)

    fig3 = Figure(size = (1600, 1200), backgroundcolor = :white)
    ax3 = Axis3(fig3[1, 1]; xlabel = "Easting (m)", ylabel = "Northing (m)",
                zlabel = "Elevation (m)", title = "Desurveyed traces, log₁₀ Cu")
    for (ex, ny, zz) in traces
        lines!(ax3, ex, ny, zz; color = (:black, 0.35), linewidth = 0.8)
    end
    if !isempty(cu_x)
        sc = scatter!(ax3, cu_x, cu_y, cu_z; color = cu_c, colormap = :plasma,
                      colorrange = extrema(cu_c), markersize = 6)
        Colorbar(fig3[1, 2], sc, label = "log₁₀ Cu (ppm)")
    end
    save(joinpath(out_dir, "traces_3d.png"), fig3)

    figp = Figure(size = (1400, 1100), backgroundcolor = :white)
    axp = Axis(figp[1, 1]; xlabel = "Easting (m)", ylabel = "Northing (m)",
               title = "Plan view, log₁₀ Cu", aspect = DataAspect())
    for (ex, ny, _) in traces
        lines!(axp, ex, ny; color = (:gray60, 0.4), linewidth = 0.6)
    end
    if !isempty(cu_x)
        scp = scatter!(axp, cu_x, cu_y; color = cu_c, colormap = :plasma,
                       colorrange = extrema(cu_c), markersize = 5)
        Colorbar(figp[1, 2], scp, label = "log₁₀ Cu (ppm)")
    end
    save(joinpath(out_dir, "traces_plan.png"), figp)

    band = SECTION_HALF_WIDTH
    figs = Figure(size = (1400, 900), backgroundcolor = :white)
    axs = Axis(figs[1, 1]; xlabel = "Easting (m)", ylabel = "Elevation (m)",
               title = @sprintf("E–W section, northing %.0f ± %.0f m", SECTION_NORTH, band))
    for (ex, ny, zz) in traces
        mask = abs.(ny .- SECTION_NORTH) .<= band
        count(mask) >= 2 || continue
        lines!(axs, ex[mask], zz[mask]; color = (:gray60, 0.45), linewidth = 0.7)
    end
    sec = abs.(cu_y .- SECTION_NORTH) .<= band
    if any(sec)
        scs = scatter!(axs, cu_x[sec], cu_z[sec]; color = cu_c[sec], colormap = :plasma,
                       colorrange = extrema(cu_c), markersize = 6)
        Colorbar(figs[1, 2], scs, label = "log₁₀ Cu (ppm)")
    end
    save(joinpath(out_dir, "section_ew_n7512250.png"), figs)
end

function write_hole_lengths(cu_holes, collars, survey, out_dir)
    pool = collect(cu_holes)
    shuffle!(RNG, pool)
    chosen = pool[1:min(5, length(pool))]
    open(joinpath(out_dir, "hole_lengths.tsv"), "w") do io
        println(io, "hole\te_collar\tn_collar\tz_collar\te_eoh\tn_eoh\tz_eoh\tmd_last_station\tcollar_length_m\tabs_delta_m")
        for hole in chosen
            c = collars[hole]
            st = survey[hole]
            path = desurvey_hole(c, st)
            md = path.depth[end]
            ee, nn, zz = positions(path, md)
            delta = abs(md - c.length)
            @printf io "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f\n" hole c.east c.north c.z ee nn zz md c.length delta
        end
    end
end

function main()
    mkpath(WORK)
    gtk = keivitsa_root()
    cfg = TOML.parsefile(SITE_TOML)
    table, covs, _ = load_site(SITE_TOML)
    _, report = keivitsa_sample_table(gtk, cfg)
    surf = only(c for c in covs if c isa DepthBelowSurface)

    collar_path = resolve_gtk(gtk, cfg, "collar",
                              "report/3_DRILLINGS/Logs/Shape_files/collar.shp")
    survey_path = resolve_gtk(gtk, cfg, "survey",
                              "report/3_DRILLINGS/Logs/Shape_files/kalte.txt")
    assay_path = resolve_gtk(gtk, cfg, "assays",
                             "report/3_DRILLINGS/Assays/Shape_files/511P.txt")
    petro_path = resolve_gtk(gtk, cfg, "petrophysics",
                             "report/3_DRILLINGS/Downhole_soundings_and_core_measurements/petro.txt")

    collars = read_collars(collar_path)
    survey = read_survey(survey_path)
    lod = Float64(cfg["properties"]["cu"]["lod"])
    cu_by = read_cu511p(assay_path, lod)

    write_counts(report, table, cfg, surf, WORK)

    nn = median_nn_collars(collars)
    open(joinpath(WORK, "collar_spacing.tsv"), "w") do io
        println(io, "metric\tvalue")
        println(io, "median_nn_collar_m\t", nn)
    end

    cu_mask = observed_mask(table, :cu)
    cu_holes = Set(table.hole[cu_mask])
    plot_traces(cu_by, collars, survey, table, WORK)
    write_hole_lengths(cu_holes, collars, survey, WORK)

    affected = affected_holes(survey)
    az_pool = azimuth_cu_pool(table, cu_by, collars, survey)
    n_source = count(s -> s.hole in affected, az_pool)
    cw = crosshole_azimuth_metrics(az_pool, affected, collars, survey; mirror = false)
    ccw = crosshole_azimuth_metrics(az_pool, affected, collars, survey; mirror = true)
    open(joinpath(WORK, "azimuth_crosshole.tsv"), "w") do io
        println(io, "metric\tvalue")
        println(io, "n_uncensored_cu_pool\t", length(az_pool))
        println(io, "n_source_samples_affected_holes\t", n_source)
        println(io, "n_affected_holes\t", length(affected))
        println(io, "convention\tcorr\tmean_abs_delta_log10_cu\tn_pairs")
        @printf io "clockwise\t%.4f\t%.4f\t%d\n" cw.corr cw.mean_abs_delta cw.n_pairs
        @printf io "counterclockwise\t%.4f\t%.4f\t%d\n" ccw.corr ccw.mean_abs_delta ccw.n_pairs
    end

    kivit_path = resolve_gtk(gtk, cfg, "lithology",
                             "report/3_DRILLINGS/Logs/Shape_files/kivit.txt")
    litho = read_kivit_lithology(kivit_path)
    ptr, dsr, n_no_rock, n_petro_join = read_petro_with_lithology(petro_path, litho)
    rocks = sort!(collect(keys(ptr)))
    rocks = filter(r -> length(get(ptr, r, NTuple{2,Float64}[])) >= 30 &&
                        length(get(dsr, r, NTuple{2,Float64}[])) >= 30, rocks)
    open(joinpath(WORK, "ptr_dsr_by_rock.tsv"), "w") do io
        println(io, "rock_type\tsource\tmedian_density_kg_m3\tmedian_susceptibility_1e-6_SI\tn")
        for rock in rocks
            pd = [p[1] for p in ptr[rock]]
            pk = [p[2] for p in ptr[rock]]
            dd = [p[1] for p in dsr[rock]]
            dk = [p[2] for p in dsr[rock]]
            @printf io "%s\tPTR\t%.2f\t%.1f\t%d\n" rock median(pd) median(pk) length(pd)
            @printf io "%s\tDSR\t%.2f\t%.1f\t%d\n" rock median(dd) median(dk) length(dd)
        end
    end
    open(joinpath(WORK, "petro_lithology_join.tsv"), "w") do io
        println(io, "metric\tvalue")
        println(io, "petro_ptr_or_dsr_rows\t", n_petro_join)
        println(io, "no_kivilaji_match\t", n_no_rock)
    end

    dbs = physical_depth_below_surface(surf, table.x[cu_mask], table.y[cu_mask],
                                     table.z[cu_mask])
    d0 = surf.d0
    open(joinpath(WORK, "depth_below_surface.tsv"), "w") do io
        println(io, "metric\tvalue")
        println(io, "d0_m\t", d0)
        println(io, "min_m\t", minimum(dbs))
        println(io, "max_m\t", maximum(dbs))
        println(io, "median_m\t", median(dbs))
        println(io, "n_cu_samples\t", length(dbs))
        println(io, "n_below_d0\t", count(<(d0), dbs))
    end

    open(joinpath(WORK, "report.txt"), "w") do io
        println(io, "Keivitsa inspect — ", WORK)
        println(io, "GTK root: ", gtk)
        println(io, report)
        println(io, "median NN collar spacing (Z≠0): ", round(nn; digits = 2), " m")
        @printf io "azimuth clockwise: corr=%.3f mean|Δ|=%.3f pairs=%d\n" cw.corr cw.mean_abs_delta cw.n_pairs
        @printf io "azimuth counterclockwise: corr=%.3f mean|Δ|=%.3f pairs=%d\n" ccw.corr ccw.mean_abs_delta ccw.n_pairs
        @printf io "depth_below_surface on Cu: [%.2f, %.2f] m; n(d < d0=%.1f) = %d\n" minimum(dbs) maximum(dbs) d0 count(<(d0), dbs)
    end

    @info "keivitsa_inspect done" work=WORK
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
