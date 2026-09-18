# Full-grid inference + VTK block-model export for a trained Keivitsa prior.
#
# Separate from examples/compare_prior_2d.jl: no gravity, no MT, no VFSA.
# Rebuilds the same feature stack as examples/train_keivitsa_prior.jl, loads
# tmp_keivitsa_prior/keivitsa_prior.jld2, and writes a VTK StructuredGrid that
# follows the NISAI hybrid grid layout (x = KKJ easting, y = northing, z = up,
# point data, Fortran / VTK axis order) so ParaView and PyVista can open it.
#
# PRELIMINARY: LUO_R resistivity mapping unverified; density and susceptibility
# are high-confidence inferences (see KEIVITSA_PETRO_STATUS).
# Not a starting model for inversion.
#
# Run:  julia --project=. examples/export_keivitsa_blockmodel.jl
# Env:  KEIVITSA_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z,
#       SMARTPRIOR_PYTHON, SMARTPRIOR_TRAIN_SEED, SMARTPRIOR_TAG, SMARTPRIOR_EPOCHS,
#       SMARTPRIOR_WIDTH, SMARTPRIOR_DEPTH

using SmartPriorMT
using Printf
using Random
using Statistics
using Plots

get!(ENV, "GKSwstype", "100")
gr()

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_keivitsa_prior"))
const CELL = parse(Float64, get(ENV, "SMARTPRIOR_CELL_M", "25"))
const CELL_Z = parse(Float64, get(ENV, "SMARTPRIOR_CELL_Z", string(CELL)))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const TAG = get(ENV, "SMARTPRIOR_TAG", "")
const EPOCHS_LABEL = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "0"))
const CUTOFF_PPM = 5000.0
const WIDTH = parse(Int, get(ENV, "SMARTPRIOR_WIDTH", "128"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const N_BANDS = 4
const PROPERTY_NAMES = ["grade", "density", "susceptibility", "resistivity"]
tagged(stem, ext) = joinpath(WORK, stem * TAG * ext)
const CHECKPOINT = joinpath(WORK, "keivitsa_prior.jld2")
const VTS_PATH = tagged("keivitsa_smartprior_blockmodel", ".vts")
const NOTE_PATH = tagged("keivitsa_smartprior_blockmodel", ".md")
const PNG_5000 = tagged("keivitsa_smartprior_cu_5000ppm_iso", ".png")
const PNG_2000 = tagged("keivitsa_smartprior_cu_2000ppm_iso", ".png")
const PNG_3000 = tagged("keivitsa_smartprior_cu_3000ppm_iso", ".png")
const PLAN_PATH = tagged("keivitsa_smartprior_cu_plan", ".png")
const DRILL_CSV = tagged("keivitsa_smartprior_drillcheck", ".csv")
const PNG_PATH = PNG_5000
const NISAI_VTS = joinpath(homedir(), "nisai", "minerai-code", "database",
                           "keivitsa", "exports", "keivitsa_cu_hybrid_grid.vts")
const NISAI_DRILLS = joinpath(homedir(), "nisai", "minerai-code", "database",
                              "keivitsa", "exports", "keivitsa_cu_hybrid_drills.npz")

mkpath(WORK)

function default_keivitsa_root()
    env = get(ENV, "KEIVITSA_ROOT", "")
    !isempty(env) && return env
    for c in (
        joinpath(homedir(), "nisai", "minerai-code", "database", "keivitsa"),
        "/Users/hayrunnisayildiz/nisai/minerai-code/database/keivitsa",
    )
        isdir(joinpath(c, "source", "gtk", "report")) && return c
    end
    error("set KEIVITSA_ROOT to database/keivitsa (the folder that contains source/gtk/report)")
end

function padded_bounds(values; pad = 0.2, fallback)
    finite = filter(isfinite, values)
    isempty(finite) && return fallback
    lo, hi = extrema(finite)
    lo == hi && return (lo - pad, hi + pad)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)
end

# VTK StructuredGrid, same dialect as PyVista's .vts: point data, x fastest.
function write_structured_grid_vts(path::AbstractString,
                                   xs::AbstractVector{<:Real},
                                   ys::AbstractVector{<:Real},
                                   zs::AbstractVector{<:Real},
                                   arrays;
                                   scalars::AbstractString = "grade")
    nx, ny, nz = length(xs), length(ys), length(zs)
    npts = nx * ny * nz
    names = String[]
    blobs = Vector{Vector{Float64}}()
    for (name, a) in arrays
        length(a) == npts || throw(DimensionMismatch(
            "write_structured_grid_vts: $(name) has $(length(a)) values, expected $npts"))
        push!(names, String(name))
        push!(blobs, collect(Float64, a))
    end
    points = Vector{Float64}(undef, 3 * npts)
    t = 1
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        points[t]     = Float64(xs[i])
        points[t + 1] = Float64(ys[j])
        points[t + 2] = Float64(zs[k])
        t += 3
    end

    header = 4
    offsets = Vector{Int}(undef, length(blobs) + 1)
    off = 0
    for i in eachindex(blobs)
        offsets[i] = off
        off += header + 8 * length(blobs[i])
    end
    offsets[end] = off

    ext = "0 $(nx - 1) 0 $(ny - 1) 0 $(nz - 1)"
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">")
        println(io, "  <StructuredGrid WholeExtent=\"$ext\">")
        println(io, "    <Piece Extent=\"$ext\">")
        println(io, "      <PointData Scalars=\"$scalars\">")
        for (name, o) in zip(names, offsets)
            @printf(io,
                    "        <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"%d\"/>\n",
                    name, o)
        end
        println(io, "      </PointData>")
        println(io, "      <Points>")
        @printf(io,
                "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n",
                offsets[end])
        println(io, "      </Points>")
        println(io, "    </Piece>")
        println(io, "  </StructuredGrid>")
        print(io, "  <AppendedData encoding=\"raw\">\n_")
        for a in blobs
            write(io, UInt32(sizeof(a)))
            write(io, a)
        end
        write(io, UInt32(sizeof(points)))
        write(io, points)
        print(io, "\n  </AppendedData>\n</VTKFile>\n")
    end
    return path
end

function collect_cu_intervals(report_root, collars, surveys)
    dir = joinpath(report_root, "3_DRILLINGS", "Assays", "Shape_files")
    files = sort(filter(p -> endswith(p, ".dbf"), readdir(dir; join = true)))
    holes = String[]
    froms = Float64[]
    tos = Float64[]
    cu = Float64[]
    east = Float64[]
    north = Float64[]
    elev = Float64[]
    seen = Set{String}()
    for path in files
        t = SmartPriorMT.read_dbf(path)
        names = collect(keys(t))
        cucol = SmartPriorMT._element_column(names, "CU")
        cucol === nothing && continue
        n = length(t["HOLE_ID"])
        cuv = SmartPriorMT._dbf_float(t, cucol)
        hid = strip.(t["HOLE_ID"])
        from = SmartPriorMT._dbf_float(t, "FROM")
        to_ = SmartPriorMT._dbf_float(t, "TO")
        by_hole = Dict{String,Vector{Int}}()
        for i in 1:n
            push!(get!(Vector{Int}, by_hole, hid[i]), i)
        end
        for (hole, idxs) in by_hole
            mids = 0.5 .* (from[idxs] .+ to_[idxs])
            e, nn, z = SmartPriorMT._desurvey_hole(collars, surveys, hole, mids)
            for (k, i) in enumerate(idxs)
                v = cuv[i]
                isfinite(v) && v > 0 || continue
                key = hole * "|" * string(from[i]) * "|" * string(to_[i])
                key in seen && continue
                push!(seen, key)
                push!(holes, hole)
                push!(froms, from[i])
                push!(tos, to_[i])
                push!(cu, v)
                push!(east, e[k])
                push!(north, nn[k])
                push!(elev, z[k])
            end
        end
    end
    return (hole = holes, from = froms, to = tos, cu_ppm = cu,
            x = east, y = north, z = elev)
end

function pick_high_cu_checks(samples, grid; n_holes = 8, min_ppm = CUTOFF_PPM)
    n = length(samples.cu_ppm)
    inside = trues(n)
    for i in 1:n
        inside[i] = containing_cell(grid, samples.x[i], samples.y[i], samples.z[i]) > 0
    end
    order = sortperm(samples.cu_ppm; rev = true)
    picked = Int[]
    used = Set{String}()
    for i in order
        inside[i] || continue
        samples.cu_ppm[i] >= min_ppm || continue
        samples.hole[i] in used && continue
        push!(picked, i)
        push!(used, samples.hole[i])
        length(picked) >= n_holes && break
    end
    if length(picked) < n_holes
        for i in order
            inside[i] || continue
            i in picked && continue
            samples.hole[i] in used && continue
            push!(picked, i)
            push!(used, samples.hole[i])
            length(picked) >= n_holes && break
        end
    end
    return picked
end

function logrmse(a, b)
    m = isfinite.(a) .& isfinite.(b) .& (a .> 0) .& (b .> 0)
    δ = log10.(a[m]) .- log10.(b[m])
    return sqrt(mean(abs2.(δ)))
end

function find_pyvista_python()
    env = get(ENV, "SMARTPRIOR_PYTHON", "")
    candidates = String[]
    isempty(env) || push!(candidates, env)
    append!(candidates, (
        joinpath(homedir(), "Desktop", "minerai", ".venv", "bin", "python"),
        joinpath(homedir(), "mtproject", ".venv", "bin", "python"),
        "python3",
    ))
    for p in candidates
        try
            cmd = pipeline(`$p -c "import pyvista, numpy"`;
                           stdout = devnull, stderr = devnull)
            success(cmd) && return p
        catch
        end
    end
    return nothing
end

# Marching-cubes isosurface, same visual dialect as NISAI orebody_viz.py:
# gold palette, isometric camera, X-Y-Z axes, drill traces overlaid.
# Contour is on the predicted grade field (light 0.25-cell blur + Taubin
# polish, matching NISAI). Maximum-filter is used only to colour the shell.
function render_orebody_pyvista(python, vts_path, png_path, drills_path,
                               shells, title)
    py = """
import os, sys
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

vts, png, drills, shell_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
title = os.environ.get("ORE_TITLE", "Keivitsa SmartPrior")
grid = pv.read(vts)
if "grade" not in grid.array_names:
    raise SystemExit("VTS has no 'grade' array")
gold = LinearSegmentedColormap.from_list(
    "minerai_gold", ["#a67c00", "#c9a227", "#dbb430", "#f0d050", "#ffe566"], N=256)
pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=(1600, 1100))
plotter.set_background("white")
shells = []
for part in shell_s.split(";"):
    iso, op = part.split(",")
    shells.append((float(iso), float(op)))
cutoff = min(iso for iso, _ in shells)
grade = np.asarray(grid["grade"], dtype=np.float64)
dims = tuple(int(v) for v in grid.dimensions)
vol = grade.reshape(dims, order="F")
try:
    from scipy.ndimage import gaussian_filter, maximum_filter
    work = gaussian_filter(np.nan_to_num(vol, nan=0.0), sigma=0.25, mode="nearest")
    grid["grade_blur"] = work.ravel(order="F")
    contour_name = "grade_blur"
    masked = np.where(vol >= cutoff, vol, 0.0)
    display = maximum_filter(masked, size=3)
    grid["DisplayGrade"] = display.ravel(order="F")
    color_name = "DisplayGrade"
    color_vals = display.ravel(order="F")
except Exception:
    contour_name = "grade"
    color_name = "grade"
    color_vals = grade
above = color_vals[np.isfinite(color_vals) & (color_vals >= cutoff)]
clim_hi = float(np.percentile(above, 97)) if above.size else cutoff + 1.0
if clim_hi <= cutoff * 1.08:
    clim_hi = float(np.max(above)) if above.size else cutoff + 1.0
if clim_hi <= cutoff:
    clim_hi = cutoff + max(50.0, cutoff * 0.15)
built = 0
bar_iso = shells[0][0]
for iso, opacity in shells:
    surf = grid.contour(isosurfaces=[iso], scalars=contour_name)
    if surf.n_points == 0:
        print("EMPTY_ISO", iso)
        continue
    n_iter = 12 if iso >= 4000.0 else 25
    taubin = getattr(surf, "smooth_taubin", None)
    if callable(taubin):
        try:
            surf = taubin(n_iter=n_iter, pass_band=0.1)
        except Exception:
            pass
    if color_name != contour_name:
        surf = surf.sample(grid)
    built += 1
    plotter.add_mesh(
        surf, scalars=color_name, cmap=gold, clim=(cutoff, clim_hi),
        smooth_shading=True, opacity=opacity,
        show_scalar_bar=(abs(iso - bar_iso) < 1e-6),
        scalar_bar_args={
            "title": "Cu (ppm) - Copper Grade",
            "vertical": True, "position_x": 0.82, "position_y": 0.22,
            "fmt": "%.0f", "title_font_size": 12, "label_font_size": 10,
        },
        name=f"iso_{int(iso)}",
    )
if built == 0:
    raise SystemExit("empty isosurface")
if drills and os.path.isfile(drills):
    d = np.load(drills, allow_pickle=True)
    x, y, z = d["x"], d["y"], d["z"]
    hids = np.asarray(d["holeid"]).astype(str)
    for hid in np.unique(hids):
        m = hids == hid
        pts = np.column_stack((x[m], y[m], z[m]))
        if len(pts) < 2:
            continue
        order = np.argsort(-pts[:, 2])
        pts = pts[order]
        plotter.add_mesh(pv.lines_from_points(pts), color="black",
                         line_width=2, opacity=0.6, name=f"dh_{hid}")
plotter.add_axes()
plotter.add_text(title, font_size=11)
plotter.view_isometric()
plotter.camera.zoom(1.15)
plotter.show(screenshot=png, auto_close=True)
print("WROTE", png)
"""
    pyfile = joinpath(WORK, "_render_orebody_iso.py")
    write(pyfile, py)
    shell_s = join(["$(Int(iso)),$(op)" for (iso, op) in shells], ";")
    run(setenv(`$python $pyfile $vts_path $png_path $drills_path $shell_s`,
               merge(ENV, Dict("ORE_TITLE" => String(title),
                               "PYVISTA_OFF_SCREEN" => "true"))))
    return isfile(png_path)
end

function epoch_tag()
    zlab = abs(CELL_Z - CELL) > 1e-9 ? @sprintf("XY %.0f m, Z %.0f m", CELL, CELL_Z) :
        @sprintf("%.0f m", CELL)
    ep = EPOCHS_LABEL > 0 ? "$(EPOCHS_LABEL) epoch" : "checkpoint"
    return "$ep, $zlab"
end

function gold_cgrad()
    return cgrad(["#a67c00", "#c9a227", "#dbb430", "#f0d050", "#ffe566"])
end

#---------- 0. checkpoint ----------

isfile(CHECKPOINT) || error("missing $CHECKPOINT; run examples/train_keivitsa_prior.jl first")

DATASET = default_keivitsa_root()
REPORT = keivitsa_report_root(DATASET)
CONFIG = joinpath(DATASET, "config.yaml")
@info "Keivitsa export" DATASET WORK CELL CELL_Z CHECKPOINT

bounds = read_keivitsa_grid_bounds(CONFIG)
grid = keivitsa_grid(bounds; cell = CELL, cell_z = CELL_Z)
nx, ny, nz = size(grid)
n_grid = ncells(grid)
@info "prior grid" size(grid) n_grid dx = grid.dx[1] dy = grid.dy[1] dz = grid.dz[1] bounds
(nx, ny) == (50, 34) || @warn "horizontal grid is $(nx)×$(ny), expected 50×34 at 25 m XY"

#---------- 1. rebuild the training inputs ----------

collars = load_keivitsa_collars(REPORT)
surveys = load_keivitsa_surveys(REPORT)
geochem_s, _ = load_keivitsa_geochemistry(REPORT)
geochem_dh = load_keivitsa_drill_geochemistry(keivitsa_cleaned_intervals_path(DATASET))
geochem, _ = combine_geochemistry(geochem_s, geochem_dh)
lith = load_keivitsa_lithology(REPORT, collars, surveys)
z_surface = fill(bounds.z_max, length(geochem_s))
coverage = (
    x = vcat(geochem_s.x, geochem_dh.x, lith.x),
    y = vcat(geochem_s.y, geochem_dh.y, lith.y),
    z = vcat(z_surface, geochem_dh.z, lith.z),
)
stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
X = encode_features(stack; n_bands = N_BANDS)

loaded = load_prior(CHECKPOINT)
size(X, 1) == loaded.nin || throw(DimensionMismatch(
    "feature width $(size(X, 1)) != checkpoint nin $(loaded.nin)"))
nprop = get(loaded.meta, "nproperties", 4)
nprop == 4 || error("checkpoint nproperties=$nprop, expected 4")

grade_anchors, _ = load_keivitsa_grade_anchors(grid, REPORT, collars, surveys)
petro = read_petro_txt(joinpath(REPORT, "3_DRILLINGS",
                                "Downhole_soundings_and_core_measurements", "petro.txt"))
petro_anchors = load_keivitsa_petrophysics_anchors(grid, petro, collars, surveys)
mu_bounds = [
    padded_bounds(grade_anchors[2]; fallback = (0.0, 5.0)),
    padded_bounds(petro_anchors.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(petro_anchors.susceptibility[2]; fallback = (-1.0, 6.0)),
    padded_bounds(petro_anchors.resistivity[2]; fallback = (-1.0, 7.0)),
]

net = PriorNet(loaded.nin;
               width = WIDTH,
               depth = DEPTH,
               nproperties = 4,
               property_names = PROPERTY_NAMES,
               mu_bounds = mu_bounds,
               sigma_bounds = loaded.sigma_bounds,
               sigma_bounds_per = loaded.sigma_bounds_per)
_, st = setup_prior(Xoshiro(TRAIN_SEED), net)
(mus, sigmas), _ = predict(net, X, loaded.params.net, st)
size(mus) == (4, n_grid) || throw(DimensionMismatch("predict returned $(size(mus))"))

grade_ppm = 10 .^ mus[1, :]
finite_counts = Dict{String,Int}()
for (name, arr) in (
        "mu_grade" => mus[1, :], "sigma_grade" => sigmas[1, :],
        "mu_density" => mus[2, :], "sigma_density" => sigmas[2, :],
        "mu_susceptibility" => mus[3, :], "sigma_susceptibility" => sigmas[3, :],
        "mu_resistivity" => mus[4, :], "sigma_resistivity" => sigmas[4, :],
        "grade" => grade_ppm)
    finite_counts[name] = count(isfinite, arr)
end
n_ok = minimum(values(finite_counts))
n_nan = n_grid - n_ok
n_ore = count(g -> isfinite(g) && g >= CUTOFF_PPM, grade_ppm)
n_ore_2000 = count(g -> isfinite(g) && g >= 2000.0, grade_ppm)
n_ore_3000 = count(g -> isfinite(g) && g >= 3000.0, grade_ppm)
max_ppm = maximum(grade_ppm)
@info "inference" n_ok n_nan mean_mu_grade = mean(mus[1, :]) mean_grade_ppm = mean(grade_ppm) n_ore n_ore_2000 n_ore_3000 max_ppm

#---------- 2. VTK StructuredGrid ----------

write_structured_grid_vts(VTS_PATH, grid.cx, grid.cy, grid.cz, [
    "grade" => grade_ppm,
    "mu_grade" => mus[1, :],
    "sigma_grade" => sigmas[1, :],
    "mu_density" => mus[2, :],
    "sigma_density" => sigmas[2, :],
    "mu_susceptibility" => mus[3, :],
    "sigma_susceptibility" => sigmas[3, :],
    "mu_resistivity" => mus[4, :],
    "sigma_resistivity" => sigmas[4, :],
]; scalars = "grade")
@info "wrote VTS" VTS_PATH

#---------- 3. drill-hole sanity check (not a validation) ----------

samples = collect_cu_intervals(REPORT, collars, surveys)
picked = pick_high_cu_checks(samples, grid)
open(DRILL_CSV, "w") do io
    println(io, "hole,from_m,to_m,x,y,z,assay_ppm,pred_ppm,pred_cu_log,assay_cu_log,cell")
    for i in picked
        cell = containing_cell(grid, samples.x[i], samples.y[i], samples.z[i])
        pred_log = mus[1, cell]
        pred_ppm = 10^pred_log
        assay_log = cu_log10(samples.cu_ppm[i])
        @printf(io, "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.4f,%.4f,%d\n",
                samples.hole[i], samples.from[i], samples.to[i],
                samples.x[i], samples.y[i], samples.z[i],
                samples.cu_ppm[i], pred_ppm, pred_log, assay_log, cell)
    end
end

anchor_cells, anchor_log, _ = grade_anchors
anchor_ppm = 10 .^ anchor_log
pred_anchor_ppm = grade_ppm[anchor_cells]
high_mask = anchor_ppm .>= CUTOFF_PPM
anchor_rmse = logrmse(pred_anchor_ppm, anchor_ppm)
high_rmse = count(high_mask) == 0 ? NaN : logrmse(pred_anchor_ppm[high_mask], anchor_ppm[high_mask])
pred_high = count(high_mask) == 0 ? NaN : mean(pred_anchor_ppm[high_mask])
pred_bg = count(.!high_mask) == 0 ? NaN : mean(pred_anchor_ppm[.!high_mask])
assay_high = count(high_mask) == 0 ? NaN : mean(anchor_ppm[high_mask])
assay_bg = count(.!high_mask) == 0 ? NaN : mean(anchor_ppm[.!high_mask])
anchor_order = sortperm(anchor_ppm; rev = true)
n_show_cells = min(8, length(anchor_order))

println()
println(repeat("=", 88))
println("  drill sanity check  (PRELIMINARY — not a validation)")
println(repeat("=", 88))
println("  A. 1 m assay spikes vs containing-cell μ  (expected mismatch at 25 m)")
@printf("%-12s %8s %8s %10s %10s %8s %8s\n",
        "hole", "from", "to", "assay_ppm", "pred_ppm", "ratio", "cell")
for i in picked
    cell = containing_cell(grid, samples.x[i], samples.y[i], samples.z[i])
    pred_ppm = 10^mus[1, cell]
    ratio = pred_ppm / samples.cu_ppm[i]
    @printf("%-12s %8.1f %8.1f %10.0f %10.0f %8.2f %8d\n",
            samples.hole[i], samples.from[i], samples.to[i],
            samples.cu_ppm[i], pred_ppm, ratio, cell)
end
println()
println("  B. highest cell-mean assay anchors  (the quantity the network sees)")
@printf("%-8s %12s %12s %8s\n", "cell", "assay_ppm", "pred_ppm", "ratio")
for k in 1:n_show_cells
    i = anchor_order[k]
    cell = anchor_cells[i]
    pred = pred_anchor_ppm[i]
    @printf("%-8d %12.0f %12.0f %8.2f\n", cell, anchor_ppm[i], pred, pred / anchor_ppm[i])
end
@printf("\n  grade-anchor cells: %d  log10-RMSE vs cell-mean assay  %.3f\n",
        length(anchor_cells), anchor_rmse)
@printf("  assay ≥ %.0f ppm cells: %d  mean assay %.0f ppm  mean pred %.0f ppm  log10-RMSE %.3f\n",
        CUTOFF_PPM, count(high_mask), assay_high, pred_high, high_rmse)
@printf("  remaining anchors:     %d  mean assay %.0f ppm  mean pred %.0f ppm\n",
        count(.!high_mask), assay_bg, pred_bg)
println("  High-assay cells are elevated vs background in the prediction, but amplitude is compressed.")
println(repeat("=", 88))

#---------- 4. figures (3D isosurface vs 2D plan are separate) ----------

colmax = reshape(grade_ppm, nx, ny, nz)
colmax = dropdims(maximum(colmax; dims = 3), dims = 3)
plan_hi = max(3000.0, maximum(colmax))
plt_plan = heatmap(grid.cx, grid.cy, permutedims(colmax, (2, 1));
                   c = gold_cgrad(),
                   clims = (0, plan_hi),
                   xlabel = "easting (m, EPSG:2393)",
                   ylabel = "northing (m)",
                   colorbar_title = "max Cu ppm",
                   title = "2D plan view (location/contour only) — column-max μ_grade, $(epoch_tag())",
                   size = (1100, 800),
                   aspect_ratio = :equal,
                   framestyle = :box)
contour!(plt_plan, grid.cx, grid.cy, permutedims(colmax, (2, 1));
         levels = [2000.0], c = :black, lw = 2, colorbar = false)
contour!(plt_plan, grid.cx, grid.cy, permutedims(colmax, (2, 1));
         levels = [3000.0], c = :white, lw = 2, colorbar = false)
if n_ore > 0
    contour!(plt_plan, grid.cx, grid.cy, permutedims(colmax, (2, 1));
             levels = [CUTOFF_PPM], c = :red, lw = 2, colorbar = false)
end
if !isempty(picked)
    scatter!(plt_plan,
             [samples.x[i] for i in picked],
             [samples.y[i] for i in picked];
             m = :diamond, ms = 7, markercolor = :black,
             markerstrokecolor = :white, markerstrokewidth = 1, label = "high-Cu holes")
end
savefig(plt_plan, PLAN_PATH)

py = find_pyvista_python()
pyvista_ok = Dict{Int,Bool}()
drills_arg = isfile(NISAI_DRILLS) ? NISAI_DRILLS : ""
iso_jobs = [
    (2000, PNG_2000, [(2000.0, 0.92)],
     "Keivitsa SmartPrior ($(epoch_tag())) — Cu >= 2000 ppm isosurface"),
    (3000, PNG_3000, [(3000.0, 0.92)],
     "Keivitsa SmartPrior ($(epoch_tag())) — Cu >= 3000 ppm isosurface"),
    (5000, PNG_5000, [(5000.0, 0.75), (6500.0, 1.00)],
     "Keivitsa SmartPrior ($(epoch_tag())) — Cu >= 5000 ppm isosurface"),
]
if py === nothing
    @warn "no PyVista python; 3D isosurface PNGs were not written"
else
    for (iso, path, shells, title) in iso_jobs
        n_at = count(g -> isfinite(g) && g >= iso, grade_ppm)
        if n_at == 0
            @info "skipping empty isosurface" iso max_ppm
            pyvista_ok[iso] = false
            continue
        end
        try
            pyvista_ok[iso] = render_orebody_pyvista(py, VTS_PATH, path, drills_arg,
                                                    shells, title)
            @info "PyVista isosurface" iso path
        catch err
            pyvista_ok[iso] = false
            @warn "PyVista isosurface failed" iso err
        end
    end
end

#---------- 5. accompanying note ----------

n_high = count(high_mask)
pred_ratio = (isfinite(pred_high) && isfinite(pred_bg) && pred_bg > 0) ?
    pred_high / pred_bg : NaN
assay_ratio = (isfinite(assay_high) && isfinite(assay_bg) && assay_bg > 0) ?
    assay_high / assay_bg : NaN

open(NOTE_PATH, "w") do io
    println(io, "# Keivitsa SmartPrior block model — PRELIMINARY")
    println(io)
    println(io, "This is an early-stage ($(epoch_tag())) neural-field prior, **not** an inversion")
    println(io, "result and **not** a VFSA starting model. Density and susceptibility are")
    println(io, "high-confidence inferences from GTK's standard combined measurement;")
    println(io, "LUO_R resistivity remains unverified (see `KEIVITSA_PETRO_STATUS` in")
    println(io, "`src/KeivitsaIO.jl`). Do not treat the resistivity field as a calibrated")
    println(io, "physical property.")
    println(io)
    println(io, "## Files")
    println(io, "- VTK StructuredGrid: `$(basename(VTS_PATH))`")
    println(io, "- 3D isosurface 2000 ppm: `$(basename(PNG_2000))`")
    println(io, "- 3D isosurface 3000 ppm: `$(basename(PNG_3000))`")
    println(io, "- 3D isosurface 5000 ppm: `$(basename(PNG_5000))`")
    println(io, "- 2D column-max plan (location/contour only): `$(basename(PLAN_PATH))`")
    println(io, "- Drill comparison: `$(basename(DRILL_CSV))`")
    println(io)
    println(io, "## Grid")
    @printf(io, "- cells: %d × %d × %d = %d (dx=dy ≈ %.1f m, dz ≈ %.1f m)\n",
            nx, ny, nz, n_grid, CELL, grid.dz[1])
    println(io, "- CRS: EPSG:2393 (KKJ Finland Zone 3); x = easting, y = northing, z = elevation up")
    @printf(io, "- bounds: x [%.1f, %.1f], y [%.1f, %.1f], z [%.1f, %.1f]\n",
            bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max,
            bounds.z_min, bounds.z_max)
    println(io, "- VTK layout matches `keivitsa_cu_hybrid_grid.vts`: StructuredGrid, point data,")
    println(io, "  x fastest (VTK / Fortran). Prediction locations are **cell centres** of this")
    @printf(io, "  mesh (dx=dy=%.0f m, dz=%.1f m), not the NISAI 60×40×35 sample lattice.\n",
            CELL, grid.dz[1])
    println(io)
    println(io, "## Arrays (per cell)")
    println(io, "| name | space | finite |")
    println(io, "|---|---|---|")
    for name in ("grade", "mu_grade", "sigma_grade", "mu_density", "sigma_density",
                 "mu_susceptibility", "sigma_susceptibility", "mu_resistivity",
                 "sigma_resistivity")
        space = name == "grade" ? "Cu ppm = 10^(Cu_Log)" :
                startswith(name, "sigma") ? "network σ (same space as μ)" :
                name == "mu_grade" ? "Cu_Log = log10(ppm)" :
                name == "mu_density" ? "g/cm³ (PTR_D/DSR_D / 1000, high-confidence inference)" :
                name == "mu_susceptibility" ? "log10(J) (PTR_J/DSR_J, high-confidence inference)" :
                "log10(Ω·m) (LUO_R, unverified)"
        @printf(io, "| `%s` | %s | %d / %d |\n", name, space, finite_counts[name], n_grid)
    end
    println(io)
    @printf(io, "- NaN / missing cells: **%d**\n", n_nan)
    @printf(io, "- predicted Cu range: %.1f – %.1f ppm (mean %.1f)\n",
            minimum(grade_ppm), max_ppm, mean(grade_ppm))
    @printf(io, "- cells with predicted Cu ≥ 2000 ppm: **%d** (%.2f%%)\n",
            n_ore_2000, 100 * n_ore_2000 / n_grid)
    @printf(io, "- cells with predicted Cu ≥ 3000 ppm: **%d** (%.2f%%)\n",
            n_ore_3000, 100 * n_ore_3000 / n_grid)
    @printf(io, "- cells with predicted Cu ≥ %.0f ppm: **%d** (%.2f%%)\n",
            CUTOFF_PPM, n_ore, 100 * n_ore / n_grid)
    println(io)
    if n_ore == 0
        println(io, "The NISAI hybrid figure thresholds at 5000 ppm. This field **does not")
        println(io, "reach 5000 ppm** (max $(round(max_ppm; digits=0)) ppm), so that isosurface is empty.")
        println(io, "Separate 2000 ppm and 3000 ppm marching-cubes isosurfaces are written instead.")
    else
        println(io, "The 5000 ppm marching-cubes isosurface is non-empty (NISAI comparison threshold).")
    end
    @printf(io, "- mean μ_grade (Cu_Log): %.4f\n", mean(mus[1, :]))
    println(io)
    println(io, "## Drill sanity check")
    println(io, "Not a hold-out validation.")
    println(io)
    println(io, "### A. 1 m assay spikes vs containing-cell μ")
    println(io)
    println(io, "| hole | from | to | assay ppm | pred ppm | ratio |")
    println(io, "|---|---|---|---|---|---|")
    for i in picked
        cell = containing_cell(grid, samples.x[i], samples.y[i], samples.z[i])
        pred_ppm = 10^mus[1, cell]
        @printf(io, "| %s | %.1f | %.1f | %.0f | %.0f | %.2f |\n",
                samples.hole[i], samples.from[i], samples.to[i],
                samples.cu_ppm[i], pred_ppm, pred_ppm / samples.cu_ppm[i])
    end
    println(io)
    println(io, "### B. highest cell-mean assay anchors")
    println(io)
    println(io, "| cell | assay ppm | pred ppm | ratio |")
    println(io, "|---|---|---|---|")
    for k in 1:n_show_cells
        i = anchor_order[k]
        pred = pred_anchor_ppm[i]
        @printf(io, "| %d | %.0f | %.0f | %.2f |\n",
                anchor_cells[i], anchor_ppm[i], pred, pred / anchor_ppm[i])
    end
    println(io)
    @printf(io, "Grade-anchor cell log10-RMSE (all %d cells): %.3f.\n",
            length(anchor_cells), anchor_rmse)
    @printf(io, "Among %d anchor cells with assay ≥ %.0f ppm: mean assay %.0f ppm, mean pred %.0f ppm, log10-RMSE %.3f.\n",
            n_high, CUTOFF_PPM, assay_high, pred_high, high_rmse)
    @printf(io, "Remaining %d anchors: mean assay %.0f ppm, mean pred %.0f ppm.\n",
            count(.!high_mask), assay_bg, pred_bg)
    @printf(io, "High/background mean ratio: pred %.2f×, assay %.2f×.\n",
            pred_ratio, assay_ratio)
    println(io)
    println(io, "A 25 m cell averages many assay intervals, so a single 1 m high-grade")
    println(io, "sample will not match the cell μ. High-assay cells are elevated versus")
    println(io, "background in the prediction; amplitude may still be compressed.")
    println(io)
    println(io, "## Checkpoint")
    println(io, "- `$CHECKPOINT`")
    println(io, "- architecture: width=$WIDTH, depth=$DEPTH, n_bands=$N_BANDS, nproperties=4")
    println(io, "- requested epochs: $EPOCHS_LABEL")
    println(io, "- μ bounds (squash, recomputed from anchors): $mu_bounds")
end

println()
@printf("VTS     %s\n", VTS_PATH)
@printf("cells   %d / %d finite  (NaN=%d)\n", n_ok, n_grid, n_nan)
@printf("Cu ppm  min %.1f  max %.1f  mean %.1f\n", minimum(grade_ppm), max_ppm, mean(grade_ppm))
@printf("ore     ≥2000: %d   ≥3000: %d   ≥5000: %d\n", n_ore_2000, n_ore_3000, n_ore)
@printf("iso2000 %s  (%s)\n", PNG_2000, get(pyvista_ok, 2000, false) ? "PyVista" : "empty/failed")
@printf("iso3000 %s  (%s)\n", PNG_3000, get(pyvista_ok, 3000, false) ? "PyVista" : "empty/failed")
@printf("iso5000 %s  (%s)\n", PNG_5000, get(pyvista_ok, 5000, false) ? "PyVista" : "empty/failed")
@printf("plan    %s\n", PLAN_PATH)
@printf("note    %s\n", NOTE_PATH)
@printf("drills  %s\n", DRILL_CSV)
@printf("high/bg pred %.2fx  assay %.2fx  log10-RMSE %.3f  mean(mu_grade) %.4f\n",
        pred_ratio, assay_ratio, anchor_rmse, mean(mus[1, :]))
