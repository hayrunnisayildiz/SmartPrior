# Full-grid inference + VTK + PyVista isosurface for a trained Cloncurry prior.
#
# Rebuilds the same feature stack as examples/train_cloncurry_prior.jl, loads
# the checkpoint, writes a StructuredGrid .vts and a 3D Cu PNG. Not a hold-out
# and not an MT starting model. No VFSA.
#
# Run (grid D):
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
#     SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 SMARTPRIOR_WIDTH=256 \
#     julia --project=. examples/export_cloncurry_blockmodel.jl

using SmartPriorMT
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_cloncurry_prior_eh"))
const CELL = parse(Float64, get(ENV, "SMARTPRIOR_CELL_M", "100"))
const CELL_Z = parse(Float64, get(ENV, "SMARTPRIOR_CELL_Z", "50"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const WIDTH = parse(Int, get(ENV, "SMARTPRIOR_WIDTH", "256"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const BOX = lowercase(strip(get(ENV, "SMARTPRIOR_BOX", "ernest_henry")))
const CHECKPOINT = joinpath(WORK, "cloncurry_prior.jld2")
const VTS_PATH = joinpath(WORK, "cloncurry_smartprior_blockmodel.vts")
const PNG_PATH = joinpath(WORK, "cloncurry_smartprior_cu_iso.png")
const SAMPLES_CSV = joinpath(WORK, "cloncurry_smartprior_cu_samples.csv")
const ASSET_PNG = joinpath(ROOT, "docs", "assets", "cloncurry_eh_cu_iso.png")
mkpath(WORK)
mkpath(joinpath(ROOT, "docs", "assets"))

function default_cloncurry_root()
    env = get(ENV, "CLONCURRY_ROOT", "")
    !isempty(env) && return env
    for c in (
        joinpath(homedir(), "Desktop", "datasets4HY",
                 "Cloncurry_integrated_2026-09-17"),
        "/Users/hayrunnisayildiz/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17",
    )
        isdir(joinpath(c, "derived")) && return c
    end
    error("set CLONCURRY_ROOT")
end

function padded_bounds(values; pad = 0.2, fallback)
    finite = filter(isfinite, values)
    isempty(finite) && return fallback
    lo, hi = extrema(finite)
    lo == hi && return (lo - pad, hi + pad)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)
end

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

function render_cu_iso(python, vts_path, png_path, samples_csv, shells, title)
    py = """
import os, sys
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

vts, png, samples, shell_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
title = os.environ.get("ORE_TITLE", "Ernest Henry SmartPrior")
grid = pv.read(vts)
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
    taubin = getattr(surf, "smooth_taubin", None)
    if callable(taubin):
        try:
            surf = taubin(n_iter=12 if iso >= 4000.0 else 25, pass_band=0.1)
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
            "title": "predicted Cu (ppm)",
            "vertical": True, "position_x": 0.82, "position_y": 0.22,
            "fmt": "%.0f", "title_font_size": 12, "label_font_size": 10,
        },
        name=f"iso_{int(iso)}",
    )
if built == 0:
    raise SystemExit("empty isosurface")
if samples and os.path.isfile(samples):
    rows = np.loadtxt(samples, delimiter=",", skiprows=1)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    cloud = pv.PolyData(rows[:, :3])
    cloud["assay_ppm"] = rows[:, 3]
    plotter.add_mesh(cloud, color="#222222", point_size=8,
                     render_points_as_spheres=True, opacity=0.85,
                     name="cu_samples")
plotter.add_axes()
plotter.add_text(title, font_size=11)
plotter.view_isometric()
plotter.camera.zoom(1.15)
plotter.show(screenshot=png, auto_close=True)
print("WROTE", png)
"""
    pyfile = joinpath(WORK, "_render_cloncurry_iso.py")
    write(pyfile, py)
    shell_s = join(["$(Int(iso)),$(op)" for (iso, op) in shells], ";")
    run(setenv(`$python $pyfile $vts_path $png_path $samples_csv $shell_s`,
               merge(ENV, Dict("ORE_TITLE" => String(title),
                               "PYVISTA_OFF_SCREEN" => "true"))))
    return isfile(png_path)
end

isfile(CHECKPOINT) || error("missing $CHECKPOINT; train grid D first")
DATASET = default_cloncurry_root()
loaded = load_prior(CHECKPOINT)
@info "Cloncurry export" DATASET WORK CELL CELL_Z CHECKPOINT loaded.nin

samples = load_cloncurry_samples(DATASET)
bounds = if BOX in ("work", "h")
    cloncurry_work_bounds()
elseif BOX in ("ernest_henry", "eh", "d")
    cloncurry_deposit_bounds(samples, "Ernest Henry")
else
    error("SMARTPRIOR_BOX must be work or ernest_henry")
end
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)
n_grid = ncells(grid)

geochem = cloncurry_geochemistry(samples)
lith = cloncurry_lithology(samples)
coverage = cloncurry_coverage_points(samples)
stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
X = encode_features(stack; n_bands = 4)
size(X, 1) == loaded.nin || error("feature width $(size(X, 1)) ≠ checkpoint nin $(loaded.nin)")

anchors = load_cloncurry_anchors(grid, samples)
property_names = copy(CLONCURRY_PROPERTY_NAMES)
mu_bounds = [
    padded_bounds(anchors.grade[2]; fallback = (0.0, 5.0)),
    padded_bounds(anchors.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(anchors.susceptibility[2]; fallback = (-7.0, 1.0)),
    padded_bounds(anchors.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
]
net = PriorNet(loaded.nin;
               width = WIDTH,
               depth = DEPTH,
               nproperties = 4,
               property_names = property_names,
               mu_bounds = mu_bounds,
               sigma_bounds = loaded.sigma_bounds,
               sigma_bounds_per = loaded.sigma_bounds_per)
_, st = setup_prior(Xoshiro(TRAIN_SEED), net)
(mus, sigmas), _ = predict(net, X, loaded.params.net, st)
size(mus) == (4, n_grid) || throw(DimensionMismatch("predict returned $(size(mus))"))

grade_ppm = 10 .^ mus[1, :]
n_500 = count(g -> isfinite(g) && g >= 500, grade_ppm)
n_2000 = count(g -> isfinite(g) && g >= 2000, grade_ppm)
n_5000 = count(g -> isfinite(g) && g >= 5000, grade_ppm)
max_ppm = maximum(grade_ppm)
@info "inference" mean_mu_grade = mean(mus[1, :]) max_ppm n_500 n_2000 n_5000

write_structured_grid_vts(VTS_PATH, grid.cx, grid.cy, grid.cz, [
    "grade" => grade_ppm,
    "mu_grade" => mus[1, :],
    "sigma_grade" => sigmas[1, :],
    "mu_density" => mus[2, :],
    "sigma_density" => sigmas[2, :],
    "mu_susceptibility" => mus[3, :],
    "sigma_susceptibility" => sigmas[3, :],
    "mu_conductivity_100kHz" => mus[4, :],
    "sigma_conductivity_100kHz" => sigmas[4, :],
]; scalars = "grade")
@info "wrote VTS" VTS_PATH

elig = cloncurry_grade_eligible(grid, samples)
open(SAMPLES_CSV, "w") do io
    println(io, "east,north,elev,cu_ppm")
    for i in 1:length(samples)
        elig[i] || continue
        @printf(io, "%.3f,%.3f,%.3f,%.6g\n",
                samples.east[i], samples.north[i], samples.elev[i], samples.cu_ppm[i])
    end
end

shells = Tuple{Float64,Float64}[]
n_500 > 0 && push!(shells, (500.0, 0.18))
n_2000 > 0 && push!(shells, (2000.0, 0.45))
n_5000 > 0 && push!(shells, (5000.0, 0.90))
isempty(shells) && error("no cells above 500 ppm; max=$(max_ppm)")

py = find_pyvista_python()
py === nothing && error("no PyVista python; set SMARTPRIOR_PYTHON")
ok = render_cu_iso(py, VTS_PATH, PNG_PATH, SAMPLES_CSV, shells,
                   "Ernest Henry D — predicted Cu (250 epoch, in-sample)")
ok || error("PyVista isosurface failed")
cp(PNG_PATH, ASSET_PNG; force = true)
@info "wrote PNG" PNG_PATH ASSET_PNG shells max_ppm n_500 n_2000 n_5000
