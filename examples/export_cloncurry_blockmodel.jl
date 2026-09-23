# Full-grid inference + VTK + PyVista isosurface for a trained Cloncurry prior.
#
# Rebuilds the same feature stack as the matching train/holdout script, loads
# the checkpoint, writes a StructuredGrid .vts and a 3D Cu PNG. Not an MT
# starting model. No VFSA.
#
# Ernest Henry (grid D, no geology maps):
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
#     SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 SMARTPRIOR_WIDTH=256 \
#     julia --project=. examples/export_cloncurry_blockmodel.jl
#
# District geology holdout (surface + structure channels):
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4_geology \
#     SMARTPRIOR_BOX=district SMARTPRIOR_WIDTH=256 SMARTPRIOR_DEPTH=4 \
#     SMARTPRIOR_PYTHON=$HOME/mtproject/.venv/bin/python \
#     julia --project=. examples/export_cloncurry_blockmodel.jl

using SmartPriorMT
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WIDTH = parse(Int, get(ENV, "SMARTPRIOR_WIDTH", "256"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const SPLIT_SEED = parse(Int, get(ENV, "SMARTPRIOR_SPLIT_SEED", "2026"))
const TARGET_CELLS = parse(Int, get(ENV, "SMARTPRIOR_TARGET_CELLS", "50000"))
const CELL_ENV = strip(get(ENV, "SMARTPRIOR_CELL_M", ""))
const CELL_Z_ENV = strip(get(ENV, "SMARTPRIOR_CELL_Z", ""))
const BOX = lowercase(strip(get(ENV, "SMARTPRIOR_BOX", "ernest_henry")))
const WORK = get(ENV, "SMARTPRIOR_WORK",
                 BOX in ("district", "samples", "aabb") ?
                 joinpath(ROOT, "tmp_cloncurry_prior_district_holdout_w$(WIDTH)_d$(DEPTH)_geology") :
                 joinpath(ROOT, "tmp_cloncurry_prior_eh"))
const FRACTIONS = (0.70, 0.15, 0.15)

function resolve_checkpoint(work)
    env = strip(get(ENV, "SMARTPRIOR_CHECKPOINT", ""))
    !isempty(env) && return abspath(env)
    holdout = joinpath(work, "cloncurry_holdout_w$(WIDTH)_d$(DEPTH).jld2")
    full = joinpath(work, "cloncurry_prior.jld2")
    isfile(holdout) && return holdout
    return full
end

const CHECKPOINT = resolve_checkpoint(WORK)
const DISTRICT = BOX in ("district", "samples", "aabb")
const VTS_PATH = joinpath(WORK, "cloncurry_smartprior_blockmodel.vts")
const PNG_PATH = joinpath(WORK, DISTRICT ?
                          "cloncurry_district_geology_cu_iso.png" :
                          "cloncurry_smartprior_cu_iso.png")
const SAMPLES_CSV = joinpath(WORK, "cloncurry_smartprior_cu_samples.csv")
const ASSET_PNG = joinpath(ROOT, "docs", "assets",
                           DISTRICT ? "cloncurry_district_geology_cu_iso.png" :
                                      "cloncurry_eh_cu_iso.png")
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

function named_hole_mask(drillhole)
    n = length(drillhole)
    m = falses(n)
    @inbounds for i in 1:n
        m[i] = !isempty(strip(String(drillhole[i])))
    end
    return m
end

# Cell-centred block model: POINTS are cell *corners* (grid edges), arrays are
# CellData in Julia vec order. Sample XYZ then falls inside a block instead of
# hanging half a cell off a centre-point mesh (100 m XY made that look like a
# collar shift).
function write_blockmodel_vts(path::AbstractString,
                              x::AbstractVector{<:Real},
                              y::AbstractVector{<:Real},
                              z::AbstractVector{<:Real},
                              arrays;
                              scalars::AbstractString = "grade")
    nx, ny, nz = length(x) - 1, length(y) - 1, length(z) - 1
    (nx > 0 && ny > 0 && nz > 0) || throw(ArgumentError(
        "write_blockmodel_vts: need at least one cell on each axis"))
    ncel = nx * ny * nz
    npts = (nx + 1) * (ny + 1) * (nz + 1)
    names = String[]
    blobs = Vector{Vector{Float64}}()
    for (name, a) in arrays
        length(a) == ncel || throw(DimensionMismatch(
            "write_blockmodel_vts: $(name) has $(length(a)) values, expected $ncel cells"))
        push!(names, String(name))
        push!(blobs, collect(Float64, a))
    end
    points = Vector{Float64}(undef, 3 * npts)
    t = 1
    @inbounds for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        points[t]     = Float64(x[i])
        points[t + 1] = Float64(y[j])
        points[t + 2] = Float64(z[k])
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
    ext = "0 $nx 0 $ny 0 $nz"
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">")
        println(io, "  <StructuredGrid WholeExtent=\"$ext\">")
        println(io, "    <Piece Extent=\"$ext\">")
        println(io, "      <CellData Scalars=\"$scalars\">")
        for (name, o) in zip(names, offsets)
            @printf(io,
                    "        <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"%d\"/>\n",
                    name, o)
        end
        println(io, "      </CellData>")
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
        joinpath(homedir(), "mtproject", ".venv", "bin", "python"),
        "python3",
    ))
    for p in candidates
        try
            # Prefer an explicit env python; avoid hanging GUIs (minerai VTK).
            cmd = pipeline(`$p -c "import pyvista, numpy"`;
                           stdout = devnull, stderr = devnull)
            success(cmd) && return p
        catch
        end
    end
    return nothing
end

function render_cu_iso(python, vts_path, png_path, samples_csv, shells, title;
                       pad = 400.0, near = 2000.0, tube_radius = 5.0,
                       draw_wire = true, keep_all_holes = false,
                       style = "iso", vexag = 1.0)
    # style: "iso" (smooth shells), "blocks" (thresholded cell cubes), "both".
    # vexag > 1 stretches Z so thin district holes read as long traces.
    py = """
import os, sys, csv
from collections import defaultdict
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

vts, png, samples, shell_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pad = float(os.environ.get("ORE_PAD", "400"))
near = float(os.environ.get("ORE_NEAR", "2000"))
tube_r = float(os.environ.get("ORE_TUBE", "5"))
draw_wire = os.environ.get("ORE_WIRE", "1") not in ("0", "false", "no")
keep_all = os.environ.get("ORE_KEEP_ALL_HOLES", "0") in ("1", "true", "yes")
style = os.environ.get("ORE_STYLE", "iso").strip().lower()  # iso|blocks|both
vexag = float(os.environ.get("ORE_VEXAG", "1"))
title = os.environ.get("ORE_TITLE", "Ernest Henry SmartPrior")
gold = LinearSegmentedColormap.from_list(
    "minerai_gold", ["#8a6500", "#c9a227", "#e0bc3a", "#f0d050", "#ffe566"], N=256)
shells = [(float(a), float(b)) for a, b in
          (p.split(",") for p in shell_s.split(";"))]
cutoff = min(iso for iso, _ in shells)

grid = pv.read(vts)
# Keep CellData for block cubes; also need point data for contours.
has_cell_grade = "grade" in grid.cell_data
if has_cell_grade and "grade" not in grid.point_data:
    grid_pts = grid.cell_data_to_point_data()
else:
    grid_pts = grid

# ---- vertical exaggeration (district XY ≫ Z; else holes look like pins) ----
bounds0 = np.asarray(grid.bounds, dtype=float)
dx = bounds0[1] - bounds0[0]
dy = bounds0[3] - bounds0[2]
dz = max(bounds0[5] - bounds0[4], 1.0)
if vexag <= 0:
    # Auto: match exaggerated ΔZ to the *longer* horizontal side so the
    # district AABB reads as a cube (XY ≫ true Z by ~100×).
    vexag = max(1.0, max(dx, dy) / dz)
print("VEXAG", vexag, "DX", dx, "DY", dy, "DZ", dz)

def exaggerate(mesh):
    m = mesh.copy(deep=True)
    if vexag != 1.0:
        m.points = m.points * np.array([1.0, 1.0, vexag])
    return m

grid_cell = exaggerate(grid)
grid_pts = exaggerate(grid_pts)

grade = np.asarray(grid_pts["grade"], dtype=np.float64)
core = np.isfinite(grade) & (grade >= cutoff)
if not np.any(core):
    raise SystemExit("no cells above cutoff")
pts = np.asarray(grid_pts.points)
iso_lo = pts[core].min(axis=0)
iso_hi = pts[core].max(axis=0)

holes = defaultdict(list)
if samples and os.path.isfile(samples):
    with open(samples, newline="") as f:
        for row in csv.DictReader(f):
            try:
                e, n, z = float(row["east"]), float(row["north"]), float(row["elev"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite([e, n, z]).all():
                continue
            try:
                cu = float(row.get("cu_ppm", "nan"))
            except ValueError:
                cu = np.nan
            hid = (row.get("drillhole") or "").strip() or "unknown"
            holes[hid].append((e, n, z * vexag, cu))
print("HOLES", len(holes), "SAMPLES", sum(len(v) for v in holes.values()))

omitted = []
if keep_all:
    print("HOLES_IN_FRAME", ",".join(sorted(holes)))
    print("HOLES_OMITTED_FAR", "")
else:
    near_lo, near_hi = iso_lo - near, iso_hi + near
    def _in(p, lo, hi):
        return bool(np.all(p >= lo) and np.all(p <= hi))
    keep = {}
    for hid, recs in holes.items():
        if any(_in(np.array(r[:3]), near_lo, near_hi) for r in recs):
            keep[hid] = recs
        else:
            omitted.append(hid)
    print("HOLES_IN_FRAME", ",".join(sorted(keep)))
    print("HOLES_OMITTED_FAR", ",".join(sorted(omitted)))
    holes = keep

# Data AABB (grid + holes), then expand to an equal-side outer cube so the
# field reads as a square cube rather than a flat XY slab.
gb = np.asarray(grid_cell.bounds, dtype=float)
frame_lo = np.array([gb[0], gb[2], gb[4]]) - pad
frame_hi = np.array([gb[1], gb[3], gb[5]]) + pad
if holes:
    hpts = np.asarray([r[:3] for recs in holes.values() for r in recs], dtype=float)
    frame_lo = np.minimum(frame_lo, hpts.min(axis=0) - pad)
    frame_hi = np.maximum(frame_hi, hpts.max(axis=0) + pad)
span = frame_hi - frame_lo
side = float(np.max(span))
center = 0.5 * (frame_lo + frame_hi)
half = 0.5 * side
cube_lo = center - half
cube_hi = center + half
frame_lo, frame_hi = cube_lo.copy(), cube_hi.copy()
print("OUTER_CUBE_SIDE_M", side, "CENTER", center.tolist())

cv = np.asarray(grid_pts["grade"])
above = cv[np.isfinite(cv) & (cv >= cutoff)]
clim_hi = float(np.percentile(above, 97)) if above.size else cutoff + 1.0
if clim_hi <= cutoff:
    clim_hi = cutoff + max(50.0, cutoff * 0.15)

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=(2000, 1400))
plotter.set_background("#f7f7f5")

# Outer square cube (equal E/N/Z visual side) — single volume, no nested AABB.
outer = pv.Cube(bounds=(cube_lo[0], cube_hi[0],
                        cube_lo[1], cube_hi[1],
                        cube_lo[2], cube_hi[2]))
# Very light faces so the wireframe is one solid cube (avoids Necker/iso
# crossing that looks like a second inner grid).
plotter.add_mesh(outer, color="#dcdcd8", opacity=0.07, show_edges=False,
                 name="field_faces")
plotter.add_mesh(outer.outline(), color="#1a1a1a", line_width=3.5,
                 name="field_outline")

corners = np.array([
    [frame_lo[0], frame_lo[1], frame_lo[2]],
    [frame_hi[0], frame_lo[1], frame_lo[2]],
    [frame_lo[0], frame_hi[1], frame_lo[2]],
    [frame_lo[0], frame_lo[1], frame_hi[2]],
    [frame_hi[0], frame_hi[1], frame_hi[2]],
], dtype=float)
plotter.add_mesh(pv.PolyData(corners), color="white", opacity=0.0,
                 point_size=1, name="frame_corners")

built = 0
bar_title = "predicted Cu (ppm)"
if style in ("blocks", "both") and has_cell_grade:
    # Thresholded cells = the actual PriorGrid cubes (not a smooth iso blob).
    blocks = grid_cell.threshold(value=cutoff, scalars="grade",
                                 preference="cell")
    if blocks.n_cells > 0:
        built += 1
        plotter.add_mesh(
            blocks, scalars="grade", cmap=gold, clim=(cutoff, clim_hi),
            opacity=0.55, show_edges=True, edge_color="#5a4500",
            line_width=0.4, smooth_shading=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": bar_title + "  [block cells]",
                "vertical": True, "position_x": 0.88, "position_y": 0.16,
                "fmt": "%.0f", "title_font_size": 11, "label_font_size": 9,
            },
            name="cu_blocks",
        )
        print("BLOCKS", blocks.n_cells)
    else:
        print("BLOCKS_EMPTY")

if style in ("iso", "both"):
    # Clip contours to Cu envelope so empty district air is not marched.
    clip_lo = iso_lo - pad
    clip_hi = iso_hi + pad
    giso = grid_pts.clip_box([clip_lo[0], clip_hi[0], clip_lo[1], clip_hi[1],
                              clip_lo[2], clip_hi[2]], invert=False)
    bar_iso = shells[0][0]
    show_bar = style == "iso"  # blocks already owns the bar in "both"
    for iso, opacity in shells:
        surf = giso.contour(isosurfaces=[iso], scalars="grade")
        if surf.n_points == 0:
            print("EMPTY_ISO", iso)
            continue
        built += 1
        plotter.add_mesh(
            surf, scalars="grade", cmap=gold, clim=(cutoff, clim_hi),
            smooth_shading=True, opacity=opacity,
            show_scalar_bar=show_bar and abs(iso - bar_iso) < 1e-6,
            scalar_bar_args={
                "title": bar_title + "  [isosurface]",
                "vertical": True, "position_x": 0.88, "position_y": 0.16,
                "fmt": "%.0f", "title_font_size": 11, "label_font_size": 9,
            },
            name="iso_%d" % int(iso),
        )
if built == 0:
    raise SystemExit("empty Cu geometry")

# Drill traces: collar→toe vertical tube (true path if multi-sample).
# Min displayed length so sparse single-sample holes still read as shafts
# (~15% of the shorter horizontal side — otherwise pins vanish in the cube).
min_len = max(400.0 * vexag, 0.15 * min(dx, dy))
assay_xyz, assay_cu = [], []
for hid, recs in sorted(holes.items()):
    recs = sorted(recs, key=lambda r: -r[2])
    xyz = np.array([(r[0], r[1], r[2]) for r in recs], dtype=float)
    if len(xyz) == 1:
        # Fabricate a short vertical shaft around the lone sample.
        z0, z1 = xyz[0, 2] + 0.5 * min_len, xyz[0, 2] - 0.5 * min_len
        xyz = np.array([[xyz[0, 0], xyz[0, 1], z0],
                        [xyz[0, 0], xyz[0, 1], z1]], dtype=float)
    else:
        # Extend slightly past collar/toe so the tube is visible end-to-end.
        span = float(xyz[0, 2] - xyz[-1, 2])
        if span < min_len:
            mid = 0.5 * (xyz[0, 2] + xyz[-1, 2])
            xyz = np.array([[xyz[0, 0], xyz[0, 1], mid + 0.5 * min_len],
                            [xyz[-1, 0], xyz[-1, 1], mid - 0.5 * min_len]],
                           dtype=float)
        else:
            pad_z = 0.05 * span
            xyz = xyz.copy()
            xyz[0, 2] += pad_z
            xyz[-1, 2] -= pad_z
    line = pv.lines_from_points(xyz)
    try:
        mesh = line.tube(radius=tube_r, n_sides=10)
    except Exception:
        mesh = line
    plotter.add_mesh(mesh, color="#1a1a1a", opacity=0.9,
                     name="dh_%s" % hid)
    for r in recs:
        if np.isfinite(r[3]) and r[3] > 0:
            assay_xyz.append((r[0], r[1], r[2]))
            assay_cu.append(r[3])
if assay_xyz:
    assay_xyz = np.asarray(assay_xyz, dtype=float)
    assay_cu = np.asarray(assay_cu, dtype=float)
    logc = np.log10(np.clip(assay_cu, 1.0, None))
    cloud = pv.PolyData(assay_xyz)
    cloud["log10_Cu"] = logc
    pt = 7 if keep_all else 11
    plotter.add_mesh(cloud, scalars="log10_Cu", cmap="plasma",
                     clim=(0.0, 4.0), point_size=pt,
                     render_points_as_spheres=True,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "assay log10 Cu",
                         "vertical": True, "position_x": 0.02, "position_y": 0.16,
                         "fmt": "%.1f", "title_font_size": 11, "label_font_size": 9,
                     },
                     name="assays")
print("TRACES_DRAWN", len(holes))

plotter.add_axes()
ztitle = ("Z ×%.0f (m ASL)" % vexag) if vexag > 1.01 else "Z (m ASL)"
# Force axis box onto the outer square cube — default show_bounds hugs the
# data AABB and drew a second nested rectangle inside the cube.
plotter.show_bounds(
    bounds=(cube_lo[0], cube_hi[0], cube_lo[1], cube_hi[1], cube_lo[2], cube_hi[2]),
    grid=False, location="outer", ticks="outside",
    n_xlabels=4, n_ylabels=4, n_zlabels=5,
    xtitle="E (m)", ytitle="N (m)", ztitle=ztitle,
    font_size=10, color="#444444", fmt="%.0f",
)
plotter.add_text(title, font_size=12)
note = "%d holes" % len(holes)
if vexag > 1.01:
    note += "  |  Z exaggerated ×%.0f (true ΔZ ≈ %.0f m)" % (vexag, dz)
note += "  |  outer frame = square cube"
if style == "blocks":
    note += "  |  gold = predicted Cu block cells"
elif style == "both":
    note += "  |  gold = Cu blocks + isosurfaces"
else:
    note += "  |  gold = predicted Cu isosurfaces"
if omitted:
    note += "  |  %d far holes omitted" % len(omitted)
plotter.add_text(note, font_size=8, position="lower_left")

plotter.view_isometric()
plotter.enable_parallel_projection()  # keep outer cube square (no perspective squash)
# Nudge off pure iso so opposite edges don't cross at the silhouette centre
# (that reading looked like a nested second grid).
plotter.camera.azimuth += 18.0
plotter.camera.elevation += 6.0
plotter.reset_camera()
plotter.camera.zoom(0.92 if keep_all else 1.1)
plotter.show(screenshot=png, auto_close=True)
print("WROTE", png)
"""
    pyfile = joinpath(WORK, "_render_cloncurry_iso.py")
    write(pyfile, py)
    shell_s = join(["$(Int(iso)),$(op)" for (iso, op) in shells], ";")
    run(setenv(`$python $pyfile $vts_path $png_path $samples_csv $shell_s`,
               merge(ENV, Dict("ORE_TITLE" => String(title),
                               "PYVISTA_OFF_SCREEN" => "true",
                               "ORE_PAD" => string(pad),
                               "ORE_NEAR" => string(near),
                               "ORE_TUBE" => string(tube_radius),
                               "ORE_WIRE" => draw_wire ? "1" : "0",
                               "ORE_KEEP_ALL_HOLES" => keep_all_holes ? "1" : "0",
                               "ORE_STYLE" => String(style),
                               "ORE_VEXAG" => string(vexag)))))
    return isfile(png_path)
end

isfile(CHECKPOINT) || error("missing $CHECKPOINT; train first")
DATASET = default_cloncurry_root()
loaded = load_prior(CHECKPOINT)

samples = load_cloncurry_samples(DATASET)
CELL, CELL_Z, bounds, grid = if DISTRICT
    aabb = cloncurry_sample_bounds(samples)
    spacing0 = cloncurry_district_spacing(aabb; target = TARGET_CELLS)
    cell = isempty(CELL_ENV) ? spacing0.cell : parse(Float64, CELL_ENV)
    cell_z = isempty(CELL_Z_ENV) ? spacing0.cell_z : parse(Float64, CELL_Z_ENV)
    b = cloncurry_sample_bounds(samples; pad_xy = cell / 2, pad_z = cell_z / 2)
    g = cloncurry_grid(b; cell = cell, cell_z = cell_z)
    cell, cell_z, b, g
elseif BOX in ("work", "h")
    cell = parse(Float64, isempty(CELL_ENV) ? "100" : CELL_ENV)
    cell_z = parse(Float64, isempty(CELL_Z_ENV) ? "50" : CELL_Z_ENV)
    b = cloncurry_work_bounds()
    cell, cell_z, b, cloncurry_grid(b; cell = cell, cell_z = cell_z)
elseif BOX in ("ernest_henry", "eh", "d")
    cell = parse(Float64, isempty(CELL_ENV) ? "100" : CELL_ENV)
    cell_z = parse(Float64, isempty(CELL_Z_ENV) ? "50" : CELL_Z_ENV)
    b = cloncurry_deposit_bounds(samples, "Ernest Henry")
    cell, cell_z, b, cloncurry_grid(b; cell = cell, cell_z = cell_z)
else
    error("SMARTPRIOR_BOX must be district, work, or ernest_henry")
end
n_grid = ncells(grid)
@info "Cloncurry export" DATASET WORK BOX CELL CELL_Z CHECKPOINT loaded.nin size(grid)

geochem = cloncurry_geochemistry(samples)
lith = cloncurry_lithology(samples)
coverage = cloncurry_coverage_points(samples)
stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
if DISTRICT || get(loaded.meta, "box", "") == "district"
    struct_c, struct_n = structure_distance_channels(grid, DATASET)
    surf_c, surf_n = surface_geology_channels(grid, DATASET)
    stack = append_channels(stack, struct_c, struct_n)
    stack = append_channels(stack, surf_c, surf_n)
    @info "geology channels" structure = struct_n n_surface = length(surf_n)
end
X = encode_features(stack; n_bands = 4)
size(X, 1) == loaded.nin || error("feature width $(size(X, 1)) ≠ checkpoint nin $(loaded.nin)")

# Holdout checkpoints squash with train-hole mu_bounds; recreate that split.
property_names = copy(CLONCURRY_PROPERTY_NAMES)
if get(loaded.meta, "holdout", "") == "drillhole"
    gkeys = cloncurry_group_keys(samples.drillhole)
    inside = cloncurry_inside_mask(grid, samples)
    eligible = inside .& named_hole_mask(samples.drillhole)
    targets = let t = get(loaded.meta, "targets", nothing)
        t === nothing ? nothing : (Int(t[1]), Int(t[2]), Int(t[3]))
    end
    split = group_holdout(gkeys, eligible; unit = :groups, targets = targets,
                          fractions = FRACTIONS, rng = Xoshiro(SPLIT_SEED))
    train_keep = group_mask(gkeys, split.groups_train) .& inside
    train_loaded = load_cloncurry_anchors(grid, samples; keep = train_keep)
    mu_bounds = [
        padded_bounds(train_loaded.grade[2]; fallback = (0.0, 5.0)),
        padded_bounds(train_loaded.density[2]; fallback = (2.0, 4.0), pad = 0.1),
        padded_bounds(train_loaded.susceptibility[2]; fallback = (-7.0, 1.0)),
        padded_bounds(train_loaded.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
    ]
else
    anchors = load_cloncurry_anchors(grid, samples)
    mu_bounds = [
        padded_bounds(anchors.grade[2]; fallback = (0.0, 5.0)),
        padded_bounds(anchors.density[2]; fallback = (2.0, 4.0), pad = 0.1),
        padded_bounds(anchors.susceptibility[2]; fallback = (-7.0, 1.0)),
        padded_bounds(anchors.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
    ]
end

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

write_blockmodel_vts(VTS_PATH, grid.x, grid.y, grid.z, [
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

gkeys = cloncurry_group_keys(samples.drillhole)
open(SAMPLES_CSV, "w") do io
    println(io, "drillhole,east,north,elev,cu_ppm")
    for i in 1:length(samples)
        isfinite(samples.east[i]) && isfinite(samples.north[i]) &&
            isfinite(samples.elev[i]) || continue
        containing_cell(grid, samples.east[i], samples.north[i], samples.elev[i]) == 0 &&
            continue
        cu = samples.cu_ppm[i]
        cu_s = isfinite(cu) ? @sprintf("%.6g", cu) : "NaN"
        @printf(io, "%s,%.3f,%.3f,%.3f,%s\n",
                gkeys[i], samples.east[i], samples.north[i], samples.elev[i], cu_s)
    end
end

shells = Tuple{Float64,Float64}[]
n_2000 > 0 && push!(shells, (2000.0, 0.35))
n_5000 > 0 && push!(shells, (5000.0, 0.95))
isempty(shells) && error("no cells above 2000 ppm; max=$(max_ppm)")

# District cells are ~2.3 km; enlarge clip/near so the envelope is readable.
pad = DISTRICT ? max(CELL, 2500.0) : 400.0
near = DISTRICT ? max(3 * CELL, 8000.0) : 2000.0
# Thicker tubes at district scale (still small vs 2.3 km cells).
tube = DISTRICT ? max(CELL / 8, 200.0) : 5.0
title = DISTRICT ?
    "Cloncurry district — predicted Cu blocks + drill traces (Z=100 m)" :
    "Ernest Henry D — predicted Cu + drill traces"
# District: show the PriorGrid as a cube/wire + thresholded cell blocks.
# Auto Z exaggeration (ORE_VEXAG≤0) so ~2 km of depth reads against ~100 km XY.
style = DISTRICT ? get(ENV, "SMARTPRIOR_RENDER_STYLE", "blocks") : "iso"
vexag = parse(Float64, get(ENV, "SMARTPRIOR_VEXAG", DISTRICT ? "0" : "1"))

py = find_pyvista_python()
py === nothing && error("no PyVista python; set SMARTPRIOR_PYTHON")
ok = render_cu_iso(py, VTS_PATH, PNG_PATH, SAMPLES_CSV, shells, title;
                   pad = pad, near = near, tube_radius = tube,
                   draw_wire = true, keep_all_holes = DISTRICT,
                   style = style, vexag = vexag)
ok || error("PyVista isosurface failed")
cp(PNG_PATH, ASSET_PNG; force = true)
@info "wrote PNG" PNG_PATH ASSET_PNG shells max_ppm n_500 n_2000 n_5000 style vexag
