# Full-grid inference + VTK + GLMakie figure for a trained Cloncurry prior.
#
# Rebuilds the same feature stack as the matching train/holdout script, loads
# the checkpoint, writes a StructuredGrid .vts and a 3D Cu PNG.
#
# Ernest Henry (grid D, no geology maps):
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
#     SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 SMARTPRIOR_WIDTH=256 \
#     julia --project=. examples/export_cloncurry_blockmodel.jl
#
# District geology holdout (surface + structure channels):
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4_geology \
#     SMARTPRIOR_BOX=district SMARTPRIOR_WIDTH=256 SMARTPRIOR_DEPTH=4 \
#     julia --project=. examples/export_cloncurry_blockmodel.jl

using SmartPrior
using GLMakie
using Printf
using Random
using Statistics
using WriteVTK

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
# hanging half a cell off a centre-point mesh.
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
    vtk_grid(path, collect(Float64, x), collect(Float64, y), collect(Float64, z)) do vtk
        for (name, a) in arrays
            length(a) == ncel || throw(DimensionMismatch(
                "write_blockmodel_vts: $(name) has $(length(a)) values, expected $ncel cells"))
            vtk[String(name), VTKCellData()] = reshape(collect(Float64, a), nx, ny, nz)
        end
    end
    return path
end

function render_cu_blocks(grid, grade_ppm, samples_csv, png_path, title;
                            cutoff = 2000.0, vexag = 1.0)
    nx, ny, nz = size(grid)
    length(grade_ppm) == nx * ny * nz || throw(DimensionMismatch(
        "render_cu_blocks: grade has $(length(grade_ppm)) values"))
    grade = reshape(grade_ppm, nx, ny, nz)
    xs = Float64[]; ys = Float64[]; zs = Float64[]; cs = Float64[]
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        g = grade[i, j, k]
        (isfinite(g) && g >= cutoff) || continue
        push!(xs, grid.cx[i])
        push!(ys, grid.cy[j])
        push!(zs, grid.cz[k])
        push!(cs, g)
    end
    isempty(xs) && error("no cells above cutoff $cutoff")

    holes = Dict{String,Vector{NTuple{4,Float64}}}()
    if isfile(samples_csv)
        open(samples_csv) do io
            readline(io)
            for line in eachline(io)
                parts = split(line, ',')
                length(parts) >= 5 || continue
                e = tryparse(Float64, parts[2])
                n = tryparse(Float64, parts[3])
                z = tryparse(Float64, parts[4])
                cu = tryparse(Float64, parts[5])
                (e === nothing || n === nothing || z === nothing) && continue
                hid = strip(parts[1])
                recs = get!(Vector{NTuple{4,Float64}}, holes, hid)
                push!(recs, (e, n, z, cu === nothing ? NaN : cu))
            end
        end
    end

    dx = grid.x[end] - grid.x[1]
    dy = grid.y[end] - grid.y[1]
    dz = max(grid.z[end] - grid.z[1], 1.0)
    if vexag <= 0
        vexag = max(1.0, max(dx, dy) / dz)
    end
    zs .*= vexag

    GLMakie.activate!(; visible = false)
    fig = Figure(size = (2000, 1400), backgroundcolor = :white)
    ax = Axis3(fig[1, 1];
               xlabel = "E (m)", ylabel = "N (m)",
               zlabel = vexag > 1.01 ? "Z ×$(round(Int, vexag)) (m ASL)" : "Z (m ASL)",
               title = title, perspectiveness = 0.0)
    if !isempty(xs)
        markersize = Vec3f(grid.h_median, grid.h_median, median(grid.dz) * vexag)
        meshscatter!(ax, xs, ys, zs; color = cs, colormap = :thermal,
                     markersize = markersize, marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
                     transparency = true, alpha = 0.55)
    end
    assay_x = Float64[]; assay_y = Float64[]; assay_z = Float64[]; assay_c = Float64[]
    for recs in values(holes)
        isempty(recs) && continue
        ordered = sort(recs; by = r -> -r[3])
        hx = [r[1] for r in ordered]
        hy = [r[2] for r in ordered]
        hz = [r[3] * vexag for r in ordered]
        if length(ordered) == 1
            half = max(400.0 * vexag, 0.05 * min(dx, dy))
            lines!(ax, [hx[1], hx[1]], [hy[1], hy[1]],
                   [hz[1] + half, hz[1] - half]; color = :black, linewidth = 1.2)
        else
            lines!(ax, hx, hy, hz; color = :black, linewidth = 1.2)
        end
        for r in ordered
            isfinite(r[4]) && r[4] > 0 || continue
            push!(assay_x, r[1]); push!(assay_y, r[2])
            push!(assay_z, r[3] * vexag)
            push!(assay_c, log10(max(r[4], 1.0)))
        end
    end
    if !isempty(assay_x)
        scatter!(ax, assay_x, assay_y, assay_z; color = assay_c, colormap = :plasma,
                 colorrange = (0.0, 4.0), markersize = 8)
    end
    Colorbar(fig[1, 2], limits = (cutoff, maximum(cs)), colormap = :thermal,
             label = "predicted Cu (ppm)")
    save(png_path, fig)
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

n_2000 == 0 && error("no cells above 2000 ppm; max=$(max_ppm)")
title = DISTRICT ?
    "Cloncurry district — predicted Cu blocks + drill traces (Z=100 m)" :
    "Ernest Henry D — predicted Cu + drill traces"
vexag = parse(Float64, get(ENV, "SMARTPRIOR_VEXAG", DISTRICT ? "0" : "1"))
ok = render_cu_blocks(grid, grade_ppm, SAMPLES_CSV, PNG_PATH, title;
                      cutoff = 2000.0, vexag = vexag)
ok || error("GLMakie Cu figure failed")
cp(PNG_PATH, ASSET_PNG; force = true)
@info "wrote PNG" PNG_PATH ASSET_PNG max_ppm n_500 n_2000 n_5000 vexag
