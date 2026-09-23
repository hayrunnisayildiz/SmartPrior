# Test-hole calibration: predicted μ vs true property values for the Z=100 m
# district drillhole hold-out (18 test holes / 230 samples).
#
# Four panels (grade, density, susceptibility, conductivity_100kHz). Points are
# coloured by σ; a 45° line marks perfect calibration. No gravity / MT /
# magnetics / VFSA.
#
# Run:
#   julia --project=. examples/calibration_plot_cloncurry.jl
#
# Env: CLONCURRY_ROOT, SMARTPRIOR_WORK (checkpoint dir; default …_z100),
#      SMARTPRIOR_OUT (PNG path), SMARTPRIOR_WIDTH, SMARTPRIOR_DEPTH,
#      SMARTPRIOR_SPLIT_SEED, SMARTPRIOR_TRAIN_SEED, SMARTPRIOR_TARGET_CELLS,
#      SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z, SMARTPRIOR_CHECKPOINT

using SmartPrior
using GLMakie
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
const FRACTIONS = (0.70, 0.15, 0.15)

const WORK = get(ENV, "SMARTPRIOR_WORK",
                 joinpath(ROOT, "tmp_cloncurry_prior_district_holdout_w$(WIDTH)_d$(DEPTH)_z100"))
const OUT_PNG = get(ENV, "SMARTPRIOR_OUT",
                    joinpath(ROOT, "tmp_cloncurry_prior_district_holdout_w$(WIDTH)_d$(DEPTH)",
                             "calibration_test.png"))

function resolve_checkpoint(work)
    env = strip(get(ENV, "SMARTPRIOR_CHECKPOINT", ""))
    !isempty(env) && return abspath(env)
    return joinpath(work, "cloncurry_holdout_w$(WIDTH)_d$(DEPTH).jld2")
end

const CHECKPOINT = resolve_checkpoint(WORK)

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
    error("set CLONCURRY_ROOT to Cloncurry_integrated_2026-09-17")
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

function property_obs(samples, i, p)
    if p == 1
        return cu_log10(samples.cu_ppm[i]; detection_limit = CLONCURRY_CU_DL_PPM)
    elseif p == 2
        d = samples.density_g_cm3[i]
        return isfinite(d) && d > 0 ? d : NaN
    elseif p == 3
        v = samples.susceptibility_SI[i]
        return isfinite(v) && v > 0 ? log10(v) : NaN
    else
        v = samples.conductivity_S_m_100kHz[i]
        isfinite(v) || return NaN
        return log10(max(v, CLONCURRY_COND_FLOOR_S_M))
    end
end

function collect_test_pairs(mus, sigmas, grid, samples, idx, p)
    obs = Float64[]
    pred = Float64[]
    sig = Float64[]
    for i in idx
        c = containing_cell(grid, samples.east[i], samples.north[i], samples.elev[i])
        c == 0 && continue
        y = property_obs(samples, i, p)
        isfinite(y) || continue
        push!(obs, y)
        push!(pred, mus[p, c])
        push!(sig, sigmas[p, c])
    end
    return obs, pred, sig
end

function pearson_r(x, y)
    n = length(x)
    n < 2 && return NaN
    sx = std(x)
    sy = std(y)
    (sx > 0 && sy > 0) || return NaN
    return cov(x, y) / (sx * sy)
end

function r_squared(obs, pred)
    n = length(obs)
    n < 2 && return NaN
    μ = mean(obs)
    ss_tot = sum(abs2, obs .- μ)
    ss_tot == 0 && return NaN
    return 1 - sum(abs2, pred .- obs) / ss_tot
end

const PANEL_META = (
    (name = "grade",
     xlabel = "true log₁₀ Cu (ppm)",
     ylabel = "μ_grade"),
    (name = "density",
     xlabel = "true density (g/cm³)",
     ylabel = "μ_density"),
    (name = "susceptibility",
     xlabel = "true log₁₀ SI",
     ylabel = "μ_susceptibility"),
    (name = "conductivity_100kHz",
     xlabel = "true log₁₀ S/m (100 kHz)",
     ylabel = "μ_conductivity_100kHz"),
)

isfile(CHECKPOINT) || error("missing $CHECKPOINT; train Z=100 hold-out first")
DATASET = default_cloncurry_root()
loaded = load_prior(CHECKPOINT)

samples = load_cloncurry_samples(DATASET)
aabb = cloncurry_sample_bounds(samples)
spacing0 = cloncurry_district_spacing(aabb; target = TARGET_CELLS)
CELL = isempty(CELL_ENV) ? spacing0.cell : parse(Float64, CELL_ENV)
CELL_Z = isempty(CELL_Z_ENV) ? spacing0.cell_z : parse(Float64, CELL_Z_ENV)
bounds = cloncurry_sample_bounds(samples; pad_xy = CELL / 2, pad_z = CELL_Z / 2)
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)
n_grid = ncells(grid)

@info "calibration" DATASET WORK CHECKPOINT CELL CELL_Z size(grid) loaded.nin OUT_PNG

geochem = cloncurry_geochemistry(samples)
lith = cloncurry_lithology(samples)
coverage = cloncurry_coverage_points(samples)
stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
struct_c, struct_n = structure_distance_channels(grid, DATASET)
surf_c, surf_n = surface_geology_channels(grid, DATASET)
stack = append_channels(stack, struct_c, struct_n)
stack = append_channels(stack, surf_c, surf_n)
X = encode_features(stack; n_bands = 4)
size(X, 1) == loaded.nin || error("feature width $(size(X, 1)) ≠ checkpoint nin $(loaded.nin)")

property_names = copy(CLONCURRY_PROPERTY_NAMES)
gkeys = cloncurry_group_keys(samples.drillhole)
inside = cloncurry_inside_mask(grid, samples)
eligible = inside .& named_hole_mask(samples.drillhole)
targets = let t = get(loaded.meta, "targets", nothing)
    t === nothing ? nothing : (Int(t[1]), Int(t[2]), Int(t[3]))
end
split = group_holdout(gkeys, eligible; unit = :groups, targets = targets,
                      fractions = FRACTIONS, rng = Xoshiro(SPLIT_SEED))
train_keep = group_mask(gkeys, split.groups_train) .& inside
test_keep = group_mask(gkeys, split.groups_test) .& inside
test_idx = findall(test_keep)

train_loaded = load_cloncurry_anchors(grid, samples; keep = train_keep)
mu_bounds = [
    padded_bounds(train_loaded.grade[2]; fallback = (0.0, 5.0)),
    padded_bounds(train_loaded.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(train_loaded.susceptibility[2]; fallback = (-7.0, 1.0)),
    padded_bounds(train_loaded.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
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

@printf("  test holes / samples        %d / %d\n",
        split.n_test_groups, length(test_idx))

GLMakie.activate!(; visible = false)
fig = Figure(size = (1600, 1200), backgroundcolor = :white)
fig[0, 1:4] = Label(fig,
    "Cloncurry district hold-out — test holes only (Z=$(round(Int, CELL_Z)) m)";
    fontsize = 18)
for p in 1:4
    meta = PANEL_META[p]
    obs, pred, sig = collect_test_pairs(mus, sigmas, grid, samples, test_idx, p)
    n = length(obs)
    r = pearson_r(obs, pred)
    r2 = r_squared(obs, pred)
    rmse = n == 0 ? NaN : sqrt(mean(abs2, pred .- obs))
    @printf("  %-22s n=%d  r=%.3f  R²=%.3f  RMSE=%.4f  mean(σ)=%.3f\n",
            meta.name, n, r, r2, rmse, n == 0 ? NaN : mean(sig))
    row = (p - 1) ÷ 2 + 1
    col0 = 2 * ((p - 1) % 2)
    ax = Axis(fig[row, col0 + 1];
              xlabel = meta.xlabel, ylabel = meta.ylabel, aspect = DataAspect(),
              title = @sprintf("%s  n=%d  r=%.3f  R²=%.3f", meta.name, n, r, r2))
    n == 0 && continue
    lo = min(minimum(obs), minimum(pred))
    hi = max(maximum(obs), maximum(pred))
    pad = 0.05 * (hi - lo + eps())
    lims = (lo - pad, hi + pad)
    σlo, σhi = extrema(sig)
    if σlo == σhi
        σlo -= 0.05
        σhi += 0.05
    end
    sc = scatter!(ax, obs, pred; color = sig, colormap = :viridis,
                  colorrange = (σlo, σhi), markersize = 8)
    lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]];
           color = :black, linestyle = :dash, linewidth = 1.5)
    xlims!(ax, lims); ylims!(ax, lims)
    Colorbar(fig[row, col0 + 2], sc; label = "σ")
end

mkpath(dirname(OUT_PNG))
save(OUT_PNG, fig)
@info "wrote" OUT_PNG bytes = filesize(OUT_PNG)
