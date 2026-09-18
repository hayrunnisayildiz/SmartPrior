# Train a four-property prior on Cloncurry–Ernest Henry (METAL).
#
# Separate from examples/compare_prior_2d.jl and from the Keivitsa scripts:
# no gravity, no magnetics, no MT, no VFSA. Features are pXRF geochemistry
# (Cu held out) and lithology_code; anchors are Cu_Concentration plus
# density / susceptibility / conductivity_100kHz (KT-20, 100 kHz specimen
# conductivity — not MT bulk resistivity).
#
# Default cell size is coarse. The work area is ~42 × 76 km; a 25 m mesh
# is not a first-run setting. See the grid-proposal table printed at start.
#
# Run:  julia --project=. examples/train_cloncurry_prior.jl
# Env:  CLONCURRY_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_BOX (work | ernest_henry),
#       SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z, SMARTPRIOR_EPOCHS,
#       SMARTPRIOR_LOG_EVERY, SMARTPRIOR_TRAIN_SEED, SMARTPRIOR_WIDTH,
#       SMARTPRIOR_DEPTH
#
# Spatial hold-out of Cu labels (train vs test RMSE): see
# examples/holdout_cloncurry_prior.jl. Do not treat a full-data RMSE as
# generalization.

using SmartPriorMT
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_cloncurry_prior"))
const CELL = parse(Float64, get(ENV, "SMARTPRIOR_CELL_M", "1000"))
const CELL_Z = parse(Float64, get(ENV, "SMARTPRIOR_CELL_Z", "100"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "50"))
const LOG_EVERY = parse(Int, get(ENV, "SMARTPRIOR_LOG_EVERY",
                                 string(max(1, EPOCHS ÷ 10))))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const WIDTH = parse(Int, get(ENV, "SMARTPRIOR_WIDTH", "256"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const BOX = lowercase(strip(get(ENV, "SMARTPRIOR_BOX", "work")))
const CUTOFF_PPM = 5000.0
mkpath(WORK)

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
    error("set CLONCURRY_ROOT to Cloncurry_integrated_2026-09-17 " *
          "(the folder that contains derived/petrophysics_samples.csv)")
end

function print_share_table(report, weights; title = "loss share")
    rows = (
        ("grade", weights.grade, report.grade),
        ("density", weights.density, report.density),
        ("susceptibility", weights.susceptibility, report.susceptibility),
        ("conductivity_100kHz", weights.conductivity, report.conductivity_100kHz),
        ("smooth", weights.smooth, report.smooth),
        ("sigma", weights.sigma, report.sigma),
    )
    println(repeat("=", 72))
    println("  ", title)
    println(repeat("=", 72))
    @printf("%-22s %8s %12s %12s %8s\n", "term", "weight", "unweighted", "weighted", "share")
    for (name, w, t) in rows
        isfinite(t) || continue
        wt = w * t
        share = report.total != 0 ? wt / report.total : NaN
        @printf("%-22s %8.3f %12.5f %12.5f %6.1f%%\n",
                name, w, t, wt, 100 * share)
        @printf("  %s share of total      %.4f / %.4f  = %.1f%%  (weight × unweighted)\n",
                name, wt, report.total, 100 * share)
    end
    @printf("total                                     %12.5f\n", report.total)
    println(repeat("=", 72))
end

function count_samples_in_grid(g, s)
    n_in = 0
    by_dep = Dict{String,Int}()
    for i in 1:length(s)
        containing_cell(g, s.east[i], s.north[i], s.elev[i]) == 0 && continue
        n_in += 1
        d = isempty(s.deposit[i]) ? "(none)" : s.deposit[i]
        by_dep[d] = get(by_dep, d, 0) + 1
    end
    return n_in, by_dep
end

function print_grid_proposal(bounds; title = "grid")
    span_x = bounds.x_max - bounds.x_min
    span_y = bounds.y_max - bounds.y_min
    span_z = bounds.z_max - bounds.z_min
    println()
    println(repeat("=", 72))
    println("  ", title)
    println(repeat("=", 72))
    @printf("  box MGA54       E %.0f–%.0f  (%.2f km)\n",
            bounds.x_min, bounds.x_max, span_x / 1000)
    @printf("                 N %.0f–%.0f  (%.2f km)\n",
            bounds.y_min, bounds.y_max, span_y / 1000)
    @printf("                 Z %.0f–%.0f m ASL  (%.2f km)\n",
            bounds.z_min, bounds.z_max, span_z / 1000)
    hasproperty(bounds, :deposit) &&
        println("  deposit filter: ", bounds.deposit, "  n_xyz=", bounds.n_samples)
    println()
    @printf("%8s %8s %10s %10s %10s %12s\n",
            "xy_m", "z_m", "nx", "ny", "nz", "ncells")
    for (cell, cz) in ((50.0, 50.0), (100.0, 50.0), (200.0, 50.0),
                       (250.0, 100.0), (400.0, 100.0), (500.0, 100.0),
                       (1000.0, 100.0), (1000.0, 200.0))
        nx = max(1, round(Int, span_x / cell))
        ny = max(1, round(Int, span_y / cell))
        nz = max(1, round(Int, span_z / cz))
        @printf("%8.0f %8.0f %10d %10d %10d %12d\n",
                cell, cz, nx, ny, nz, nx * ny * nz)
    end
    println(repeat("=", 72))
end

const DATASET = default_cloncurry_root()
@info "Cloncurry paths" DATASET WORK BOX CELL CELL_Z EPOCHS LOG_EVERY WIDTH DEPTH

samples = load_cloncurry_samples(DATASET)
bounds = if BOX in ("work", "h")
    cloncurry_work_bounds()
elseif BOX in ("ernest_henry", "eh", "d")
    cloncurry_deposit_bounds(samples, "Ernest Henry")
else
    error("SMARTPRIOR_BOX must be work or ernest_henry, got $(repr(BOX))")
end
print_grid_proposal(bounds; title = "grid box $(BOX)")
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)
@info "prior grid" size(grid) ncells(grid) dx = grid.dx[1] dy = grid.dy[1] dz = grid.dz[1] bounds

#---------- 1. samples, features ----------

geochem = cloncurry_geochemistry(samples)
lith = cloncurry_lithology(samples)
coverage = cloncurry_coverage_points(samples)
@info "samples" nsamples = length(samples) deposits = length(unique(samples.deposit))
@info "geochemistry" nsamples = length(geochem) channels = geochem.names
println("  overlap policy: Cu_Concentration is the grade target and is not a feature")
@info "lithology" nsamples = length(lith)

stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
X = encode_features(stack; n_bands = 4)
@info "features" nchannels(stack) size(X) names = stack.names

#---------- 2. anchors ----------

anchors = load_cloncurry_anchors(grid, samples)
cov = anchors.stats

println()
println(repeat("=", 72))
println("  conductivity_100kHz documentation status")
println(repeat("=", 72))
println("  ", CLONCURRY_CONDUCTIVITY_STATUS)
println(repeat("=", 72))
@printf("  METAL rows                   %d\n", cov.n)
@printf("  finite (x, y, z)             %d\n", cov.n_xyz)
@printf("  grade (Cu, incl. <LOD)       %d  (%.1f%%)\n", cov.n_grade, 100 * cov.grade)
@printf("  density g/cm³                %d  (%.1f%%)\n", cov.n_density, 100 * cov.density)
@printf("  susceptibility SI (>0)       %d  (%.1f%%)\n",
        cov.n_susceptibility, 100 * cov.susceptibility)
@printf("  conductivity_100kHz          %d  (%.1f%%)  of which zeros (lifted to floor) %d\n",
        cov.n_conductivity_100kHz, 100 * cov.conductivity_100kHz, cov.n_conductivity_zero)
println(repeat("=", 72))

n_grid = ncells(grid)
function cover_line(name, n_cells, extra = "")
    @printf("  %-22s  %6d cells  (%5.2f%% of grid)%s\n",
            name, n_cells, 100 * n_cells / n_grid, extra)
end

println("  anchor cell coverage (inside the current grid only)")
println(repeat("-", 72))
cover_line("grade (Cu_Log)", length(anchors.grade[1]),
           @sprintf("  %d mapped samples", cov.n_grade_mapped))
cover_line("density", length(anchors.density[1]),
           @sprintf("  %d mapped samples", cov.n_density_mapped))
cover_line("susceptibility", length(anchors.susceptibility[1]),
           @sprintf("  %d mapped samples", cov.n_susceptibility_mapped))
cover_line("conductivity_100kHz", length(anchors.conductivity_100kHz[1]),
           @sprintf("  %d mapped samples", cov.n_conductivity_100kHz_mapped))
println("  samples outside this box are dropped as anchors (containing_cell=0)")
n_in_grid, dep_in = count_samples_in_grid(grid, samples)
@printf("  samples inside this box      %d / %d\n", n_in_grid, length(samples))
for d in sort!(collect(keys(dep_in)); by = k -> -dep_in[k])
    @printf("    %-22s %4d\n", d, dep_in[d])
end
@printf("  grade mapped / METAL Cu rows %d / %d\n",
        cov.n_grade_mapped, cov.n_grade)
println("  NLL is a per-group mean, so coverage imbalance does not reweight")
println(repeat("=", 72))

property_names = copy(CLONCURRY_PROPERTY_NAMES)
targets = PriorTargets(
    anchors_grade = anchors.grade,
    anchors_density = anchors.density,
    anchors_susceptibility = anchors.susceptibility,
    anchors_conductivity = anchors.conductivity_100kHz,
    property_names = property_names,
    sigma_target = 0.5,
)

weights = LossWeights(
    grade = 1.0,
    density = 1.0,
    susceptibility = 1.0,
    conductivity = 1.0,
    smooth = 1.0e-2,
    sigma = 1.0e-2,
)

function padded_bounds(values; pad = 0.2, fallback)
    finite = filter(isfinite, values)
    isempty(finite) && return fallback
    lo, hi = extrema(finite)
    lo == hi && return (lo - pad, hi + pad)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)
end

# NLL spaces: Cu_Log, g/cm³, log10 SI, log10 (S/m at 100 kHz)
mu_bounds = [
    padded_bounds(anchors.grade[2]; fallback = (0.0, 5.0)),
    padded_bounds(anchors.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(anchors.susceptibility[2]; fallback = (-7.0, 1.0)),
    padded_bounds(anchors.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
]
@info "mu_bounds" property_names mu_bounds

sigma_hi = 1.2
sigma_bounds_per = [
    sigma_bounds_from_anchors(anchors.grade; hi = sigma_hi),
    sigma_bounds_from_anchors(anchors.density; hi = sigma_hi),
    sigma_bounds_from_anchors(anchors.susceptibility; hi = sigma_hi),
    sigma_bounds_from_anchors(anchors.conductivity_100kHz; hi = sigma_hi),
]
println()
println(repeat("=", 72))
println("  sigma squash (data-driven floor = 1 × group std)")
println(repeat("=", 72))
@printf("%-22s %10s %10s %10s %12s\n", "property", "std s", "σ lo", "σ hi", "2 log(lo/s)")
for (name, (lo, hi)) in zip(property_names, sigma_bounds_per)
    @printf("%-22s %10.4f %10.4f %10.4f %12.4f\n", name, lo, lo, hi, 0.0)
end
println(repeat("=", 72))

net = PriorNet(size(X, 1);
               width = WIDTH,
               depth = DEPTH,
               nproperties = 4,
               property_names = property_names,
               mu_bounds = mu_bounds,
               sigma_bounds = (0.05, sigma_hi),
               sigma_bounds_per = sigma_bounds_per)

cfg = TrainConfig(
    epochs = EPOCHS,
    learning_rate = 1.0e-3,
    weights = weights,
    log_every = LOG_EVERY,
    checkpoint_path = joinpath(WORK, "cloncurry_prior.jld2"),
    checkpoint_every = LOG_EVERY,
    seed = TRAIN_SEED,
    verbose = true,
)

#---------- 3. train (no VFSA, no MT, no gravity) ----------

t_train = time()
result = train_prior(net, X, grid, targets; config = cfg)
train_s = time() - t_train
first_hist = result.history[1]
last_hist = result.history[end]
best_hist = something(findfirst(h -> h.epoch == result.best_epoch, result.history),
                      lastindex(result.history))
best_hist = result.history[best_hist]

share_log = joinpath(WORK, "cloncurry_prior_shares.tsv")
open(share_log, "w") do io
    println(io, "epoch\ttotal\tgrade\tdensity\tsusceptibility\tconductivity_100kHz\tsmooth\tsigma\tgrade_share\tdensity_share\tsusceptibility_share\tconductivity_100kHz_share\tsmooth_share\tsigma_share")
    for h in result.history
        rows = (
            ("grade", weights.grade, h.grade),
            ("density", weights.density, h.density),
            ("susceptibility", weights.susceptibility, h.susceptibility),
            ("conductivity_100kHz", weights.conductivity, h.conductivity_100kHz),
            ("smooth", weights.smooth, h.smooth),
            ("sigma", weights.sigma, h.sigma),
        )
        shares = Float64[]
        for (_, w, t) in rows
            push!(shares, (isfinite(t) && h.total != 0) ? w * t / h.total : NaN)
        end
        @printf(io, "%d\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                h.epoch, h.total, h.grade, h.density, h.susceptibility, h.conductivity_100kHz,
                h.smooth, h.sigma, shares...)
        print_share_table(h, weights; title = "epoch $(h.epoch)")
    end
end

@printf("best loss %.5e at epoch %d\n", result.best_loss, result.best_epoch)
@printf("train_prior wall %.1f s  (width=%d depth=%d)\n", train_s, WIDTH, DEPTH)

(mus, sigmas), _ = predict(net, X, result.params.net, result.state)

anchor_cells, anchor_log, _ = anchors.grade
anchor_ppm = 10 .^ anchor_log
pred_anchor_ppm = 10 .^ mus[1, anchor_cells]
high_mask = anchor_ppm .>= CUTOFF_PPM
anchor_rmse = sqrt(mean(abs2, mus[1, anchor_cells] .- anchor_log))
high_rmse = count(high_mask) == 0 ? NaN :
    sqrt(mean(abs2, log10.(pred_anchor_ppm[high_mask]) .- log10.(anchor_ppm[high_mask])))
pred_high = count(high_mask) == 0 ? NaN : mean(pred_anchor_ppm[high_mask])
pred_bg = count(.!high_mask) == 0 ? NaN : mean(pred_anchor_ppm[.!high_mask])
assay_high = count(high_mask) == 0 ? NaN : mean(anchor_ppm[high_mask])
assay_bg = count(.!high_mask) == 0 ? NaN : mean(anchor_ppm[.!high_mask])
pred_ratio = (isfinite(pred_high) && isfinite(pred_bg) && pred_bg > 0) ?
    pred_high / pred_bg : NaN
assay_ratio = (isfinite(assay_high) && isfinite(assay_bg) && assay_bg > 0) ?
    assay_high / assay_bg : NaN
println()
println(repeat("=", 72))
println("  grade-anchor fit (best checkpoint)")
println(repeat("=", 72))
@printf("  log10-RMSE (all %d cells)     %.3f\n", length(anchor_cells), anchor_rmse)
@printf("  assay ≥ %.0f ppm cells        %d  log10-RMSE %.3f\n",
        CUTOFF_PPM, count(high_mask), high_rmse)
@printf("  high/bg pred                  %.2fx   assay %.2fx\n",
        pred_ratio, assay_ratio)
@printf("  mean(μ_grade)                 %.4f\n", mean(mus[1, :]))
println(repeat("=", 72))

println()
println(repeat("=", 72))
println("  sigma occupancy vs data-driven floor  (all cells / anchor cells)")
println(repeat("=", 72))
@printf("%-22s %8s %8s %8s %10s %10s\n",
        "property", "min", "mean", "max", "% at lo", "% anc lo")
sigma_occ = []
anchor_groups = (anchors.grade, anchors.density,
                 anchors.susceptibility, anchors.conductivity_100kHz)
for (p, name) in enumerate(property_names)
    lo, hi = sigma_bounds_per[p]
    col = sigmas[p, :]
    at = count(s -> s <= lo + 1e-3, col) / length(col)
    acells = anchor_groups[p][1]
    at_anc = isempty(acells) ? NaN :
        count(i -> col[i] <= lo + 1e-3, acells) / length(acells)
    @printf("%-22s %8.4f %8.4f %8.4f %9.1f%% %9.1f%%\n",
            name, minimum(col), mean(col), maximum(col), 100 * at, 100 * at_anc)
    push!(sigma_occ, (name, lo, hi, minimum(col), mean(col), maximum(col), at, at_anc))
end
println(repeat("=", 72))

open(joinpath(WORK, "cloncurry_prior_report.txt"), "w") do io
    println(io, "dataset\tcloncurry")
    println(io, "box\t", BOX)
    println(io, "grid\t", size(grid), "\t", ncells(grid),
            "\tcell_xy=", CELL, "\tcell_z=", CELL_Z, "\tdz=", grid.dz[1])
    println(io, "geochem_channels\t", join(geochem.names, ","))
    println(io, "geochem_n\t", length(geochem))
    println(io, "feature_names\t", join(stack.names, ","))
    println(io, "grade_cells\t", length(anchors.grade[1]))
    println(io, "density_cells\t", length(anchors.density[1]))
    println(io, "susceptibility_cells\t", length(anchors.susceptibility[1]))
    println(io, "conductivity_100kHz_cells\t", length(anchors.conductivity_100kHz[1]))
    println(io, "n_in_grid\t", n_in_grid)
    println(io, "n_samples\t", cov.n)
    println(io, "n_xyz\t", cov.n_xyz)
    println(io, "n_grade\t", cov.n_grade)
    println(io, "n_grade_mapped\t", cov.n_grade_mapped)
    println(io, "n_density_mapped\t", cov.n_density_mapped)
    println(io, "n_susceptibility_mapped\t", cov.n_susceptibility_mapped)
    println(io, "n_conductivity_100kHz_mapped\t", cov.n_conductivity_100kHz_mapped)
    println(io, "n_density\t", cov.n_density)
    println(io, "n_susceptibility\t", cov.n_susceptibility)
    println(io, "n_conductivity_100kHz\t", cov.n_conductivity_100kHz)
    println(io, "n_conductivity_zero\t", cov.n_conductivity_zero)
    println(io, "epochs_requested\t", EPOCHS)
    println(io, "log_every\t", LOG_EVERY)
    println(io, "width\t", WIDTH)
    println(io, "depth\t", DEPTH)
    println(io, "train_prior_s\t", train_s)
    println(io, "first_epoch\t", first_hist.epoch, "\t", first_hist.total)
    println(io, "last_epoch\t", last_hist.epoch, "\t", last_hist.total)
    println(io, "best_epoch\t", result.best_epoch, "\t", result.best_loss)
    println(io, "best_nll_grade\t", best_hist.grade)
    println(io, "best_nll_density\t", best_hist.density)
    println(io, "best_nll_susceptibility\t", best_hist.susceptibility)
    println(io, "best_nll_conductivity_100kHz\t", best_hist.conductivity_100kHz)
    println(io, "log10_rmse\t", anchor_rmse)
    println(io, "high_bg_pred\t", pred_ratio)
    println(io, "high_bg_assay\t", assay_ratio)
    for (name, w, t) in (
            ("grade", weights.grade, last_hist.grade),
            ("density", weights.density, last_hist.density),
            ("susceptibility", weights.susceptibility, last_hist.susceptibility),
            ("conductivity_100kHz", weights.conductivity, last_hist.conductivity_100kHz),
            ("smooth", weights.smooth, last_hist.smooth),
            ("sigma", weights.sigma, last_hist.sigma))
        share = (isfinite(t) && last_hist.total != 0) ? w * t / last_hist.total : NaN
        println(io, "last_share_", name, "\t", share, "\t", t)
    end
    println(io, "conductivity_100kHz_status\t", CLONCURRY_CONDUCTIVITY_STATUS)
    println(io, "mean_mu_grade\t", mean(mus[1, :]))
    println(io, "mean_mu_density\t", mean(mus[2, :]))
    println(io, "mean_mu_susceptibility\t", mean(mus[3, :]))
    println(io, "mean_mu_conductivity_100kHz\t", mean(mus[4, :]))
    for (name, lo, hi, mn, μσ, mx, at, at_anc) in sigma_occ
        println(io, "sigma_bounds_", name, "\t", lo, "\t", hi)
        println(io, "sigma_stats_", name, "\t", mn, "\t", μσ, "\t", mx, "\t", at, "\t", at_anc)
    end
end

save_prior(joinpath(WORK, "cloncurry_prior.jld2"), net, result.params, result.history;
           meta = Dict(
               "dataset" => "cloncurry",
               "box" => BOX,
               "nproperties" => 4,
               "property_names" => property_names,
               "cell_m" => CELL,
               "cell_z" => CELL_Z,
               "width" => WIDTH,
               "depth" => DEPTH,
               "epochs" => EPOCHS,
               "log_every" => LOG_EVERY,
               "sigma_bounds_per" => sigma_bounds_per,
               "conductivity_status" => CLONCURRY_CONDUCTIVITY_STATUS,
           ))
@info "wrote" joinpath(WORK, "cloncurry_prior_report.txt") joinpath(WORK, "cloncurry_prior.jld2")
