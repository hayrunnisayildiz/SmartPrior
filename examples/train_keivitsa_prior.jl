# Train a four-property prior on the Keivitsa (GTK) package.
#
# Separate from examples/compare_prior_2d.jl: no gravity, no MT, no VFSA.
# Features are till/bedrock geochemistry and drillhole lithology; anchors are
# Cu_Log assays plus petrophysical density / susceptibility / resistivity.
#
# PTR_D/DSR_D and PTR_J/DSR_J are a high-confidence inference from GTK's
# standard density–susceptibility–remanence measurement (see
# KEIVITSA_PETRO_STATUS). LUO_R remains unverified. PTR_K/DSR_K are unused
# (remanence candidate, not a fifth anchor group).
#
# Geochemistry: till/bedrock shapefiles plus drill assays. Surface wins on
# overlapping elements (Cu, Ni, Co, Pd, Au); drill supplies S, Fe, Cr, Pt.
#
# Run:  julia --project=. examples/train_keivitsa_prior.jl
# After training, export a block model (no VFSA):
#   julia --project=. examples/export_keivitsa_blockmodel.jl
# Env:  KEIVITSA_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z,
#       SMARTPRIOR_EPOCHS, SMARTPRIOR_LOG_EVERY, SMARTPRIOR_TRAIN_SEED,
#       SMARTPRIOR_WIDTH, SMARTPRIOR_DEPTH

using SmartPriorMT
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_keivitsa_prior"))
const CELL = parse(Float64, get(ENV, "SMARTPRIOR_CELL_M", "25"))
const CELL_Z = parse(Float64, get(ENV, "SMARTPRIOR_CELL_Z", string(CELL)))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "100"))
const LOG_EVERY = parse(Int, get(ENV, "SMARTPRIOR_LOG_EVERY",
                                 string(max(1, EPOCHS ÷ 10))))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const WIDTH = parse(Int, get(ENV, "SMARTPRIOR_WIDTH", "128"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const CUTOFF_PPM = 5000.0
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

function print_share_table(report, weights; title = "loss share")
    rows = (
        ("grade", weights.grade, report.grade),
        ("density", weights.density, report.density),
        ("susceptibility", weights.susceptibility, report.susceptibility),
        ("resistivity", weights.resistivity, report.resistivity),
        ("smooth", weights.smooth, report.smooth),
        ("sigma", weights.sigma, report.sigma),
    )
    println(repeat("=", 72))
    println("  ", title)
    println(repeat("=", 72))
    @printf("%-18s %8s %12s %12s %8s\n", "term", "weight", "unweighted", "weighted", "share")
    for (name, w, t) in rows
        isfinite(t) || continue
        wt = w * t
        share = report.total != 0 ? wt / report.total : NaN
        @printf("%-18s %8.3f %12.5f %12.5f %6.1f%%\n",
                name, w, t, wt, 100 * share)
        @printf("  %s share of total      %.4f / %.4f  = %.1f%%  (weight × unweighted)\n",
                name, wt, report.total, 100 * share)
    end
    @printf("total                                 %12.5f\n", report.total)
    println(repeat("=", 72))
end

const DATASET = default_keivitsa_root()
const REPORT = keivitsa_report_root(DATASET)
const CONFIG = joinpath(DATASET, "config.yaml")
@info "Keivitsa paths" DATASET WORK CELL CELL_Z EPOCHS LOG_EVERY WIDTH DEPTH

bounds = read_keivitsa_grid_bounds(CONFIG)
grid = keivitsa_grid(bounds; cell = CELL, cell_z = CELL_Z)
@info "prior grid" size(grid) ncells(grid) dx = grid.dx[1] dy = grid.dy[1] dz = grid.dz[1] bounds

CELL < 10 && @warn """keivitsa grid cell is $(CELL) m. The config block model is 5 m
(~2.9e6 cells). Full-batch training at that size is not the intended first run."""

#---------- 1. collars, surveys, features ----------

collars = load_keivitsa_collars(REPORT)
surveys = load_keivitsa_surveys(REPORT)
@info "holes" n_collar = length(collars) n_survey = length(surveys)

geochem_s, geochem_missing = load_keivitsa_geochemistry(REPORT)
intervals_csv = keivitsa_cleaned_intervals_path(DATASET)
geochem_dh = load_keivitsa_drill_geochemistry(intervals_csv)
geochem, geochem_skipped = combine_geochemistry(geochem_s, geochem_dh)
@info "geochemistry" surface = length(geochem_s) drill = length(geochem_dh) combined = length(geochem) channels = geochem.names missing_surface = geochem_missing skipped_overlap = geochem_skipped
println("  overlap policy: surface (till/bedrock) wins; drill Cu is the grade target and is not a feature")

lith = load_keivitsa_lithology(REPORT, collars, surveys)
@info "lithology" nsamples = length(lith)

# Surface till/bedrock shapefiles miss S/Fe/Cr/Pt; drill assays supply those
# as a second geochemistry source. Overlapping elements (Cu, Ni, Co, Pd, Au)
# keep the surface channel: till and core are different media, and drill Cu is
# already the grade target.
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
X = encode_features(stack; n_bands = 4)
@info "features" nchannels(stack) size(X) names = stack.names

#---------- 2. anchors ----------

grade_anchors, grade_stats = load_keivitsa_grade_anchors(grid, REPORT, collars, surveys)
petro_path = joinpath(REPORT, "3_DRILLINGS", "Downhole_soundings_and_core_measurements", "petro.txt")
petro = read_petro_txt(petro_path)
petro_frac = petro_coverage(petro)
petro_anchors = load_keivitsa_petrophysics_anchors(grid, petro, collars, surveys)

println()
println(repeat("=", 72))
println("  petrophysics documentation status")
println(repeat("=", 72))
for (k, v) in pairs(KEIVITSA_PETRO_STATUS)
    @printf("  %-8s  %s\n", k, v)
end
println(repeat("=", 72))
@printf("  petro.txt rows              %d\n", petro_frac.n)
@printf("  PTR_D finite                %.1f%%\n", 100 * petro_frac.ptr_d)
@printf("  PTR_J finite                %.1f%%\n", 100 * petro_frac.ptr_j)
@printf("  DSR_D finite                %.1f%%\n", 100 * petro_frac.dsr_d)
@printf("  DSR_J finite                %.1f%%\n", 100 * petro_frac.dsr_j)
@printf("  LUO_R finite                %.1f%%\n", 100 * petro_frac.luo_r)
@printf("  merged density finite       %.1f%%\n", 100 * petro_frac.density)
@printf("  merged susceptibility finite%.1f%%\n", 100 * petro_frac.susceptibility)
println(repeat("=", 72))

n_grid = ncells(grid)
function cover_line(name, anchors, extra = "")
    n = length(anchors[1])
    @printf("  %-18s  %6d cells  (%5.2f%% of grid)%s\n",
            name, n, 100 * n / n_grid, extra)
end

println("  anchor cell coverage")
println(repeat("-", 72))
cover_line("grade (Cu_Log)", grade_anchors,
           @sprintf("  %d holes, %d unique intervals, %d CU rows",
                    grade_stats.n_holes, grade_stats.n_unique_intervals,
                    grade_stats.n_cu_rows))
cover_line("density", petro_anchors.density)
cover_line("susceptibility", petro_anchors.susceptibility)
cover_line("resistivity", petro_anchors.resistivity)
@printf("  grade / density cell ratio  %.1fx  (NLL is a per-group mean, so this\n",
        length(grade_anchors[1]) / max(length(petro_anchors.density[1]), 1))
println("  does not automatically down-weight the sparse petro groups)")
println(repeat("=", 72))

property_names = ["grade", "density", "susceptibility", "resistivity"]
targets = PriorTargets(
    anchors_grade = grade_anchors,
    anchors_density = petro_anchors.density,
    anchors_susceptibility = petro_anchors.susceptibility,
    anchors_resistivity = petro_anchors.resistivity,
    property_names = property_names,
    sigma_target = 0.5,
)

weights = LossWeights(
    grade = 1.0,
    density = 1.0,
    susceptibility = 1.0,
    resistivity = 1.0,
    smooth = 1.0e-2,
    sigma = 1.0e-2,
)

# bounds in the spaces the NLL sees: Cu_Log, g/cm³, log10 J, log10 Ω·m
function padded_bounds(values; pad = 0.2, fallback)
    finite = filter(isfinite, values)
    isempty(finite) && return fallback
    lo, hi = extrema(finite)
    lo == hi && return (lo - pad, hi + pad)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)
end

mu_bounds = [
    padded_bounds(grade_anchors[2]; fallback = (0.0, 5.0)),
    padded_bounds(petro_anchors.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(petro_anchors.susceptibility[2]; fallback = (-1.0, 6.0)),
    padded_bounds(petro_anchors.resistivity[2]; fallback = (-1.0, 7.0)),
]
@info "mu_bounds" property_names mu_bounds

# Per-property σ squash. The historical global floor of 0.05 is below density's
# group std (~0.1) and far below resistivity's (~2), which is how calibrated
# NLL went negative around epoch 100–150. Floor = 1 × group std so σ cannot
# undercut the scale the NLL is scored in.
sigma_hi = 1.2
sigma_bounds_per = [
    sigma_bounds_from_anchors(grade_anchors; hi = sigma_hi),
    sigma_bounds_from_anchors(petro_anchors.density; hi = sigma_hi),
    sigma_bounds_from_anchors(petro_anchors.susceptibility; hi = sigma_hi),
    sigma_bounds_from_anchors(petro_anchors.resistivity; hi = sigma_hi),
]
println()
println(repeat("=", 72))
println("  sigma squash (data-driven floor = 1 × group std)")
println(repeat("=", 72))
@printf("%-18s %10s %10s %10s %12s\n", "property", "std s", "σ lo", "σ hi", "2 log(lo/s)")
for (name, (lo, hi)) in zip(property_names, sigma_bounds_per)
    # fraction = 1 ⇒ lo = s, so 2 log(lo/s) = 0 by construction
    @printf("%-18s %10.4f %10.4f %10.4f %12.4f\n", name, lo, lo, hi, 0.0)
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
    checkpoint_path = joinpath(WORK, "keivitsa_prior.jld2"),
    checkpoint_every = LOG_EVERY,
    seed = TRAIN_SEED,
    verbose = true,
)

#---------- 3. train (no VFSA) ----------

t_train = time()
result = train_prior(net, X, grid, targets; config = cfg)
train_s = time() - t_train
first_hist = result.history[1]
last_hist = result.history[end]
best_hist = something(findfirst(h -> h.epoch == result.best_epoch, result.history),
                      lastindex(result.history))
best_hist = result.history[best_hist]

share_log = joinpath(WORK, "keivitsa_prior_shares.tsv")
open(share_log, "w") do io
    println(io, "epoch\ttotal\tgrade\tdensity\tsusceptibility\tresistivity\tsmooth\tsigma\tgrade_share\tdensity_share\tsusceptibility_share\tresistivity_share\tsmooth_share\tsigma_share")
    for h in result.history
        rows = (
            ("grade", weights.grade, h.grade),
            ("density", weights.density, h.density),
            ("susceptibility", weights.susceptibility, h.susceptibility),
            ("resistivity", weights.resistivity, h.resistivity),
            ("smooth", weights.smooth, h.smooth),
            ("sigma", weights.sigma, h.sigma),
        )
        shares = Float64[]
        for (_, w, t) in rows
            push!(shares, (isfinite(t) && h.total != 0) ? w * t / h.total : NaN)
        end
        @printf(io, "%d\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                h.epoch, h.total, h.grade, h.density, h.susceptibility, h.resistivity,
                h.smooth, h.sigma, shares...)
        print_share_table(h, weights; title = "epoch $(h.epoch)")
    end
end

@printf("best loss %.5e at epoch %d\n", result.best_loss, result.best_epoch)
@printf("train_prior wall %.1f s  (width=%d depth=%d)\n", train_s, WIDTH, DEPTH)

(mus, sigmas), _ = predict(net, X, result.params.net, result.state)

anchor_cells, anchor_log, _ = grade_anchors
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
@printf("%-18s %8s %8s %8s %10s %10s\n",
        "property", "min", "mean", "max", "% at lo", "% anc lo")
sigma_occ = []
anchor_groups = (grade_anchors, petro_anchors.density,
                 petro_anchors.susceptibility, petro_anchors.resistivity)
for (p, name) in enumerate(property_names)
    lo, hi = sigma_bounds_per[p]
    col = sigmas[p, :]
    at = count(s -> s <= lo + 1e-3, col) / length(col)
    acells = anchor_groups[p][1]
    at_anc = isempty(acells) ? NaN :
        count(i -> col[i] <= lo + 1e-3, acells) / length(acells)
    @printf("%-18s %8.4f %8.4f %8.4f %9.1f%% %9.1f%%\n",
            name, minimum(col), mean(col), maximum(col), 100 * at, 100 * at_anc)
    push!(sigma_occ, (name, lo, hi, minimum(col), mean(col), maximum(col), at, at_anc))
end
println(repeat("=", 72))

open(joinpath(WORK, "keivitsa_prior_report.txt"), "w") do io
    println(io, "grid\t", size(grid), "\t", ncells(grid),
            "\tcell_xy=", CELL, "\tcell_z=", CELL_Z, "\tdz=", grid.dz[1])
    println(io, "geochem_channels\t", join(geochem.names, ","))
    println(io, "geochem_missing_surface\t", join(geochem_missing, ","))
    println(io, "geochem_surface_n\t", length(geochem_s))
    println(io, "geochem_drill_n\t", length(geochem_dh))
    println(io, "geochem_skipped_overlap\t", join(geochem_skipped, ","))
    println(io, "feature_names\t", join(stack.names, ","))
    println(io, "grade_cells\t", length(grade_anchors[1]))
    println(io, "density_cells\t", length(petro_anchors.density[1]))
    println(io, "susceptibility_cells\t", length(petro_anchors.susceptibility[1]))
    println(io, "resistivity_cells\t", length(petro_anchors.resistivity[1]))
    println(io, "petro_n\t", petro_frac.n)
    println(io, "petro_ptr_d\t", petro_frac.ptr_d)
    println(io, "petro_ptr_j\t", petro_frac.ptr_j)
    println(io, "petro_dsr_d\t", petro_frac.dsr_d)
    println(io, "petro_dsr_j\t", petro_frac.dsr_j)
    println(io, "petro_luo_r\t", petro_frac.luo_r)
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
    println(io, "best_nll_resistivity\t", best_hist.resistivity)
    println(io, "log10_rmse\t", anchor_rmse)
    println(io, "high_bg_pred\t", pred_ratio)
    println(io, "high_bg_assay\t", assay_ratio)
    for (name, w, t) in (
            ("grade", weights.grade, last_hist.grade),
            ("density", weights.density, last_hist.density),
            ("susceptibility", weights.susceptibility, last_hist.susceptibility),
            ("resistivity", weights.resistivity, last_hist.resistivity),
            ("smooth", weights.smooth, last_hist.smooth),
            ("sigma", weights.sigma, last_hist.sigma))
        share = (isfinite(t) && last_hist.total != 0) ? w * t / last_hist.total : NaN
        println(io, "last_share_", name, "\t", share, "\t", t)
    end
    println(io, "LUO_R_status\t", KEIVITSA_PETRO_STATUS.LUO_R)
    println(io, "PTR_D_status\t", KEIVITSA_PETRO_STATUS.PTR_D)
    println(io, "PTR_J_status\t", KEIVITSA_PETRO_STATUS.PTR_J)
    println(io, "PTR_K_status\t", KEIVITSA_PETRO_STATUS.PTR_K)
    println(io, "mean_mu_grade\t", mean(mus[1, :]))
    println(io, "mean_mu_density\t", mean(mus[2, :]))
    println(io, "mean_mu_susceptibility\t", mean(mus[3, :]))
    println(io, "mean_mu_resistivity\t", mean(mus[4, :]))
    for (name, lo, hi, mn, μσ, mx, at, at_anc) in sigma_occ
        println(io, "sigma_bounds_", name, "\t", lo, "\t", hi)
        println(io, "sigma_stats_", name, "\t", mn, "\t", μσ, "\t", mx, "\t", at, "\t", at_anc)
    end
end

save_prior(joinpath(WORK, "keivitsa_prior.jld2"), net, result.params, result.history;
           meta = Dict(
               "dataset" => "keivitsa",
               "nproperties" => 4,
               "property_names" => property_names,
               "cell_m" => CELL,
               "cell_z" => CELL_Z,
               "width" => WIDTH,
               "depth" => DEPTH,
               "epochs" => EPOCHS,
               "log_every" => LOG_EVERY,
               "sigma_bounds_per" => sigma_bounds_per,
               "petro_status" => Dict(string(k) => v for (k, v) in pairs(KEIVITSA_PETRO_STATUS)),
           ))
@info "wrote" joinpath(WORK, "keivitsa_prior_report.txt") joinpath(WORK, "keivitsa_prior.jld2")
