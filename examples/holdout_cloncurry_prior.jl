# Spatial hold-out of Ernest Henry Cu anchors.
#
# Trains only on the train split (same architecture as D: depth=4, σ floor =
# 1× group std, 100 m × 50 m grid). Reports train vs test log10-RMSE and
# high/bg. A second, narrower net (default width 64) uses the same split.
#
# No gravity, magnetics, MT, or VFSA. Cu is never a feature.
#
# Run:
#   SMARTPRIOR_WORK=tmp_cloncurry_prior_eh_holdout \
#   julia --project=. examples/holdout_cloncurry_prior.jl
#
# Env:  CLONCURRY_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z,
#       SMARTPRIOR_EPOCHS, SMARTPRIOR_LOG_EVERY, SMARTPRIOR_TRAIN_SEED,
#       SMARTPRIOR_WIDTHS (comma-separated, default 256,64), SMARTPRIOR_DEPTH,
#       SMARTPRIOR_HOLDOUT_FRAC, SMARTPRIOR_BUFFER_M, SMARTPRIOR_SPLIT_SEED

using SmartPriorMT
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_cloncurry_prior_eh_holdout"))
const CELL = parse(Float64, get(ENV, "SMARTPRIOR_CELL_M", "100"))
const CELL_Z = parse(Float64, get(ENV, "SMARTPRIOR_CELL_Z", "50"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "250"))
const LOG_EVERY = parse(Int, get(ENV, "SMARTPRIOR_LOG_EVERY",
                                 string(max(1, EPOCHS ÷ 10))))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const SPLIT_SEED = parse(Int, get(ENV, "SMARTPRIOR_SPLIT_SEED", "2026"))
const DEPTH = parse(Int, get(ENV, "SMARTPRIOR_DEPTH", "4"))
const HOLDOUT_FRAC = parse(Float64, get(ENV, "SMARTPRIOR_HOLDOUT_FRAC", "0.2"))
const BUFFER_M = parse(Float64, get(ENV, "SMARTPRIOR_BUFFER_M", "300"))
const CUTOFF_PPM = 5000.0
const WIDTHS = [parse(Int, strip(w))
                for w in split(get(ENV, "SMARTPRIOR_WIDTHS", "256,64"), ',')
                if !isempty(strip(w))]
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

function min_pair_distance(x, y, z, a, b)
    (isempty(a) || isempty(b)) && return NaN
    m = Inf
    for i in a, j in b
        d = hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j])
        d < m && (m = d)
    end
    return m
end

function cell_fit(mus, anchors; cutoff = CUTOFF_PPM)
    cells, logs, _ = anchors
    isempty(cells) && return (n = 0, rmse = NaN, n_high = 0, pred_ratio = NaN,
                              assay_ratio = NaN)
    pred_log = mus[1, cells]
    ppm = 10 .^ logs
    pred_ppm = 10 .^ pred_log
    rmse = sqrt(mean(abs2, pred_log .- logs))
    high = ppm .>= cutoff
    n_high = count(high)
    pred_high = n_high == 0 ? NaN : mean(pred_ppm[high])
    pred_bg = count(.!high) == 0 ? NaN : mean(pred_ppm[.!high])
    assay_high = n_high == 0 ? NaN : mean(ppm[high])
    assay_bg = count(.!high) == 0 ? NaN : mean(ppm[.!high])
    pred_ratio = (isfinite(pred_high) && isfinite(pred_bg) && pred_bg > 0) ?
        pred_high / pred_bg : NaN
    assay_ratio = (isfinite(assay_high) && isfinite(assay_bg) && assay_bg > 0) ?
        assay_high / assay_bg : NaN
    return (n = length(cells), rmse = rmse, n_high = n_high,
            pred_ratio = pred_ratio, assay_ratio = assay_ratio)
end

function sample_fit(mus, grid, samples, idx; cutoff = CUTOFF_PPM)
    logs = Float64[]
    pred = Float64[]
    for i in idx
        c = containing_cell(grid, samples.east[i], samples.north[i], samples.elev[i])
        c == 0 && continue
        isfinite(samples.cu_ppm[i]) && samples.cu_ppm[i] > 0 || continue
        push!(logs, log10(samples.cu_ppm[i]))
        push!(pred, mus[1, c])
    end
    isempty(logs) && return (n = 0, rmse = NaN, n_high = 0, pred_ratio = NaN,
                             assay_ratio = NaN)
    ppm = 10 .^ logs
    pred_ppm = 10 .^ pred
    rmse = sqrt(mean(abs2, pred .- logs))
    high = ppm .>= cutoff
    n_high = count(high)
    pred_high = n_high == 0 ? NaN : mean(pred_ppm[high])
    pred_bg = count(.!high) == 0 ? NaN : mean(pred_ppm[.!high])
    assay_high = n_high == 0 ? NaN : mean(ppm[high])
    assay_bg = count(.!high) == 0 ? NaN : mean(ppm[.!high])
    pred_ratio = (isfinite(pred_high) && isfinite(pred_bg) && pred_bg > 0) ?
        pred_high / pred_bg : NaN
    assay_ratio = (isfinite(assay_high) && isfinite(assay_bg) && assay_bg > 0) ?
        assay_high / assay_bg : NaN
    return (n = length(logs), rmse = rmse, n_high = n_high,
            pred_ratio = pred_ratio, assay_ratio = assay_ratio)
end

function print_fit(title, fit)
    println(repeat("-", 72))
    println("  ", title)
    @printf("  n                         %d\n", fit.n)
    @printf("  log10-RMSE                %.3f\n", fit.rmse)
    @printf("  assay ≥ %.0f ppm           %d\n", CUTOFF_PPM, fit.n_high)
    @printf("  high/bg pred              %.2fx   assay %.2fx\n",
            fit.pred_ratio, fit.assay_ratio)
end

function keep_mask(n, idx)
    m = falses(n)
    m[idx] .= true
    return m
end

const DATASET = default_cloncurry_root()
@info "holdout paths" DATASET WORK CELL CELL_Z EPOCHS WIDTHS DEPTH BUFFER_M HOLDOUT_FRAC

samples = load_cloncurry_samples(DATASET)
bounds = cloncurry_deposit_bounds(samples, "Ernest Henry")
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)
@info "prior grid" size(grid) ncells(grid) dx = grid.dx[1] dy = grid.dy[1] dz = grid.dz[1]

geochem = cloncurry_geochemistry(samples)
lith = cloncurry_lithology(samples)
coverage = cloncurry_coverage_points(samples)
stack = build_features(grid;
                       coordinates = true,
                       geochemistry = geochem,
                       lithology = lith,
                       coverage_points = coverage)
X = encode_features(stack; n_bands = 4)
@info "features" nchannels(stack) size(X)
println("  overlap policy: Cu_Concentration is the grade target and is not a feature")
println("  hold-out hides Cu labels only; pXRF / lithology / coverage still use all samples")

eligible = cloncurry_grade_eligible(grid, samples)
split = spatial_holdout(samples.east, samples.north, samples.elev, eligible;
                        fraction = HOLDOUT_FRAC, buffer = BUFFER_M,
                        rng = Xoshiro(SPLIT_SEED))
dmin = min_pair_distance(samples.east, samples.north, samples.elev,
                         split.train, split.test)

println()
println(repeat("=", 72))
println("  spatial hold-out (Cu samples inside the D grid)")
println(repeat("=", 72))
@printf("  eligible (in-grid Cu)      %d\n", split.n_eligible)
@printf("  target test fraction       %.2f  (%d samples)\n",
        split.fraction, split.n_test_target)
@printf("  train                      %d\n", length(split.train))
@printf("  test                       %d\n", length(split.test))
@printf("  buffer zone (dropped)      %d   (<%.0f m from a test point)\n",
        length(split.buffer), BUFFER_M)
@printf("  min train–test distance    %.0f m\n", dmin)
@printf("  split seed                 %d\n", SPLIT_SEED)
println("  density / susc / cond anchors are not held out")
println(repeat("=", 72))

open(joinpath(WORK, "cloncurry_holdout_split.tsv"), "w") do io
    println(io, "sample\tdeposit\teast\tnorth\telev\tcu_ppm\trole")
    role = fill("unused", length(samples))
    role[split.train] .= "train"
    role[split.test] .= "test"
    role[split.buffer] .= "buffer"
    for i in 1:length(samples)
        eligible[i] || continue
        @printf(io, "%s\t%s\t%.3f\t%.3f\t%.3f\t%.6g\t%s\n",
                samples.sample[i], samples.deposit[i],
                samples.east[i], samples.north[i], samples.elev[i],
                samples.cu_ppm[i], role[i])
    end
end

train_keep = keep_mask(length(samples), split.train)
test_keep = keep_mask(length(samples), split.test)
train_loaded = load_cloncurry_anchors(grid, samples; grade_keep = train_keep)
test_loaded = load_cloncurry_anchors(grid, samples; grade_keep = test_keep)
full_loaded = load_cloncurry_anchors(grid, samples)

@printf("  grade cells train/test/full  %d / %d / %d\n",
        length(train_loaded.grade[1]), length(test_loaded.grade[1]),
        length(full_loaded.grade[1]))

property_names = copy(CLONCURRY_PROPERTY_NAMES)
weights = LossWeights(
    grade = 1.0,
    density = 1.0,
    susceptibility = 1.0,
    conductivity = 1.0,
    smooth = 1.0e-2,
    sigma = 1.0e-2,
)

# Grade squash from the train labels only — test Cu must not set the range.
mu_bounds = [
    padded_bounds(train_loaded.grade[2]; fallback = (0.0, 5.0)),
    padded_bounds(train_loaded.density[2]; fallback = (2.0, 4.0), pad = 0.1),
    padded_bounds(train_loaded.susceptibility[2]; fallback = (-7.0, 1.0)),
    padded_bounds(train_loaded.conductivity_100kHz[2]; fallback = (-3.0, 3.0)),
]
sigma_hi = 1.2
sigma_bounds_per = [
    sigma_bounds_from_anchors(train_loaded.grade; hi = sigma_hi),
    sigma_bounds_from_anchors(train_loaded.density; hi = sigma_hi),
    sigma_bounds_from_anchors(train_loaded.susceptibility; hi = sigma_hi),
    sigma_bounds_from_anchors(train_loaded.conductivity_100kHz; hi = sigma_hi),
]

targets = PriorTargets(
    anchors_grade = train_loaded.grade,
    anchors_density = train_loaded.density,
    anchors_susceptibility = train_loaded.susceptibility,
    anchors_conductivity = train_loaded.conductivity_100kHz,
    property_names = property_names,
    sigma_target = 0.5,
)

rows_out = []

for width in WIDTHS
    println()
    println(repeat("=", 72))
    @printf("  train width=%d depth=%d epochs=%d  (grade NLL on train Cu only)\n",
            width, DEPTH, EPOCHS)
    println(repeat("=", 72))

    net = PriorNet(size(X, 1);
                   width = width,
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
        checkpoint_path = joinpath(WORK, "cloncurry_holdout_w$(width).jld2"),
        checkpoint_every = LOG_EVERY,
        seed = TRAIN_SEED,
        verbose = true,
    )
    t_train = time()
    result = train_prior(net, X, grid, targets; config = cfg)
    train_s = time() - t_train
    (mus, _), _ = predict(net, X, result.params.net, result.state)

    train_cells = cell_fit(mus, train_loaded.grade)
    test_cells = cell_fit(mus, test_loaded.grade)
    train_samp = sample_fit(mus, grid, samples, split.train)
    test_samp = sample_fit(mus, grid, samples, split.test)
    ratio = (isfinite(test_samp.rmse) && isfinite(train_samp.rmse) &&
             train_samp.rmse > 0) ? test_samp.rmse / train_samp.rmse : NaN

    println()
    println(repeat("=", 72))
    @printf("  hold-out fit  width=%d  best epoch %d  wall %.1f s\n",
            width, result.best_epoch, train_s)
    println(repeat("=", 72))
    print_fit("train samples", train_samp)
    print_fit("test samples", test_samp)
    print_fit("train cells (for comparison to D's 104-cell RMSE)", train_cells)
    print_fit("test cells", test_cells)
    @printf("  test/train sample RMSE     %.2f×   (5–10× ⇒ memorization)\n", ratio)
    println(repeat("=", 72))

    row = (width = width, depth = DEPTH, epochs = EPOCHS,
           best_epoch = result.best_epoch, wall_s = train_s,
           best_loss = result.best_loss,
           nll_grade = result.history[end].grade,
           train_n = train_samp.n, train_rmse = train_samp.rmse,
           train_pred_ratio = train_samp.pred_ratio,
           test_n = test_samp.n, test_rmse = test_samp.rmse,
           test_pred_ratio = test_samp.pred_ratio,
           test_assay_ratio = test_samp.assay_ratio,
           train_cells = train_cells.n, train_cell_rmse = train_cells.rmse,
           test_cells = test_cells.n, test_cell_rmse = test_cells.rmse,
           rmse_ratio = ratio)
    push!(rows_out, row)

    save_prior(joinpath(WORK, "cloncurry_holdout_w$(width).jld2"),
               net, result.params, result.history;
               meta = Dict(
                   "dataset" => "cloncurry",
                   "box" => "ernest_henry",
                   "holdout" => true,
                   "width" => width,
                   "depth" => DEPTH,
                   "epochs" => EPOCHS,
                   "buffer_m" => BUFFER_M,
                   "holdout_frac" => HOLDOUT_FRAC,
                   "n_train" => length(split.train),
                   "n_test" => length(split.test),
                   "n_buffer" => length(split.buffer),
                   "train_rmse" => train_samp.rmse,
                   "test_rmse" => test_samp.rmse,
               ))
end

open(joinpath(WORK, "cloncurry_holdout_report.txt"), "w") do io
    println(io, "dataset\tcloncurry")
    println(io, "box\ternest_henry")
    println(io, "grid\t", size(grid), "\t", ncells(grid),
            "\tcell_xy=", CELL, "\tcell_z=", CELL_Z)
    println(io, "n_eligible\t", split.n_eligible)
    println(io, "n_train\t", length(split.train))
    println(io, "n_test\t", length(split.test))
    println(io, "n_buffer\t", length(split.buffer))
    println(io, "buffer_m\t", BUFFER_M)
    println(io, "holdout_frac\t", HOLDOUT_FRAC)
    println(io, "min_train_test_m\t", dmin)
    println(io, "split_seed\t", SPLIT_SEED)
    println(io, "train_seed\t", TRAIN_SEED)
    println(io, "d_full_data_rmse\t0.005")
    println(io, "keivitsa_250ep_rmse\t0.249")
    println(io, "keivitsa_250ep_contrast\t5.81")
    println(io, "note\tCu labels held out; geochem/lithology/coverage still use all samples")
    println(io, "width\ttrain_n\ttrain_rmse\ttest_n\ttest_rmse\ttest_over_train\ttrain_highbg\ttest_highbg\ttrain_cell_rmse\ttest_cell_rmse\twall_s")
    for r in rows_out
        @printf(io, "%d\t%d\t%.6f\t%d\t%.6f\t%.4f\t%.4f\t%.4f\t%.6f\t%.6f\t%.1f\n",
                r.width, r.train_n, r.train_rmse, r.test_n, r.test_rmse,
                r.rmse_ratio, r.train_pred_ratio, r.test_pred_ratio,
                r.train_cell_rmse, r.test_cell_rmse, r.wall_s)
    end
end

println()
@info "wrote" joinpath(WORK, "cloncurry_holdout_report.txt") joinpath(WORK, "cloncurry_holdout_split.tsv")
