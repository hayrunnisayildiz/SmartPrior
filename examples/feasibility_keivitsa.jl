# Keivitsa feasibility: 10-fold hole-grouped CV for one target property.
# Methods: mean, kriging, nn_xyz (E2), nn_cov (E2).
#
# E2 = Gaussian NLL training, early stop on validation RMSE of μ (see docs/keivitsa_data_notes.md).
#
# Target: SMARTPRIOR_TARGET = cu (default) | density | susceptibility.
# The Cu run is unchanged: same work directory, same bootstrap tags, same tables.
#
# Run:  julia --project=. examples/feasibility_keivitsa.jl
#       SMARTPRIOR_TARGET=density julia --project=. examples/feasibility_keivitsa.jl
# Out:  tmp_feasibility_keivitsa/            (cu, gitignored)
#       tmp_feasibility_keivitsa_<target>/   (other targets, gitignored)

const ROOT = dirname(@__DIR__)

const ALLOWED_TARGETS = (:cu, :density, :susceptibility)
const TARGET = let t = Symbol(lowercase(strip(get(ENV, "SMARTPRIOR_TARGET", "cu"))))
    t in ALLOWED_TARGETS || error(
        "SMARTPRIOR_TARGET must be one of $(join(ALLOWED_TARGETS, ", ")), got $t")
    t
end

# Honor SMARTPRIOR_WORK when set (relative paths are from the repo root).
# Default: tmp_feasibility_keivitsa for cu, tmp_feasibility_keivitsa_<target> otherwise.
if !haskey(ENV, "SMARTPRIOR_WORK") || isempty(ENV["SMARTPRIOR_WORK"])
    ENV["SMARTPRIOR_WORK"] = joinpath(ROOT,
        TARGET === :cu ? "tmp_feasibility_keivitsa" : "tmp_feasibility_keivitsa_$(TARGET)")
elseif !isabspath(ENV["SMARTPRIOR_WORK"])
    ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, ENV["SMARTPRIOR_WORK"])
end
include(joinpath(@__DIR__, "feasibility_loho.jl"))

const SITE_PATH = joinpath(ROOT, "sites", "keivitsa.toml")
const DEPOSIT = "keivitsa"

const N_FOLDS = 10
const KFOLD_SEED = 2026
# Hole count checked against the frozen Cu run. Other targets have no frozen count yet.
const EXPECTED_HOLES = Dict(:cu => 261)

const NN_TRAIN_KW = (loss = :nll, stop_on = :val_rmse)
const LOSS_LOG_EVERY = 50

const SKILL_MIN = 0.10
const KRIGING_LAG = 0.05
const BOOT_SEED_TAG = "feasibility-keivitsa-bootstrap-v1"

const METHODS_BASE = ("mean", "kriging")
const METHODS_NN = ("nn_xyz", "nn_cov")
const UNCERTAIN = ("kriging", "nn_xyz", "nn_cov")

struct KFoldResult
    fold_id::Int
    test_holes::Vector{String}
    train_rows::Vector{Int}
    test_rows::Vector{Int}
    train_mean::Float64
end

function pooled_skill_multifold(obs, pred, mu_per_sample)
    n = length(obs)
    n == length(pred) == length(mu_per_sample) || fail("skill: length mismatch")
    n == 0 && return NaN
    rmse = sqrt(mean(abs2, pred .- obs))
    rmse_mean = sqrt(mean(abs2, obs .- mu_per_sample))
    return rmse_mean == 0 ? NaN : 1 - rmse / rmse_mean
end

function keivitsa_sample_ids(table)
    return [string(i) for i in 1:nsamples(table)]
end

function hole_kfold_assignments(holes::Vector{String}, n_folds::Int, seed::Int)
    n_folds >= 2 || fail("k-fold needs at least 2 folds")
    perm = shuffle(Xoshiro(seed), collect(holes))
    folds = [String[] for _ in 1:n_folds]
    for (i, h) in enumerate(perm)
        push!(folds[mod1(i, n_folds)], h)
    end
    return folds
end

function prepare_fold_k(table, col_of, y_all, train_rows, test_rows, fold_id::Int,
                        deposit, target)
    y_tr = y_all[train_rows]
    m = mean(y_tr)
    s = std(y_tr)
    (isfinite(m) && isfinite(s) && s > 0) || fail(
        "fold $fold_id: training std is not positive")
    holes = sort!(unique(table.hole[train_rows]))
    tag = "val|$(deposit)|$(target)|fold$(fold_id)"
    fit_holes, val_holes = split_train_val(holes, rng_for(tag))
    fit_set = Set(fit_holes)
    val_set = Set(val_holes)
    fit_rows = [i for i in train_rows if table.hole[i] in fit_set]
    val_rows = [i for i in train_rows if table.hole[i] in val_set]
    (isempty(fit_rows) || isempty(val_rows)) && fail(
        "fold $fold_id: empty neural-net hole split")
    y_fit = (y_all[fit_rows] .- m) ./ s
    y_val = (y_all[val_rows] .- m) ./ s
    label = "fold$(fold_id)"
    return FoldRef(deposit, string(target), label, test_rows, m, s,
                   columns_of(col_of, fit_rows), columns_of(col_of, val_rows),
                   columns_of(col_of, test_rows), y_fit, y_val, length(val_holes))
end

function loss_at_step(history, step::Int)
    for h in history
        h.step == step || continue
        return h.train_loss, h.val_loss, h.val_rmse
    end
    return NaN, NaN, NaN
end

function append_loss_curves(io, fold_id::Int, method::AbstractString, seed::Int, history)
    for h in history
        println(io, join((
            string(fold_id), string(seed), method, string(h.step),
            tsv_num(h.train_loss), tsv_num(h.val_loss), tsv_num(h.val_rmse),
        ), '\t'))
    end
    flush(io)
end

function train_ensemble(model, Xtr, ytr, Xva, yva, ctx, fold_id::Int, method::AbstractString,
                        loss_io)
    members = Tuple{Any,Any}[]
    epochs = Int[]
    for seed in NN_SEEDS
        history = Any[]
        ps, st, best_ep, last_ep, best_stop = train_member(
            model, seed, Xtr, ytr, Xva, yva, ctx;
            NN_TRAIN_KW..., log_every = LOSS_LOG_EVERY, history = history)
        append_loss_curves(loss_io, fold_id, method, seed, history)
        tr_b, va_b, rm_b = loss_at_step(history, best_ep)
        tr_f, va_f, rm_f = loss_at_step(history, last_ep)
        logmsg(@sprintf("%s seed %d: best_step=%d train_loss=%.5e val_loss=%.5e val_rmse=%.5e | final_step=%d train_loss=%.5e val_loss=%.5e val_rmse=%.5e (stop_metric=%.5e)",
                        ctx, seed, best_ep, tr_b, va_b, rm_b, last_ep, tr_f, va_f, rm_f, best_stop))
        push!(members, (ps, st))
        push!(epochs, best_ep)
        last_ep == NN_EPOCHS &&
            logmsg("$ctx seed $seed ran all $(NN_EPOCHS) epochs (best $best_ep)")
    end
    return members, epochs
end

function row_train_means(folds::Vector{KFoldResult})
    mu = Dict{Int,Float64}()
    for fr in folds
        for i in fr.test_rows
            mu[i] = fr.train_mean
        end
    end
    return mu
end

function select_preds(preds, method, subset)
    rows = [r for r in preds if r.target == string(TARGET) && r.method == method]
    if subset == "uncensored"
        rows = [r for r in rows if r.censored == 0]
    end
    return rows
end

function pooled_skill_from_rows(rows, mu_by_sample::Dict{Int,Float64})
    obs = Float64[]
    pred = Float64[]
    mu = Float64[]
    for r in rows
        row_idx = parse(Int, r.sample)
        haskey(mu_by_sample, row_idx) || continue
        push!(obs, r.obs)
        push!(pred, r.pred)
        push!(mu, mu_by_sample[row_idx])
    end
    return pooled_skill_multifold(obs, pred, mu)
end

function bootstrap_skill_ci(rows, mu_by_sample::Dict{Int,Float64}, tag::AbstractString)
    by_hole = Dict{String,Vector{Pred}}()
    for r in rows
        push!(get!(by_hole, r.hole, Pred[]), r)
    end
    holes = sort!(collect(keys(by_hole)))
    n = length(holes)
    n >= 1 || fail("bootstrap skill: no holes")
    rng = rng_for(tag)
    skills = Vector{Float64}(undef, N_BOOT)
    for b in 1:N_BOOT
        obs = Float64[]
        pred = Float64[]
        mu = Float64[]
        for _ in 1:n
            h = holes[rand(rng, 1:n)]
            for r in by_hole[h]
                row_idx = parse(Int, r.sample)
                push!(obs, r.obs)
                push!(pred, r.pred)
                push!(mu, mu_by_sample[row_idx])
            end
        end
        skills[b] = pooled_skill_multifold(obs, pred, mu)
    end
    point = pooled_skill_from_rows(rows, mu_by_sample)
    return point, quantile(skills, 0.025), quantile(skills, 0.975)
end

function hole_skill(rows::AbstractVector{Pred}, mu_by_sample::Dict{Int,Float64})
    sse = Dict{String,Float64}()
    sst = Dict{String,Float64}()
    cnt = Dict{String,Int}()
    for r in rows
        row_idx = parse(Int, r.sample)
        μ0 = mu_by_sample[row_idx]
        e = r.obs - μ0
        sse[r.hole] = get(sse, r.hole, 0.0) + (r.pred - r.obs)^2
        sst[r.hole] = get(sst, r.hole, 0.0) + e^2
        cnt[r.hole] = get(cnt, r.hole, 0) + 1
    end
    holes = sort!(collect(keys(cnt)))
    sk = Dict{String,Float64}()
    for h in holes
        rmse = sqrt(sse[h] / cnt[h])
        rmse0 = sqrt(sst[h] / cnt[h])
        sk[h] = rmse0 == 0 ? NaN : 1 - rmse / rmse0
    end
    return holes, sk
end

function run_fold(table, ids, col_of, fold_id, test_holes, all_holes, eligible,
                  y_all, cens_all, pred_io, param_io, loss_io,
                  model_xyz, model_cov, Xxyz, Xcov; methods)
    test_set = Set(test_holes)
    train_holes = [h for h in all_holes if h ∉ test_set]
    test_rows = [i for i in eligible if table.hole[i] in test_set]
    train_rows = [i for i in eligible if table.hole[i] in Set(train_holes)]
    isempty(test_rows) && fail("fold $fold_id: no test rows")
    isempty(train_rows) && fail("fold $fold_id: no train rows")
    y_tr = y_all[train_rows]
    y_te = y_all[test_rows]
    cens_te = cens_all[test_rows]
    m = mean(y_tr)
    s = std(y_tr)
    (isfinite(m) && isfinite(s) && s > 0) || fail("fold $fold_id: bad train moments")

    ctx = "$DEPOSIT $(TARGET) fold $fold_id"
    t0 = time()
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    n_merged > 0 && logmsg("$ctx: merged $n_merged coincident train locations")
    fit = fit_variogram(dx, dy, dz, dv, dh; context = ctx)
    ratio = fit.range_h / fit.range_v
    (isfinite(ratio) && ratio > 0) || fail("$ctx: bad anisotropy ratio")

    rows = Pred[]
    if "mean" in methods
        append!(rows, rows_for(DEPOSIT, string(TARGET), "mean", table, ids, test_rows,
                               y_te, cens_te, fill(m, length(test_rows)), fill(NaN, length(test_rows))))
    end
    if "kriging" in methods
        μ_ok, σ_ok = kriging_predict(dx, dy, dz, dv,
                                     table.x[test_rows], table.y[test_rows], table.z[test_rows], fit)
        append!(rows, rows_for(DEPOSIT, string(TARGET), "kriging", table, ids, test_rows,
                               y_te, cens_te, μ_ok, σ_ok))
    end
    !isempty(rows) && append_preds(pred_io, rows)

    println(param_io, join((
        DEPOSIT, string(TARGET), "fold$(fold_id)", fit.model,
        tsv_num(fit.nugget), tsv_num(fit.partial_sill), tsv_num(fit.total_sill),
        tsv_num(fit.range_h), tsv_num(fit.range_v), tsv_num(ratio),
        tsv_num(fit.fit_rmse), string(fit.n_pairs_h), string(fit.n_pairs_v),
        string(fit.n_bins_h), string(fit.n_bins_v),
        fit.degenerate ? "1" : "0", string(n_merged),
    ), '\t'))
    flush(param_io)
    t_base = time() - t0

    t_nn = 0.0
    if !isempty(intersect(methods, METHODS_NN))
        t_nn = time()
        fold_ref = prepare_fold_k(table, col_of, y_all, train_rows, test_rows,
                                  fold_id, DEPOSIT, TARGET)
        ctx_nn = ctx * " nn"
        if "nn_xyz" in methods
            Xtr = take_cols(Xxyz, fold_ref.fit_cols)
            Xva = take_cols(Xxyz, fold_ref.val_cols)
            Xte = take_cols(Xxyz, fold_ref.test_cols)
            members, _ = train_ensemble(model_xyz, Xtr, fold_ref.y_fit, Xva, fold_ref.y_val,
                                        ctx_nn * "_xyz", fold_id, "nn_xyz", loss_io)
            μ, σ = ensemble_predict(model_xyz, members, Xte, fold_ref.y_shift, fold_ref.y_scale,
                                    ctx_nn * "_xyz test")
            append_preds(pred_io, rows_for(DEPOSIT, string(TARGET), "nn_xyz", table, ids,
                                          test_rows, y_te, cens_te, μ, σ))
        end
        if "nn_cov" in methods
            Xtr = take_cols(Xcov, fold_ref.fit_cols)
            Xva = take_cols(Xcov, fold_ref.val_cols)
            Xte = take_cols(Xcov, fold_ref.test_cols)
            members, _ = train_ensemble(model_cov, Xtr, fold_ref.y_fit, Xva, fold_ref.y_val,
                                        ctx_nn * "_cov", fold_id, "nn_cov", loss_io)
            μ, σ = ensemble_predict(model_cov, members, Xte, fold_ref.y_shift, fold_ref.y_scale,
                                    ctx_nn * "_cov test")
            append_preds(pred_io, rows_for(DEPOSIT, string(TARGET), "nn_cov", table, ids,
                                          test_rows, y_te, cens_te, μ, σ))
        end
        t_nn = time() - t_nn
    end

    fr = KFoldResult(fold_id, test_holes, train_rows, test_rows, m)
    return fr, t_base + t_nn, (baseline = t_base, nn = t_nn)
end

function write_fold_split(path, fold_holes)
    open(path, "w") do io
        println(io, "fold\thole")
        for (k, holes) in enumerate(fold_holes)
            for h in sort(holes)
                println(io, "$k\t$h")
            end
        end
    end
    logmsg("wrote $path")
end

function write_runtime_estimate(path, t_one_fold, n_folds_done, n_folds_total)
    proj = t_one_fold * n_folds_total / n_folds_done
    open(path, "w") do io
        println(io, "folds_timed\tfold_seconds\tprojected_total_seconds\tprojected_hours")
        println(io, join(("1", tsv_num(t_one_fold), tsv_num(proj), tsv_num(proj / 3600)), '\t'))
    end
    logmsg(@sprintf("runtime estimate from 1 fold: %.1f s → projected %.1f s (%.2f h) for %d folds",
                    t_one_fold, proj, proj / 3600, n_folds_total))
end

function write_summaries(preds::Vector{Pred}, folds::Vector{KFoldResult})
    mu_by = row_train_means(folds)
    skill_krig = pooled_skill_from_rows(select_preds(preds, "kriging", "all"), mu_by)

    summary_path = joinpath(WORK, "summary.tsv")
    paired_path = joinpath(WORK, "paired.tsv")
    decision_path = joinpath(WORK, "decision.tsv")

    open(summary_path, "w") do io
        println(io, "target\tmethod\tsubset\tn\trmse\tskill\tskill_ci95_low\tskill_ci95_high\tcoverage90")
        for method in (METHODS_BASE..., METHODS_NN...)
            for subset in ("all", "uncensored")
                rows = select_preds(preds, method, subset)
                want_cov = method in UNCERTAIN
                met = pooled_metrics([r.obs for r in rows], [r.pred for r in rows],
                                     [r.sd for r in rows], want_cov && !isempty(rows))
                tag = BOOT_SEED_TAG * "|$(TARGET)|$(method)|$(subset)|skill"
                sk, lo, hi = bootstrap_skill_ci(rows, mu_by, tag)
                cov = want_cov ? tsv_num(met.coverage) : ""
                println(io, join((
                    string(TARGET), method, subset, string(met.n),
                    tsv_num(met.rmse), tsv_num(sk), tsv_num(lo), tsv_num(hi), cov,
                ), '\t'))
            end
        end
    end

    open(paired_path, "w") do io
        println(io, "target\tnn_method\tsubset\tn_holes\tmean_skill_diff\tci95_low\tci95_high\tn_win\tn_total")
        for nn in METHODS_NN
            for subset in ("all", "uncensored")
                rows_nn = select_preds(preds, nn, subset)
                rows_ok = select_preds(preds, "kriging", subset)
                h_nn, sk_nn = hole_skill(rows_nn, mu_by)
                h_ok, sk_ok = hole_skill(rows_ok, mu_by)
                h_nn == h_ok || fail("paired holes differ for $nn $subset")
                diffs = [sk_nn[h] - sk_ok[h] for h in h_nn]
                tag = BOOT_SEED_TAG * "|$(TARGET)|$(nn)|$(subset)|paired_skill"
                μ, lo, hi = bootstrap_mean_ci(diffs, tag)
                n_win = count(>(0), diffs)
                println(io, join((
                    string(TARGET), nn, subset, string(length(h_nn)),
                    tsv_num(μ), tsv_num(lo), tsv_num(hi),
                    string(n_win), string(length(h_nn)),
                ), '\t'))
            end
        end
    end

    open(decision_path, "w") do io
        println(io, "nn_method\tsubset\tskill\tskill_ci95_low\tskill_kriging\tpass_skill_min\tpass_ci_above_zero\tpass_vs_kriging\tuseful")
        for nn in METHODS_NN
            for subset in ("all", "uncensored")
                rows = select_preds(preds, nn, subset)
                rows_k = select_preds(preds, "kriging", subset)
                tag = BOOT_SEED_TAG * "|$(TARGET)|$(nn)|$(subset)|skill"
                sk, lo, _ = bootstrap_skill_ci(rows, mu_by, tag)
                sk_k = pooled_skill_from_rows(rows_k, mu_by)
                p1 = sk >= SKILL_MIN
                p2 = lo > 0
                p3 = sk >= sk_k - KRIGING_LAG
                useful = p1 && p2 && p3
                println(io, join((
                    nn, subset, tsv_num(sk), tsv_num(lo), tsv_num(sk_k),
                    p1 ? "1" : "0", p2 ? "1" : "0", p3 ? "1" : "0", useful ? "1" : "0",
                ), '\t'))
            end
        end
    end

    logmsg("wrote $summary_path")
    logmsg("wrote $paired_path")
    logmsg("wrote $decision_path")
    logmsg(@sprintf("kriging pooled skill (all) = %.4f", skill_krig))
    return nothing
end

function claim_work_dir()
    lock_path = joinpath(WORK, ".run.lock")
    if isfile(lock_path)
        fail("refusing to start: lock file present at $lock_path")
    end
    if isdir(WORK)
        names = readdir(WORK)
        if !isempty(names)
            fail("refusing to start: work directory exists and is non-empty ($WORK)")
        end
    end
    mkpath(WORK)
    open(lock_path, "w") do io
        println(io, getpid())
    end
    return lock_path
end

function release_work_lock(lock_path)
    isfile(lock_path) && rm(lock_path)
    return nothing
end

function main_keivitsa()
    lock_path = claim_work_dir()
    try
        main_keivitsa_locked()
    finally
        release_work_lock(lock_path)
        if isassigned(LOG) && isopen(LOG[])
            close(LOG[])
        end
    end
    return nothing
end

function main_keivitsa_locked()
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("Keivitsa $(TARGET) feasibility: $N_FOLDS-fold hole-grouped CV")
    logmsg("skill = 1 - RMSE/RMSE_mean (pooled test; mean from fold training set)")
    logmsg("NN E2: loss=:nll stop_on=:val_rmse seeds=$(NN_SEEDS)")
    logmsg("NN loss curves every $(LOSS_LOG_EVERY) steps → loss_curves.tsv")
    logmsg("julia " * string(VERSION))
    logmsg("kriging self-check: GeoStats returns variance as σ = $(kriging_sigma_is_variance())")
    logmsg("work $WORK")

    table, covs, cfg = load_site(SITE_PATH)
    ids = keivitsa_sample_ids(table)
    Xxyz, Xcov, col_of, names = finite_features(covs, table)
    model_xyz = build_mlp(size(Xxyz, 1))
    model_cov = build_mlp(size(Xcov, 1))
    logmsg(@sprintf("rows=%d xyz_nin=%d cov_nin=%d channels=%s",
                    nsamples(table), size(Xxyz, 1), size(Xcov, 1), join(names, ",")))

    spec = spec_of(table, TARGET)
    eligible = findall(
        training_mask(table) .& real_hole_mask(table) .& observed_mask(table, TARGET))
    all_holes = sort!(unique(table.hole[eligible]))
    if haskey(EXPECTED_HOLES, TARGET)
        want = EXPECTED_HOLES[TARGET]
        length(all_holes) == want ||
            logmsg("WARNING: expected $want $(TARGET) holes, got $(length(all_holes))")
    end
    logmsg("$(TARGET): $(length(eligible)) samples on $(length(all_holes)) holes")

    fold_holes = hole_kfold_assignments(all_holes, N_FOLDS, KFOLD_SEED)
    write_fold_split(joinpath(WORK, "fold_split.tsv"), fold_holes)
    for (k, fh) in enumerate(fold_holes)
        logmsg("fold $k: $(length(fh)) test holes")
    end

    y_all = fill(NaN, nsamples(table))
    y_all[eligible] = transformed(spec, table.values[TARGET][eligible])
    cens = table.censored[TARGET]

    pred_path = joinpath(WORK, "predictions.tsv")
    param_path = joinpath(WORK, "kriging_params.tsv")
    pred_io = open(pred_path, "w")
    param_io = open(param_path, "w")
    println(pred_io, "deposit\ttarget\tmethod\thole\tsample\tx\ty\tz\tobs\tcensored\tpred\tsd")
    println(param_io, "deposit\ttarget\ttest_hole\tmodel\tnugget\tpartial_sill\ttotal_sill\t" *
            "range_horizontal_m\trange_vertical_m\tanisotropy_ratio\tfit_rmse\t" *
            "n_pairs_horizontal\tn_pairs_downhole\tn_bins_horizontal\tn_bins_downhole\t" *
            "degenerate\tn_merged_locations")
    flush(pred_io)
    flush(param_io)

    loss_path = joinpath(WORK, "loss_curves.tsv")
    loss_io = open(loss_path, "w")
    println(loss_io, "fold\tseed\tmethod\tstep\ttrain_loss\tval_loss\tval_rmse")
    flush(loss_io)

    methods = ("mean", "kriging", "nn_xyz", "nn_cov")
    fold_results = KFoldResult[]
    for (k, test_holes) in enumerate(fold_holes)
        fr, t_fold, _ = run_fold(
            table, ids, col_of, k, test_holes, all_holes, eligible,
            y_all, cens, pred_io, param_io, loss_io,
            model_xyz, model_cov, Xxyz, Xcov; methods = methods)
        push!(fold_results, fr)
        logmsg(@sprintf("fold %d/%d done in %.1f s", k, N_FOLDS, t_fold))
        if k == 1
            write_runtime_estimate(joinpath(WORK, "runtime_estimate.tsv"), t_fold, 1, N_FOLDS)
        end
    end

    close(pred_io)
    close(param_io)
    close(loss_io)
    logmsg("wrote $loss_path")

    preds = load_predictions(pred_path)
    write_summaries(preds, fold_results)

    elapsed = time() - t0
    logmsg(@sprintf("finished in %.1f s (%.2f h)", elapsed, elapsed / 3600))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_keivitsa()
end
