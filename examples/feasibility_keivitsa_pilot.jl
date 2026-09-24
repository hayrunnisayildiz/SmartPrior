# Keivitsa Cu pilot: 30 holes, 3-fold hole-grouped CV + pipeline checks A/B.
# Full experiment will use all kept Cu holes and 10 folds (same code path).
#
# Run:  julia --project=. examples/feasibility_keivitsa_pilot.jl
# Out:  tmp_keivitsa_pilot/  (gitignored)

const ROOT = dirname(@__DIR__)
ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, "tmp_keivitsa_pilot")
include(joinpath(@__DIR__, "feasibility_loho.jl"))

const SITE_PATH = joinpath(ROOT, "sites", "keivitsa.toml")
const DEPOSIT = "keivitsa"
const TARGET = :cu

const PILOT_N_HOLES = 30
const PILOT_N_FOLDS = 3
const FULL_N_FOLDS = 10
const HOLE_SELECT_SEED = 2026
const KFOLD_SEED = 2026

const SYNTH_LAMBDA = 0.5
const SYNTH_NOISE = 0.1
const SYNTH_SEED = 2026

const TRAIN_SKILL_MIN = 0.10
const SYNTH_SKILL_MIN = 0.50
const SYNTH_NN_LAG = 0.15

# skill = 1 − RMSE / RMSE_mean  (RMSE_mean vs the fold training mean)
function pooled_skill(obs::AbstractVector{<:Real}, pred::AbstractVector{<:Real},
                      mu_train::Real)
    n = length(obs)
    n == length(pred) || fail("skill: length mismatch")
    n == 0 && return NaN
    rmse = sqrt(mean(abs2, pred .- obs))
    rmse_mean = sqrt(mean(abs2, obs .- mu_train))
    return rmse_mean == 0 ? NaN : 1 - rmse / rmse_mean
end

function pooled_skill_multifold(obs, pred, mu_per_sample)
    n = length(obs)
    n == length(pred) == length(mu_per_sample) || fail("skill: length mismatch")
    n == 0 && return NaN
    rmse = sqrt(mean(abs2, pred .- obs))
    rmse_mean = sqrt(mean(abs2, obs .- mu_per_sample))
    return rmse_mean == 0 ? NaN : 1 - rmse / rmse_mean
end

function synthetic_target(table, rows, xlo, xhi, ylo, yhi, zlo, zhi, rng::AbstractRNG)
    y = Vector{Float64}(undef, length(rows))
    λ = SYNTH_LAMBDA
    @inbounds for (k, i) in enumerate(rows)
        xp = 2 * (table.x[i] - xlo) / (xhi - xlo) - 1
        yp = 2 * (table.y[i] - ylo) / (yhi - ylo) - 1
        zp = 2 * (table.z[i] - zlo) / (zhi - zlo) - 1
        ε = SYNTH_NOISE * randn(rng)
        y[k] = sin(2π * xp / λ) * cos(2π * yp / λ) + 0.5 * zp + ε
    end
    return y
end

function keivitsa_sample_ids(table)
    return [string(i) for i in 1:nsamples(table)]
end

function bounds_from_cfg(cfg)
    b = cfg["bounds"]
    return (Float64(b["x"][1]), Float64(b["x"][2]),
            Float64(b["y"][1]), Float64(b["y"][2]),
            Float64(b["z"][1]), Float64(b["z"][2]))
end

function select_pilot_holes(all_holes::Vector{String}, n::Int, seed::Int)
    length(all_holes) >= n || fail(
        "need at least $n Cu holes for pilot, got $(length(all_holes))")
    perm = shuffle(Xoshiro(seed), collect(all_holes))
    return sort!(perm[1:n])
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

function nn_xyz_fold(model, Xxyz, col_of, fold::FoldRef, train_rows, y_tr, ctx)
    Xtr = take_cols(Xxyz, fold.fit_cols)
    Xva = take_cols(Xxyz, fold.val_cols)
    Xte = take_cols(Xxyz, fold.test_cols)
    Xtrain = take_cols(Xxyz, columns_of(col_of, train_rows))
    members = Tuple{Any,Any}[]
    epochs = Int[]
    for seed in NN_SEEDS
        ps, st, best_ep, last_ep, _ = train_member(
            model, seed, Xtr, fold.y_fit, Xva, fold.y_val, ctx * " nn_xyz")
        push!(members, (ps, st))
        push!(epochs, best_ep)
        last_ep == NN_EPOCHS &&
            logmsg("$ctx nn_xyz seed $seed ran all $(NN_EPOCHS) epochs (best $best_ep)")
    end
    m = mean(y_tr)
    μ_te, σ_te = ensemble_predict(model, members, Xte, fold.y_shift, fold.y_scale, ctx * " test")
    μ_tr, _ = ensemble_predict(model, members, Xtrain, fold.y_shift, fold.y_scale, ctx * " train")
    tr_skill = pooled_skill(y_tr, μ_tr, m)
    return μ_te, σ_te, μ_tr, tr_skill, epochs
end

struct KFoldResult
    fold_id::Int
    train_rows::Vector{Int}
    test_rows::Vector{Int}
    train_mean::Float64
    fold_ref::FoldRef
    y_te::Vector{Float64}
    cens_te::BitVector
end

function run_fold(table, ids, col_of, fold_id, test_holes, pilot_hole_set,
                  eligible, y_all, cens_all, pred_io, param_io, target_label,
                  methods, model_xyz, Xxyz; track_train_nn = false)
    test_set = Set(test_holes)
    train_holes = [h for h in pilot_hole_set if h ∉ test_set]
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

    ctx = "$DEPOSIT $target_label fold $fold_id"
    t_base = time()
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    n_merged > 0 && logmsg("$ctx: merged $n_merged coincident train locations")
    fit = fit_variogram(dx, dy, dz, dv, dh; context = ctx)
    ratio = fit.range_h / fit.range_v
    (isfinite(ratio) && ratio > 0) || fail("$ctx: bad anisotropy ratio")

    rows = Pred[]
    if "mean" in methods
        append!(rows, rows_for(DEPOSIT, target_label, "mean", table, ids, test_rows,
                               y_te, cens_te, fill(m, length(test_rows)), fill(NaN, length(test_rows))))
    end
    if "kriging" in methods
        μ_ok, σ_ok = kriging_predict(dx, dy, dz, dv,
                                     table.x[test_rows], table.y[test_rows], table.z[test_rows], fit)
        append!(rows, rows_for(DEPOSIT, target_label, "kriging", table, ids, test_rows,
                               y_te, cens_te, μ_ok, σ_ok))
    end
    !isempty(rows) && append_preds(pred_io, rows)
    t_base = time() - t_base

    println(param_io, join((
        DEPOSIT, target_label, "fold$(fold_id)", fit.model,
        tsv_num(fit.nugget), tsv_num(fit.partial_sill), tsv_num(fit.total_sill),
        tsv_num(fit.range_h), tsv_num(fit.range_v), tsv_num(ratio),
        tsv_num(fit.fit_rmse), string(fit.n_pairs_h), string(fit.n_pairs_v),
        string(fit.n_bins_h), string(fit.n_bins_v),
        fit.degenerate ? "1" : "0", string(n_merged),
    ), '\t'))
    flush(param_io)

    t_nn = 0.0
    train_skill = NaN
    μ_tr = Float64[]
    if "nn_xyz" in methods
        t_nn = time()
        fold_ref = prepare_fold_k(table, col_of, y_all, train_rows, test_rows,
                                  fold_id, DEPOSIT, TARGET)
        μ_nn, σ_nn, μ_tr, train_skill, _ = nn_xyz_fold(
            model_xyz, Xxyz, col_of, fold_ref, train_rows, y_tr, ctx)
        append_preds(pred_io, rows_for(DEPOSIT, target_label, "nn_xyz", table, ids,
                                      test_rows, y_te, cens_te, μ_nn, σ_nn))
        t_nn = time() - t_nn
        fold_out = fold_ref
    else
        fold_out = prepare_fold_k(table, col_of, y_all, train_rows, test_rows,
                                  fold_id, DEPOSIT, TARGET)
    end

    return KFoldResult(fold_id, train_rows, test_rows, m, fold_out, y_te, cens_te),
           (baseline = t_base, nn_xyz = t_nn),
           (train_skill = train_skill, train_obs = track_train_nn ? y_tr : Float64[],
            train_pred = μ_tr)
end

function pooled_test_skill_preds(preds, method, target_label, fold_results)
    row_mu = Dict{Int,Float64}()
    for fr in fold_results
        for i in fr.test_rows
            row_mu[i] = fr.train_mean
        end
    end
    obs = Float64[]
    pred = Float64[]
    mu = Float64[]
    for r in preds
        r.target == target_label && r.method == method || continue
        row_idx = parse(Int, r.sample)
        haskey(row_mu, row_idx) || continue
        push!(obs, r.obs)
        push!(pred, r.pred)
        push!(mu, row_mu[row_idx])
    end
    return pooled_skill_multifold(obs, pred, mu)
end

function write_pipeline_checks(path, rows)
    open(path, "w") do io
        println(io, "check\tmetric\tvalue\tthreshold\tpass\tnote")
        for r in rows
            println(io, join(r, '\t'))
        end
    end
    logmsg("wrote $path")
end

function projected_full_runtime(t_pilot, n_pilot_samples, n_full_samples,
                                n_pilot_folds, n_full_folds)
    # Scale by fold count and mean train-set size (dominant cost for NN / kriging).
    train_frac_pilot = (PILOT_N_FOLDS - 1) / PILOT_N_FOLDS
    train_frac_full = (n_full_folds - 1) / n_full_folds
    train_n_pilot = train_frac_pilot * n_pilot_samples
    train_n_full = train_frac_full * n_full_samples
    train_n_pilot > 0 || return NaN
    scale = (n_full_folds / n_pilot_folds) * (train_n_full / train_n_pilot)
    return t_pilot * scale
end

function main_pilot()
    mkpath(WORK)
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("Keivitsa pilot: $(PILOT_N_HOLES) holes, $(PILOT_N_FOLDS)-fold grouped CV")
    logmsg("skill = 1 - RMSE/RMSE_mean (pooled test samples; mean from train fold)")
    logmsg("julia " * string(VERSION))
    logmsg("work $WORK")

    table, covs, cfg = load_site(SITE_PATH)
    ids = keivitsa_sample_ids(table)
    Xxyz, _, col_of, names = finite_features(covs, table)
    model_xyz = build_mlp(size(Xxyz, 1))
    logmsg(@sprintf("rows=%d xyz_nin=%d channels=%s", nsamples(table), size(Xxyz, 1), join(names, ",")))

    spec = spec_of(table, TARGET)
    eligible = findall(
        training_mask(table) .& real_hole_mask(table) .& observed_mask(table, TARGET))
    all_holes = sort!(unique(table.hole[eligible]))
    logmsg("cu: $(length(eligible)) samples on $(length(all_holes)) holes")
    pilot_holes = select_pilot_holes(all_holes, PILOT_N_HOLES, HOLE_SELECT_SEED)
    pilot_set = Set(pilot_holes)
    pilot_eligible = [i for i in eligible if table.hole[i] in pilot_set]
    n_pilot_samples = length(pilot_eligible)
    n_full_samples = length(eligible)
    logmsg("pilot subset: $(length(pilot_holes)) holes, $n_pilot_samples samples")
    open(joinpath(WORK, "pilot_holes.tsv"), "w") do io
        println(io, "hole")
        for h in pilot_holes
            println(io, h)
        end
    end

    fold_holes = hole_kfold_assignments(pilot_holes, PILOT_N_FOLDS, KFOLD_SEED)
    for (k, fh) in enumerate(fold_holes)
        logmsg("fold $k: $(length(fh)) test holes, $(length(pilot_holes)-length(fh)) train holes")
    end

    y_real = fill(NaN, nsamples(table))
    y_real[eligible] = transformed(spec, table.values[TARGET][eligible])
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

    real_folds = KFoldResult[]
    train_skills = Dict{Int,Float64}()
    times_real = Dict{Symbol,Float64}(:baseline => 0.0, :nn_xyz => 0.0)

    for (k, test_holes) in enumerate(fold_holes)
        fr, t, extra = run_fold(
            table, ids, col_of, k, test_holes, pilot_holes, pilot_eligible,
            y_real, cens, pred_io, param_io, string(TARGET),
            ("mean", "kriging", "nn_xyz"), model_xyz, Xxyz; track_train_nn = true)
        push!(real_folds, fr)
        train_skills[k] = extra.train_skill
        times_real[:baseline] += t.baseline
        times_real[:nn_xyz] += t.nn_xyz
        logmsg(@sprintf("real Cu fold %d train_skill=%.4f (nn_xyz)", k, extra.train_skill))
    end

    preds_real = load_predictions(pred_path)
    skill_real_mean = pooled_test_skill_preds(preds_real, "mean", "cu", real_folds)
    skill_real_krig = pooled_test_skill_preds(preds_real, "kriging", "cu", real_folds)
    skill_real_nn = pooled_test_skill_preds(preds_real, "nn_xyz", "cu", real_folds)

    check_rows = Vector{Vector{String}}()
    pass_a = true
    for k in 1:PILOT_N_FOLDS
        sk = train_skills[k]
        ok = sk >= TRAIN_SKILL_MIN
        pass_a &= ok
        push!(check_rows, ["A", "train_skill_fold$k", tsv_num(sk), tsv_num(TRAIN_SKILL_MIN),
                           ok ? "1" : "0", "real Cu nn_xyz train pooled"])
    end
    push!(check_rows, ["pilot", "test_skill_mean", tsv_num(skill_real_mean), "", "", "real Cu"])
    push!(check_rows, ["pilot", "test_skill_kriging", tsv_num(skill_real_krig), "", "", "real Cu"])
    push!(check_rows, ["pilot", "test_skill_nn_xyz", tsv_num(skill_real_nn), "", "", "real Cu"])
    push!(check_rows, ["pilot", "time_baseline_s", tsv_num(times_real[:baseline]), "", "", "mean+kriging"])
    push!(check_rows, ["pilot", "time_nn_xyz_s", tsv_num(times_real[:nn_xyz]), "", "", "real Cu"])

    if !pass_a
        elapsed = time() - t0
        push!(check_rows, ["runtime", "pilot_seconds", tsv_num(elapsed), "", "", ""])
        write_pipeline_checks(joinpath(WORK, "pipeline_checks.tsv"), check_rows)
        close(pred_io); close(param_io); close(LOG[])
        fail("check A failed: train skill below $(TRAIN_SKILL_MIN) on at least one fold")
    end

    xlo, xhi, ylo, yhi, zlo, zhi = bounds_from_cfg(cfg)
    y_synth = fill(NaN, nsamples(table))
    y_synth[pilot_eligible] = synthetic_target(
        table, pilot_eligible, xlo, xhi, ylo, yhi, zlo, zhi, Xoshiro(SYNTH_SEED))
    cens_zero = falses(nsamples(table))

    pred_synth_path = joinpath(WORK, "predictions_synthetic.tsv")
    pred_synth_io = open(pred_synth_path, "w")
    println(pred_synth_io, "deposit\ttarget\tmethod\thole\tsample\tx\ty\tz\tobs\tcensored\tpred\tsd")
    flush(pred_synth_io)
    synth_folds = KFoldResult[]
    times_synth = Dict{Symbol,Float64}(:baseline => 0.0, :nn_xyz => 0.0)

    for (k, test_holes) in enumerate(fold_holes)
        fr, t, _ = run_fold(
            table, ids, col_of, k, test_holes, pilot_holes, pilot_eligible,
            y_synth, cens_zero, pred_synth_io, param_io, "synthetic_cu",
            ("mean", "kriging", "nn_xyz"), model_xyz, Xxyz)
        push!(synth_folds, fr)
        times_synth[:baseline] += t.baseline
        times_synth[:nn_xyz] += t.nn_xyz
    end
    close(pred_synth_io)

    preds_synth = load_predictions(pred_synth_path)
    skill_b_mean = pooled_test_skill_preds(preds_synth, "mean", "synthetic_cu", synth_folds)
    skill_b_krig = pooled_test_skill_preds(preds_synth, "kriging", "synthetic_cu", synth_folds)
    skill_b_nn = pooled_test_skill_preds(preds_synth, "nn_xyz", "synthetic_cu", synth_folds)

    pass_b_krig = skill_b_krig >= SYNTH_SKILL_MIN
    pass_b_nn = skill_b_nn >= SYNTH_SKILL_MIN
    pass_b_gap = skill_b_nn >= skill_b_krig - SYNTH_NN_LAG
    pass_b = pass_b_krig && pass_b_nn && pass_b_gap

    push!(check_rows, ["B", "test_skill_kriging", tsv_num(skill_b_krig),
                       tsv_num(SYNTH_SKILL_MIN), pass_b_krig ? "1" : "0", "synthetic pooled"])
    push!(check_rows, ["B", "test_skill_nn_xyz", tsv_num(skill_b_nn),
                       tsv_num(SYNTH_SKILL_MIN), pass_b_nn ? "1" : "0", "synthetic pooled"])
    push!(check_rows, ["B", "nn_lag_vs_kriging", tsv_num(skill_b_krig - skill_b_nn),
                       tsv_num(SYNTH_NN_LAG), pass_b_gap ? "1" : "0", "kriging - nn_xyz"])
    push!(check_rows, ["B", "test_skill_mean", tsv_num(skill_b_mean), "", "", "synthetic"])

    elapsed = time() - t0
    t_real_total = times_real[:baseline] + times_real[:nn_xyz]
    t_synth_total = times_synth[:baseline] + times_synth[:nn_xyz]
    proj = projected_full_runtime(elapsed, n_pilot_samples, n_full_samples,
                                  PILOT_N_FOLDS, FULL_N_FOLDS)
    push!(check_rows, ["runtime", "pilot_seconds", tsv_num(elapsed), "", "", ""])
    push!(check_rows, ["runtime", "projected_full_seconds", tsv_num(proj), "", "",
                       "10-fold × $(length(all_holes)) holes, train-size scaled"])
    push!(check_rows, ["runtime", "synth_baseline_s", tsv_num(times_synth[:baseline]), "", "", ""])
    push!(check_rows, ["runtime", "synth_nn_xyz_s", tsv_num(times_synth[:nn_xyz]), "", "", ""])

    write_pipeline_checks(joinpath(WORK, "pipeline_checks.tsv"), check_rows)
    write_summary(preds_real, preds_synth, real_folds, synth_folds)

    close(pred_io)
    close(param_io)
    logmsg(@sprintf("pilot done in %.1f s; projected full ~%.1f s (%.2f h)",
                    elapsed, proj, proj / 3600))
    close(LOG[])

    if !pass_b
        fail("check B failed: synthetic krig=$(skill_b_krig) nn=$(skill_b_nn); " *
             "do not run the full 10-fold experiment")
    end
    return nothing
end

function write_summary(preds_real, preds_synth, real_folds, synth_folds)
    open(joinpath(WORK, "summary.tsv"), "w") do io
        println(io, "subset\tmethod\tn\trmse\tskill_pooled")
        for (label, preds, folds, tgt) in (
            ("real_cu", preds_real, real_folds, "cu"),
            ("synthetic_cu", preds_synth, synth_folds, "synthetic_cu"),
        )
            for method in ("mean", "kriging", "nn_xyz")
                rows = [r for r in preds if r.target == tgt && r.method == method]
                isempty(rows) && continue
                obs = [r.obs for r in rows]
                pred = [r.pred for r in rows]
                sk = pooled_test_skill_preds(preds, method, tgt, folds)
                rmse = sqrt(mean(abs2, pred .- obs))
                println(io, join((label, method, string(length(rows)),
                                  tsv_num(rmse), tsv_num(sk)), '\t'))
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_pilot()
end
