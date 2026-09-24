# Keivitsa NN diagnosis on the synthetic target (replaces E(-1)/E0–E4 instructions).
#
# Run:  julia --project=. examples/diagnose_nn_keivitsa.jl
# Out:  tmp_keivitsa_diag/  (gitignored via tmp_keivitsa*/)
#
# Synthetic target only. Stops after the results table; no real-Cu run.

const ROOT = dirname(@__DIR__)
ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, "tmp_keivitsa_diag")
include(joinpath(@__DIR__, "feasibility_loho.jl"))

const SITE_PATH = joinpath(ROOT, "sites", "keivitsa.toml")
const DEPOSIT = "keivitsa"

const N_FOLDS = 3
const KFOLD_SEED = 2026
const DIAG_SEEDS = (1, 2, 3)

const SYNTH_LAMBDA = 0.5
const SYNTH_NOISE = 0.1
const SYNTH_SEED = 2026

const MEM_N_SAMPLES = 200
const MEM_STEPS = 5000
const MEM_SEED = 1
const FIXED_STEPS = 2000
const LOG_EVERY = 100
const BETA_NLL = 0.5
const SYNTH_NN_LAG = 0.15
const FOURIER_SCALE_MAX_WIDE = 128.0
const MEM_SKILL_PASS = 0.95

#---------- helpers (match feasibility_keivitsa_pilot.jl) ----------

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

function bounds_from_cfg(cfg)
    b = cfg["bounds"]
    return (Float64(b["x"][1]), Float64(b["x"][2]),
            Float64(b["y"][1]), Float64(b["y"][2]),
            Float64(b["z"][1]), Float64(b["z"][2]))
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
                   columns_of(col_of, test_rows), y_fit, y_val, length(val_holes)),
           fit_rows, val_rows
end

#---------- diagnosis types ----------

struct FoldPack
    fold_id::Int
    train_rows::Vector{Int}
    test_rows::Vector{Int}
    fit_rows::Vector{Int}
    val_rows::Vector{Int}
    train_mean::Float64
    fold_ref::FoldRef
    y_te::Vector{Float64}
    μ_krig::Vector{Float64}
end

struct VariantResult
    variant::String
    train_skill::Float64
    val_skill::Float64
    test_skill::Float64
    kriging_test_skill::Float64
    median_best_step::Float64
    frac_stop_le5::Float64
    mean_sigma::Float64
    fit_rmse::Float64
    runtime_s::Float64
    best_steps::Vector{Int}
end

function skill_pool(obs_blocks, pred_blocks, mu_blocks)
    obs = Float64[]
    pred = Float64[]
    mu = Float64[]
    for (o, p, m) in zip(obs_blocks, pred_blocks, mu_blocks)
        append!(obs, o)
        append!(pred, p)
        append!(mu, m)
    end
    return pooled_skill_multifold(obs, pred, mu)
end

function train_ensemble(model, Xtr, ytr, Xva, yva, ctx; train_kw...)
    members = Tuple{Any,Any}[]
    best_steps = Int[]
    histories = Vector{Any}[]
    log_every = Int(get(train_kw, :log_every, 0))
    for seed in DIAG_SEEDS
        hist = log_every > 0 ? Any[] : nothing
        kw = (; train_kw...)
        if hist !== nothing
            kw = merge(kw, (; history = hist))
        end
        ps, st, best_ep, last_ep, _ = train_member(
            model, seed, Xtr, ytr, Xva, yva, ctx; kw...)
        push!(members, (ps, st))
        push!(best_steps, best_ep)
        push!(histories, hist === nothing ? Any[] : hist)
        logmsg(@sprintf("%s seed %d best_step=%d last_step=%d", ctx, seed, best_ep, last_ep))
    end
    return members, best_steps, histories
end

function channel_minmax(X::AbstractMatrix)
    nin = size(X, 1)
    lo = Vector{Float64}(undef, nin)
    hi = Vector{Float64}(undef, nin)
    for i in 1:nin
        row = @view X[i, :]
        lo[i] = minimum(row)
        hi[i] = maximum(row)
    end
    return lo, hi
end

function write_history(path, hist)
    open(path, "w") do io
        println(io, "step\ttrain_rmse\tval_rmse\tmean_sigma")
        for h in hist
            println(io, join((string(h.step), tsv_num(h.train_rmse),
                              tsv_num(h.val_rmse), tsv_num(h.mean_sigma)), '\t'))
        end
    end
end

#---------- E(-1) memorisation ----------

function run_memorisation(Xxyz, fold0::FoldPack)
    logmsg("="^60)
    logmsg("E(-1) memorisation: $(MEM_N_SAMPLES) fit samples, $(MEM_STEPS) steps, early stop OFF")
    fr = fold0.fold_ref
    n_fit = length(fr.fit_cols)
    n_fit >= MEM_N_SAMPLES || fail("fold 1 has only $n_fit fit columns, need $(MEM_N_SAMPLES)")
    pick = sort(shuffle(Xoshiro(MEM_SEED), collect(1:n_fit))[1:MEM_N_SAMPLES])
    cols = fr.fit_cols[pick]
    X = take_cols(Xxyz, cols)
    y = fr.y_fit[pick]
    Xva = X[:, 1:1]
    yva = y[1:1]
    model = build_mlp(size(X, 1))
    lo, hi = channel_minmax(X)
    open(joinpath(WORK, "e_minus1_channel_minmax.tsv"), "w") do io
        println(io, "channel\tmin\tmax")
        for i in eachindex(lo)
            println(io, join((string(i), tsv_num(lo[i]), tsv_num(hi[i])), '\t'))
        end
    end

    function one_mem(tag, loss)
        hist = Any[]
        t0 = time()
        ps, st, best_ep, last_ep, best_val = train_member(
            model, MEM_SEED, X, y, Xva, yva, "E(-1) $tag";
            loss = loss, stop_on = :none, max_epochs = MEM_STEPS,
            log_every = LOG_EVERY, history = hist)
        elapsed = time() - t0
        μ, σ = predict_mu_sigma(model, ps, st, X)
        rmse = sqrt(mean(abs2, μ .- y))
        sk = pooled_skill(y, μ, 0.0)
        write_history(joinpath(WORK, "e_minus1_$(tag)_curve.tsv"), hist)
        logmsg(@sprintf("E(-1) %s: train_skill=%.4f rmse=%.4f mean_σ=%.4f best_step=%d (%.1f s)",
                        tag, sk, rmse, mean(σ), best_ep, elapsed))
        return (skill = sk, rmse = rmse, mean_sigma = mean(σ), best_step = best_ep,
                last_step = last_ep, best_val = best_val, elapsed = elapsed)
    end

    mse = one_mem("mse", :mse)
    nll = one_mem("nll", :nll)
    open(joinpath(WORK, "e_minus1_summary.tsv"), "w") do io
        println(io, "loss\ttrain_skill\trmse\tmean_sigma\tbest_step\truntime_s")
        for (tag, r) in (("mse", mse), ("nll", nll))
            println(io, join((tag, tsv_num(r.skill), tsv_num(r.rmse), tsv_num(r.mean_sigma),
                              string(r.best_step), tsv_num(r.elapsed)), '\t'))
        end
    end
    return mse, nll, lo, hi
end

#---------- folds + baselines ----------

function build_folds(table, col_of, eligible, y_all, all_holes, fold_holes)
    packs = FoldPack[]
    for (k, test_holes) in enumerate(fold_holes)
        test_set = Set(test_holes)
        train_holes = [h for h in all_holes if h ∉ test_set]
        test_rows = [i for i in eligible if table.hole[i] in test_set]
        train_rows = [i for i in eligible if table.hole[i] in Set(train_holes)]
        y_tr = y_all[train_rows]
        y_te = y_all[test_rows]
        m = mean(y_tr)
        ctx = "$DEPOSIT synthetic fold $k"
        dx, dy, dz, dv, dh, n_merged = dedupe_locations(
            table.x[train_rows], table.y[train_rows], table.z[train_rows],
            y_tr, table.hole[train_rows])
        n_merged > 0 && logmsg("$ctx: merged $n_merged coincident train locations")
        fit = fit_variogram(dx, dy, dz, dv, dh; context = ctx)
        μ_krig, _ = kriging_predict(dx, dy, dz, dv,
                                    table.x[test_rows], table.y[test_rows], table.z[test_rows], fit)
        # Use :cu in the val-split tag (same as the pilot) so the hole split RNG matches.
        fr, fit_rows, val_rows = prepare_fold_k(
            table, col_of, y_all, train_rows, test_rows, k, DEPOSIT, :cu)
        push!(packs, FoldPack(k, train_rows, test_rows, fit_rows, val_rows, m, fr, y_te, μ_krig))
        logmsg(@sprintf("fold %d: train_holes=%d test_holes=%d fit=%d val=%d test=%d",
                        k, length(train_holes), length(test_holes),
                        length(fit_rows), length(val_rows), length(test_rows)))
    end
    return packs
end

function run_variant(name::AbstractString, packs::Vector{FoldPack}, Xxyz, model;
                     train_kw...)
    logmsg("="^60)
    logmsg("Variant $name  kwargs=$(NamedTuple(train_kw))")
    t0 = time()
    obs_fit = Vector{Float64}[]; pred_fit = Vector{Float64}[]; mu_fit = Vector{Float64}[]
    obs_val = Vector{Float64}[]; pred_val = Vector{Float64}[]; mu_val = Vector{Float64}[]
    obs_te = Vector{Float64}[]; pred_te = Vector{Float64}[]; mu_te = Vector{Float64}[]
    obs_kr = Vector{Float64}[]; pred_kr = Vector{Float64}[]; mu_kr = Vector{Float64}[]
    all_steps = Int[]
    sigma_sum = 0.0
    sigma_n = 0
    rmse_num = 0.0
    rmse_den = 0

    log_every = Int(get(train_kw, :log_every, 0))
    curve_io = nothing
    if log_every > 0
        curve_io = open(joinpath(WORK, "variant_$(name)_curves.tsv"), "w")
        println(curve_io, "fold\tseed\tstep\ttrain_rmse\tval_rmse\tmean_sigma")
    end

    for pack in packs
        fr = pack.fold_ref
        ctx = "$name fold $(pack.fold_id)"
        Xtr = take_cols(Xxyz, fr.fit_cols)
        Xva = take_cols(Xxyz, fr.val_cols)
        Xte = take_cols(Xxyz, fr.test_cols)
        members, steps, histories = train_ensemble(
            model, Xtr, fr.y_fit, Xva, fr.y_val, ctx; train_kw...)
        append!(all_steps, steps)
        if curve_io !== nothing
            for (si, seed) in enumerate(DIAG_SEEDS)
                for h in histories[si]
                    println(curve_io, join((string(pack.fold_id), string(seed),
                                            string(h.step), tsv_num(h.train_rmse),
                                            tsv_num(h.val_rmse), tsv_num(h.mean_sigma)), '\t'))
                end
            end
        end

        μ_f, σ_f = ensemble_predict(model, members, Xtr, fr.y_shift, fr.y_scale, ctx * " fit")
        μ_v, _ = ensemble_predict(model, members, Xva, fr.y_shift, fr.y_scale, ctx * " val")
        μ_t, _ = ensemble_predict(model, members, Xte, fr.y_shift, fr.y_scale, ctx * " test")
        y_fit_obs = fr.y_fit .* fr.y_scale .+ fr.y_shift
        y_val_obs = fr.y_val .* fr.y_scale .+ fr.y_shift

        push!(obs_fit, y_fit_obs); push!(pred_fit, μ_f)
        push!(mu_fit, fill(pack.train_mean, length(y_fit_obs)))
        push!(obs_val, y_val_obs); push!(pred_val, μ_v)
        push!(mu_val, fill(pack.train_mean, length(y_val_obs)))
        push!(obs_te, pack.y_te); push!(pred_te, μ_t)
        push!(mu_te, fill(pack.train_mean, length(pack.y_te)))
        push!(obs_kr, pack.y_te); push!(pred_kr, pack.μ_krig)
        push!(mu_kr, fill(pack.train_mean, length(pack.y_te)))

        sigma_sum += sum(σ_f)
        sigma_n += length(σ_f)
        rmse_num += sum(abs2, μ_f .- y_fit_obs)
        rmse_den += length(μ_f)

        logmsg(@sprintf("%s: fit_skill=%.4f val_skill=%.4f test_skill=%.4f steps=%s",
                        ctx,
                        pooled_skill(y_fit_obs, μ_f, pack.train_mean),
                        pooled_skill(y_val_obs, μ_v, pack.train_mean),
                        pooled_skill(pack.y_te, μ_t, pack.train_mean),
                        join(string.(steps), ",")))
    end
    curve_io !== nothing && close(curve_io)

    elapsed = time() - t0
    tr_sk = skill_pool(obs_fit, pred_fit, mu_fit)
    va_sk = skill_pool(obs_val, pred_val, mu_val)
    te_sk = skill_pool(obs_te, pred_te, mu_te)
    kr_sk = skill_pool(obs_kr, pred_kr, mu_kr)
    med = median(Float64.(all_steps))
    frac = count(<=(5), all_steps) / length(all_steps)
    mean_σ = sigma_n == 0 ? NaN : sigma_sum / sigma_n
    fit_rmse = rmse_den == 0 ? NaN : sqrt(rmse_num / rmse_den)
    logmsg(@sprintf("%s DONE: train=%.4f val=%.4f test=%.4f krig=%.4f med_step=%.0f frac≤5=%.2f σ̄=%.4f fit_rmse=%.4f (%.1f s)",
                    name, tr_sk, va_sk, te_sk, kr_sk, med, frac, mean_σ, fit_rmse, elapsed))
    return VariantResult(String(name), tr_sk, va_sk, te_sk, kr_sk, med, frac,
                         mean_σ, fit_rmse, elapsed, all_steps)
end

function write_results_table(path, results::Vector{VariantResult})
    open(path, "w") do io
        println(io, "variant\ttrain_skill\tval_skill\tsynthetic_test_skill\t" *
                "kriging_test_skill\tmedian_best_step\tfrac_stop_le5\t" *
                "mean_sigma\tfit_rmse\truntime_s")
        for r in results
            println(io, join((
                r.variant,
                tsv_num(r.train_skill), tsv_num(r.val_skill),
                tsv_num(r.test_skill), tsv_num(r.kriging_test_skill),
                tsv_num(r.median_best_step), tsv_num(r.frac_stop_le5),
                tsv_num(r.mean_sigma), tsv_num(r.fit_rmse),
                tsv_num(r.runtime_s),
            ), '\t'))
        end
    end
    logmsg("wrote $path")
end

function propose_frozen(results::Vector{VariantResult})
    # Prefer simplest variant with nn_xyz ≥ kriging − 0.15 on synthetic test.
    order = ("E0", "E1", "E2", "E3", "E4")
    by_name = Dict(r.variant => r for r in results)
    candidates = VariantResult[]
    for name in order
        haskey(by_name, name) || continue
        r = by_name[name]
        if r.test_skill >= r.kriging_test_skill - SYNTH_NN_LAG
            push!(candidates, r)
        end
    end
    if isempty(candidates)
        logmsg("No variant reaches kriging − $(SYNTH_NN_LAG) on synthetic test.")
        best = results[argmax([r.test_skill for r in results])]
        logmsg(@sprintf("Closest: %s test=%.4f krig=%.4f (lag=%.4f)",
                        best.variant, best.test_skill, best.kriging_test_skill,
                        best.kriging_test_skill - best.test_skill))
        return nothing
    end
    frozen = candidates[1]
    logmsg("="^60)
    logmsg(@sprintf("PROPOSED FROZEN CONFIG: %s  (simplest with test ≥ krig − %.2f)",
                    frozen.variant, SYNTH_NN_LAG))
    logmsg(@sprintf("  train=%.4f val=%.4f test=%.4f krig=%.4f med_step=%.0f",
                    frozen.train_skill, frozen.val_skill, frozen.test_skill,
                    frozen.kriging_test_skill, frozen.median_best_step))
    if frozen.variant == "E0"
        logmsg("  settings: loss=:nll, stop_on=:val_nll (current defaults), seeds=1:5 later")
    elseif frozen.variant == "E1"
        logmsg("  settings: loss=:nll, stop_on=:none, max_epochs=$FIXED_STEPS")
    elseif frozen.variant == "E2"
        logmsg("  settings: loss=:nll, stop_on=:val_rmse")
    elseif frozen.variant == "E3"
        logmsg("  settings: loss=:nll, stop_on=:val_rmse, beta=$BETA_NLL")
    elseif frozen.variant == "E4"
        logmsg("  settings: loss=:nll, stop_on=:val_rmse, beta=$BETA_NLL, " *
               "fourier scales 1–$FOURIER_SCALE_MAX_WIDE")
    end
    logmsg("STOP: confirm before any real-Cu run. Note: sites/keivitsa.toml d0=25 → 2.5 before nn_cov.")
    return frozen
end

#---------- main ----------

function main()
    mkpath(WORK)
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("Keivitsa NN diagnosis (synthetic only)")
    logmsg("seeds=$(DIAG_SEEDS) folds=$N_FOLDS kfold_seed=$KFOLD_SEED")
    logmsg("julia " * string(VERSION))
    logmsg("work $WORK")

    table, covs, cfg = load_site(SITE_PATH)
    Xxyz, _, col_of, names = finite_features(covs, table)
    model_xyz = build_mlp(size(Xxyz, 1))
    logmsg(@sprintf("rows=%d xyz_nin=%d channels=%s",
                    nsamples(table), size(Xxyz, 1), join(names, ",")))

    eligible = findall(
        training_mask(table) .& real_hole_mask(table) .& observed_mask(table, :cu))
    all_holes = sort!(unique(table.hole[eligible]))
    logmsg("cu: $(length(eligible)) samples on $(length(all_holes)) holes")
    length(all_holes) >= N_FOLDS || fail("need at least $N_FOLDS holes")

    xlo, xhi, ylo, yhi, zlo, zhi = bounds_from_cfg(cfg)
    y_synth = fill(NaN, nsamples(table))
    y_synth[eligible] = synthetic_target(
        table, eligible, xlo, xhi, ylo, yhi, zlo, zhi, Xoshiro(SYNTH_SEED))

    fold_holes = hole_kfold_assignments(all_holes, N_FOLDS, KFOLD_SEED)
    packs = build_folds(table, col_of, eligible, y_synth, all_holes, fold_holes)

    # ---- E(-1) ----
    mse, nll, lo, hi = run_memorisation(Xxyz, packs[1])
    if !(mse.skill > MEM_SKILL_PASS)
        logmsg("E(-1) MSE FAILED: train_skill=$(mse.skill) (need > $MEM_SKILL_PASS)")
        logmsg("Pipeline bug suspected. Channel min/max written; loss curve written.")
        logmsg("Input channel min range: $(minimum(lo)) .. $(maximum(lo)); max range: $(minimum(hi)) .. $(maximum(hi))")
        close(LOG[])
        fail("E(-1) MSE memorisation failed; not running E0–E4")
    end
    logmsg(@sprintf("E(-1) MSE passed (skill=%.4f). NLL skill=%.4f%s",
                    mse.skill, nll.skill,
                    nll.skill > MEM_SKILL_PASS ? "" : " — H1 supported (NLL underfits μ)"))

    results = VariantResult[]

    # ---- E0: current configuration ----
    push!(results, run_variant("E0", packs, Xxyz, model_xyz;
                               loss = :nll, stop_on = :val_nll))
    open(joinpath(WORK, "e0_member_steps.tsv"), "w") do io
        println(io, "member_index\tbest_step")
        for (i, s) in enumerate(results[end].best_steps)
            println(io, "$i\t$s")
        end
    end
    r0 = results[end]
    logmsg(@sprintf("E0: mean_σ=%.4f fit_rmse=%.4f  (σ vs RMSE for H1)",
                    r0.mean_sigma, r0.fit_rmse))

    # ---- E1: early stopping OFF, 2000 fixed steps ----
    push!(results, run_variant("E1", packs, Xxyz, model_xyz;
                               loss = :nll, stop_on = :none,
                               max_epochs = FIXED_STEPS, log_every = LOG_EVERY))

    # ---- E2: early stop on validation RMSE of μ ----
    push!(results, run_variant("E2", packs, Xxyz, model_xyz;
                               loss = :nll, stop_on = :val_rmse))

    # ---- E3: E2 + β-NLL ----
    push!(results, run_variant("E3", packs, Xxyz, model_xyz;
                               loss = :nll, stop_on = :val_rmse, beta = BETA_NLL))

    # ---- E4: only if E2 and E3 both lag kriging by > 0.15 ----
    by = Dict(r.variant => r for r in results)
    lag2 = by["E2"].kriging_test_skill - by["E2"].test_skill
    lag3 = by["E3"].kriging_test_skill - by["E3"].test_skill
    if lag2 > SYNTH_NN_LAG && lag3 > SYNTH_NN_LAG
        logmsg(@sprintf("E4 triggered: E2 lag=%.4f E3 lag=%.4f (both > %.2f)",
                        lag2, lag3, SYNTH_NN_LAG))
        Xxyz_w, _, col_of_w, _ = finite_features(
            covs, table; fourier_scale_max = FOURIER_SCALE_MAX_WIDE)
        # Rebuild fold column indices against the same sample order (identical col_of).
        col_of_w == col_of || fail("E4: column map changed with wider Fourier features")
        model_w = build_mlp(size(Xxyz_w, 1))
        push!(results, run_variant("E4", packs, Xxyz_w, model_w;
                                   loss = :nll, stop_on = :val_rmse, beta = BETA_NLL))
    else
        logmsg(@sprintf("E4 skipped: E2 lag=%.4f E3 lag=%.4f", lag2, lag3))
    end

    write_results_table(joinpath(WORK, "diagnosis_summary.tsv"), results)
    propose_frozen(results)

    # Human-readable table to the log / stdout
    logmsg("="^60)
    logmsg(@sprintf("%-4s %8s %8s %8s %8s %8s %8s %8s",
                    "var", "train", "val", "test", "krig", "med_ep", "≤5", "runtime"))
    for r in results
        logmsg(@sprintf("%-4s %8.4f %8.4f %8.4f %8.4f %8.0f %8.2f %8.1f",
                        r.variant, r.train_skill, r.val_skill, r.test_skill,
                        r.kriging_test_skill, r.median_best_step, r.frac_stop_le5,
                        r.runtime_s))
    end
    logmsg(@sprintf("diagnosis finished in %.1f s", time() - t0))
    close(LOG[])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
