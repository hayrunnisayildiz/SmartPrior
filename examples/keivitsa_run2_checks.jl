# Post-run checks for the clean Keivitsa Cu feasibility rerun.
# Does not refit the neural net and does not change variogram bounds.
#
#   KEIVITSA_ROOT=... julia --project=. examples/keivitsa_run2_checks.jl

const ROOT = dirname(@__DIR__)
ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, "tmp_feasibility_keivitsa_run2")
include(joinpath(@__DIR__, "feasibility_keivitsa.jl"))

using GLMakie

const RUN = WORK
const MIXED = joinpath(ROOT, "tmp_feasibility_keivitsa_mixed")
const PLOT_FOLDS = (6, 7, 9)

function file_equal(a, b)
    return read(a) == read(b)
end

function compare_tables()
    println("=== file compare (full bytes) ===")
    for name in ("summary.tsv", "paired.tsv", "decision.tsv")
        a = joinpath(RUN, name)
        b = joinpath(MIXED, name)
        same = file_equal(a, b)
        println(name, same ? " IDENTICAL" : " DIFFER")
        if !same
            println("--- run2")
            println(read(a, String))
            println("--- mixed (previously reported)")
            println(read(b, String))
        end
    end
end

function bootstrap_pooled_skill_diff(rows_a, rows_b, mu_by, tag)
    by_a = Dict{String,Vector{Pred}}()
    by_b = Dict{String,Vector{Pred}}()
    for r in rows_a
        push!(get!(by_a, r.hole, Pred[]), r)
    end
    for r in rows_b
        push!(get!(by_b, r.hole, Pred[]), r)
    end
    holes = sort!(collect(keys(by_a)))
    holes == sort!(collect(keys(by_b))) || fail("pooled-diff holes differ")
    n = length(holes)
    rng = rng_for(tag)
    diffs = Vector{Float64}(undef, N_BOOT)
    for b in 1:N_BOOT
        oa = Float64[]
        pa = Float64[]
        ma = Float64[]
        ob = Float64[]
        pb = Float64[]
        mb = Float64[]
        for _ in 1:n
            h = holes[rand(rng, 1:n)]
            for r in by_a[h]
                push!(oa, r.obs)
                push!(pa, r.pred)
                push!(ma, mu_by[parse(Int, r.sample)])
            end
            for r in by_b[h]
                push!(ob, r.obs)
                push!(pb, r.pred)
                push!(mb, mu_by[parse(Int, r.sample)])
            end
        end
        diffs[b] = pooled_skill_multifold(oa, pa, ma) - pooled_skill_multifold(ob, pb, mb)
    end
    point = pooled_skill_from_rows(rows_a, mu_by) - pooled_skill_from_rows(rows_b, mu_by)
    return point, quantile(diffs, 0.025), quantile(diffs, 0.975)
end

function report_skill_diffs(preds, folds)
    mu_by = row_train_means(folds)
    println("=== pooled skill difference vs per-hole paired mean ===")
    for nn in METHODS_NN
        for subset in ("all", "uncensored")
            rows_nn = select_preds(preds, nn, subset)
            rows_k = select_preds(preds, "kriging", subset)
            tag = BOOT_SEED_TAG * "|cu|$(nn)|$(subset)|pooled_skill_diff"
            d, lo, hi = bootstrap_pooled_skill_diff(rows_nn, rows_k, mu_by, tag)
            h_nn, sk_nn = hole_skill(rows_nn, mu_by)
            h_k, sk_k = hole_skill(rows_k, mu_by)
            h_nn == h_k || fail("paired holes differ")
            hole_diffs = [sk_nn[h] - sk_k[h] for h in h_nn]
            μ, plo, phi = bootstrap_mean_ci(hole_diffs, BOOT_SEED_TAG * "|cu|$(nn)|$(subset)|paired_skill")
            println(nn, " ", subset,
                    " pooled_diff=", tsv_num(d),
                    " ci95=", tsv_num(lo), " ", tsv_num(hi),
                    " paired_mean=", tsv_num(μ),
                    " paired_ci95=", tsv_num(plo), " ", tsv_num(phi),
                    " n_holes=", length(h_nn))
        end
    end
end

function fold_of_holes(path)
    out = Dict{String,Int}()
    open(path) do io
        readline(io)
        for line in eachline(io)
            p = split(line, '\t')
            out[p[2]] = parse(Int, p[1])
        end
    end
    return out
end

function report_coverage(preds, hole_fold)
    println("=== coverage90 per fold and median sd/|error| ===")
    for method in ("kriging", "nn_xyz", "nn_cov")
        rows = [r for r in preds if r.method == method]
        ratios = Float64[]
        n_zero = 0
        for r in rows
            ae = abs(r.pred - r.obs)
            if ae == 0
                n_zero += 1
            else
                push!(ratios, r.sd / ae)
            end
        end
        println(method, " n=", length(rows),
                " median_sd_over_abserr=", tsv_num(median(ratios)),
                " n_zero_abserr=", n_zero)
        if method == "nn_cov"
            continue
        end
        for fold in 1:10
            sub = [r for r in rows if hole_fold[r.hole] == fold]
            n = length(sub)
            cov = count(r -> abs(r.obs - r.pred) <= Z90 * r.sd, sub) / n
            println("  fold", fold, " n=", n, " coverage90=", tsv_num(cov))
        end
    end
end

function variogram_curves(table, y_all, train_rows)
    y_tr = y_all[train_rows]
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    hspan = max(maximum(dx) - minimum(dx), maximum(dy) - minimum(dy))
    vspan = maximum(dz) - minimum(dz)
    maxlag_h = max(10.0, 0.5 * hspan)
    maxlag_v = max(10.0, 0.5 * vspan)
    dist_h, semi_h, dist_v, semi_v = collect_pairs(dx, dy, dz, dv, dh;
                                                    maxlag_h = maxlag_h, maxlag_v = maxlag_v,
                                                    dip_deg = HORIZONTAL_DIP_DEG)
    bins_h, exp_h = bin_experimental(dist_h, semi_h, N_LAGS, maxlag_h)
    bins_v, exp_v = bin_experimental(dist_v, semi_v, N_LAGS, maxlag_v)
    vvar = var(dv; corrected = false)
    rmax_h = max(bins_h[end] * 2, RANGE_MIN_M * 4)
    rmax_v = max(bins_v[end] * 2, RANGE_MIN_M * 4)
    sill_cap = max(2 * max(vvar, 1.0e-8), maximum(exp_h), maximum(exp_v), 1.0e-6)
    lo = [RANGE_MIN_M, RANGE_MIN_M, 1.0e-8, 0.0]
    hi = [max(rmax_h, RANGE_MIN_M * 1.01), max(rmax_v, RANGE_MIN_M * 1.01), sill_cap, sill_cap]
    fit = fit_variogram(dx, dy, dz, dv, dh; context = "plot")
    return (; bins_h, exp_h, bins_v, exp_v, fit, lo, hi, maxlag_h, maxlag_v,
            rmax_h, rmax_v, sill_cap, hspan, vspan, n_merged, vvar)
end

function save_variogram_plot(path, fold_id, curves)
    fit = curves.fit
    hs = range(0.0, curves.maxlag_h; length = 200)
    vs = range(0.0, curves.maxlag_v; length = 200)
    gh = [model_gamma(fit.model, h, fit.range_h, fit.partial_sill, fit.nugget) for h in hs]
    gv = [model_gamma(fit.model, h, fit.range_v, fit.partial_sill, fit.nugget) for h in vs]
    fig = GLMakie.Figure(size = (900, 380))
    axh = GLMakie.Axis(fig[1, 1]; xlabel = "horizontal lag (m)", ylabel = "semivariance",
                       title = "fold $fold_id horizontal")
    axv = GLMakie.Axis(fig[1, 2]; xlabel = "downhole lag (m)", ylabel = "semivariance",
                       title = "fold $fold_id downhole")
    GLMakie.scatter!(axh, curves.bins_h, curves.exp_h; label = "experimental")
    GLMakie.lines!(axh, collect(hs), gh; label = fit.model)
    GLMakie.scatter!(axv, curves.bins_v, curves.exp_v; label = "experimental")
    GLMakie.lines!(axv, collect(vs), gv; label = fit.model)
    GLMakie.axislegend(axh; position = :rb)
    GLMakie.axislegend(axv; position = :rb)
    GLMakie.save(path, fig)
    return nothing
end

function report_variograms(table, y_all, eligible, fold_holes)
    println("=== variogram bounds ===")
    println("N_LAGS=", N_LAGS, " RANGE_MIN_M=", RANGE_MIN_M,
            " HORIZONTAL_DIP_DEG=", HORIZONTAL_DIP_DEG)
    println("parameter vector [range_h, range_v, partial_sill, nugget]")
    println("lo = [RANGE_MIN_M, RANGE_MIN_M, 1e-8, 0]")
    println("hi = [max(rmax_h, RANGE_MIN_M*1.01), max(rmax_v, RANGE_MIN_M*1.01), sill_cap, sill_cap]")
    println("rmax = max(2*last_bin_center, RANGE_MIN_M*4)")
    println("maxlag = max(10, 0.5*span) on the deduped training coordinates")
    GLMakie.activate!(; visible = false)
    for (k, test_holes) in enumerate(fold_holes)
        k in PLOT_FOLDS || continue
        test_set = Set(test_holes)
        train_rows = [i for i in eligible if table.hole[i] ∉ test_set]
        c = variogram_curves(table, y_all, train_rows)
        on_hi_v = c.fit.range_v == c.hi[2]
        on_lo_v = c.fit.range_v == c.lo[2]
        on_hi_h = c.fit.range_h == c.hi[1]
        on_maxlag_v = c.fit.range_v == c.maxlag_v
        println("fold", k,
                " model=", c.fit.model,
                " range_h=", tsv_num(c.fit.range_h),
                " range_v=", tsv_num(c.fit.range_v),
                " lo_h=", tsv_num(c.lo[1]), " hi_h=", tsv_num(c.hi[1]),
                " lo_v=", tsv_num(c.lo[2]), " hi_v=", tsv_num(c.hi[2]),
                " maxlag_h=", tsv_num(c.maxlag_h), " maxlag_v=", tsv_num(c.maxlag_v),
                " rmax_h=", tsv_num(c.rmax_h), " rmax_v=", tsv_num(c.rmax_v),
                " sill_cap=", tsv_num(c.sill_cap),
                " hspan=", tsv_num(c.hspan), " vspan=", tsv_num(c.vspan),
                " range_v_equals_hi=", on_hi_v,
                " range_v_equals_lo=", on_lo_v,
                " range_h_equals_hi=", on_hi_h,
                " range_v_equals_maxlag=", on_maxlag_v,
                " last_bin_h=", tsv_num(c.bins_h[end]),
                " last_bin_v=", tsv_num(c.bins_v[end]))
        path = joinpath(RUN, "variogram_fold$(k).png")
        save_variogram_plot(path, k, c)
        println("wrote ", path)
    end
end

function rebuild_folds(table, y_all, eligible, fold_holes)
    folds = KFoldResult[]
    for (k, test_holes) in enumerate(fold_holes)
        test_set = Set(test_holes)
        test_rows = [i for i in eligible if table.hole[i] in test_set]
        train_rows = [i for i in eligible if table.hole[i] ∉ test_set]
        m = mean(y_all[train_rows])
        push!(folds, KFoldResult(k, test_holes, train_rows, test_rows, m))
    end
    return folds
end

function main_checks()
    isfile(joinpath(RUN, "summary.tsv")) || fail("run2 summary.tsv missing")
    compare_tables()
    table, _, _ = load_site(SITE_PATH)
    spec = spec_of(table, TARGET)
    eligible = findall(training_mask(table) .& real_hole_mask(table) .& observed_mask(table, TARGET))
    y_all = fill(NaN, nsamples(table))
    y_all[eligible] = transformed(spec, table.values[TARGET][eligible])
    all_holes = sort!(unique(table.hole[eligible]))
    fold_holes = hole_kfold_assignments(all_holes, N_FOLDS, KFOLD_SEED)
    preds = load_predictions(joinpath(RUN, "predictions.tsv"))
    folds = rebuild_folds(table, y_all, eligible, fold_holes)
    report_skill_diffs(preds, folds)
    report_coverage(preds, fold_of_holes(joinpath(RUN, "fold_split.tsv")))
    report_variograms(table, y_all, eligible, fold_holes)
    return nothing
end

main_checks()
