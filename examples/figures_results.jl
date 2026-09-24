# Result figures for the Keivitsa Cu feasibility run2.
#
#   julia --project=. examples/figures_results.jl
#
# Reads metrics from tmp_feasibility_keivitsa_run2/ and writes PNGs to docs/figures/.
# Skill matches the feasibility definition: 1 - RMSE / RMSE_mean, with the
# fold training mean taken from the "mean" method predictions.

using Random
using Statistics
using Printf

# Prefer GLMakie (headless via visible=false). Fall back to CairoMakie only if
# GLMakie fails; do not add packages — stop if CairoMakie is also unavailable.
const MAKIE_BACKEND = Ref{Symbol}(:none)
try
    using GLMakie
    GLMakie.activate!(; visible = false)
    MAKIE_BACKEND[] = :GLMakie
catch err_gl
    @warn "GLMakie failed; trying CairoMakie" exception = err_gl
    try
        using CairoMakie
        MAKIE_BACKEND[] = :CairoMakie
    catch err_ca
        error("Neither GLMakie nor CairoMakie could be loaded. " *
              "GLMakie is listed in Project.toml but failed: $err_gl. " *
              "CairoMakie is not in Project.toml (do not add packages): $err_ca")
    end
end

const ROOT = dirname(@__DIR__)
const RUN = joinpath(ROOT, "tmp_feasibility_keivitsa_run2")
const OUTDIR = joinpath(ROOT, "docs", "figures")

const N_BOOT = 2000
const BOOT_SEED_TAG = "feasibility-keivitsa-bootstrap-v1"
const Z90 = 1.6448536269514722  # Φ^{-1}(0.95); central 90% interval
const SKILL_REF = 0.10
const COVERAGE_REF = 0.90
const PX_PER_UNIT = 150 / 96

# RGB tuples (Makie accepts these without a separate Colors import).
const COL_MEAN = (0.50, 0.50, 0.50)
const COL_KRIG = (0.12, 0.47, 0.71)   # blue
const COL_NN = (0.84, 0.15, 0.16)     # red
const COL_NN_COV = (1.00, 0.50, 0.05) # orange

# Lightweight row — spatial columns from predictions.tsv are not loaded.
struct PredRow
    method::String
    hole::String
    sample::String
    obs::Float64
    censored::Int
    pred::Float64
    sd::Float64
end

function fail(msg::AbstractString)
    error(msg)
end

function fnv1a(s::AbstractString)
    h = UInt64(14695981039346656037)
    for c in codeunits(String(s))
        h = xor(h, UInt64(c))
        h *= UInt64(1099511628211)
    end
    return h
end

function rng_for(tag::AbstractString)
    return Xoshiro(Int(fnv1a(tag) & typemax(Int)))
end

# Acklam rational approximation to Φ^{-1}(p) (no extra deps).
function norminv(p::Float64)
    (0.0 < p < 1.0) || fail("norminv: p out of (0,1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614736e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p > phigh
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    else
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    end
end

function z_for_nominal(alpha::Float64)
    return norminv(0.5 * (1 + alpha))
end

function is_gtk_hole(hole::AbstractString)
    return occursin(r"(?i)gtk", hole)
end

function load_predictions(path)
    rows = PredRow[]
    open(path) do io
        header = readline(io)
        cols = split(header, '\t')
        length(cols) == 12 || fail("predictions.tsv: expected 12 columns, got $(length(cols))")
        for line in eachline(io)
            isempty(strip(line)) && continue
            p = split(line, '\t')
            length(p) == 12 || fail("predictions.tsv: bad row width $(length(p))")
            hole = p[4]
            is_gtk_hole(hole) && continue
            # Columns: deposit,target,method,hole,sample, <skip 2>, z,obs,censored,pred,sd
            push!(rows, PredRow(p[3], hole, p[5],
                                parse(Float64, p[9]), parse(Int, p[10]),
                                parse(Float64, p[11]), parse(Float64, p[12])))
        end
    end
    isempty(rows) && fail("no prediction rows loaded")
    return rows
end

function load_fold_of_holes(path)
    out = Dict{String,Int}()
    open(path) do io
        header = readline(io)
        startswith(header, "fold\thole") || fail("fold_split.tsv header mismatch")
        for line in eachline(io)
            isempty(strip(line)) && continue
            p = split(line, '\t')
            length(p) >= 2 || fail("fold_split.tsv bad row")
            hole = p[2]
            is_gtk_hole(hole) && continue
            out[hole] = parse(Int, p[1])
        end
    end
    return out
end

function select_method(preds::Vector{PredRow}, method::AbstractString)
    return [r for r in preds if r.method == method]
end

function train_mean_by_sample(preds::Vector{PredRow})
    mu = Dict{String,Float64}()
    for r in select_method(preds, "mean")
        mu[r.sample] = r.pred
    end
    isempty(mu) && fail("no mean-method rows for train-fold baseline")
    return mu
end

function pooled_skill(obs, pred, mu_per_sample)
    n = length(obs)
    n == length(pred) == length(mu_per_sample) || fail("skill: length mismatch")
    n == 0 && return NaN
    rmse = sqrt(mean(abs2, pred .- obs))
    rmse_mean = sqrt(mean(abs2, obs .- mu_per_sample))
    return rmse_mean == 0 ? NaN : 1 - rmse / rmse_mean
end

function skill_from_rows(rows::Vector{PredRow}, mu_by::Dict{String,Float64})
    obs = Float64[]
    pred = Float64[]
    mu = Float64[]
    for r in rows
        haskey(mu_by, r.sample) || continue
        push!(obs, r.obs)
        push!(pred, r.pred)
        push!(mu, mu_by[r.sample])
    end
    return pooled_skill(obs, pred, mu)
end

function bootstrap_skill_ci(rows::Vector{PredRow}, mu_by::Dict{String,Float64},
                            tag::AbstractString)
    by_hole = Dict{String,Vector{PredRow}}()
    for r in rows
        haskey(mu_by, r.sample) || continue
        push!(get!(by_hole, r.hole, PredRow[]), r)
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
                push!(obs, r.obs)
                push!(pred, r.pred)
                push!(mu, mu_by[r.sample])
            end
        end
        skills[b] = pooled_skill(obs, pred, mu)
    end
    point = skill_from_rows(rows, mu_by)
    return point, quantile(skills, 0.025), quantile(skills, 0.975)
end

function coverage_at(rows::Vector{PredRow}, z::Float64)
    n = length(rows)
    n == 0 && return NaN
    return count(r -> isfinite(r.sd) && abs(r.obs - r.pred) <= z * r.sd, rows) / n
end

function folds_present(hole_fold::Dict{String,Int})
    return sort!(unique(values(hole_fold)))
end

function rows_in_fold(rows::Vector{PredRow}, hole_fold::Dict{String,Int}, fold::Int)
    return [r for r in rows if get(hole_fold, r.hole, -1) == fold]
end

function per_fold_skill(rows::Vector{PredRow}, mu_by::Dict{String,Float64},
                        hole_fold::Dict{String,Int}, folds::Vector{Int})
    return [skill_from_rows(rows_in_fold(rows, hole_fold, f), mu_by) for f in folds]
end

function per_fold_coverage90(rows::Vector{PredRow}, hole_fold::Dict{String,Int},
                             folds::Vector{Int})
    return [coverage_at(rows_in_fold(rows, hole_fold, f), Z90) for f in folds]
end

function calibration_curve(rows::Vector{PredRow}, alphas)
    return [coverage_at(rows, z_for_nominal(a)) for a in alphas]
end

function save_fig(path, fig)
    save(path, fig; px_per_unit = PX_PER_UNIT)
    println("wrote ", path)
end

function fig_skill_ci(methods, labels, colors, points, los, his)
    fig = Figure(size = (720, 420))
    ax = Axis(fig[1, 1];
              title = "Pooled skill with 95% hole-bootstrap CI",
              ylabel = "skill (1 − RMSE / RMSE_mean)",
              xlabel = "method",
              xticks = (1:length(methods), labels),
              xticklabelrotation = π / 8)
    for i in eachindex(methods)
        y = points[i]
        lo = los[i]
        hi = his[i]
        errorbars!(ax, [i], [y], [y - lo], [hi - y]; color = colors[i], whiskerwidth = 12)
        scatter!(ax, [i], [y]; color = colors[i], markersize = 14)
    end
    hlines!(ax, [SKILL_REF]; color = :black, linestyle = :dot, label = "skill = 0.10")
    xlims!(ax, 0.4, length(methods) + 0.6)
    axislegend(ax; position = :rb)
    return fig
end

function fig_grouped_bars(folds, y_k, y_n, title, ylabel, href)
    fig = Figure(size = (820, 420))
    ax = Axis(fig[1, 1];
              title = title,
              xlabel = "fold",
              ylabel = ylabel,
              xticks = (folds, string.(folds)))
    n = length(folds)
    xs = Float64.(folds)
    # Grouped bars via dodge cateogry
    xpos = vcat(xs, xs)
    heights = vcat(y_k, y_n)
    grp = vcat(fill(1, n), fill(2, n))
    cols = [grp[i] == 1 ? COL_KRIG : COL_NN for i in eachindex(grp)]
    barplot!(ax, xpos, heights; dodge = grp, color = cols)
    if href !== nothing
        hlines!(ax, [href]; color = :black, linestyle = :dash,
                label = href == COVERAGE_REF ? "nominal 0.90" : "ref")
    end
    # Manual legend swatches
    scatter!(ax, [NaN], [NaN]; color = COL_KRIG, marker = :rect, markersize = 14,
             label = "kriging (v1)")
    scatter!(ax, [NaN], [NaN]; color = COL_NN, marker = :rect, markersize = 14,
             label = "nn_xyz")
    axislegend(ax; position = :rb)
    return fig
end

function fig_calibration(alphas, cov_k, cov_n)
    fig = Figure(size = (560, 520))
    ax = Axis(fig[1, 1];
              title = "Predictive calibration (central intervals)",
              xlabel = "nominal coverage",
              ylabel = "empirical coverage",
              aspect = DataAspect())
    lines!(ax, [0.0, 1.0], [0.0, 1.0]; color = :gray, linestyle = :dash, label = "1:1")
    lines!(ax, alphas, cov_k; color = COL_KRIG, linewidth = 2, label = "kriging (v1)")
    scatter!(ax, alphas, cov_k; color = COL_KRIG, markersize = 10)
    lines!(ax, alphas, cov_n; color = COL_NN, linewidth = 2, label = "neural field (nn_xyz)")
    scatter!(ax, alphas, cov_n; color = COL_NN, markersize = 10)
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)
    axislegend(ax; position = :lt)
    return fig
end

function fig_pred_vs_obs(obs_k, pred_k, obs_n, pred_n)
    fig = Figure(size = (900, 420))
    lo = min(minimum(obs_k), minimum(pred_k), minimum(obs_n), minimum(pred_n))
    hi = max(maximum(obs_k), maximum(pred_k), maximum(obs_n), maximum(pred_n))
    pad = 0.05 * (hi - lo + eps())
    lims = (lo - pad, hi + pad)
    ax1 = Axis(fig[1, 1];
               title = "kriging (v1)",
               xlabel = "observed log₁₀ Cu",
               ylabel = "predicted log₁₀ Cu",
               aspect = DataAspect())
    ax2 = Axis(fig[1, 2];
               title = "neural field (nn_xyz)",
               xlabel = "observed log₁₀ Cu",
               ylabel = "predicted log₁₀ Cu",
               aspect = DataAspect())
    hexbin!(ax1, obs_k, pred_k; bins = 40, colormap = :blues)
    hexbin!(ax2, obs_n, pred_n; bins = 40, colormap = :reds)
    for ax in (ax1, ax2)
        lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]];
               color = :black, linestyle = :dash, linewidth = 1.5)
        xlims!(ax, lims)
        ylims!(ax, lims)
    end
    Label(fig[0, :], "Predicted vs observed log₁₀ Cu"; fontsize = 16, font = :bold)
    return fig
end

function main()
    isfile(joinpath(RUN, "predictions.tsv")) || fail("missing $(joinpath(RUN, "predictions.tsv"))")
    isfile(joinpath(RUN, "fold_split.tsv")) || fail("missing $(joinpath(RUN, "fold_split.tsv"))")
    mkpath(OUTDIR)

    backend = MAKIE_BACKEND[]
    println("Makie backend: ", backend)

    preds = load_predictions(joinpath(RUN, "predictions.tsv"))
    hole_fold = load_fold_of_holes(joinpath(RUN, "fold_split.tsv"))
    mu_by = train_mean_by_sample(preds)
    folds = folds_present(hole_fold)

    methods = ["mean", "kriging", "nn_xyz", "nn_cov"]
    labels = ["mean", "kriging (v1)", "nn_xyz", "nn_cov"]
    colors = [COL_MEAN, COL_KRIG, COL_NN, COL_NN_COV]
    points = Float64[]
    los = Float64[]
    his = Float64[]

    println("=== pooled skill (subset=all) with 95% hole-bootstrap CI ===")
    for method in methods
        rows = select_method(preds, method)
        tag = BOOT_SEED_TAG * "|cu|$(method)|all|skill"
        sk, lo, hi = bootstrap_skill_ci(rows, mu_by, tag)
        push!(points, sk)
        push!(los, lo)
        push!(his, hi)
        @printf("%s  skill=%.10g  ci95=(%.10g, %.10g)  n=%d\n",
                method, sk, lo, hi, length(rows))
    end

    fig1 = fig_skill_ci(methods, labels, colors, points, los, his)
    save_fig(joinpath(OUTDIR, "skill_ci.png"), fig1)

    rows_k = select_method(preds, "kriging")
    rows_n = select_method(preds, "nn_xyz")
    sk_k = per_fold_skill(rows_k, mu_by, hole_fold, folds)
    sk_n = per_fold_skill(rows_n, mu_by, hole_fold, folds)
    println("=== per-fold skill ===")
    for (i, f) in enumerate(folds)
        @printf("fold %d  kriging=%.10g  nn_xyz=%.10g\n", f, sk_k[i], sk_n[i])
    end
    fig2 = fig_grouped_bars(folds, sk_k, sk_n,
                            "Per-fold skill: kriging (v1) vs nn_xyz",
                            "skill (1 − RMSE / RMSE_mean)", nothing)
    save_fig(joinpath(OUTDIR, "skill_per_fold.png"), fig2)

    cov_k = per_fold_coverage90(rows_k, hole_fold, folds)
    cov_n = per_fold_coverage90(rows_n, hole_fold, folds)
    println("=== per-fold coverage90 ===")
    for (i, f) in enumerate(folds)
        @printf("fold %d  kriging=%.10g  nn_xyz=%.10g\n", f, cov_k[i], cov_n[i])
    end
    fig3 = fig_grouped_bars(folds, cov_k, cov_n,
                            "Per-fold 90% predictive coverage",
                            "empirical coverage", COVERAGE_REF)
    save_fig(joinpath(OUTDIR, "coverage_per_fold.png"), fig3)

    alphas = Float64[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal_k = calibration_curve(rows_k, alphas)
    cal_n = calibration_curve(rows_n, alphas)
    println("=== calibration curve ===")
    for (i, a) in enumerate(alphas)
        @printf("nominal=%.2f  kriging=%.10g  nn_xyz=%.10g\n", a, cal_k[i], cal_n[i])
    end
    fig4 = fig_calibration(alphas, cal_k, cal_n)
    save_fig(joinpath(OUTDIR, "calibration_curve.png"), fig4)

    obs_k = [r.obs for r in rows_k]
    pred_k = [r.pred for r in rows_k]
    obs_n = [r.obs for r in rows_n]
    pred_n = [r.pred for r in rows_n]
    fig5 = fig_pred_vs_obs(obs_k, pred_k, obs_n, pred_n)
    save_fig(joinpath(OUTDIR, "pred_vs_obs.png"), fig5)

    println("done; backend=", backend)
    return backend, methods, points, los, his
end

main()
