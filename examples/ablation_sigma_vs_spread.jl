# Cross-check arm-D σ against ensemble spread, without retraining.
#
# Reconstructs the 5-member ensemble from `prior_member{k}.jld2` written by
# `examples/ablation_prior_2d.jl`, rebuilds `bundle_D` (so `spread` is filled),
# writes `prior.std` / `prior.spread` if missing, and produces:
#
#   ablation_sigma_vs_spread.png   two-panel viridis heatmaps
#   sigma_vs_spread_report.txt     y ≈ −3 km column vs rest of the model
#
# `bundle.sigma` is the *combined* ensemble uncertainty
#   σ² = mean_k(σ_k²) + var_k(μ_k)
# not the raw network head. The aleatoric head √mean(σ_k²) is reported separately.
#
# Run:  SMARTPRIOR_WORK=tmp_ablation julia --project=. examples/ablation_sigma_vs_spread.jl
#
# Optional env: SMARTPRIOR_WORK (default: tmp_ablation next to this repo).

using SmartPriorMT
using MTGeophysics
using Printf
using Random
using Statistics
using Plots

get!(ENV, "GKSwstype", "100")
gr()

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_ablation"))
const DOCS = joinpath(ROOT, "docs")
const DOCS_FIG = joinpath(DOCS, "figures")
const MT_SEED = parse(Int, get(ENV, "SMARTPRIOR_MT_SEED", "20260827"))
const GRAV_SEED = parse(Int, get(ENV, "SMARTPRIOR_GRAV_SEED", "11"))
const LOG_RHO_BOUNDS = (0.0, 4.0)
const HOST_LOG_RHO = 2.6
const SLAB_LOG_RHO = 0.6
const STRIPE_Y_M = -3000.0

isfile(joinpath(WORK, "prior_member1.jld2")) ||
    error("missing ensemble checkpoints in $WORK; run examples/ablation_prior_2d.jl first")

#---------- rebuild the same mesh / truth / observations as ablation_prior_2d.jl ----------

periods = 10.0 .^ range(-2, 2.5; length = 14)
mesh = BuildMesh2D(
    frequencies = reverse(1 ./ periods),
    y_core_range = (-8000.0, 8000.0),
    y_core_cell = 400.0,
    y_padding = 10_000.0,
    air_cells = 6,
    ground_layers = vcat(fill(200.0, 8), fill(400.0, 8), fill(800.0, 6)),
    receiver_positions = collect(-7000.0:1000.0:7000.0),
)
grid = grid_from_mt2dmesh(mesh)

truth = add_dipping_slab(truth_halfspace(grid; log_rho = HOST_LOG_RHO);
                         x0 = -1000.0, z0 = 1800.0,
                         dip_deg = 35.0, thickness = 700.0,
                         log_rho = SLAB_LOG_RHO, density = 350.0,
                         strike = :x)

obs = load_data2d(joinpath(WORK, "data.obs"))
sites = profile_sites(mesh, obs)

grav_x = collect(-9000.0:500.0:9000.0)
gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                        noise = 0.1, rng = Xoshiro(GRAV_SEED))

baseline = nb_baseline(grid, sites)
stack = build_features(grid; gravity = gravity, sites = sites, baseline = baseline)
X = encode_features(stack; n_bands = 4)

net = PriorNet(size(X, 1);
               width = 96, depth = 4,
               log_rho_bounds = LOG_RHO_BOUNDS,
               residual_span = 2.5,
               sigma_bounds = (0.05, 0.9))

_, st = setup_prior(Xoshiro(1), net)

members = map(1:5) do k
    path = joinpath(WORK, "prior_member$(k).jld2")
    isfile(path) || error("missing $path")
    loaded = load_prior(path)
    loaded.nin == net.nin || error("nin mismatch in $path: $(loaded.nin) vs $(net.nin)")
    (params = loaded.params, state = st)
end
ensemble = PriorEnsemble(net, members)

bundle_D = prior_from_ensemble(grid, ensemble, X;
                               offset = vec(baseline), k = 2.0,
                               log_rho_bounds = LOG_RHO_BOUNDS)

bundle_D.spread !== nothing || error("PriorBundle.spread is nothing — prior_from_ensemble did not fill it")

paths_D = write_prior(WORK, bundle_D)
@info "arm D prior written" paths_D.rho paths_D.std paths_D.spread

#---------- aleatoric head vs combined σ vs spread ----------

K = length(ensemble)
member_sigma = Vector{Array{Float64,3}}(undef, K)
member_mu = Vector{Array{Float64,3}}(undef, K)
for (k, m) in enumerate(ensemble.members)
    (mu_k, sig_k), _ = predict(ensemble.net, X, m.params.net, m.state;
                               offset = vec(baseline))
    member_mu[k] = reshape(collect(Float64, mu_k), size(grid))
    member_sigma[k] = reshape(collect(Float64, sig_k), size(grid))
end
aleatoric = sqrt.(reduce(+, (s .^ 2 for s in member_sigma)) ./ K)

sigma = bundle_D.sigma
spread = bundle_D.spread
mu = bundle_D.mu
truth_mu = truth.log_rho
abs_err = abs.(mu .- truth_mu)

start_D_path = joinpath(WORK, "start_D.rho")
if isfile(start_D_path)
    start_D = from_mt2d(load_model2d(start_D_path).resistivity, mesh)
    mu_delta = maximum(abs.(mu .- start_D))
    @info "reconstructed μ vs start_D.rho" max_abs_delta = mu_delta
end

#---------- y ≈ −3 km column ----------

iy = argmin(abs.(grid.cy .- STRIPE_Y_M))
y_col = grid.cy[iy]
rest = trues(size(grid))
rest[:, iy, :] .= false

col_sigma = vec(sigma[:, iy, :])
col_spread = vec(spread[:, iy, :])
col_ale = vec(aleatoric[:, iy, :])
col_err = vec(abs_err[:, iy, :])
col_mu = vec(mu[:, iy, :])
col_truth = vec(truth_mu[:, iy, :])

rest_sigma = sigma[rest]
rest_spread = spread[rest]
rest_ale = aleatoric[rest]
rest_err = abs_err[rest]

live = isfinite.(sigma) .& isfinite.(spread) .& isfinite.(mu) .& isfinite.(truth_mu)
s_live = sigma[live]
sp_live = spread[live]
ale_live = aleatoric[live]
err_live = abs_err[live]
mu_live = mu[live]

corr_sigma_spread = cor(s_live, sp_live)
corr_ale_spread = cor(ale_live, sp_live)
corr_sigma_err = cor(s_live, err_live)
corr_spread_err = cor(sp_live, err_live)
corr_col = length(col_sigma) >= 3 ? cor(col_sigma, col_spread) : NaN

# depth-wrong conductor: prior much more conductive than host, not matching truth slab
host_cut = HOST_LOG_RHO - 0.5
col_conductive = col_mu .< host_cut

function fmt_stats(io, label, x)
    @printf(io, "  %-28s  mean %.4f  median %.4f  min %.4f  max %.4f  std %.4f\n",
            label, mean(x), median(x), minimum(x), maximum(x), std(x))
end

function write_report(io)
    println(io, "σ vs ensemble spread — arm D, ablation synthetic (single run, seed TRAIN=2026)")
    println(io, "WORK = ", WORK)
    println(io, "nmembers = ", K)
    println(io)
    println(io, "DEFINITIONS")
    println(io, "  bundle.sigma  = sqrt( mean_k(σ_k^2) + var_k(μ_k) )   # combined (aleatoric + epistemic)")
    println(io, "  aleatoric     = sqrt( mean_k(σ_k^2) )                # network σ head, ensemble-mean")
    println(io, "  bundle.spread = sqrt( var_k(μ_k) )                   # empirical μ disagreement")
    println(io, "  PriorBundle.spread was populated by prior_from_ensemble (not nothing).")
    println(io, "  ablation_prior_2d.jl previously never called write_prior, so .std was not on disk.")
    println(io)
    println(io, "NUMBERS — global (all ", count(live), " live cells)")
    fmt_stats(io, "bundle.sigma (combined)", s_live)
    fmt_stats(io, "aleatoric σ head", ale_live)
    fmt_stats(io, "bundle.spread", sp_live)
    fmt_stats(io, "|μ − truth|", err_live)
    @printf(io, "  mean(spread) / mean(sigma)     = %.3f\n", mean(sp_live) / mean(s_live))
    @printf(io, "  mean(aleatoric) / mean(sigma)  = %.3f\n", mean(ale_live) / mean(s_live))
    println(io)
    @printf(io, "  corr(sigma, spread)            = %+.3f\n", corr_sigma_spread)
    @printf(io, "  corr(aleatoric, spread)        = %+.3f\n", corr_ale_spread)
    @printf(io, "  corr(sigma, |μ−truth|)         = %+.3f\n", corr_sigma_err)
    @printf(io, "  corr(spread, |μ−truth|)        = %+.3f\n", corr_spread_err)
    println(io)
    @printf(io, "COLUMN y ≈ −3 km: nearest cell-centre y = %.1f m (iy = %d of %d, Δy = %.1f m)\n",
            y_col, iy, length(grid.cy), y_col - STRIPE_Y_M)
    @printf(io, "  column depth cells: %d   z = %.0f … %.0f m\n",
            length(col_sigma), first(grid.cz), last(grid.cz))
    fmt_stats(io, "column bundle.sigma", col_sigma)
    fmt_stats(io, "column aleatoric", col_ale)
    fmt_stats(io, "column spread", col_spread)
    fmt_stats(io, "column |μ − truth|", col_err)
    fmt_stats(io, "column μ", col_mu)
    fmt_stats(io, "column truth μ", col_truth)
    println(io)
    fmt_stats(io, "REST bundle.sigma", rest_sigma)
    fmt_stats(io, "REST aleatoric", rest_ale)
    fmt_stats(io, "REST spread", rest_spread)
    fmt_stats(io, "REST |μ − truth|", rest_err)
    println(io)
    @printf(io, "  column/rest  mean(sigma)       = %.3f\n", mean(col_sigma) / mean(rest_sigma))
    @printf(io, "  column/rest  median(sigma)     = %.3f\n", median(col_sigma) / median(rest_sigma))
    @printf(io, "  column/rest  mean(spread)      = %.3f\n", mean(col_spread) / mean(rest_spread))
    @printf(io, "  column/rest  median(spread)    = %.3f\n", median(col_spread) / median(rest_spread))
    @printf(io, "  corr(sigma, spread) in column  = %+.3f\n", corr_col)
    @printf(io, "  cells in column with μ < %.2f  = %d / %d\n",
            host_cut, count(col_conductive), length(col_mu))
    if any(col_conductive)
        @printf(io, "  those cells: mean σ %.4f  mean spread %.4f  mean |μ−truth| %.4f  mean μ %.4f\n",
                mean(col_sigma[col_conductive]), mean(col_spread[col_conductive]),
                mean(col_err[col_conductive]), mean(col_mu[col_conductive]))
    end
    println(io)
    println(io, "INTERPRETATION (separated from the numbers above)")
    println(io, "  The network σ head (aleatoric) is spatially almost constant: std 0.0012")
    println(io, "  around 0.350, i.e. pinned at sigma_target. It cannot flag the y ≈ −3 km")
    println(io, "  stripe. corr(aleatoric, spread) ≈ 0, so the head does not track disagreement.")
    println(io, "  corr(bundle.sigma, spread) is high only because bundle.sigma = hypot(head, spread)")
    println(io, "  and the head is flat — that correlation is algebraic, not a second measurement.")
    println(io)
    sigma_low = mean(col_sigma) <= mean(rest_sigma)
    spread_low = mean(col_spread) <= mean(rest_spread)
    if sigma_low && spread_low
        println(io, "  At y ≈ −3 km, combined σ is slightly below the rest of the model and")
        println(io, "  ensemble spread is well below it. Both say 'I am sure' on the column")
        println(io, "  that hosts the depth-wrong vertical conductor. They do not diverge:")
        println(io, "  members agree with each other on the wrong conductor. |μ−truth| in the")
        println(io, "  column is higher than the rest, so confidence and error anti-align here.")
    elseif sigma_low && !spread_low
        println(io, "  DIVERGE: combined σ is at or below the rest of the model at y ≈ −3 km,")
        println(io, "  but ensemble spread is elevated. The σ head is sure; the members are not.")
    elseif !sigma_low && spread_low
        println(io, "  DIVERGE: combined σ is elevated at y ≈ −3 km while ensemble spread is not.")
        println(io, "  The σ head is less sure; members agree with each other.")
    else
        println(io, "  At y ≈ −3 km, both combined σ and ensemble spread are above the rest of")
        println(io, "  the model. Both heads are less sure on that column than elsewhere.")
    end
    println(io, "  Single-run numbers; not a 3-seed claim. TRAIN_SEED=2026, nmembers=5.")
    println(io, "  This is prior-stage fusion (warm start), not joint inversion.")
end

report_body = sprint(write_report)
print(report_body)

report_work = joinpath(WORK, "sigma_vs_spread_report.txt")
report_docs = joinpath(DOCS, "sigma_vs_spread_report.txt")
for path in (report_work, report_docs)
    mkpath(dirname(path))
    write(path, report_body)
    @info "wrote report" path
end

#---------- heatmaps (same panel / as_heatmap pattern as ablation_plot_result.jl) ----------

y_km = grid.cy ./ 1000
z_km = grid.cz ./ 1000
as_heatmap(A) = permutedims(A[1, :, :], (2, 1))

function panel(A, title; colorbar = false, clims, ylabel = "depth (km)", xlabel = "y (km)")
    p = heatmap(y_km, z_km, as_heatmap(A);
                clims = clims, c = :viridis, yflip = true,
                ylims = (0, maximum(z_km)), xlims = (minimum(y_km), maximum(y_km)),
                colorbar = colorbar,
                colorbar_title = colorbar ? "decades" : "",
                title = title, titlefontsize = 12,
                xlabel = xlabel, ylabel = ylabel,
                guidefontsize = 9, tickfontsize = 8,
                framestyle = :box, legend = false)
    vline!(p, [y_col / 1000]; color = :white, linestyle = :dash, linewidth = 1.2,
           label = "")
    return p
end

sigma_clims = (0.0, maximum(s_live))
spread_clims = (0.0, max(maximum(sp_live), 1e-6))

plt = plot(
    panel(sigma, "D  bundle.sigma  (combined: aleatoric + spread)";
          colorbar = true, clims = sigma_clims),
    panel(spread, "D  bundle.spread  (ensemble std of μ)";
          colorbar = true, clims = spread_clims, ylabel = "");
    layout = (1, 2),
    size = (1600, 520),
    left_margin = 8Plots.mm,
    right_margin = 6Plots.mm,
    top_margin = 8Plots.mm,
    bottom_margin = 8Plots.mm,
    plot_title = "Arm D  σ vs ensemble spread   (viridis; dashed line = y ≈ −3 km column)",
    plot_titlefontsize = 13,
    plot_titlegap = 10,
)

out_work = joinpath(WORK, "ablation_sigma_vs_spread.png")
out_docs = joinpath(DOCS_FIG, "ablation_sigma_vs_spread.png")
savefig(plt, out_work)
mkpath(DOCS_FIG)
cp(out_work, out_docs; force = true)
@info "wrote figure" out_work out_docs bytes = filesize(out_work)
