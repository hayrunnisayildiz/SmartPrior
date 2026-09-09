# Step 1/7: retrain the 2D prior with reference_penalty ON. Training only — no VFSA.
#
# Copied from examples/compare_prior_2d.jl. The single changed setting is
# LossWeights.reference. Everything else (mesh, truth, features, network,
# ensemble size, epochs, TrainConfig seed) is the same.
#
# TrainConfig.seed is 2026 in compare_prior_2d.jl; 4242 is the VFSA seed.
# Changing the training seed would add a second variable, so 2026 is kept.
#
# Run:  julia --project=. examples/train_prior_calib.jl
# Env:  SMARTPRIOR_WORK, SMARTPRIOR_REFERENCE, SMARTPRIOR_EPOCHS, SMARTPRIOR_NMEMBERS

using SmartPriorMT
using MTGeophysics
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_calib"))
const REFERENCE_WEIGHT = parse(Float64, get(ENV, "SMARTPRIOR_REFERENCE", "0.1"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "3000"))
const NMEMBERS = parse(Int, get(ENV, "SMARTPRIOR_NMEMBERS", "5"))
mkpath(WORK)
@info "working directory" WORK REFERENCE_WEIGHT EPOCHS NMEMBERS

#---------- 1. mesh and truth (identical to compare_prior_2d.jl) ----------

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
@info "prior grid" size(grid) ncells(grid)

const HOST_LOG_RHO = 2.6
const SLAB_LOG_RHO = 0.6

truth = add_dipping_slab(truth_halfspace(grid; log_rho = HOST_LOG_RHO);
                         x0 = -1000.0, z0 = 1800.0,
                         dip_deg = 35.0, thickness = 700.0,
                         log_rho = SLAB_LOG_RHO, density = 350.0,
                         strike = :x)

@info "truth" truth

#---------- 2. MT observations from the 2D solver ----------

rho_true_2d = to_mt2d(truth, mesh)
true_model_path = write_model2d(joinpath(WORK, "true.rho"), mesh, rho_true_2d;
                                title = "dipping conductive slab")

template_path = write_mt2d_data_template(joinpath(WORK, "template.ref"), mesh;
                                         impedance_error_fraction = 0.05)

obs_path = ForwardSolve2D(true_model_path, template_path;
                          add_noise = true, rng_seed = 20260827,
                          output_path = joinpath(WORK, "data.obs"))
@info "observed data written" obs_path

sites = profile_sites(mesh, load_data2d(obs_path))

#---------- 3. gravity observations from the truth's density ----------

grav_x = collect(-9000.0:500.0:9000.0)

gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                        noise = 0.1, rng = Xoshiro(11))
@printf("gravity anomaly: %.2f to %.2f mGal, error %.2f mGal\n",
        minimum(gravity.value), maximum(gravity.value), gravity.err[1])

#---------- 4. train the prior ----------

baseline = nb_baseline(grid, sites)
@printf("Niblett-Bostick baseline: rmse %.3f, correlation %.3f\n",
        rmse(truth.log_rho, baseline), anomaly_correlation(truth.log_rho, baseline))

stack = build_features(grid;
                       gravity = gravity,
                       sites = sites,
                       baseline = baseline)
X = encode_features(stack; n_bands = 4)
@info "features" nchannels(stack) size(X)

A_grav = gravity_matrix(grid, gravity.x, gravity.y, gravity.z)

site_cells = [(1, clamp(searchsortedlast(grid.y, y), 1, size(grid, 2)))
              for y in sites.y]

targets = PriorTargets(
    gravity = (A_grav, gravity.value, gravity.err),
    mt = (sites, site_cells),
    reference = vec(baseline),
    sigma_target = 0.35,
)

net = PriorNet(size(X, 1);
               width = 96, depth = 4,
               log_rho_bounds = (0.0, 4.0),
               residual_span = 2.5,
               sigma_bounds = (0.05, 0.9))

# reference is the only changed field vs compare_prior_2d.jl (which left it at 0.0).
# 0.1 is the value the LossWeights docstring uses as the residual-mode starting
# point: one decade of departure from the baseline must buy 0.1 of data misfit.
config = TrainConfig(
    epochs = EPOCHS,
    learning_rate = 3.0e-3,
    log_every = 250,
    seed = 2026,
    slope_bounds = (-4.0, -0.2),
    weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = 10.0, sigma = 0.1,
                          reference = REFERENCE_WEIGHT),
    checkpoint_path = joinpath(WORK, "prior.jld2"),
    checkpoint_every = 1000,
)

damping_warns = config.weights.reference <= 0 && targets.reference !== nothing

ensemble, results = train_ensemble(net, X, grid, targets;
                                   nmembers = NMEMBERS,
                                   config = config,
                                   offset = vec(baseline))

bundle = prior_from_ensemble(grid, ensemble, X;
                             offset = vec(baseline),
                             k = 2.0,
                             log_rho_bounds = (0.0, 4.0))
paths = write_prior(joinpath(WORK, "prior"), bundle)
@info "prior written" paths.rho paths.lo paths.hi

#---------- 5. score against truth (scoring only; truth was not a training target) ----------

println()
print_report(prior_report(truth, bundle); label = "smart prior vs truth")

half_start = fill(HOST_LOG_RHO, size(grid))
print_report(prior_report(truth,
                          PriorBundle(grid, half_start, fill(1.0, size(grid));
                                      k = 2.0, log_rho_bounds = (0.0, 4.0)));
             label = "half-space start vs truth")

function live_mask(arrays...)
    mask = trues(size(first(arrays)))
    for a in arrays
        mask .&= isfinite.(a)
    end
    return mask
end

function ols_truth_on_mu(truth_field, mu)
    mask = live_mask(truth_field, mu)
    t = truth_field[mask]
    m = mu[mask]
    m̄, t̄ = mean(m), mean(t)
    b = sum((m .- m̄) .* (t .- t̄)) / sum(abs2, m .- m̄)
    a = t̄ - b * m̄
    return a, b
end

mu = bundle.mu
sig = bundle.sigma
spr = bundle.spread
tfield = truth.log_rho
mask = live_mask(tfield, mu, sig)
t_live = tfield[mask]
mu_live = mu[mask]
sig_live = sig[mask]
absres = abs.(mu_live .- t_live)

a, b = ols_truth_on_mu(tfield, mu)
raw_rmse = rmse(tfield, mu)
bias_mu_minus_truth = mean(mu_live) - mean(t_live)
bias_truth_minus_mu = mean(t_live) - mean(mu_live)
corr_mu = cor(mu_live, t_live)
mean_sigma = mean(sig_live)
corr_sigma = cor(sig_live, absres)
mean_spread = spr === nothing ? NaN : mean(spr[live_mask(tfield, spr)])

last_hist = results[1].history[end]
w_ref = REFERENCE_WEIGHT * last_hist.reference
ref_share = last_hist.total > 0 ? w_ref / last_hist.total : NaN
@printf("\n")
println(repeat("=", 62))
println("  calibration report  (reference weight = $(REFERENCE_WEIGHT))")
println(repeat("=", 62))
@printf("OLS truth = a + b·μ          a = %+.4f   b = %+.4f\n", a, b)
@printf("μ ham RMSE (no affine)       %.4f\n", raw_rmse)
@printf("bias mean(μ)−mean(truth)     %+.4f\n", bias_mu_minus_truth)
@printf("bias mean(truth)−mean(μ)     %+.4f  (package calibration.bias)\n", bias_truth_minus_mu)
@printf("corr(μ, truth)               %.4f\n", corr_mu)
@printf("mean(σ)                      %.4f\n", mean_sigma)
@printf("corr(σ, |μ − truth|)         %+.4f\n", corr_sigma)
@printf("ensemble spread mean         %.4f\n", mean_spread)
@printf("damping warning fired        %s\n", damping_warns ? "yes" : "no")
@printf("last-epoch unweighted terms  grav %.3f  mt %.3f  smooth %.3e  ref %.3f\n",
        last_hist.gravity, last_hist.mt, last_hist.smooth, last_hist.reference)
@printf("ref term share of total      %.4f / %.4f  = %.1f%%  (weight × unweighted)\n",
        w_ref, last_hist.total, 100 * ref_share)
println(repeat("=", 62))
@printf("previous (reference = 0):    a = +1.720  b = +0.370  RMSE = 0.628\n")
@printf("                             bias = +0.396  corr = 0.328\n")
@printf("                             mean(σ) = 0.361  corr(σ,|res|) = +0.09  spread = 0.082\n")
@printf("acceptance: ham RMSE < 0.456 (half-space). this run: %.4f  %s\n",
        raw_rmse, raw_rmse < 0.456 ? "PASS" : "FAIL")
println(repeat("=", 62))

metrics_path = joinpath(WORK, "metrics.txt")
open(metrics_path, "w") do io
    println(io, "reference_weight\t", REFERENCE_WEIGHT)
    println(io, "ols_a\t", a)
    println(io, "ols_b\t", b)
    println(io, "rmse\t", raw_rmse)
    println(io, "bias_mu_minus_truth\t", bias_mu_minus_truth)
    println(io, "bias_truth_minus_mu\t", bias_truth_minus_mu)
    println(io, "corr_mu_truth\t", corr_mu)
    println(io, "mean_sigma\t", mean_sigma)
    println(io, "corr_sigma_absres\t", corr_sigma)
    println(io, "mean_spread\t", mean_spread)
    println(io, "damping_warning\t", damping_warns)
    println(io, "epochs\t", EPOCHS)
    println(io, "nmembers\t", NMEMBERS)
    println(io, "train_seed\t", config.seed)
    println(io, "last_total\t", last_hist.total)
    println(io, "last_gravity\t", last_hist.gravity)
    println(io, "last_mt\t", last_hist.mt)
    println(io, "last_smooth\t", last_hist.smooth)
    println(io, "last_sigma\t", last_hist.sigma)
    println(io, "last_reference\t", last_hist.reference)
    println(io, "weighted_reference\t", w_ref)
    println(io, "ref_share\t", ref_share)
end
@info "metrics written" metrics_path
