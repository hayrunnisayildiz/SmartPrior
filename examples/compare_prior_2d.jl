# End-to-end validation: does a smart prior beat a half-space start?
#
# This is the experiment the package exists to justify. It runs entirely in Julia
# with no external solver, so it can be reproduced anywhere MTGeophysics.jl
# installs.
#
#   1. Build a truth on MTGeophysics' own 2D mesh: a dipping conductive slab in a
#      resistive host. Dipping structure is the case a half-space start handles
#      worst, because no single depth is right across the profile.
#   2. Generate MT observations with MTGeophysics' 2D finite-volume solver and add
#      noise.
#   3. Generate gravity observations from the truth's density, which is set
#      independently of its resistivity.
#   4. Train a smart prior from the gravity and the Niblett-Bostick transform of
#      the observed MT data. The prior never sees the truth.
#   5. Invert twice with identical VFSA settings and seed, from a half-space and
#      from the prior, and score both against the truth.
#
# Why this is not circular. The prior is trained against a closed-form prism
# gravity operator and a 1D impedance recursion. The data it is scored on comes
# from a 2D finite-volume induction solve -- a different discretisation, and
# physics the 1D operator cannot represent, since it has no lateral induction at
# all. The density contrast of the slab does not follow from its resistivity by
# the linear law the network is allowed to learn, so the gravity term has to
# discover an approximate relation rather than invert a known one. The truth
# enters only in the final scoring.
#
# Run with:  julia --project=. examples/compare_prior_2d.jl
# Takes a few minutes, most of it in the two inversions.
#
# Optional env (defaults are the published 2D comparison):
#   SMARTPRIOR_WORK, SMARTPRIOR_MT_SEED, SMARTPRIOR_TRAIN_SEED,
#   SMARTPRIOR_VFSA_SEED, SMARTPRIOR_GRAV_SEED

using SmartPriorMT
using MTGeophysics
using Printf
using Random
using Statistics

const WORK = get(ENV, "SMARTPRIOR_WORK", mktempdir(; prefix = "smartprior_2d_"))
const MT_SEED = parse(Int, get(ENV, "SMARTPRIOR_MT_SEED", "20260827"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const VFSA_SEED = parse(Int, get(ENV, "SMARTPRIOR_VFSA_SEED", "4242"))
const GRAV_SEED = parse(Int, get(ENV, "SMARTPRIOR_GRAV_SEED", "11"))
mkpath(WORK)
@info "working directory" WORK
@info "seeds" MT_SEED TRAIN_SEED VFSA_SEED GRAV_SEED
open(joinpath(WORK, "seeds.txt"), "w") do io
    println(io, "mt_seed\t", MT_SEED)
    println(io, "train_seed\t", TRAIN_SEED)
    println(io, "vfsa_seed\t", VFSA_SEED)
    println(io, "grav_seed\t", GRAV_SEED)
end

#---------- 1. mesh and truth ----------

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

# A resistive host with a conductive slab dipping at 35 degrees. The slab is
# dense, so gravity sees it; that pairing is realistic for a sulphide or a
# graphitic shear zone, and it is deliberately not the relation the network
# would get for free from a resistivity-density proportionality.
const HOST_LOG_RHO = 2.6
const SLAB_LOG_RHO = 0.6

# strike = :x, so the slab is invariant along the single wide strike cell and
# dips across the profile. strike = :y would make it uniform in the one x cell
# and leave a flat layer -- a 1D model, which is not the problem being posed.
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

# a .ref template makes ForwardSolve2D read the stored errors as fractions
template_path = write_mt2d_data_template(joinpath(WORK, "template.ref"), mesh;
                                         impedance_error_fraction = 0.05)

obs_path = ForwardSolve2D(true_model_path, template_path;
                          add_noise = true, rng_seed = MT_SEED,
                          output_path = joinpath(WORK, "data.obs"))
@info "observed data written" obs_path

# The prior must see the same noisy observations VFSA inverts. load_data2d
# already converts z_xy to rho_xy / phase_xy with the identity in
# mt1d_apparent: ρ_a = |Z|² / (ω μ₀), φ = atan(Im Z, Re Z) in degrees.
# (synth_mt_sites perturbs ρ_a and phase directly, not Z, so it is not the
# conversion to reuse here.)
obs = load_data2d(obs_path)
sites = profile_sites(mesh, obs)

#---------- 3. gravity observations from the truth's density ----------

grav_x = collect(-9000.0:500.0:9000.0)

# 0.1 mGal is about what a careful ground survey achieves after terrain and
# Bouguer corrections. It matters that this is realistic: a much smaller error
# makes the gravity term demand an exact fit, and since gravity has no depth
# resolution at all the only way to deliver one is to contort the field.
gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                        noise = 0.1, rng = Xoshiro(GRAV_SEED))
@printf("gravity anomaly: %.2f to %.2f mGal, error %.2f mGal\n",
        minimum(gravity.value), maximum(gravity.value), gravity.err[1])

#---------- 4. train the prior ----------

# The Niblett-Bostick transform of the observed TE curves is a data-driven
# depth-resistivity estimate. It knows nothing about the 2D solver's physics and
# smears a dipping body badly, which is exactly why it is a baseline and not an
# answer: the prior's job is to sharpen it using gravity.
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

# The two data types resolve different things, and the prior exists to fuse them.
# Gravity fixes lateral position and total mass but has no depth resolution
# whatsoever: any number of density distributions produce the same surface
# anomaly. The MT curves carry the depth information, through the skin-depth
# dependence on period. Fitting gravity alone gives a prior with the structure in
# the right place horizontally and smeared through the whole column.
#
# The MT term uses a 1D response per column, which the 2D observations were not
# generated by. That is a modelling approximation, not circularity: the data comes
# from a 2D finite-volume induction solve, and the 1D operator cannot represent
# the lateral induction in it. The term is there to reject columns whose own 1D
# response contradicts the measurement, which is enough to pin depth.
site_cells = [(1, clamp(searchsortedlast(grid.y, y), 1, size(grid, 2)))
              for y in sites.y]

targets = PriorTargets(
    gravity = (A_grav, gravity.value, gravity.err),
    mt = (sites, site_cells),
    reference = vec(baseline),
    sigma_target = 0.35,
)

# residual_span has to exceed the largest departure from the baseline that the
# truth actually contains. The slab is two decades below the host, and a baseline
# sitting near the host level therefore needs a span well past 2; a span of 1.5
# would make the correct answer unreachable and the network would saturate
# against the bound instead.
net = PriorNet(size(X, 1);
               width = 96, depth = 4,
               log_rho_bounds = (0.0, 4.0),
               residual_span = 2.5,
               sigma_bounds = (0.05, 0.9))

# Weights set by watching the terms, not guessed. Both data terms are
# chi-squared-per-datum, so a value near one means fitted to the noise; the
# smoothness weight is then raised until the reported gravity term stops falling
# below one, which is the point where it would be fitting noise by contorting the
# field. `saturation` in the log should stay near zero throughout.
# The slope bound is the single most important setting here. The target is a
# conductive body that is also dense, so the coupling must be negative; the
# magnitude stays free within a range that spans plausible petrophysics. Without
# this the fit reaches the same gravity misfit with a positive slope, putting
# resistive material where the mass is and returning a prior anti-correlated with
# the truth. Gravity alone cannot tell the two apart.
config = TrainConfig(
    epochs = 3000,
    learning_rate = 3.0e-3,
    log_every = 250,
    seed = TRAIN_SEED,
    slope_bounds = (-4.0, -0.2),
    weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = 10.0, sigma = 0.1),
    checkpoint_path = joinpath(WORK, "prior.jld2"),
    checkpoint_every = 1000,
)

# residual mode: the network predicts a correction to the baseline rather than an
# absolute resistivity, so it starts from something already roughly right and
# spends its capacity on the part gravity can explain
ensemble, results = train_ensemble(net, X, grid, targets;
                                   nmembers = 5,
                                   config = config,
                                   offset = vec(baseline))

bundle = prior_from_ensemble(grid, ensemble, X;
                             offset = vec(baseline),
                             k = 2.0,
                             log_rho_bounds = (0.0, 4.0))
paths = write_prior(joinpath(WORK, "prior"), bundle)
@info "prior written" paths.rho paths.lo paths.hi

#---------- 5. score the prior itself, before any inversion ----------

println()
print_report(prior_report(truth, bundle); label = "smart prior vs truth")

# OLS of truth on μ: amplitude collapse shows up as b < 1
mu_live = bundle.mu[isfinite.(bundle.mu) .& isfinite.(truth.log_rho)]
t_live = truth.log_rho[isfinite.(bundle.mu) .& isfinite.(truth.log_rho)]
m̄, t̄ = mean(mu_live), mean(t_live)
b_ols = sum((mu_live .- m̄) .* (t_live .- t̄)) / sum(abs2, mu_live .- m̄)
a_ols = t̄ - b_ols * m̄
@printf("prior OLS truth = a + b·μ:  a = %+.4f   b = %+.4f   RMSE = %.4f   corr = %.4f\n",
        a_ols, b_ols, rmse(truth.log_rho, bundle.mu),
        anomaly_correlation(truth.log_rho, bundle.mu))

half_start = fill(HOST_LOG_RHO, size(grid))
print_report(prior_report(truth,
                          PriorBundle(grid, half_start, fill(1.0, size(grid));
                                      k = 2.0, log_rho_bounds = (0.0, 4.0)));
             label = "half-space start vs truth")

#---------- 6. two inversions, identical settings ----------

half_path = write_model2d(joinpath(WORK, "start_half.rho"), mesh,
                          to_mt2d(reshape(half_start, size(grid)), mesh);
                          title = "homogeneous half-space start")

prior_path = write_model2d(joinpath(WORK, "start_prior.rho"), mesh,
                           to_mt2d(bundle.mu, mesh);
                           title = "smart prior start")

# identical seed and schedule, so the only difference between the two runs is
# where they started
inv_config = VFSA2DMTConfig(
    n_chains = 2,
    n_ctrl = 250,
    max_iter = 400,
    log_bounds = (0.0, 4.0),
    step_scale = 0.11,
    cool_ratio = 1.0e-3,
    target_rms = 1.0,
    seed = VFSA_SEED,
    keep_models = false,
    output_root = WORK,
)

@info "inverting from the half-space"
run_half = VFSA2DMT(half_path, obs_path;
                    run_dir = joinpath(WORK, "inv_half"),
                    true_model_path = true_model_path,
                    config = inv_config)

@info "inverting from the smart prior"
run_prior = VFSA2DMT(prior_path, obs_path;
                     run_dir = joinpath(WORK, "inv_prior"),
                     true_model_path = true_model_path,
                     config = inv_config)

#---------- 7. compare ----------

result_half = from_mt2d(run_half.best_chain.best_resistivity, mesh)
result_prior = from_mt2d(run_prior.best_chain.best_resistivity, mesh)

cmp_half = compare_starts(truth.log_rho, half_start, result_half)
cmp_prior = compare_starts(truth.log_rho, bundle.mu, result_prior)

rms_best_at(chain, iter) = begin
    rec = findfirst(r -> r.iteration == iter, chain.iterations)
    rec === nothing ? NaN : chain.iterations[rec].best_rms
end
half_c1 = run_half.chains[1]
prior_c1 = run_prior.chains[1]

println()
println(repeat("=", 62))
println("  inversion outcome")
println(repeat("=", 62))
@printf("%-26s %12s %12s\n", "", "half-space", "smart prior")
@printf("%-26s %12.4f %12.4f\n", "data rms (best chain)",
        run_half.best_chain.best_rms, run_prior.best_chain.best_rms)
@printf("%-26s %12.4f %12.4f\n", "data rms chain1 iter 1",
        rms_best_at(half_c1, 1), rms_best_at(prior_c1, 1))
@printf("%-26s %12.4f %12.4f\n", "data rms chain1 iter 400",
        rms_best_at(half_c1, 400), rms_best_at(prior_c1, 400))
@printf("%-26s %12.4f %12.4f\n", "model rmse at start",
        cmp_half.rmse_start, cmp_prior.rmse_start)
@printf("%-26s %12.4f %12.4f\n", "model rmse after",
        cmp_half.rmse_final, cmp_prior.rmse_final)
@printf("%-26s %12.3f %12.3f\n", "improvement",
        cmp_half.improvement, cmp_prior.improvement)
@printf("%-26s %12.3f %12.3f\n", "correlation after",
        cmp_half.correlation_final, cmp_prior.correlation_final)
println(repeat("=", 62))

# The honest reading. A lower data rms with a worse model rmse is the signature
# of the problem this package is about: the inversion found a different model
# that fits the data just as well, which is non-uniqueness doing its work. The
# number that matters is the model rmse against the truth, and the correlation
# next to it, which says whether the structure is in the right place at all.
if cmp_prior.rmse_final < cmp_half.rmse_final
    @printf("\nthe smart prior ended %.1f%% closer to the truth\n",
            100 * (1 - cmp_prior.rmse_final / cmp_half.rmse_final))
else
    @printf("\nthe half-space ended closer this time; prior rmse %.4f vs %.4f\n",
            cmp_prior.rmse_final, cmp_half.rmse_final)
end

#---------- 8. what the per-cell bounds would buy a 3D run ----------

# The 2D driver clamps against a scalar interval, so this run cannot use the
# bound files. Report what they would confine a 3D search to, since that is the
# other half of the contribution.
lo, hi = prior_bounds(bundle)
bc = SmartPriorMT.BoundedCore(lo, hi, 1:1, 1:size(grid, 2), 1:size(grid, 3))
r = SmartPriorMT.bound_report(bc; reference = inv_config.log_bounds)

println()
@printf("per-cell bounds: %d cells, width %.2f-%.2f decades (geometric mean %.2f)\n",
        r.nbounded, r.width_min, r.width_max, r.width_geomean)
@printf("search volume per cell: %.1f%% of the scalar %.1f-%.1f interval\n",
        100 * r.volume_ratio, inv_config.log_bounds...)
@printf("coverage of the truth by those bounds: %.3f\n", coverage(truth.log_rho, lo, hi))

println()
@info "all outputs kept" WORK
