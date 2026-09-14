# Geology sweep: does the prior's gain over NB survive changes to the truth
# geometry, not just to the noise seed?
#
# compare_prior_2d.jl and the 3-seed table in docs/ARA_RAPOR.md vary
# (MT noise, train init, VFSA schedule, gravity noise) but hold the truth fixed:
# one dip angle (35 deg), one polarity (dense body is the conductor). This
# script holds the noise seeds fixed and instead varies the truth itself, so a
# "generalises" claim has more than one geometry behind it.
#
# No VFSA here -- this only scores the trained prior against the truth, the
# same thing train_prior_sensitivity.jl does for hyperparameters. That keeps a
# 4-scenario sweep to the cost of ~4 training runs instead of ~4 training runs
# plus 8 VFSA chains. Once a scenario's prior/NB gap looks interesting, rerun
# it through compare_prior_2d.jl (SMARTPRIOR_WORK=... and the same seeds) to
# see whether the gain also shows up in VFSA convergence speed.
#
# One scenario below ("dip35_conductive_published") uses the exact truth from
# compare_prior_2d.jl / the published 3-seed table, as an internal check: its
# prior_rmse / prior_corr should land close to the t1 row in ARA_RAPOR.md
# (0.546 / 0.438) run-to-run noise aside. If it doesn't, something in this
# script diverged from the published pipeline and the other rows should not be
# trusted yet.
#
# Env overrides:
#   SMARTPRIOR_WORK              output root (default tmp_scenario_sweep)
#   SMARTPRIOR_EPOCHS            default 3000 (set 300-500 for a fast scout run)
#   SMARTPRIOR_NMEMBERS          default 5 (set 1 for a fast scout run)
#   SMARTPRIOR_MT_SEED / SMARTPRIOR_TRAIN_SEED / SMARTPRIOR_GRAV_SEED
#   SMARTPRIOR_SCENARIOS         comma-separated scenario names to isolate
#                                 (e.g. "dip55_conductive") instead of running
#                                 all four -- ~4x faster for a single-scenario
#                                 diagnostic
#   SMARTPRIOR_SMOOTH_WEIGHT     default 10.0 -- lower it to test whether the
#                                 smoothness penalty is why a scenario fails
#   SMARTPRIOR_REFERENCE_WEIGHT  default 0.0 -- raise it (e.g. 0.3) to test
#                                 whether penalising drift from NB is why a
#                                 scenario fails (see the "unpenalised
#                                 residual" warning at train time)
#
# Writes:
#   $WORK/sweep_metrics.tsv
#   $WORK/<scenario>/metrics.txt
#
# Run:
#   julia --project=. examples/scenario_sweep_prior_2d.jl
# Scout run (few minutes total instead of ~20-40):
#   SMARTPRIOR_EPOCHS=400 SMARTPRIOR_NMEMBERS=1 \
#     julia --project=. examples/scenario_sweep_prior_2d.jl

using SmartPriorMT
using MTGeophysics
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_scenario_sweep"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "3000"))
const NMEMBERS = parse(Int, get(ENV, "SMARTPRIOR_NMEMBERS", "5"))
const MT_SEED = parse(Int, get(ENV, "SMARTPRIOR_MT_SEED", "20260827"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const GRAV_SEED = parse(Int, get(ENV, "SMARTPRIOR_GRAV_SEED", "11"))
const LOG_RHO_BOUNDS = (0.0, 4.0)
const HOST_LOG_RHO = 2.6

# Diagnostic overrides -- for isolating *why* a scenario fails, not for the
# main sweep. Defaults reproduce the published/sweep weights exactly.
#   SMARTPRIOR_SMOOTH_WEIGHT     default 10.0 (published value)
#   SMARTPRIOR_REFERENCE_WEIGHT  default 0.0  (published value -- the
#                                 "unpenalised residual" warning comes from
#                                 this being 0; set >0, e.g. 0.3, to test
#                                 whether penalising drift away from NB fixes
#                                 a failing scenario)
#   SMARTPRIOR_SCENARIOS         comma-separated scenario names to run, e.g.
#                                 "dip55_conductive" to isolate one instead of
#                                 sweeping all four. Default: all.
const SMOOTH_WEIGHT = parse(Float64, get(ENV, "SMARTPRIOR_SMOOTH_WEIGHT", "10.0"))
const REFERENCE_WEIGHT = parse(Float64, get(ENV, "SMARTPRIOR_REFERENCE_WEIGHT", "0.0"))

mkpath(WORK)

# Each row is a full truth geometry, not just a hyperparameter. dip_deg holds
# the slab's dip (see add_dipping_slab; the same case a half-space handles
# worst is the 35 deg published case, here bracketed by a shallower and a
# steeper dip). polarity flips which sign of the density-resistivity coupling
# is true: "conductive" matches everything published so far (dense = low rho,
# slope_bounds negative); "resistive" is dense AND resistive (an intrusion,
# not a sulphide) which the published slope_bounds = (-4, -0.2) would forbid
# the network from fitting at all, so it needs the mirrored bound below.
# thickness / x0 / z0 unchanged from the published truth in every row.
SCENARIOS = [
    (name = "dip15_conductive", dip_deg = 15.0,
     slab_log_rho = 0.6, density = 350.0, slope_bounds = (-4.0, -0.2)),
    (name = "dip35_conductive_published", dip_deg = 35.0,
     slab_log_rho = 0.6, density = 350.0, slope_bounds = (-4.0, -0.2)),
    (name = "dip55_conductive", dip_deg = 55.0,
     slab_log_rho = 0.6, density = 350.0, slope_bounds = (-4.0, -0.2)),
    (name = "dip35_resistive", dip_deg = 35.0,
     slab_log_rho = 3.8, density = 350.0, slope_bounds = (0.2, 4.0)),
]

let raw = get(ENV, "SMARTPRIOR_SCENARIOS", "")
    if !isempty(raw)
        wanted = Set(strip.(split(raw, ",")))
        global SCENARIOS = filter(sc -> sc.name in wanted, SCENARIOS)
        isempty(SCENARIOS) && error("SMARTPRIOR_SCENARIOS matched no scenario name")
    end
end
@info "scenario sweep" WORK EPOCHS NMEMBERS length(SCENARIOS)
if SMOOTH_WEIGHT != 10.0 || REFERENCE_WEIGHT != 0.0
    @info "diagnostic weight override" SMOOTH_WEIGHT REFERENCE_WEIGHT
end
@info "running scenarios" [sc.name for sc in SCENARIOS]

#---------- mesh (same for every scenario; only the truth on it changes) ----------

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
grav_x = collect(-9000.0:500.0:9000.0)

function live_mask(arrays...)
    mask = trues(size(first(arrays)))
    for a in arrays
        mask .&= isfinite.(a)
    end
    return mask
end

function score_mu(truth_field, mu)
    mask = live_mask(truth_field, mu)
    t = truth_field[mask]
    m = mu[mask]
    m̄, t̄ = mean(m), mean(t)
    b = sum((m .- m̄) .* (t .- t̄)) / sum(abs2, m .- m̄)
    a = t̄ - b * m̄
    return (rmse = rmse(truth_field, mu),
            corr = cor(m, t),
            ols_a = a,
            ols_b = b)
end

sweep_path = joinpath(WORK, "sweep_metrics.tsv")
open(sweep_path, "w") do io
    println(io, "scenario\tdip_deg\tslab_log_rho\tslope_lo\tslope_hi\t",
            "nb_rmse\tnb_corr\tprior_rmse\tprior_corr\tprior_gain_corr\tols_b")
end

for sc in SCENARIOS
    run_dir = joinpath(WORK, sc.name)
    mkpath(run_dir)
    @info "scenario" sc.name sc.dip_deg sc.slab_log_rho sc.slope_bounds

    #---------- truth for this scenario ----------

    truth = add_dipping_slab(truth_halfspace(grid; log_rho = HOST_LOG_RHO);
                             x0 = -1000.0, z0 = 1800.0,
                             dip_deg = sc.dip_deg, thickness = 700.0,
                             log_rho = sc.slab_log_rho, density = sc.density,
                             strike = :x)

    #---------- MT observations from the 2D solver ----------

    rho_true_2d = to_mt2d(truth, mesh)
    true_model_path = write_model2d(joinpath(run_dir, "true.rho"), mesh,
                                    rho_true_2d; title = sc.name)
    template_path = write_mt2d_data_template(joinpath(run_dir, "template.ref"),
                                             mesh; impedance_error_fraction = 0.05)
    obs_path = ForwardSolve2D(true_model_path, template_path;
                              add_noise = true, rng_seed = MT_SEED,
                              output_path = joinpath(run_dir, "data.obs"))
    sites = profile_sites(mesh, load_data2d(obs_path))

    #---------- gravity observations from this scenario's density ----------

    gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                            noise = 0.1, rng = Xoshiro(GRAV_SEED))

    #---------- NB baseline and features ----------

    baseline = nb_baseline(grid, sites)
    nb_rmse = rmse(truth.log_rho, baseline)
    nb_corr = anomaly_correlation(truth.log_rho, baseline)

    stack = build_features(grid; gravity = gravity, sites = sites, baseline = baseline)
    X = encode_features(stack; n_bands = 4)
    A_grav = gravity_matrix(grid, gravity.x, gravity.y, gravity.z)
    site_cells = [(1, clamp(searchsortedlast(grid.y, y), 1, size(grid, 2)))
                  for y in sites.y]

    targets = PriorTargets(
        gravity = (A_grav, gravity.value, gravity.err),
        mt = (sites, site_cells),
        reference = vec(baseline),
        sigma_target = 0.35,
        sigma_drive = SigmaDriveConfig(),
    )

    #---------- train ----------

    net = PriorNet(size(X, 1);
                   width = 96, depth = 4,
                   log_rho_bounds = LOG_RHO_BOUNDS,
                   residual_span = 2.5,
                   sigma_bounds = (0.05, 0.9))
    config = TrainConfig(
        epochs = EPOCHS,
        learning_rate = 3.0e-3,
        log_every = 250,
        seed = TRAIN_SEED,
        slope_bounds = sc.slope_bounds,
        weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = SMOOTH_WEIGHT,
                              sigma = 0.1, reference = REFERENCE_WEIGHT),
        checkpoint_path = joinpath(run_dir, "prior.jld2"),
        checkpoint_every = 1000,
    )
    ensemble, results = train_ensemble(net, X, grid, targets;
                                       nmembers = NMEMBERS,
                                       config = config,
                                       offset = vec(baseline))
    bundle = prior_from_ensemble(grid, ensemble, X;
                                 offset = vec(baseline),
                                 k = 2.0,
                                 log_rho_bounds = LOG_RHO_BOUNDS)
    write_prior(joinpath(run_dir, "prior"), bundle)

    #---------- score ----------

    sc_score = score_mu(truth.log_rho, bundle.mu)
    gain_corr = sc_score.corr - nb_corr
    @printf("  %-28s nb corr %.4f  prior corr %.4f  gain %+.4f  prior rmse %.4f\n",
            sc.name, nb_corr, sc_score.corr, gain_corr, sc_score.rmse)

    open(joinpath(run_dir, "metrics.txt"), "w") do io
        println(io, "scenario\t", sc.name)
        println(io, "dip_deg\t", sc.dip_deg)
        println(io, "slab_log_rho\t", sc.slab_log_rho)
        println(io, "slope_bounds\t", sc.slope_bounds)
        println(io, "nb_rmse\t", nb_rmse)
        println(io, "nb_corr\t", nb_corr)
        println(io, "prior_rmse\t", sc_score.rmse)
        println(io, "prior_corr\t", sc_score.corr)
        println(io, "prior_gain_corr\t", gain_corr)
        println(io, "ols_a\t", sc_score.ols_a)
        println(io, "ols_b\t", sc_score.ols_b)
        println(io, "epochs\t", EPOCHS)
        println(io, "nmembers\t", NMEMBERS)
    end
    open(sweep_path, "a") do io
        @printf(io, "%s\t%.1f\t%.2f\t%.2f\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f\t%.6f\n",
                sc.name, sc.dip_deg, sc.slab_log_rho,
                sc.slope_bounds[1], sc.slope_bounds[2],
                nb_rmse, nb_corr, sc_score.rmse, sc_score.corr, gain_corr,
                sc_score.ols_b)
    end
end

println()
println("wrote ", sweep_path)
println("sanity check: dip35_conductive_published prior_corr should land near 0.438",
        " (ARA_RAPOR.md t1) -- if it does not, this script's pipeline has",
        " diverged from compare_prior_2d.jl and the other rows are not yet trustworthy.")
