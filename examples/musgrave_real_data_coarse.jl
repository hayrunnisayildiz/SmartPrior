# Train a smart prior on the Musgrave 2-D profile (WA55–SA347).
#
# One-off coarse-mesh copy of examples/musgrave_real_data.jl. The only
# intended difference is y_core_cell = 20000.0 (instead of 5000.0).
# build_musgrave_mesh is not modified; it hardcodes 5000.0 and takes no
# kwargs, so the BuildMesh2D call is inlined here. Outputs go to
# tmp_musgrave_coarse/, not tmp_musgrave/.
#
# Parallel to examples/compare_prior_2d.jl, but the observations come from the
# field adapters rather than a synthetic truth: AusLAMP EDI soundings, a Bouguer
# corridor CSV, and a DEM sampled onto the mesh. There is no truth to score
# against; the report is whether training produced a finite field and how far
# that field moved from the Niblett-Bostick baseline.
#
# This is the first training path that supplies surface_z to build_features.
# Slope bounds are positive: clay cover / basin fill is conductive and less
# dense than the host, the opposite of the sulphide slab in the synthetic
# comparison (which used slope_bounds = (-4, -0.2)).
#
# Run:  julia --project=. examples/musgrave_real_data_coarse.jl
# Smoke-test defaults (a few minutes, one member): SMARTPRIOR_EPOCHS=300,
# SMARTPRIOR_NMEMBERS=1. A full survey run is 3000 / 5 via those env vars.
#
# Optional env:
#   SMARTPRIOR_WORK, SMARTPRIOR_EPOCHS, SMARTPRIOR_NMEMBERS,
#   SMARTPRIOR_GRAVITY_CSV, SMARTPRIOR_DEM

using SmartPriorMT
using MTGeophysics: BuildMesh2D
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_musgrave_coarse"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "300"))
const NMEMBERS = parse(Int, get(ENV, "SMARTPRIOR_NMEMBERS", "1"))
const GRAVITY_CSV = get(ENV, "SMARTPRIOR_GRAVITY_CSV",
                        joinpath(ROOT, "gravity_profile_corridor.csv"))
const DEM_PATH = get(ENV, "SMARTPRIOR_DEM",
                     joinpath(ROOT, "data_aust",
                              "appRasterSelectAPIService1788341855327-480261704.tif"))
mkpath(WORK)
@info "working directory" WORK EPOCHS NMEMBERS

#---------- 1. mesh ----------
# Same as build_musgrave_mesh, except y_core_cell = 20000.0.

sites = build_musgrave_profile()
frequencies = collect(1 ./ sites.periods)
issorted(frequencies) || reverse!(frequencies)
mesh = BuildMesh2D(;
    frequencies = frequencies,
    y_core_range = (-50000.0, 494727.4),
    y_core_cell = 20000.0,
    y_padding = 100000.0,
    air_cells = 6,
    ground_layers = vcat(fill(200.0, 10), fill(1000.0, 10),
                         fill(5000.0, 10), fill(20000.0, 10)),
    receiver_positions = sites.y,
)
grid = grid_from_mt2dmesh(mesh)
@info "prior grid" size(grid) ncells(grid)

#---------- 2. MT profile ----------

@info "MT sites" nsites(sites) nperiods = length(sites.periods)

#---------- 3. gravity corridor ----------

gravity = build_musgrave_gravity(GRAVITY_CSV)
@printf("gravity: %d stations, %.2f to %.2f mGal, err %.3f to %.3f mGal (mean %.3f)\n",
        length(gravity.value), minimum(gravity.value), maximum(gravity.value),
        extrema(gravity.err)..., mean(gravity.err))

#---------- 4. DEM onto the profile ----------

surface_z = build_musgrave_surface_z(mesh, DEM_PATH)
@printf("surface_z: %s, %.1f to %.1f m\n",
        string(size(surface_z)), minimum(surface_z), maximum(surface_z))

#---------- 5–7. features ----------

baseline = nb_baseline(grid, sites)

stack = build_features(grid;
                       gravity = gravity,
                       sites = sites,
                       baseline = baseline,
                       surface_z = surface_z)
X = encode_features(stack; n_bands = 4)
@info "features" nchannels(stack) size(X) names = stack.names

#---------- 8–10. targets ----------

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

#---------- 11–13. train ----------

# residual_span = 2.0: two decades of departure from the Niblett-Bostick
# baseline, which is the range log_rho_bounds = (0.5, 4.5) can actually host.
net = PriorNet(size(X, 1);
               width = 96, depth = 4,
               log_rho_bounds = (0.5, 4.5),
               residual_span = 2.0,
               sigma_bounds = (0.05, 0.9))

config = TrainConfig(
    epochs = EPOCHS,
    learning_rate = 3.0e-3,
    log_every = 50,
    seed = 2026,
    slope_bounds = (0.0, 1.5),
    weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = 50.0, sigma = 0.1,
                          reference = 0.1),
    checkpoint_path = joinpath(WORK, "prior.jld2"),
    checkpoint_every = 1000,
)

damping_warns = !isnothing(targets.reference) && config.weights.reference <= 0

ensemble, results = train_ensemble(net, X, grid, targets;
                                   nmembers = NMEMBERS,
                                   config = config,
                                   offset = vec(baseline))

#---------- 14. bundle ----------

bundle = prior_from_ensemble(grid, ensemble, X;
                             offset = vec(baseline),
                             k = 2.0,
                             log_rho_bounds = (0.5, 4.5))
paths = write_prior(joinpath(WORK, "prior"), bundle)
@info "prior written" paths.rho paths.lo paths.hi

#---------- report (no truth) ----------

@show any(isnan, bundle.mu) any(isnan, bundle.sigma)

last_hist = results[1].history[end]

println()
println(repeat("=", 62))
println("  musgrave real-data prior  (no truth, coarse mesh)")
println(repeat("=", 62))
@printf("epochs / members             %d / %d\n", EPOCHS, NMEMBERS)
@printf("last epoch                   %d\n", last_hist.epoch)
@printf("last-epoch unweighted terms  grav %.3f  mt %.3f  smooth %.3e  sigma %.3f  ref %.3f\n",
        last_hist.gravity, last_hist.mt, last_hist.smooth, last_hist.sigma,
        last_hist.reference)
@printf("last-epoch total             %.5e  slope %+.3f  sat %.3f\n",
        last_hist.total, last_hist.slope, last_hist.saturation)
@printf("mean |μ − baseline|          %.4f decades\n", mae(bundle.mu, baseline))
@printf("corr(μ, baseline)            %.4f\n", anomaly_correlation(bundle.mu, baseline))
@printf("damping warning fired        %s\n", damping_warns ? "yes" : "no")
println(repeat("=", 62))

println()
println("per-member  mean|μ − baseline|  corr(μ, baseline)")
off = vec(baseline)
for (k, m) in enumerate(ensemble.members)
    (mu, _), _ = predict(net, X, m.params.net, m.state; offset = off)
    mu3 = reshape(mu, size(grid))
    @printf("  member %d  seed %d  mae %.4f  corr %.4f\n",
            k, config.seed + k - 1, mae(mu3, baseline),
            anomaly_correlation(mu3, baseline))
end

metrics_path = joinpath(WORK, "metrics.txt")
open(metrics_path, "w") do io
    println(io, "epochs\t", EPOCHS)
    println(io, "nmembers\t", NMEMBERS)
    println(io, "train_seed\t", config.seed)
    println(io, "y_core_cell\t", 20000.0)
    println(io, "nan_mu\t", any(isnan, bundle.mu))
    println(io, "nan_sigma\t", any(isnan, bundle.sigma))
    println(io, "last_epoch\t", last_hist.epoch)
    println(io, "last_total\t", last_hist.total)
    println(io, "last_gravity\t", last_hist.gravity)
    println(io, "last_mt\t", last_hist.mt)
    println(io, "last_smooth\t", last_hist.smooth)
    println(io, "last_sigma\t", last_hist.sigma)
    println(io, "last_reference\t", last_hist.reference)
    println(io, "last_slope\t", last_hist.slope)
    println(io, "last_saturation\t", last_hist.saturation)
    println(io, "mae_mu_baseline\t", mae(bundle.mu, baseline))
    println(io, "corr_mu_baseline\t", anomaly_correlation(bundle.mu, baseline))
    println(io, "damping_warning\t", damping_warns)
    for (k, m) in enumerate(ensemble.members)
        (mu, _), _ = predict(net, X, m.params.net, m.state; offset = off)
        mu3 = reshape(mu, size(grid))
        println(io, "member$(k)_mae\t", mae(mu3, baseline))
        println(io, "member$(k)_corr\t", anomaly_correlation(mu3, baseline))
        println(io, "member$(k)_seed\t", config.seed + k - 1)
    end
end
@info "metrics written" metrics_path
@info "all outputs kept" WORK
