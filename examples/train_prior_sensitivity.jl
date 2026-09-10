# Hyperparameter sensitivity: residual_span × slope_bounds width.
#
# Same mesh / truth / features as compare_prior_2d.jl. One fixed seed; only
# residual_span and slope_bounds change. Desired shape: a broad plateau, not a
# sharp peak at the published (2.5, (−4,−0.2)) point.
#
# Default grid:
#   residual_span ∈ {1.0, 1.5, 2.0, 2.5, 3.5, 5.0}
#   slope_bounds  ∈ {(−2,−0.5), (−4,−0.2), (−10,−0.1)}  (sign fixed)
#
# Env overrides:
#   SMARTPRIOR_WORK          output root (default tmp_sensitivity)
#   SMARTPRIOR_EPOCHS        default 3000
#   SMARTPRIOR_NMEMBERS      default 5 (set 1 for a cheap scout)
#   SMARTPRIOR_SPANS         comma list, e.g. "1.0,2.0,2.5"
#   SMARTPRIOR_SLOPE_SETS    semicolon-separated lo,hi pairs, e.g. "-4,-0.2;-10,-0.1"
#   SMARTPRIOR_MT_SEED / TRAIN_SEED / GRAV_SEED / REFERENCE
#
# Writes:
#   $WORK/sweep_metrics.tsv
#   $WORK/span_<s>_slope_<lo>_<hi>/metrics.txt
#
# Run:
#   julia --project=. examples/train_prior_sensitivity.jl

using SmartPriorMT
using MTGeophysics
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_sensitivity"))
const EPOCHS = parse(Int, get(ENV, "SMARTPRIOR_EPOCHS", "3000"))
const NMEMBERS = parse(Int, get(ENV, "SMARTPRIOR_NMEMBERS", "5"))
const MT_SEED = parse(Int, get(ENV, "SMARTPRIOR_MT_SEED", "20260827"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const GRAV_SEED = parse(Int, get(ENV, "SMARTPRIOR_GRAV_SEED", "11"))
const REFERENCE_WEIGHT = parse(Float64, get(ENV, "SMARTPRIOR_REFERENCE", "0.0"))
const LOG_RHO_BOUNDS = (0.0, 4.0)
const HOST_LOG_RHO = 2.6
const SLAB_LOG_RHO = 0.6

function parse_spans(s::AbstractString)
    return [parse(Float64, strip(x)) for x in split(s, ',') if !isempty(strip(x))]
end

function parse_slope_sets(s::AbstractString)
    out = Tuple{Float64,Float64}[]
    for chunk in split(s, ';')
        isempty(strip(chunk)) && continue
        parts = split(chunk, ',')
        length(parts) == 2 || error("bad slope pair: $(chunk)")
        push!(out, (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2]))))
    end
    return out
end

const SPANS = parse_spans(get(ENV, "SMARTPRIOR_SPANS", "1.0,1.5,2.0,2.5,3.5,5.0"))
const SLOPE_SETS = parse_slope_sets(get(ENV, "SMARTPRIOR_SLOPE_SETS",
                                        "-2,-0.5;-4,-0.2;-10,-0.1"))

mkpath(WORK)
@info "sensitivity sweep" WORK EPOCHS NMEMBERS SPANS SLOPE_SETS

#---------- shared data (built once) ----------

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

rho_true_2d = to_mt2d(truth, mesh)
true_model_path = write_model2d(joinpath(WORK, "true.rho"), mesh, rho_true_2d;
                                title = "dipping conductive slab")
template_path = write_mt2d_data_template(joinpath(WORK, "template.ref"), mesh;
                                         impedance_error_fraction = 0.05)
obs_path = ForwardSolve2D(true_model_path, template_path;
                          add_noise = true, rng_seed = MT_SEED,
                          output_path = joinpath(WORK, "data.obs"))
sites = profile_sites(mesh, load_data2d(obs_path))

grav_x = collect(-9000.0:500.0:9000.0)
gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                        noise = 0.1, rng = Xoshiro(GRAV_SEED))

baseline = nb_baseline(grid, sites)
nb_rmse = rmse(truth.log_rho, baseline)
nb_corr = anomaly_correlation(truth.log_rho, baseline)
lat_std = nb_baseline_lateral_std(baseline)
span_half = residual_span_half_band(LOG_RHO_BOUNDS)
span_data = residual_span_from_baseline(baseline; k = 3.0, floor = 1.0, ceil = 5.0)
@printf("NB rmse=%.4f corr=%.4f  σ_lat=%.4f  half_band=%.3f  from_baseline=%.3f\n",
        nb_rmse, nb_corr, lat_std, span_half, span_data)

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
    println(io, "residual_span\tslope_lo\tslope_hi\tprior_rmse\tprior_corr\tols_b\tnb_rmse\tnb_corr\tlast_total")
end

for span in SPANS, (slo, shi) in SLOPE_SETS
    tag = @sprintf("span_%.1f_slope_%.1f_%.1f", span, slo, shi)
    run_dir = joinpath(WORK, tag)
    mkpath(run_dir)
    @info "training cell" span slo shi run_dir

    net = PriorNet(size(X, 1);
                   width = 96, depth = 4,
                   log_rho_bounds = LOG_RHO_BOUNDS,
                   residual_span = span,
                   sigma_bounds = (0.05, 0.9))
    config = TrainConfig(
        epochs = EPOCHS,
        learning_rate = 3.0e-3,
        log_every = 250,
        seed = TRAIN_SEED,
        slope_bounds = (slo, shi),
        weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = 10.0, sigma = 0.1,
                              reference = REFERENCE_WEIGHT),
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

    sc = score_mu(truth.log_rho, bundle.mu)
    last_hist = results[1].history[end]
    open(joinpath(run_dir, "metrics.txt"), "w") do io
        println(io, "residual_span\t", span)
        println(io, "slope_lo\t", slo)
        println(io, "slope_hi\t", shi)
        println(io, "prior_rmse\t", sc.rmse)
        println(io, "prior_corr\t", sc.corr)
        println(io, "ols_a\t", sc.ols_a)
        println(io, "ols_b\t", sc.ols_b)
        println(io, "nb_rmse\t", nb_rmse)
        println(io, "nb_corr\t", nb_corr)
        println(io, "last_total\t", last_hist.total)
        println(io, "epochs\t", EPOCHS)
        println(io, "nmembers\t", NMEMBERS)
        println(io, "train_seed\t", TRAIN_SEED)
    end
    open(sweep_path, "a") do io
        @printf(io, "%.4f\t%.4f\t%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\n",
                span, slo, shi, sc.rmse, sc.corr, sc.ols_b, nb_rmse, nb_corr,
                last_hist.total)
    end
    @printf("DONE span=%.1f slope=(%.1f,%.1f)  rmse=%.4f  corr=%.4f  b=%.3f\n",
            span, slo, shi, sc.rmse, sc.corr, sc.ols_b)
end

@info "sweep complete" sweep_path
