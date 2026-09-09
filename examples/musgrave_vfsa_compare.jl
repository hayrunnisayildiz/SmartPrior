# Smoke-test VFSA from a half-space vs the Musgrave smart prior.
#
# Parallel to examples/compare_prior_2d.jl, but there is no truth: the
# observations are the TE-only AusLAMP file written here by
# build_musgrave_datafile2d() (default model_err_frac=0.4; TM was examined
# and left out of v0.1.0 — see README). The prior is loaded from the
# 5-member ensemble already on disk (tmp_musgrave/prior/prior.rho).
# VFSA2DMT accepts true_model_path=nothing; this script omits the keyword.
# The public VFSA2DMT wrapper then plots data maps and crashes on TE-only
# NaN TM, so the call is MTGeophysics.run_mt2d_vfsa — the same inversion,
# without that plot.
#
# max_iter=100 is deliberate. This is a smoke test, not the published
# comparison; a 400-iteration run is a later step.
#
# Run:  julia --project=. examples/musgrave_vfsa_compare.jl
#
# Optional env: SMARTPRIOR_WORK (default tmp_musgrave next to this repo).

using SmartPriorMT
using MTGeophysics
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_musgrave"))
const OBS_PATH = joinpath(WORK, "musgrave_data.obs")
const PRIOR_RHO = joinpath(WORK, "prior", "prior.rho")
const MAX_ITER = 400
mkpath(WORK)

isfile(PRIOR_RHO) || error("missing $PRIOR_RHO; run examples/musgrave_real_data.jl (5 members) first")

obs = build_musgrave_datafile2d()
write_data2d(OBS_PATH, obs)
@info "working directory" WORK OBS_PATH

#---------- mesh, baseline, starts ----------

mesh = build_musgrave_mesh()
grid = grid_from_mt2dmesh(mesh)
sites = build_musgrave_profile()
baseline = nb_baseline(grid, sites)

half_start = fill(mean(vec(baseline)), size(grid))

ws = load_ws3d_model(PRIOR_RHO)
size(ws.A) == size(grid) || error(
    "prior.rho is $(size(ws.A)) but the Musgrave grid is $(size(grid))")
prior_start = ws.A

@printf("half-space start: log10 ρ = %.4f  (mean of NB baseline)\n", half_start[1])
@printf("prior μ:          %.4f to %.4f\n", extrema(filter(isfinite, prior_start))...)

half_path = write_model2d(joinpath(WORK, "start_half.rho"), mesh,
                          to_mt2d(reshape(half_start, size(grid)), mesh);
                          title = "Musgrave NB-mean half-space start")

prior_path = write_model2d(joinpath(WORK, "start_prior.rho"), mesh,
                           to_mt2d(prior_start, mesh);
                           title = "Musgrave smart prior start")

inv_config = VFSA2DMTConfig(
    n_chains = 2,
    n_ctrl = 250,
    max_iter = 400,
    log_bounds = (0.5, 4.5),
    step_scale = 0.11,
    cool_ratio = 1.0e-3,
    target_rms = 0.1,
    seed = 4242,
    keep_models = false,
    output_root = WORK,
)

# true_model_path omitted: VFSA2DMT defaults it to nothing.
#
# VFSA2DMT is the public wrapper around run_mt2d_vfsa plus Makie plots.
# plot_mt2d_data_maps cannot colour a TE-only file (rho_yx is all NaN) and
# throws after the inversion has already finished. Call the inversion
# itself; the chain records and best_rms are the same objects.

@info "inverting from the half-space"
run_half = MTGeophysics.run_mt2d_vfsa(half_path, OBS_PATH;
                                      run_dir = joinpath(WORK, "inv_half"),
                                      config = inv_config)

@info "inverting from the smart prior"
run_prior = MTGeophysics.run_mt2d_vfsa(prior_path, OBS_PATH;
                                       run_dir = joinpath(WORK, "inv_prior"),
                                       config = inv_config)

#---------- data-fit RMS per iteration, both chains ----------

rms_best_at(chain, iter) = begin
    rec = findfirst(r -> r.iteration == iter, chain.iterations)
    rec === nothing ? NaN : chain.iterations[rec].best_rms
end

half_c1, half_c2 = run_half.chains[1], run_half.chains[2]
prior_c1, prior_c2 = run_prior.chains[1], run_prior.chains[2]

table_path = joinpath(WORK, "vfsa_smoke_rms.txt")
open(table_path, "w") do io
    println(io, "iter\thalf_c1\thalf_c2\thalf_best\tprior_c1\tprior_c2\tprior_best")
    println()
    println(repeat("=", 78))
    println("  data-fit best_rms  (two chains each; half vs prior)")
    println(repeat("=", 78))
    @printf("%6s %10s %10s %10s %10s %10s %10s\n",
            "iter", "half_c1", "half_c2", "half_best",
            "prior_c1", "prior_c2", "prior_best")
    for iter in 1:MAX_ITER
        h1 = rms_best_at(half_c1, iter)
        h2 = rms_best_at(half_c2, iter)
        p1 = rms_best_at(prior_c1, iter)
        p2 = rms_best_at(prior_c2, iter)
        hb = min(h1, h2)
        pb = min(p1, p2)
        @printf("%6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
                iter, h1, h2, hb, p1, p2, pb)
        @printf(io, "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                iter, h1, h2, hb, p1, p2, pb)
    end
    println(repeat("=", 78))
    @printf("best chain overall     half %.4f   prior %.4f\n",
            run_half.best_chain.best_rms, run_prior.best_chain.best_rms)
end
@info "rms table written" table_path
@info "all outputs kept" WORK
