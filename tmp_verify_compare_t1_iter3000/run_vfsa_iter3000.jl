# One-off t1 VFSA with max_iter=3000. Does not modify examples/compare_prior_2d.jl.
# Reuses t1 start models and observations so only the VFSA schedule changes.

using MTGeophysics
using Dates
using Printf

const WORK = @__DIR__
const VFSA_SEED = 4242

inv_config = VFSA2DMTConfig(
    n_chains = 2,
    n_ctrl = 250,
    max_iter = 3000,
    log_bounds = (0.0, 4.0),
    step_scale = 0.11,
    cool_ratio = 1.0e-3,
    target_rms = 1.0,
    seed = VFSA_SEED,
    keep_models = false,
    output_root = WORK,
)

half_path = joinpath(WORK, "start_half.rho")
prior_path = joinpath(WORK, "start_prior.rho")
obs_path = joinpath(WORK, "data.obs")
true_model_path = joinpath(WORK, "true.rho")

open(joinpath(WORK, "timing.txt"), "w") do io
    println(io, "start\t", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
end

t_all = time()

@info "inverting from the half-space" max_iter = inv_config.max_iter
t_half = time()
run_half = VFSA2DMT(half_path, obs_path;
                    run_dir = joinpath(WORK, "inv_half"),
                    true_model_path = true_model_path,
                    config = inv_config)
half_s = time() - t_half

@info "inverting from the smart prior" max_iter = inv_config.max_iter
t_prior = time()
run_prior = VFSA2DMT(prior_path, obs_path;
                     run_dir = joinpath(WORK, "inv_prior"),
                     true_model_path = true_model_path,
                     config = inv_config)
prior_s = time() - t_prior

all_s = time() - t_all

open(joinpath(WORK, "timing.txt"), "a") do io
    @printf(io, "half_s\t%.3f\n", half_s)
    @printf(io, "prior_s\t%.3f\n", prior_s)
    @printf(io, "total_s\t%.3f\n", all_s)
    println(io, "end\t", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    @printf(io, "half_best_rms\t%.6f\n", run_half.best_chain.best_rms)
    @printf(io, "prior_best_rms\t%.6f\n", run_prior.best_chain.best_rms)
end

@printf("wall half  %.1f min\n", half_s / 60)
@printf("wall prior %.1f min\n", prior_s / 60)
@printf("wall total %.1f min\n", all_s / 60)
@printf("best RMS half  %.4f\n", run_half.best_chain.best_rms)
@printf("best RMS prior %.4f\n", run_prior.best_chain.best_rms)
