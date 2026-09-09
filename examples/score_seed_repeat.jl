# Score one compare_prior_2d.jl working directory: prior vs truth, both VFSA
# chains, and the distribution-based dip (lowest 10% of live cells).
#
# Run:  SMARTPRIOR_WORK=tmp_noisy julia --project=. examples/score_seed_repeat.jl
# Does not train or invert. Writes metrics.txt next to the run.

using SmartPriorMT
using MTGeophysics
using LinearAlgebra
using Printf
using Statistics

const WORK = get(ENV, "SMARTPRIOR_WORK") do
    error("score_seed_repeat.jl needs SMARTPRIOR_WORK")
end
isdir(WORK) || error("working directory does not exist: $WORK")

#---------- mesh and truth (identical to compare_prior_2d.jl) ----------

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
truth = add_dipping_slab(truth_halfspace(grid; log_rho = 2.6);
                         x0 = -1000.0, z0 = 1800.0,
                         dip_deg = 35.0, thickness = 700.0,
                         log_rho = 0.6, density = 350.0,
                         strike = :x)
tfield = truth.log_rho

#---------- helpers ----------

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

# lowest `p` of finite cells; level-independent, matches the published dip definition
function select_lowest(field, p::Real)
    idxs = [i for i in eachindex(field) if isfinite(field[i])]
    n_keep = clamp(round(Int, p * length(idxs)), 2, length(idxs))
    order = sort(idxs; by = i -> (field[i], i))
    mask = falses(size(field))
    for i in @view order[1:n_keep]
        mask[i] = true
    end
    return mask, field[order[n_keep]], n_keep
end

function slab_geometry(field, mask)
    ys, zs = Float64[], Float64[]
    for j in 1:size(grid, 2), k in 1:size(grid, 3)
        mask[1, j, k] || continue
        push!(ys, grid.cy[j])
        push!(zs, grid.cz[k])
    end
    n = length(ys)
    n < 2 && return (depth_m = NaN, dip_deg = NaN, n = n, r2 = NaN)
    A = [ys ones(n)]
    coef = A \ zs
    pred = A * coef
    ss_res = sum(abs2, zs .- pred)
    ss_tot = sum(abs2, zs .- mean(zs))
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    return (depth_m = coef[2], dip_deg = rad2deg(atan(abs(coef[1]))), n = n, r2 = r2)
end

function load_logrho(path)
    return from_mt2d(load_model2d(path).resistivity, mesh)
end

function parse_rms_best(log_path, iter::Int)
    isfile(log_path) || return NaN
    for line in eachline(log_path)
        startswith(strip(line), "#") && continue
        occursin("Chain", line) && continue
        occursin("---", line) && continue
        toks = split(line)
        length(toks) < 9 && continue
        it = tryparse(Int, toks[2])
        it == iter || continue
        return parse(Float64, toks[9])
    end
    return NaN
end

function parse_acceptance(summary_path, chain_id::Int)
    isfile(summary_path) || return NaN
    pat = Regex("chain $(chain_id): .*acceptance_fraction=([0-9.]+)")
    for line in eachline(summary_path)
        m = match(pat, line)
        m === nothing || return parse(Float64, m.captures[1])
    end
    return NaN
end

dip10(field) = slab_geometry(field, first(select_lowest(field, 0.10))).dip_deg

#---------- prior / baseline ----------

obs = load_data2d(joinpath(WORK, "data.obs"))
sites = profile_sites(mesh, obs)
baseline = nb_baseline(grid, sites)
nb_rmse = rmse(tfield, baseline)
nb_corr = anomaly_correlation(tfield, baseline)

mu = load_logrho(joinpath(WORK, "start_prior.rho"))
a_ols, b_ols = ols_truth_on_mu(tfield, mu)
prior_rmse = rmse(tfield, mu)
prior_corr = anomaly_correlation(tfield, mu)

#---------- inversions: both chains ----------

half = Dict{Int,Any}()
prior = Dict{Int,Any}()
for c in (1, 2)
    hmod = load_logrho(joinpath(WORK, "inv_half", "model.c$(c)best"))
    pmod = load_logrho(joinpath(WORK, "inv_prior", "model.c$(c)best"))
    hlog = joinpath(WORK, "inv_half", "chain_0$c", "0vfsa2DMT.log")
    plog = joinpath(WORK, "inv_prior", "chain_0$c", "0vfsa2DMT.log")
    hsum = joinpath(WORK, "inv_half", "Summary.md")
    psum = joinpath(WORK, "inv_prior", "Summary.md")
    half[c] = (
        rms1 = parse_rms_best(hlog, 1),
        rms400 = parse_rms_best(hlog, 400),
        acc = parse_acceptance(hsum, c),
        dip = dip10(hmod),
    )
    prior[c] = (
        rms1 = parse_rms_best(plog, 1),
        rms400 = parse_rms_best(plog, 400),
        acc = parse_acceptance(psum, c),
        dip = dip10(pmod),
    )
end

truth_dip = dip10(tfield)

#---------- write ----------

metrics_path = joinpath(WORK, "metrics.txt")
open(metrics_path, "w") do io
    println(io, "nb_rmse\t", nb_rmse)
    println(io, "nb_corr\t", nb_corr)
    println(io, "prior_ols_a\t", a_ols)
    println(io, "prior_ols_b\t", b_ols)
    println(io, "prior_rmse\t", prior_rmse)
    println(io, "prior_corr\t", prior_corr)
    println(io, "truth_dip10\t", truth_dip)
    for c in (1, 2)
        println(io, "half_c$(c)_rms1\t", half[c].rms1)
        println(io, "half_c$(c)_rms400\t", half[c].rms400)
        println(io, "half_c$(c)_acc\t", half[c].acc)
        println(io, "half_c$(c)_dip10\t", half[c].dip)
        println(io, "prior_c$(c)_rms1\t", prior[c].rms1)
        println(io, "prior_c$(c)_rms400\t", prior[c].rms400)
        println(io, "prior_c$(c)_acc\t", prior[c].acc)
        println(io, "prior_c$(c)_dip10\t", prior[c].dip)
    end
    println(io, "dip_diff_c1\t", prior[1].dip - half[1].dip)
    println(io, "dip_diff_c2\t", prior[2].dip - half[2].dip)
end

println(repeat("=", 72))
@printf("score  WORK = %s\n", WORK)
println(repeat("=", 72))
@printf("NB baseline          RMSE %.3f   corr %.3f\n", nb_rmse, nb_corr)
@printf("prior OLS            a = %+.4f   b = %+.4f\n", a_ols, b_ols)
@printf("prior ham            RMSE %.3f   corr %.3f\n", prior_rmse, prior_corr)
@printf("truth dip (lowest 10%%)  %.1f deg\n", truth_dip)
println()
@printf("%-22s %10s %10s %10s %10s\n", "", "half c1", "half c2", "prior c1", "prior c2")
@printf("%-22s %10.3f %10.3f %10.3f %10.3f\n", "RMS iter 1",
        half[1].rms1, half[2].rms1, prior[1].rms1, prior[2].rms1)
@printf("%-22s %10.3f %10.3f %10.3f %10.3f\n", "RMS iter 400",
        half[1].rms400, half[2].rms400, prior[1].rms400, prior[2].rms400)
@printf("%-22s %10.1f %10.1f %10.1f %10.1f\n", "dip lowest 10% (deg)",
        half[1].dip, half[2].dip, prior[1].dip, prior[2].dip)
@printf("%-22s %10.3f %10.3f %10.3f %10.3f\n", "acceptance",
        half[1].acc, half[2].acc, prior[1].acc, prior[2].acc)
@printf("\ndip difference (prior − half):  c1 %+.1f deg   c2 %+.1f deg\n",
        prior[1].dip - half[1].dip, prior[2].dip - half[2].dip)
println(repeat("=", 72))
@info "metrics written" metrics_path
