# Shared code for the Keivitsa examples: variograms, ordinary kriging,
# the pointwise Lux field (build_mlp / train_member / ensemble_predict),
# covariate matrices, prediction bookkeeping, and bootstrap intervals.
#
# Not a runnable script. Include it after setting ENV["SMARTPRIOR_WORK"]:
#   include(joinpath(@__DIR__, "keivitsa_common.jl"))
#
# Hyperparameters below are fixed; the scripts that include this file do
# not retune them.

using SmartPrior
using Lux
using Lux: Chain, Dense, gelu, softplus
using Optimisers
using Zygote
using GeoStats
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_feasibility"))

# Fixed procedure. Not tuned on the results of this run.
const N_LAGS = 15
const RANGE_MIN_M = 10.0
const HORIZONTAL_DIP_DEG = 22.5
const MIN_PAIRS = 30
const DEDUPE_TOL_M = 1.0e-3
const IDW_POWER = 2
const N_NEIGHBORS = 16
const FOURIER_BANDS = 16
const FOURIER_SCALE_MIN = 1.0
const FOURIER_SCALE_MAX = 16.0
const MLP_WIDTH = 64
const MLP_DEPTH = 3
const NN_LR = 1.0e-3
const NN_WEIGHT_DECAY = 1.0e-4
const NN_EPOCHS = 2000
const NN_PATIENCE = 200
const NN_SEEDS = (1, 2, 3, 4, 5)
const VAL_HOLE_FRACTION = 0.20
const SIGMA_FLOOR = 1.0e-3
const Z90 = 1.6448536269514722   # Φ^{-1}(0.95)
const N_BOOT = 2000

const LOG = Ref{IO}()

function logmsg(msg::AbstractString)
    println(msg)
    flush(stdout)
    if isassigned(LOG)
        io = LOG[]
        println(io, msg)
        flush(io)
    end
    return nothing
end

function fail(msg::AbstractString)
    logmsg("ABORT: " * msg)
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

function spec_of(table, prop::Symbol)
    for s in table.specs
        s.name === prop && return s
    end
    fail("no property $(prop)")
end

function transformed(spec, values::AbstractVector)
    if spec.transform === :identity
        return collect(Float64, values)
    elseif spec.transform === :log10
        any(v -> !(isfinite(v) && v > 0), values) && fail(
            "$(spec.name): log10 transform needs finite positive values")
        return log10.(Float64.(values))
    else
        fail("unsupported transform $(spec.transform)")
    end
end

#---------- variogram ----------

function model_gamma(name::AbstractString, h::Real, range::Real, partial::Real, nugget::Real)
    hh = Float64(h)
    hh <= 0 && return 0.0
    a = max(Float64(range), 1.0e-6)
    p = Float64(partial)
    n = Float64(nugget)
    if name == "exponential"
        return n + p * (1 - exp(-3 * hh / a))
    end
    hh >= a && return n + p
    r = hh / a
    return n + p * (1.5 * r - 0.5 * r^3)
end

function collect_pairs(x, y, z, v, hole; maxlag_h, maxlag_v, dip_deg)
    dh = Float64[]
    gh = Float64[]
    dv = Float64[]
    gv = Float64[]
    n = length(v)
    @inbounds for j in 2:n
        xj, yj, zj, vj, hj = x[j], y[j], z[j], v[j], hole[j]
        for i in 1:(j - 1)
            dx = xj - x[i]
            dy = yj - y[i]
            dz = zj - z[i]
            horiz = hypot(dx, dy)
            d3 = hypot(horiz, dz)
            sv = 0.5 * (v[i] - vj)^2
            if !isempty(hj) && hj == hole[i] && d3 > 0 && d3 <= maxlag_v
                push!(dv, d3)
                push!(gv, sv)
            end
            if horiz > 0 && horiz <= maxlag_h && atand(abs(dz), horiz) <= dip_deg
                push!(dh, horiz)
                push!(gh, sv)
            end
        end
    end
    return dh, gh, dv, gv
end

function bin_experimental(dist, semi, n_lags, maxlag)
    maxlag > 0 || return Float64[], Float64[]
    edges = range(0.0, Float64(maxlag); length = n_lags + 1)
    centers = Float64[]
    experimental = Float64[]
    for k in 1:n_lags
        lo, hi = edges[k], edges[k + 1]
        s = 0.0
        c = 0
        @inbounds for t in eachindex(dist)
            d = dist[t]
            if d > lo && d <= hi
                s += semi[t]
                c += 1
            end
        end
        if c > 0
            push!(centers, 0.5 * (lo + hi))
            push!(experimental, s / c)
        end
    end
    return centers, experimental
end

# Same result as `collect_pairs` followed by `bin_experimental`, without
# storing every pair. Each bin sums its pairs in the same order as the two-step
# version, so the bins are bit-identical. Memory is O(n_lags), not O(n^2):
# the stored-pair version runs out of memory on dense tables (Keivitsa density).
function binned_variogram(x, y, z, v, hole; maxlag_h, maxlag_v, dip_deg, n_lags)
    # Collected so the bin search compares the same edge values `bin_experimental` indexes.
    edges_h = collect(range(0.0, Float64(maxlag_h); length = n_lags + 1))
    edges_v = collect(range(0.0, Float64(maxlag_v); length = n_lags + 1))
    sum_h = zeros(Float64, n_lags)
    cnt_h = zeros(Int, n_lags)
    sum_v = zeros(Float64, n_lags)
    cnt_v = zeros(Int, n_lags)
    n = length(v)
    @inbounds for j in 2:n
        xj, yj, zj, vj, hj = x[j], y[j], z[j], v[j], hole[j]
        for i in 1:(j - 1)
            dx = xj - x[i]
            dy = yj - y[i]
            dz = zj - z[i]
            horiz = hypot(dx, dy)
            d3 = hypot(horiz, dz)
            sv = 0.5 * (v[i] - vj)^2
            if !isempty(hj) && hj == hole[i] && d3 > 0 && d3 <= maxlag_v
                k = _lag_bin(edges_v, d3, n_lags)
                if k > 0
                    sum_v[k] += sv
                    cnt_v[k] += 1
                end
            end
            if horiz > 0 && horiz <= maxlag_h && atand(abs(dz), horiz) <= dip_deg
                k = _lag_bin(edges_h, horiz, n_lags)
                if k > 0
                    sum_h[k] += sv
                    cnt_h[k] += 1
                end
            end
        end
    end
    bins_h, exp_h = _bins_from_sums(edges_h, sum_h, cnt_h, n_lags)
    bins_v, exp_v = _bins_from_sums(edges_v, sum_v, cnt_v, n_lags)
    return bins_h, exp_h, bins_v, exp_v, sum(cnt_h), sum(cnt_v)
end

# The bin k with edges[k] < d <= edges[k + 1], as in `bin_experimental`; 0 if none.
function _lag_bin(edges, d, n_lags)
    i = searchsortedfirst(edges, d)
    k = i - 1
    (1 <= k <= n_lags && edges[k] < d && d <= edges[k + 1]) || return 0
    return k
end

function _bins_from_sums(edges, s, c, n_lags)
    centers = Float64[]
    experimental = Float64[]
    for k in 1:n_lags
        if c[k] > 0
            push!(centers, 0.5 * (edges[k] + edges[k + 1]))
            push!(experimental, s[k] / c[k])
        end
    end
    return centers, experimental
end

function sse_variogram(name, bins_h, exp_h, bins_v, exp_v, p)
    ah, av, partial, nugget = p
    s = 0.0
    @inbounds for i in eachindex(bins_h)
        s += (model_gamma(name, bins_h[i], ah, partial, nugget) - exp_h[i])^2
    end
    @inbounds for i in eachindex(bins_v)
        s += (model_gamma(name, bins_v[i], av, partial, nugget) - exp_v[i])^2
    end
    return s
end

function refine_variogram!(p, lo, hi, sill_cap, sse)
    best = sse(p)
    step = 0.35
    for _ in 1:120
        improved = false
        for j in 1:4
            for sgn in (-1.0, 1.0)
                cand = copy(p)
                cand[j] = j == 4 ?
                          clamp(p[j] + sgn * step * sill_cap, lo[j], hi[j]) :
                          clamp(p[j] * exp(sgn * step), lo[j], hi[j])
                ss = sse(cand)
                if ss + 1.0e-15 < best
                    best = ss
                    p .= cand
                    improved = true
                end
            end
        end
        improved || (step *= 0.5)
        step < 1.0e-4 && break
    end
    return best
end

function fit_one_model(name, bins_h, exp_h, bins_v, exp_v, vvar, rmin, rmax_h, rmax_v)
    sill_cap = max(2 * max(vvar, 1.0e-8), maximum(exp_h), maximum(exp_v), 1.0e-6)
    lo = [rmin, rmin, 1.0e-8, 0.0]
    hi = [max(rmax_h, rmin * 1.01), max(rmax_v, rmin * 1.01), sill_cap, sill_cap]
    function clamp_start(ah, av, partial, nugget)
        return [clamp(ah, lo[1], hi[1]), clamp(av, lo[2], hi[2]),
                clamp(partial, lo[3], hi[3]), clamp(nugget, lo[4], hi[4])]
    end
    nug0 = clamp(exp_h[1] * 0.5, 0.0, sill_cap)
    starts = [
        clamp_start(0.30 * bins_h[end], 0.30 * bins_v[end], 0.80 * vvar, 0.10 * vvar),
        clamp_start(0.80 * bins_h[end], 0.40 * bins_v[end], 0.50 * vvar, 0.30 * vvar),
        clamp_start(bins_h[end], bins_v[end], vvar, 0.05 * vvar),
        clamp_start(0.50 * hi[1], 0.50 * hi[2], max(maximum(exp_h), vvar), nug0),
    ]
    sse = p -> sse_variogram(name, bins_h, exp_h, bins_v, exp_v, p)
    best_p = copy(starts[1])
    best_s = Inf
    for p0 in starts
        p = copy(p0)
        ss = refine_variogram!(p, lo, hi, sill_cap, sse)
        if ss < best_s
            best_s = ss
            best_p = copy(p)
        end
    end
    nb = length(bins_h) + length(bins_v)
    rmse = sqrt(best_s / nb)
    return best_p, rmse
end

struct VarioFit
    model::String
    nugget::Float64
    partial_sill::Float64
    total_sill::Float64
    range_h::Float64
    range_v::Float64
    fit_rmse::Float64
    n_pairs_h::Int
    n_pairs_v::Int
    n_bins_h::Int
    n_bins_v::Int
    degenerate::Bool
end

function dedupe_locations(x, y, z, v, hole; tol = DEDUPE_TOL_M)
    n = length(v)
    used = falses(n)
    ox = Float64[]
    oy = Float64[]
    oz = Float64[]
    ov = Float64[]
    oh = String[]
    n_merged = 0
    for i in 1:n
        used[i] && continue
        acc = v[i]
        cnt = 1
        holes = Set{String}([hole[i]])
        used[i] = true
        for j in (i + 1):n
            used[j] && continue
            if hypot(x[j] - x[i], y[j] - y[i], z[j] - z[i]) <= tol
                acc += v[j]
                cnt += 1
                push!(holes, hole[j])
                used[j] = true
            end
        end
        cnt > 1 && (n_merged += 1)
        push!(ox, x[i])
        push!(oy, y[i])
        push!(oz, z[i])
        push!(ov, acc / cnt)
        # A point mixed from two holes is not a downhole sample.
        push!(oh, length(holes) == 1 ? first(holes) : "")
    end
    return ox, oy, oz, ov, oh, n_merged
end

function fit_variogram(x, y, z, v, hole; context::AbstractString)
    n = length(v)
    n >= 3 || fail("$context: fewer than 3 locations for a variogram")
    hspan = max(maximum(x) - minimum(x), maximum(y) - minimum(y))
    vspan = maximum(z) - minimum(z)
    maxlag_h = max(10.0, 0.5 * hspan)
    maxlag_v = max(10.0, 0.5 * vspan)
    bins_h, exp_h, bins_v, exp_v, n_pairs_h, n_pairs_v =
        binned_variogram(x, y, z, v, hole;
                         maxlag_h = maxlag_h, maxlag_v = maxlag_v,
                         dip_deg = HORIZONTAL_DIP_DEG, n_lags = N_LAGS)
    if n_pairs_h < MIN_PAIRS || n_pairs_v < MIN_PAIRS
        fail("$context: variogram pair count below $(MIN_PAIRS) " *
             "(horizontal $(n_pairs_h), downhole $(n_pairs_v); " *
             "maxlag_h $(maxlag_h) m, maxlag_v $(maxlag_v) m)")
    end
    if length(bins_h) < 3 || length(bins_v) < 3
        fail("$context: fewer than 3 occupied variogram bins " *
             "(horizontal $(length(bins_h)), downhole $(length(bins_v)))")
    end
    vvar = var(v; corrected = false)
    rmax_h = max(bins_h[end] * 2, RANGE_MIN_M * 4)
    rmax_v = max(bins_v[end] * 2, RANGE_MIN_M * 4)
    best = nothing
    best_rmse = Inf
    for name in ("spherical", "exponential")
        p, rmse = fit_one_model(name, bins_h, exp_h, bins_v, exp_v, vvar,
                                RANGE_MIN_M, rmax_h, rmax_v)
        all(isfinite, p) && isfinite(rmse) || fail(
            "$context: $(name) variogram fit is not finite")
        if rmse < best_rmse
            best_rmse = rmse
            best = (name = name, p = p, rmse = rmse)
        end
    end
    best === nothing && fail("$context: no variogram model fitted")
    nugget = best.p[4]
    partial = best.p[3]
    total = nugget + partial
    total > 0 || fail("$context: variogram total sill is not positive")
    ratio = partial / total
    return VarioFit(best.name, nugget, partial, total, best.p[1], best.p[2],
                    best.rmse, n_pairs_h, n_pairs_v, length(bins_h), length(bins_v),
                    ratio < 0.05)
end

#---------- baselines ----------

function idw_predict(tx, ty, tz, tv, qx, qy, qz, ratio)
    ntr = length(tv)
    nq = length(qx)
    ntr >= 1 || fail("IDW has no training locations")
    ratio > 0 && isfinite(ratio) || fail("IDW anisotropy ratio is not positive and finite")
    pred = Vector{Float64}(undef, nq)
    d = Vector{Float64}(undef, ntr)
    @inbounds for q in 1:nq
        for i in 1:ntr
            d[i] = hypot(tx[i] - qx[q], ty[i] - qy[q], (tz[i] - qz[q]) * ratio)
        end
        zero = findall(di -> di == 0 || di < 1.0e-9, d)
        if !isempty(zero)
            pred[q] = mean(@view tv[zero])
            continue
        end
        k = min(N_NEIGHBORS, ntr)
        idx = partialsortperm(d, 1:k)
        wsum = 0.0
        vsum = 0.0
        for i in idx
            w = inv(d[i]^IDW_POWER)
            wsum += w
            vsum += w * tv[i]
        end
        pred[q] = vsum / wsum
    end
    return pred
end

# GeoStats' ordinary kriging with `prob = true` returns a Normal whose second
# parameter is, in the installed version, the kriging VARIANCE, not the standard
# deviation (checked 2026-09-25: two uncorrelated points, sill 100, query far away
# gives std(d) = 150 = variance; the standard deviation is √150 ≈ 12.25).
# The behaviour is measured once on that known case, so a future GeoStats fix
# is picked up automatically instead of being silently square-rooted twice.
const _KRIGING_SIGMA_IS_VARIANCE = Ref{Union{Nothing, Bool}}(nothing)

function kriging_sigma_is_variance()
    flag = _KRIGING_SIGMA_IS_VARIANCE[]
    flag === nothing || return flag
    gtb = georef((value = [0.0, 1.0],), [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)])
    γ = SphericalVariogram(; ranges = (1.0, 1.0, 1.0), sill = 100.0, nugget = 0.0)
    out = gtb |> InterpolateNeighbors([Point(1000.0, 0.0, 0.0)];
                                      model = Kriging(γ), prob = true)
    s = std(out.value[1])
    flag = if isapprox(s, 150.0; rtol = 1.0e-6)
        true
    elseif isapprox(s, sqrt(150.0); rtol = 1.0e-6)
        false
    else
        fail("kriging self-check: std of the known case is $s, expected 150 (variance) or √150")
    end
    _KRIGING_SIGMA_IS_VARIANCE[] = flag
    return flag
end

function kriging_predict(tx, ty, tz, tv, qx, qy, qz, fit::VarioFit)
    gtb = georef((value = tv,), Point.(tx, ty, tz))
    ranges = (fit.range_h, fit.range_h, fit.range_v)
    γ = fit.model == "exponential" ?
        ExponentialVariogram(; ranges = ranges, sill = fit.total_sill, nugget = fit.nugget) :
        SphericalVariogram(; ranges = ranges, sill = fit.total_sill, nugget = fit.nugget)
    queries = Point.(qx, qy, qz)
    local out
    try
        out = gtb |> InterpolateNeighbors(queries;
                                          model = Kriging(γ),
                                          maxneighbors = N_NEIGHBORS,
                                          minneighbors = 1,
                                          prob = true)
    catch e
        fail("GeoStats ordinary kriging failed: " * sprint(showerror, e))
    end
    dists = out.value
    n = length(qx)
    length(dists) == n || fail("kriging returned $(length(dists)) rows for $n queries")
    μ = Vector{Float64}(undef, n)
    σ = Vector{Float64}(undef, n)
    is_var = kriging_sigma_is_variance()
    for i in 1:n
        d = dists[i]
        if ismissing(d)
            fail("kriging prediction $i is missing")
        end
        μ[i] = mean(d)
        σ[i] = is_var ? sqrt(std(d)) : std(d)
    end
    all(isfinite, μ) && all(isfinite, σ) && all(>=(0), σ) || fail(
        "kriging mean or standard deviation is not finite and non-negative")
    return μ, σ
end

#---------- neural field ----------

function fourier_scales(; scale_min::Real = FOURIER_SCALE_MIN,
                        scale_max::Real = FOURIER_SCALE_MAX,
                        bands::Int = FOURIER_BANDS)
    return exp.(range(log(Float64(scale_min)), log(Float64(scale_max)); length = bands))
end

"Keep every channel. Append sin/cos of π·scale·x on the named xyz rows."
function fourier_append(M::AbstractMatrix, rows::AbstractVector{Int};
                        scales = fourier_scales())
    nrow, ncol = size(M)
    extra = 2 * length(rows) * length(scales)
    out = Matrix{Float64}(undef, nrow + extra, ncol)
    out[1:nrow, :] .= M
    r = nrow
    for row in rows, s in scales
        x = @view M[row, :]
        @views out[r + 1, :] .= sin.(π * s .* x)
        @views out[r + 2, :] .= cos.(π * s .* x)
        r += 2
    end
    return out
end

function build_mlp(nin::Int)
    hidden = ntuple(_ -> Dense(MLP_WIDTH => MLP_WIDTH, gelu), MLP_DEPTH - 1)
    return Chain(Dense(nin => MLP_WIDTH, gelu), hidden..., Dense(MLP_WIDTH => 2))
end

function gauss_nll(model, ps, st, X, y; beta::Real = 0.0)
    raw, _ = model(X, ps, st)
    μ = raw[1, :]
    σ = softplus.(raw[2, :]) .+ SIGMA_FLOOR
    nll = @. 0.5 * ((y - μ) / σ)^2 + log(σ)
    β = Float64(beta)
    if β == 0.0
        return mean(nll)
    end
    # β-NLL (Seitzer et al.): weight by detached σ^(2β) so μ still sees a 1/σ² gradient.
    w = Zygote.ignore_derivatives(σ .^ (2 * β))
    return mean(w .* nll)
end

"Plain MSE on μ; the σ head is ignored."
function mse_mu(model, ps, st, X, y)
    raw, _ = model(X, ps, st)
    μ = raw[1, :]
    return mean(abs2, μ .- y)
end

function predict_mu_sigma(model, ps, st, X)
    raw, _ = model(X, ps, st)
    μ = collect(Float64, raw[1, :])
    σ = collect(Float64, softplus.(raw[2, :]) .+ SIGMA_FLOOR)
    return μ, σ
end

function _train_objective(loss::Symbol, beta::Real)
    loss === :nll || loss === :mse || fail("train_member: unknown loss $loss (expected :nll or :mse)")
    if loss === :mse
        beta == 0.0 || fail("train_member: beta is only valid with loss=:nll")
        return (model, ps, st, X, y) -> mse_mu(model, ps, st, X, y)
    end
    return (model, ps, st, X, y) -> gauss_nll(model, ps, st, X, y; beta = beta)
end

function _stop_metric(stop_on::Symbol, loss_fn, model, ps, st, Xva, yva)
    if stop_on === :val_nll
        return Float64(loss_fn(model, ps, st, Xva, yva))
    elseif stop_on === :val_rmse
        μ, _ = predict_mu_sigma(model, ps, st, Xva)
        return sqrt(mean(abs2, μ .- yva))
    elseif stop_on === :none
        return NaN
    else
        fail("train_member: unknown stop_on $stop_on (expected :val_nll, :val_rmse, or :none)")
    end
end

"""
    train_member(...; loss=:nll, stop_on=:val_nll, beta=0.0, max_epochs=NN_EPOCHS,
                 patience=NN_PATIENCE, log_every=0, history=nothing)

Keyword defaults reproduce the previous fixed behaviour (Gaussian NLL, early
stop on validation NLL). `loss=:mse` fits μ only. `stop_on=:none` runs a fixed
number of full-batch steps. `beta>0` applies β-NLL with detached σ^(2β).
When `log_every>0` and `history` is a Vector, each logged step pushes a NamedTuple
`(step, train_loss, val_loss, train_rmse, val_rmse, mean_sigma)`. After training,
best- and final-step snapshots are appended if those steps were not already logged.
"""
function _member_loss_snapshot(model, loss_fn, ps, st, Xtr, ytr, Xva, yva, step::Int)
    μtr, σtr = predict_mu_sigma(model, ps, st, Xtr)
    μva, _ = predict_mu_sigma(model, ps, st, Xva)
    return (
        step = step,
        train_loss = Float64(loss_fn(model, ps, st, Xtr, ytr)),
        val_loss = Float64(loss_fn(model, ps, st, Xva, yva)),
        train_rmse = sqrt(mean(abs2, μtr .- ytr)),
        val_rmse = sqrt(mean(abs2, μva .- yva)),
        mean_sigma = mean(σtr),
    )
end

function train_member(model, seed::Int, Xtr, ytr, Xva, yva, context::AbstractString;
                      loss::Symbol = :nll,
                      stop_on::Symbol = :val_nll,
                      beta::Real = 0.0,
                      max_epochs::Int = NN_EPOCHS,
                      patience::Int = NN_PATIENCE,
                      log_every::Int = 0,
                      history = nothing)
    loss_fn = _train_objective(loss, beta)
    ps, st = Lux.setup(Xoshiro(seed), model)
    ps = Lux.f64(ps)
    opt_state = Optimisers.setup(AdamW(NN_LR, (0.9, 0.999), NN_WEIGHT_DECAY), ps)
    best_ps = deepcopy(ps)
    best_val = Inf
    best_epoch = 0
    wait = 0
    last_epoch = 0
    early_stop = stop_on !== :none
    for epoch in 1:max_epochs
        last_epoch = epoch
        loss_val, grads = Zygote.withgradient(p -> loss_fn(model, p, st, Xtr, ytr), ps)
        isfinite(loss_val) || fail("$context seed $seed: training loss is not finite at epoch $epoch")
        grads[1] === nothing && fail("$context seed $seed: missing gradient at epoch $epoch")
        opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])

        if log_every > 0 && history !== nothing && (epoch % log_every == 0 || epoch == max_epochs)
            push!(history, _member_loss_snapshot(model, loss_fn, ps, st, Xtr, ytr, Xva, yva, epoch))
        end

        if !early_stop
            continue
        end
        v = _stop_metric(stop_on, loss_fn, model, ps, st, Xva, yva)
        isfinite(v) || fail("$context seed $seed: validation metric is not finite at epoch $epoch")
        if v < best_val
            best_val = Float64(v)
            best_epoch = epoch
            best_ps = deepcopy(ps)
            wait = 0
        else
            wait += 1
            wait >= patience && break
        end
    end
    if !early_stop
        best_ps = deepcopy(ps)
        best_epoch = last_epoch
        best_val = Float64(loss_fn(model, ps, st, Xtr, ytr))
    end
    if log_every > 0 && history !== nothing
        logged = Set(h.step for h in history)
        best_epoch ∉ logged &&
            push!(history, _member_loss_snapshot(model, loss_fn, best_ps, st, Xtr, ytr, Xva, yva, best_epoch))
        last_epoch ∉ logged &&
            push!(history, _member_loss_snapshot(model, loss_fn, ps, st, Xtr, ytr, Xva, yva, last_epoch))
        sort!(history, by = h -> h.step)
    end
    return best_ps, st, best_epoch, last_epoch, best_val
end

function ensemble_predict(model, members, X, shift, scale, context::AbstractString)
    M = length(members)
    n = size(X, 2)
    mus = Matrix{Float64}(undef, M, n)
    vars = Matrix{Float64}(undef, M, n)
    for (i, (ps, st)) in enumerate(members)
        μs, σs = predict_mu_sigma(model, ps, st, X)
        (all(isfinite, μs) && all(isfinite, σs)) || fail(
            "$context: ensemble member $i predicted a non-finite value")
        mus[i, :] .= μs .* scale .+ shift
        vars[i, :] .= (σs .* scale) .^ 2
    end
    μ = vec(mean(mus; dims = 1))
    # Population variance of the member means (divide by M, not M-1).
    var_μ = vec(sum((mus .- reshape(μ, 1, :)) .^ 2; dims = 1) ./ M)
    v = vec(mean(vars; dims = 1)) .+ var_μ
    all(v .>= 0) || fail("$context: ensemble variance is negative")
    return μ, sqrt.(v)
end

function split_train_val(holes::AbstractVector{<:AbstractString}, rng)
    n = length(holes)
    n >= 2 || fail("internal validation needs at least 2 training holes, got $n")
    order = shuffle(rng, collect(String, holes))
    n_val = clamp(round(Int, VAL_HOLE_FRACTION * n), 1, n - 1)
    return order[(n_val + 1):end], order[1:n_val]
end

#---------- bookkeeping ----------

struct Pred
    deposit::String
    target::String
    method::String
    hole::String
    sample::String
    x::Float64
    y::Float64
    z::Float64
    obs::Float64
    censored::Int
    pred::Float64
    sd::Float64
end

struct FoldRef
    deposit::String
    target::String
    test_hole::String
    test_rows::Vector{Int}
    y_shift::Float64
    y_scale::Float64
    fit_cols::Vector{Int}
    val_cols::Vector{Int}
    test_cols::Vector{Int}
    y_fit::Vector{Float64}
    y_val::Vector{Float64}
    n_val_holes::Int
end

function tsv_num(x)
    return isfinite(x) ? @sprintf("%.10g", x) : "NaN"
end

function tsv_field(s::AbstractString)
    if occursin('\t', s) || occursin('\n', s)
        fail("TSV field contains a tab or newline: $(repr(s))")
    end
    return s
end

function append_preds(io, rows::AbstractVector{Pred})
    for r in rows
        println(io, join((
            tsv_field(r.deposit), tsv_field(r.target), tsv_field(r.method),
            tsv_field(r.hole), tsv_field(r.sample),
            tsv_num(r.x), tsv_num(r.y), tsv_num(r.z),
            tsv_num(r.obs), string(r.censored), tsv_num(r.pred), tsv_num(r.sd),
        ), '\t'))
    end
    flush(io)
    return nothing
end

function rows_for(deposit, target, method, table, ids, rows, obs, cens, pred, sd)
    n = length(rows)
    (length(obs) == n && length(cens) == n && length(pred) == n && length(sd) == n) ||
        fail("$deposit $target $method: prediction length mismatch")
    all(isfinite, pred) || fail("$deposit $target $method: non-finite prediction")
    out = Vector{Pred}(undef, n)
    for i in 1:n
        r = rows[i]
        out[i] = Pred(deposit, target, method, table.hole[r], ids[r],
                      table.x[r], table.y[r], table.z[r],
                      obs[i], cens[i] ? 1 : 0, pred[i], sd[i])
    end
    return out
end

function finite_features(covs, table;
                         fourier_scale_min::Real = FOURIER_SCALE_MIN,
                         fourier_scale_max::Real = FOURIER_SCALE_MAX,
                         fourier_bands::Int = FOURIER_BANDS)
    mask = training_mask(table)
    idx = findall(mask)
    xyz = Matrix{Float64}(undef, 3, length(idx))
    for (k, i) in enumerate(idx)
        xyz[1, k] = table.x[i]
        xyz[2, k] = table.y[i]
        xyz[3, k] = table.z[i]
    end
    M, names = evaluate_all(covs, xyz)
    xyz_rows = Int[]
    for n in ("x_norm", "y_norm", "z_norm")
        k = findfirst(==(n), names)
        k === nothing && fail("covariate channels have no $n")
        push!(xyz_rows, k)
    end
    any(c -> c isa CoordinateCovariate, covs) || fail("site has no CoordinateCovariate")
    any(c -> c isa DepthCovariate || c isa DepthBelowSurface, covs) ||
        fail("site has no depth covariate (depth or depth_below_surface)")
    scales = fourier_scales(; scale_min = fourier_scale_min, scale_max = fourier_scale_max,
                            bands = fourier_bands)
    Xcov = fourier_append(M, xyz_rows; scales = scales)
    Xxyz = fourier_append(M[xyz_rows, :], [1, 2, 3]; scales = scales)
    col_of = Dict{Int,Int}(i => k for (k, i) in enumerate(idx))
    return Xxyz, Xcov, col_of, names
end

function columns_of(col_of, rows)
    return [col_of[i] for i in rows]
end

function take_cols(X, cols)
    return X[:, cols]
end

#---------- metrics ----------

function pooled_metrics(obs, pred, sd, want_coverage::Bool)
    n = length(obs)
    n == 0 && return (n = 0, rmse = NaN, mae = NaN, r2 = NaN, pearson = NaN, coverage = NaN)
    err = pred .- obs
    rmse = sqrt(mean(abs2, err))
    mae = mean(abs, err)
    mu = mean(obs)
    sst = sum(abs2, obs .- mu)
    r2 = sst == 0 ? NaN : 1 - sum(abs2, err) / sst
    pearson = (std(obs) == 0 || std(pred) == 0) ? NaN : cor(obs, pred)
    coverage = NaN
    if want_coverage
        length(sd) == n || fail("coverage length mismatch")
        all(isfinite, sd) && all(sd .>= 0) || fail("coverage standard deviation is not finite")
        coverage = count(abs.(err) .<= Z90 .* sd) / n
    end
    return (n = n, rmse = rmse, mae = mae, r2 = r2, pearson = pearson, coverage = coverage)
end

function hole_rmses(rows::AbstractVector{Pred})
    sse = Dict{String,Float64}()
    cnt = Dict{String,Int}()
    for r in rows
        sse[r.hole] = get(sse, r.hole, 0.0) + (r.pred - r.obs)^2
        cnt[r.hole] = get(cnt, r.hole, 0) + 1
    end
    holes = sort!(collect(keys(cnt)))
    rmse = Dict{String,Float64}(h => sqrt(sse[h] / cnt[h]) for h in holes)
    return holes, rmse
end

function bootstrap_mean_ci(diffs::AbstractVector{<:Real}, tag::AbstractString)
    n = length(diffs)
    n >= 1 || fail("bootstrap has no holes")
    rng = rng_for(tag)
    means = Vector{Float64}(undef, N_BOOT)
    for b in 1:N_BOOT
        s = 0.0
        for _ in 1:n
            s += diffs[rand(rng, 1:n)]
        end
        means[b] = s / n
    end
    return mean(diffs), quantile(means, 0.025), quantile(means, 0.975)
end

function load_predictions(path)
    preds = Pred[]
    open(path) do io
        header = readline(io)
        startswith(header, "deposit\t") || fail("predictions.tsv header missing")
        for line in eachline(io)
            isempty(strip(line)) && continue
            p = split(line, '\t')
            length(p) == 12 || fail("predictions.tsv has $(length(p)) columns")
            push!(preds, Pred(p[1], p[2], p[3], p[4], p[5],
                              parse(Float64, p[6]), parse(Float64, p[7]), parse(Float64, p[8]),
                              parse(Float64, p[9]), parse(Int, p[10]),
                              parse(Float64, p[11]), parse(Float64, p[12])))
        end
    end
    return preds
end

