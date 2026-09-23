# Semi-synthetic Experiment 1: sample known Gaussian fields at real
# Cloncurry collar locations and compare mean / kriging / kriging_oracle /
# nn_xyz.
#
# Hyperparameters are fixed (same NN and fitted-variogram procedure as
# examples/feasibility_loho.jl). Do not retune on the results.
#
# measurement fix: stratified evaluation distances, R2_global vs uniform
# var_ref, kriging_oracle with the true exponential covariance, and a
# re-defined SAĞLAMLIK check on oracle R2_global (see run.log header).
#
# Run:  julia --project=. examples/synthetic_exp1.jl
# Out:  tmp_synthetic_exp1/{metrics,nn_epochs,kriging_params}.tsv
#       tmp_synthetic_exp1/run.log
#       tmp_synthetic_exp1/r2_global_panels.png

using SmartPrior
using Lux
using Lux: Chain, Dense, gelu, softplus
using Optimisers
using Zygote
using GeoStats
import GLMakie
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = joinpath(ROOT, "tmp_synthetic_exp1")

const DEPOSITS = (
    (key = "ernest_henry", label = "Ernest Henry", note = ""),
    (key = "cannington", label = "Cannington", note = ""),
    (key = "starra", label = "Starra", note = "4 hole — borderline"),
)

const L_H_GRID = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
const ETA_GRID = (0.1, 0.5)
const FIELD_SEEDS = (1, 2, 3)
const METHODS = ("mean", "kriging", "kriging_oracle", "nn_xyz")

# Stratified distance bins (nearest training sample, 3D).
const STRAT_BINS = (
    (label = "[0,25)", lo = 0.0, hi = 25.0),
    (label = "[25,50)", lo = 25.0, hi = 50.0),
    (label = "[50,100)", lo = 50.0, hi = 100.0),
    (label = "[100,200)", lo = 100.0, hi = 200.0),
    (label = "[200,inf)", lo = 200.0, hi = Inf),
)
const FIGURE_BINS = ("[0,50)", "[50,100)", "[100,200)", "[200,inf)")
const METRIC_BINS = ("uniform", "[0,25)", "[25,50)", "[0,50)",
                     "[50,100)", "[100,200)", "[200,inf)")

# Fixed procedure — copied from feasibility_loho.jl; not tuned here.
const N_LAGS = 15
const RANGE_MIN_M = 10.0
const HORIZONTAL_DIP_DEG = 22.5
const MIN_PAIRS = 30
const DEDUPE_TOL_M = 1.0e-3
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
const Z90 = 1.6448536269514722
const N_UNIFORM = 5000
const N_PER_BIN = 1000
const STRAT_MAX_TRIES = 500_000
const EVAL_SEED_TAG = "synthetic-exp1-eval-v2"
const VAL_SEED_TAG = "synthetic-exp1-val-v1"

const LOG = Ref{IO}()

function logmsg(msg::AbstractString)
    println(msg)
    flush(stdout)
    if isassigned(LOG)
        println(LOG[], msg)
        flush(LOG[])
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

function tsv_num(x)
    return isfinite(x) ? @sprintf("%.10g", x) : "NaN"
end

#---------- variogram (same as feasibility_loho.jl) ----------

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
    dh, gh, dv, gv = collect_pairs(x, y, z, v, hole;
                                   maxlag_h = maxlag_h, maxlag_v = maxlag_v,
                                   dip_deg = HORIZONTAL_DIP_DEG)
    if length(dh) < MIN_PAIRS || length(dv) < MIN_PAIRS
        fail("$context: variogram pair count below $(MIN_PAIRS) " *
             "(horizontal $(length(dh)), downhole $(length(dv)); " *
             "maxlag_h $(maxlag_h) m, maxlag_v $(maxlag_v) m)")
    end
    bins_h, exp_h = bin_experimental(dh, gh, N_LAGS, maxlag_h)
    bins_v, exp_v = bin_experimental(dv, gv, N_LAGS, maxlag_v)
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
                    best.rmse, length(dh), length(dv), length(bins_h), length(bins_v),
                    ratio < 0.05)
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
    for i in 1:n
        d = dists[i]
        if ismissing(d)
            fail("kriging prediction $i is missing")
        end
        μ[i] = mean(d)
        σ[i] = std(d)
    end
    all(isfinite, μ) && all(isfinite, σ) && all(>=(0), σ) || fail(
        "kriging mean or standard deviation is not finite and non-negative")
    return μ, σ
end

"""
True exponential covariance for the generative field.

GeoStats `ExponentialVariogram` uses the practical-range form
`1 - exp(-3 h / range)`, while the synthetic field has correlation
`exp(-h / L)`. Therefore oracle ranges are `3 L_h` and `3 L_v`.
"""
function oracle_variogram(L_h::Real, η::Real, σ::Real)
    L_h > 0 || fail("oracle_variogram: L_h must be positive")
    (0 <= η < 1) || fail("oracle_variogram: η must be in [0, 1)")
    σ >= 0 || fail("oracle_variogram: σ must be non-negative")
    total = Float64(σ)^2
    nugget = Float64(η) * total
    partial = total - nugget
    L_v = Float64(L_h) / 2
    return VarioFit("exponential", nugget, partial, total,
                    3 * Float64(L_h), 3 * L_v, NaN, 0, 0, 0, 0, false)
end

#---------- neural field (nn_xyz only; feasibility settings) ----------

function fourier_scales()
    return exp.(range(log(FOURIER_SCALE_MIN), log(FOURIER_SCALE_MAX); length = FOURIER_BANDS))
end

function fourier_append(M::AbstractMatrix, rows::AbstractVector{Int})
    scales = fourier_scales()
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

function xyz_features(covs, x, y, z)
    i = findfirst(c -> c isa CoordinateCovariate, covs)
    i === nothing && fail("site has no CoordinateCovariate")
    xyz = Matrix{Float64}(undef, 3, length(x))
    xyz[1, :] .= x
    xyz[2, :] .= y
    xyz[3, :] .= z
    M = SmartPrior.evaluate(covs[i], xyz)
    return fourier_append(M, [1, 2, 3])
end

function build_mlp(nin::Int)
    hidden = ntuple(_ -> Dense(MLP_WIDTH => MLP_WIDTH, gelu), MLP_DEPTH - 1)
    return Chain(Dense(nin => MLP_WIDTH, gelu), hidden..., Dense(MLP_WIDTH => 2))
end

function gauss_nll(model, ps, st, X, y)
    raw, _ = model(X, ps, st)
    μ = raw[1, :]
    σ = softplus.(raw[2, :]) .+ SIGMA_FLOOR
    return mean(@. 0.5 * ((y - μ) / σ)^2 + log(σ))
end

function predict_mu_sigma(model, ps, st, X)
    raw, _ = model(X, ps, st)
    μ = collect(Float64, raw[1, :])
    σ = collect(Float64, softplus.(raw[2, :]) .+ SIGMA_FLOOR)
    return μ, σ
end

function train_member(model, seed::Int, Xtr, ytr, Xva, yva, context::AbstractString)
    ps, st = Lux.setup(Xoshiro(seed), model)
    ps = Lux.f64(ps)
    opt_state = Optimisers.setup(AdamW(NN_LR, (0.9, 0.999), NN_WEIGHT_DECAY), ps)
    best_ps = deepcopy(ps)
    best_val = Inf
    best_epoch = 0
    wait = 0
    last_epoch = 0
    for epoch in 1:NN_EPOCHS
        last_epoch = epoch
        loss, grads = Zygote.withgradient(p -> gauss_nll(model, p, st, Xtr, ytr), ps)
        isfinite(loss) || fail("$context seed $seed: training loss is not finite at epoch $epoch")
        grads[1] === nothing && fail("$context seed $seed: missing gradient at epoch $epoch")
        opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
        v = gauss_nll(model, ps, st, Xva, yva)
        isfinite(v) || fail("$context seed $seed: validation loss is not finite at epoch $epoch")
        if v < best_val
            best_val = Float64(v)
            best_epoch = epoch
            best_ps = deepcopy(ps)
            wait = 0
        else
            wait += 1
            wait >= NN_PATIENCE && break
        end
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

#---------- geometry / metrics ----------

function deposit_box(table)
    return (xlo = minimum(table.x), xhi = maximum(table.x),
            ylo = minimum(table.y), yhi = maximum(table.y),
            zlo = minimum(table.z), zhi = maximum(table.z))
end

function in_box(box, x, y, z)
    return box.xlo <= x <= box.xhi &&
           box.ylo <= y <= box.yhi &&
           box.zlo <= z <= box.zhi
end

function nearest_train_dist_one(x, y, z, tx, ty, tz)
    best = Inf
    @inbounds for i in eachindex(tx)
        di = hypot(x - tx[i], y - ty[i], z - tz[i])
        di < best && (best = di)
    end
    return best
end

function nearest_train_dist(qx, qy, qz, tx, ty, tz)
    nq = length(qx)
    d = Vector{Float64}(undef, nq)
    @inbounds for q in 1:nq
        d[q] = nearest_train_dist_one(qx[q], qy[q], qz[q], tx, ty, tz)
    end
    return d
end

function random_unit_vector(rng)
    x = randn(rng)
    y = randn(rng)
    z = randn(rng)
    n = hypot(x, y, z)
    n < 1.0e-15 && return random_unit_vector(rng)
    return x / n, y / n, z / n
end

function uniform_eval_points(box, n::Int, rng)
    xyz = Matrix{Float64}(undef, 3, n)
    @inbounds for j in 1:n
        xyz[1, j] = box.xlo + (box.xhi - box.xlo) * rand(rng)
        xyz[2, j] = box.ylo + (box.yhi - box.ylo) * rand(rng)
        xyz[3, j] = box.zlo + (box.zhi - box.zlo) * rand(rng)
    end
    return xyz
end

function sample_distance_bin!(xs, ys, zs, box, tx, ty, tz, lo, hi, n, rng; far::Bool)
    nt = length(tx)
    nt >= 1 || fail("stratified eval: no training locations")
    target = length(xs) + n
    tries = 0
    while length(xs) < target
        tries += 1
        tries > STRAT_MAX_TRIES && fail(
            "stratified eval: failed to fill bin [$lo,$hi) after $STRAT_MAX_TRIES tries " *
            "(have $(length(xs) - (target - n))/$n)")
        if far
            x = box.xlo + (box.xhi - box.xlo) * rand(rng)
            y = box.ylo + (box.yhi - box.ylo) * rand(rng)
            z = box.zlo + (box.zhi - box.zlo) * rand(rng)
        else
            i = rand(rng, 1:nt)
            ux, uy, uz = random_unit_vector(rng)
            r = lo + (hi - lo) * rand(rng)
            x = tx[i] + r * ux
            y = ty[i] + r * uy
            z = tz[i] + r * uz
            in_box(box, x, y, z) || continue
        end
        d = nearest_train_dist_one(x, y, z, tx, ty, tz)
        ok = far ? (d >= lo) : (d >= lo && d < hi)
        ok || continue
        push!(xs, x)
        push!(ys, y)
        push!(zs, z)
    end
    return nothing
end

"""
Stratified eval points (1000 / distance bin) plus a uniform 5000-point cloud.

Returns `(xyz, dist, bin_of)` where `bin_of[j]` is the metric-bin label for
column `j` (`uniform` or a stratified label). Points for `[0,50)` are not
stored separately; that bin is the union of `[0,25)` and `[25,50)`.
"""
function build_eval_set(box, tx, ty, tz, tag::AbstractString)
    rng = rng_for(tag)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    labels = String[]
    for b in STRAT_BINS
        n0 = length(xs)
        far = !isfinite(b.hi)
        sample_distance_bin!(xs, ys, zs, box, tx, ty, tz, b.lo, b.hi, N_PER_BIN, rng;
                             far = far)
        for _ in 1:N_PER_BIN
            push!(labels, b.label)
        end
        length(xs) - n0 == N_PER_BIN || fail("stratified bin $(b.label) size mismatch")
    end
    xyz_u = uniform_eval_points(box, N_UNIFORM, rng)
    for j in 1:N_UNIFORM
        push!(xs, xyz_u[1, j])
        push!(ys, xyz_u[2, j])
        push!(zs, xyz_u[3, j])
        push!(labels, "uniform")
    end
    n = length(xs)
    xyz = Matrix{Float64}(undef, 3, n)
    xyz[1, :] .= xs
    xyz[2, :] .= ys
    xyz[3, :] .= zs
    dist = nearest_train_dist(xs, ys, zs, tx, ty, tz)
    return xyz, dist, labels
end

function bin_indices(labels::Vector{String}, name::AbstractString)
    if name == "[0,50)"
        return findall(l -> l == "[0,25)" || l == "[25,50)", labels)
    end
    return findall(==(name), labels)
end

function metrics_block(truth, pred, sd, mean_pred, var_ref; want_coverage::Bool)
    n = length(truth)
    n == 0 && return (n = 0, rmse = NaN, r2_local = NaN, r2_global = NaN,
                     skill = NaN, coverage90 = NaN)
    err = pred .- truth
    mse = mean(abs2, err)
    rmse = sqrt(mse)
    mu_t = mean(truth)
    sst = sum(abs2, truth .- mu_t)
    r2_local = sst == 0 ? NaN : 1 - sum(abs2, err) / sst
    r2_global = (!isfinite(var_ref) || var_ref <= 0) ? NaN : 1 - mse / var_ref
    mse_mean = mean(abs2, mean_pred .- truth)
    skill = mse_mean == 0 ? NaN : 1 - mse / mse_mean
    coverage90 = NaN
    if want_coverage
        length(sd) == n || fail("coverage length mismatch")
        all(isfinite, sd) && all(sd .>= 0) || fail("coverage sd is not finite")
        coverage90 = count(abs.(err) .<= Z90 .* sd) / n
    end
    return (n = n, rmse = rmse, r2_local = r2_local, r2_global = r2_global,
            skill = skill, coverage90 = coverage90)
end

function cu_moments(table)
    mask = training_mask(table) .& real_hole_mask(table) .& observed_mask(table, :cu)
    idx = findall(mask)
    isempty(idx) && fail("no real-hole Cu observations for μ/σ")
    vals = table.values[:cu][idx]
    any(v -> !(isfinite(v) && v > 0), vals) && fail("Cu needs finite positive values for log10")
    y = log10.(vals)
    μ = mean(y)
    σ = std(y)
    (isfinite(μ) && isfinite(σ) && σ > 0) || fail("Cu log10 mean/std not usable")
    return μ, σ, idx
end

function write_r2_figure(metrics_rows)
    nd = length(DEPOSITS)
    nb = length(FIGURE_BINS)
    fig = GLMakie.Figure(size = (280 * nb, 220 * nd))
    for (id, d) in enumerate(DEPOSITS), (ib, b) in enumerate(FIGURE_BINS)
        title = ib == 1 ?
                (isempty(d.note) ? d.key : "$(d.key) ($(d.note))") :
                d.key
        ax = GLMakie.Axis(fig[id, ib];
                          xlabel = ib == nb ? "L_h (m)" : "",
                          ylabel = ib == 1 ? "R2_global" : "",
                          title = "$title | $b",
                          xscale = log10)
        for η in ETA_GRID, method in METHODS
            xs = Float64[]
            ys = Float64[]
            for L_h in L_H_GRID
                sub = [r for r in metrics_rows if r.deposit == d.key &&
                       r.method == method && r.eta == η && r.L_h == L_h &&
                       r.dist_bin == b]
                isempty(sub) && continue
                push!(xs, L_h)
                push!(ys, mean(r.r2_global for r in sub))
            end
            isempty(xs) && continue
            GLMakie.lines!(ax, xs, ys; label = "$(method) η=$(η)")
        end
        id == 1 && ib == nb && GLMakie.axislegend(ax; position = :rb, labelsize = 10)
    end
    path = joinpath(WORK, "r2_global_panels.png")
    GLMakie.save(path, fig)
    logmsg("wrote $path")
    return nothing
end

#---------- one configuration ----------

struct MetricRow
    deposit::String
    L_h::Float64
    eta::Float64
    seed::Int
    method::String
    dist_bin::String
    n::Int
    rmse::Float64
    r2_local::Float64
    r2_global::Float64
    var_ref::Float64
    skill::Float64
    coverage90::Float64
end

function run_config(deposit_key, table, covs, idx, μ_cu, σ_cu, L_h, η, seed,
                    metric_io, epoch_io, param_io, flags, oracle_r2_050)
    ctx = @sprintf("%s L_h=%.0f η=%.1f seed=%d", deposit_key, L_h, η, seed)
    t0 = time()
    field = gaussian_field(; μ = μ_cu, σ = σ_cu, η = η, L_h = L_h, seed = seed)
    xyz_tr = Matrix{Float64}(undef, 3, length(idx))
    holes = Vector{String}(undef, length(idx))
    @inbounds for (k, i) in enumerate(idx)
        xyz_tr[1, k] = table.x[i]
        xyz_tr[2, k] = table.y[i]
        xyz_tr[3, k] = table.z[i]
        holes[k] = table.hole[i]
    end
    y_tr = SmartPrior.evaluate(field, xyz_tr)

    box = deposit_box(subset(table, idx))
    tag = "$(EVAL_SEED_TAG)|$(deposit_key)"
    xyz_ev, _, labels = build_eval_set(box, xyz_tr[1, :], xyz_tr[2, :], xyz_tr[3, :], tag)
    n_ev = size(xyz_ev, 2)
    truth = smooth_field(field, xyz_ev)

    iu = bin_indices(labels, "uniform")
    length(iu) == N_UNIFORM || fail("$ctx: expected $N_UNIFORM uniform points, got $(length(iu))")
    var_ref = var(truth[iu]; corrected = false)
    (isfinite(var_ref) && var_ref > 0) || fail("$ctx: var_ref is not positive ($var_ref)")

    m_tr = mean(y_tr)
    pred_mean = fill(m_tr, n_ev)
    sd_na = fill(NaN, n_ev)

    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        xyz_tr[1, :], xyz_tr[2, :], xyz_tr[3, :], y_tr, holes)
    n_merged > 0 && logmsg("$ctx: averaged $n_merged coincident training locations")
    fit = fit_variogram(dx, dy, dz, dv, dh; context = ctx)
    μ_ok, σ_ok = kriging_predict(dx, dy, dz, dv,
                                 xyz_ev[1, :], xyz_ev[2, :], xyz_ev[3, :], fit)
    fit_or = oracle_variogram(L_h, η, σ_cu)
    μ_or, σ_or = kriging_predict(dx, dy, dz, dv,
                                 xyz_ev[1, :], xyz_ev[2, :], xyz_ev[3, :], fit_or)
    println(param_io, join((
        deposit_key, tsv_num(L_h), tsv_num(η), string(seed), fit.model,
        tsv_num(fit.nugget), tsv_num(fit.partial_sill), tsv_num(fit.total_sill),
        tsv_num(fit.range_h), tsv_num(fit.range_v), tsv_num(L_h),
        tsv_num(fit.fit_rmse), string(fit.n_pairs_h), string(fit.n_pairs_v),
        string(fit.n_bins_h), string(fit.n_bins_v),
        fit.degenerate ? "1" : "0", string(n_merged),
    ), '\t'))
    flush(param_io)

    Xall_tr = xyz_features(covs, xyz_tr[1, :], xyz_tr[2, :], xyz_tr[3, :])
    Xall_ev = xyz_features(covs, xyz_ev[1, :], xyz_ev[2, :], xyz_ev[3, :])
    unique_holes = sort!(unique(holes))
    val_tag = "$(VAL_SEED_TAG)|$(deposit_key)|$(L_h)|$(η)|$(seed)"
    fit_holes, val_holes = split_train_val(unique_holes, rng_for(val_tag))
    fit_set = Set(fit_holes)
    val_set = Set(val_holes)
    fit_cols = Int[k for k in eachindex(holes) if holes[k] in fit_set]
    val_cols = Int[k for k in eachindex(holes) if holes[k] in val_set]
    (isempty(fit_cols) || isempty(val_cols)) && fail("$ctx: empty NN split")
    s_tr = std(y_tr)
    (isfinite(m_tr) && isfinite(s_tr) && s_tr > 0) || fail("$ctx: train std not positive")
    y_fit = (y_tr[fit_cols] .- m_tr) ./ s_tr
    y_val = (y_tr[val_cols] .- m_tr) ./ s_tr
    model = build_mlp(size(Xall_tr, 1))
    members = Tuple{Any,Any}[]
    epochs = Int[]
    n_early = 0
    for nn_seed in NN_SEEDS
        ps, st, best_ep, last_ep, _ = train_member(
            model, nn_seed, Xall_tr[:, fit_cols], y_fit,
            Xall_tr[:, val_cols], y_val, ctx * " nn_xyz")
        push!(members, (ps, st))
        push!(epochs, best_ep)
        best_ep <= 5 && (n_early += 1)
        println(epoch_io, join((
            deposit_key, tsv_num(L_h), tsv_num(η), string(seed),
            string(nn_seed), string(best_ep), string(last_ep),
        ), '\t'))
    end
    flush(epoch_io)
    μ_nn, σ_nn = ensemble_predict(model, members, Xall_ev, m_tr, s_tr, ctx * " nn_xyz")

    preds = Dict(
        "mean" => (pred_mean, sd_na, false),
        "kriging" => (μ_ok, σ_ok, true),
        "kriging_oracle" => (μ_or, σ_or, true),
        "nn_xyz" => (μ_nn, σ_nn, true),
    )
    rows = MetricRow[]
    for method in METHODS
        pred, sd, want = preds[method]
        for bname in METRIC_BINS
            ii = bin_indices(labels, bname)
            met = metrics_block(truth[ii], pred[ii], sd[ii], pred_mean[ii], var_ref;
                                want_coverage = want && !isempty(ii))
            row = MetricRow(deposit_key, L_h, η, seed, method, bname,
                            met.n, met.rmse, met.r2_local, met.r2_global, var_ref,
                            met.skill, met.coverage90)
            push!(rows, row)
            covs_str = want ? tsv_num(met.coverage90) : ""
            println(metric_io, join((
                deposit_key, tsv_num(L_h), tsv_num(η), string(seed), method, bname,
                string(met.n), tsv_num(met.rmse), tsv_num(met.r2_local),
                tsv_num(met.r2_global), tsv_num(var_ref), tsv_num(met.skill), covs_str,
            ), '\t'))
        end
    end
    flush(metric_io)

    if L_h >= 800 && η == 0.1
        orow = only(r for r in rows if r.method == "kriging_oracle" && r.dist_bin == "[0,50)")
        key = (deposit_key, L_h, η)
        push!(get!(oracle_r2_050, key, Float64[]), orow.r2_global)
        frac_early = n_early / length(NN_SEEDS)
        if frac_early > 0.20
            msg = @sprintf("NN EĞİTİMİ FLAG: %s L_h=%.0f η=%.1f seed=%d: epoch≤5 fraction=%.2f > 0.20 (epochs=%s)",
                           deposit_key, L_h, η, seed, frac_early, join(string.(epochs), ","))
            logmsg(msg)
            push!(flags, msg)
        end
    end

    logmsg(@sprintf("%s done in %.1f s | fit range_h=%.1f (true L_h=%.0f) oracle ranges=%.1f/%.1f | NN epochs %s | var_ref=%.4g",
                    ctx, time() - t0, fit.range_h, L_h, fit_or.range_h, fit_or.range_v,
                    join(string.(epochs), ","), var_ref))
    return rows
end

function main()
    mkpath(WORK)
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("synthetic exp1 started")
    logmsg("measurement fix: stratified eval (1000/bin) + uniform 5000; " *
           "R2_global = 1 - MSE/var_ref(uniform); kriging_oracle uses true " *
           "exponential covariance (GeoStats practical range = 3 L); " *
           "SAĞLAMLIK on mean oracle R2_global[0,50) > 0.7 for L_h≥800, η=0.1; " *
           "Osborne dropped; Starra marked 4-hole borderline; method hyperparameters unchanged")
    logmsg("julia " * string(VERSION))
    logmsg("deposits=$(join((d.key for d in DEPOSITS), ","))")
    logmsg("L_h=$(collect(L_H_GRID)) η=$(collect(ETA_GRID)) seeds=$(collect(FIELD_SEEDS))")
    logmsg("NN: fourier=$FOURIER_BANDS scales=[$(FOURIER_SCALE_MIN),$(FOURIER_SCALE_MAX)] " *
           "width=$MLP_WIDTH depth=$MLP_DEPTH lr=$NN_LR wd=$NN_WEIGHT_DECAY " *
           "epochs=$NN_EPOCHS patience=$NN_PATIENCE ensemble=$(NN_SEEDS)")

    metric_path = joinpath(WORK, "metrics.tsv")
    epoch_path = joinpath(WORK, "nn_epochs.tsv")
    param_path = joinpath(WORK, "kriging_params.tsv")
    metric_io = open(metric_path, "w")
    epoch_io = open(epoch_path, "w")
    param_io = open(param_path, "w")
    println(metric_io, "deposit\tL_h\teta\tseed\tmethod\tdist_bin\tn\trmse\t" *
            "R2_local\tR2_global\tvar_ref\tskill\tcoverage90")
    println(epoch_io, "deposit\tL_h\teta\tseed\tnn_seed\tbest_epoch\tlast_epoch")
    println(param_io, "deposit\tL_h_true\teta\tseed\tmodel\tnugget\tpartial_sill\ttotal_sill\t" *
            "range_horizontal_m\trange_vertical_m\tL_h_true_again\tfit_rmse\t" *
            "n_pairs_horizontal\tn_pairs_downhole\tn_bins_horizontal\tn_bins_downhole\t" *
            "degenerate\tn_merged_locations")
    flush(metric_io); flush(epoch_io); flush(param_io)

    all_rows = MetricRow[]
    flags = String[]
    oracle_r2_050 = Dict{Tuple{String,Float64,Float64},Vector{Float64}}()

    for d in DEPOSITS
        path = joinpath(ROOT, "sites", d.key * ".toml")
        logmsg("load_site $path")
        table, covs, cfg = load_site(path)
        String(cfg["deposit"]) == d.label || fail("deposit label mismatch for $(d.key)")
        μ_cu, σ_cu, idx = cu_moments(table)
        n_holes = length(unique(table.hole[idx]))
        note = isempty(d.note) ? "" : " [$(d.note)]"
        logmsg(@sprintf("%s%s: real Cu samples=%d holes=%d μ=%.4g σ=%.4g (log10 Cu)",
                        d.key, note, length(idx), n_holes, μ_cu, σ_cu))
        n_holes >= 2 || fail("$(d.key): need ≥2 real holes for NN validation, got $n_holes")

        for L_h in L_H_GRID, η in ETA_GRID, seed in FIELD_SEEDS
            rows = run_config(d.key, table, covs, idx, μ_cu, σ_cu, L_h, η, seed,
                              metric_io, epoch_io, param_io, flags, oracle_r2_050)
            append!(all_rows, rows)
            if L_h >= 800 && η == 0.1 && length(oracle_r2_050[(d.key, L_h, η)]) == length(FIELD_SEEDS)
                vals = oracle_r2_050[(d.key, L_h, η)]
                μ = mean(vals)
                logmsg(@sprintf("SAĞLAMLIK oracle R2_global[0,50) %s L_h=%.0f η=%.1f seeds=%s mean=%.4g",
                                d.key, L_h, η, join((@sprintf("%.4g", v) for v in vals), ","), μ))
                if !(μ > 0.7)
                    fail(@sprintf("SAĞLAMLIK FAIL: %s L_h=%.0f η=%.1f: mean oracle R2_global[0,50)=%.4g ≤ 0.7",
                                  d.key, L_h, η, μ))
                end
            end
        end
    end

    close(metric_io)
    close(epoch_io)
    close(param_io)

    write_r2_figure(all_rows)
    for msg in flags
        logmsg(msg)
    end
    elapsed = time() - t0
    logmsg(@sprintf("finished in %.1f s (%.2f h)", elapsed, elapsed / 3600))
    logmsg("outputs: $metric_path")
    logmsg("outputs: $epoch_path")
    logmsg("outputs: $param_path")
    close(LOG[])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
