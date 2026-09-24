# Leave-one-hole-out feasibility: a pointwise Lux field against kriging.
#
# One deposit at a time, each hole is the test set once. The test hole is
# excluded from variogram fitting, target standardisation, early stopping,
# and the training loss. Hyperparameters below are fixed; this script does
# not retune them.
#
# Run:  julia --project=. examples/feasibility_loho.jl
# Out:  tmp_feasibility/{predictions,summary,paired,kriging_params}.tsv
#       tmp_feasibility/run.log

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
const TEMPLATE = joinpath(ROOT, "sites", "ernest_henry.toml")

const DEPOSITS = (
    (key = "ernest_henry", label = "Ernest Henry"),
    (key = "cannington", label = "Cannington"),
    (key = "starra", label = "Starra"),
    (key = "osborne", label = "Osborne"),
)

const TARGETS = (:cu, :density, :susceptibility)

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
const BOOT_SEED_TAG = "feasibility-loho-bootstrap-v1"

const METHODS = ("mean", "idw", "kriging", "nn_xyz", "nn_cov")
const UNCERTAIN = ("kriging", "nn_xyz", "nn_cov")

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

function toml_float(x::Real)
    return @sprintf("%.17g", Float64(x))
end

"Site file from ernest_henry.toml, with this deposit's cloncurry_deposit_bounds."
function render_site(template::AbstractString, name::AbstractString,
                     deposit::AbstractString, b)
    lines = split(template, '\n')
    out = String[]
    push!(out, "# $deposit, Cloncurry district (METAL / GDA94 MGA zone 54).")
    push!(out, "# Bounds are cloncurry_deposit_bounds(samples, \"$(deposit)\") on")
    push!(out, "# Cloncurry_integrated_2026-09-17, with that function's default pads")
    push!(out, "# (50 m in x/y, 25 m in z). They frame the covariates; they do not drop rows.")
    push!(out, "")
    started = false
    for line in lines
        if !started
            startswith(line, "name ") || continue
            started = true
        end
        if startswith(line, "name ")
            push!(out, "name = \"$(name)\"")
        elseif startswith(line, "deposit ")
            push!(out, "deposit = \"$(deposit)\"")
        elseif startswith(line, "x = ")
            push!(out, "x = [$(toml_float(b.x_min)), $(toml_float(b.x_max))]")
        elseif startswith(line, "y = ")
            push!(out, "y = [$(toml_float(b.y_min)), $(toml_float(b.y_max))]")
        elseif startswith(line, "z = ")
            push!(out, "z = [$(toml_float(b.z_min)), $(toml_float(b.z_max))]")
        else
            push!(out, line)
        end
    end
    return join(out, "\n") * "\n"
end

function ensure_sites(samples)
    template = read(TEMPLATE, String)
    paths = Dict{String,String}()
    for d in DEPOSITS
        path = joinpath(ROOT, "sites", d.key * ".toml")
        b = cloncurry_deposit_bounds(samples, d.label)
        if d.key == "ernest_henry" && isfile(path)
            paths[d.key] = path
            logmsg(@sprintf("site %s kept (existing); bounds x [%.3f, %.3f] y [%.3f, %.3f] z [%.3f, %.3f] n=%d",
                            d.key, b.x_min, b.x_max, b.y_min, b.y_max, b.z_min, b.z_max, b.n_samples))
        else
            write(path, render_site(template, d.key, d.label, b))
            paths[d.key] = path
            logmsg(@sprintf("wrote %s from ernest_henry.toml; bounds x [%.3f, %.3f] y [%.3f, %.3f] z [%.3f, %.3f] n=%d",
                            path, b.x_min, b.x_max, b.y_min, b.y_max, b.z_min, b.z_max, b.n_samples))
        end
    end
    return paths
end

function filled_id(raw, i::Int, prefix::AbstractString)
    t = strip(String(raw))
    if isempty(t) || lowercase(t) == "missing"
        return String(prefix) * string(i)
    end
    return t
end

function dataset_root()
    env = get(ENV, "CLONCURRY_ROOT", "")
    isempty(env) || return env
    for c in (
        joinpath(homedir(), "Desktop", "datasets4HY", "Cloncurry_integrated_2026-09-17"),
        "/Users/hayrunnisayildiz/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17",
    )
        isdir(joinpath(c, "derived")) && return c
    end
    fail("set CLONCURRY_ROOT to Cloncurry_integrated_2026-09-17")
end

"METAL sample ids in the same row order as load_site's deposit filter."
function sample_ids_for(samples, table, deposit::AbstractString)
    idx = findall(samples.deposit .== deposit)
    length(idx) == nsamples(table) || fail(
        "sample-id alignment: $(length(idx)) raw rows vs $(nsamples(table)) table rows for $deposit")
    ids = Vector{String}(undef, length(idx))
    for (k, i) in enumerate(idx)
        hole = filled_id(samples.drillhole[i], i, "__ungrouped_")
        hole == table.hole[k] || fail(
            "sample-id alignment: hole mismatch at filtered row $k ($(hole) vs $(table.hole[k]))")
        same_x = isfinite(samples.east[i]) && isfinite(table.x[k]) ?
                 samples.east[i] == table.x[k] :
                 !isfinite(samples.east[i]) && !isfinite(table.x[k])
        same_x || fail("sample-id alignment: easting mismatch at filtered row $k")
        ids[k] = String(samples.sample[i])
    end
    return ids
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

#---------- neural field ----------

function fourier_scales()
    return exp.(range(log(FOURIER_SCALE_MIN), log(FOURIER_SCALE_MAX); length = FOURIER_BANDS))
end

"Keep every channel. Append sin/cos of π·scale·x on the named xyz rows."
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

function finite_features(covs, table)
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
    Xcov = fourier_append(M, xyz_rows)
    Xxyz = fourier_append(M[xyz_rows, :], [1, 2, 3])
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

#---------- one fold of baselines ----------

function baseline_fold(table, ids, deposit, target, test_hole, eligible, y_all, cens_all,
                       pred_io, param_io)
    test_rows = [i for i in eligible if table.hole[i] == test_hole]
    train_rows = [i for i in eligible if table.hole[i] != test_hole]
    isempty(test_rows) && fail("$deposit $target: hole $test_hole has no test rows")
    isempty(train_rows) && fail("$deposit $target: hole $test_hole leaves no training rows")
    y_tr = y_all[train_rows]
    y_te = y_all[test_rows]
    cens_te = cens_all[test_rows]
    m = mean(y_tr)
    s = std(y_tr)
    (isfinite(m) && isfinite(s) && s > 0) || fail(
        "$deposit $target hole $test_hole: training mean/std is not usable (mean=$m std=$s)")

    ctx = "$deposit $target hole $test_hole"
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    n_merged > 0 && logmsg("$ctx: averaged $n_merged coincident training locations (tol $(DEDUPE_TOL_M) m)")
    fit = fit_variogram(dx, dy, dz, dv, dh; context = ctx)
    ratio = fit.range_h / fit.range_v
    (isfinite(ratio) && ratio > 0) || fail("$ctx: anisotropy ratio is not positive")

    μ_mean = fill(m, length(test_rows))
    sd_na = fill(NaN, length(test_rows))
    μ_idw = idw_predict(dx, dy, dz, dv,
                        table.x[test_rows], table.y[test_rows], table.z[test_rows], ratio)
    all(isfinite, μ_idw) || fail("$ctx: IDW prediction is not finite")
    μ_ok, σ_ok = kriging_predict(dx, dy, dz, dv,
                                 table.x[test_rows], table.y[test_rows], table.z[test_rows], fit)

    rows = Pred[]
    append!(rows, rows_for(deposit, target, "mean", table, ids, test_rows, y_te, cens_te, μ_mean, sd_na))
    append!(rows, rows_for(deposit, target, "idw", table, ids, test_rows, y_te, cens_te, μ_idw, sd_na))
    append!(rows, rows_for(deposit, target, "kriging", table, ids, test_rows, y_te, cens_te, μ_ok, σ_ok))
    append_preds(pred_io, rows)

    println(param_io, join((
        deposit, target, test_hole, fit.model,
        tsv_num(fit.nugget), tsv_num(fit.partial_sill), tsv_num(fit.total_sill),
        tsv_num(fit.range_h), tsv_num(fit.range_v), tsv_num(ratio),
        tsv_num(fit.fit_rmse), string(fit.n_pairs_h), string(fit.n_pairs_v),
        string(fit.n_bins_h), string(fit.n_bins_v),
        fit.degenerate ? "1" : "0", string(n_merged),
    ), '\t'))
    flush(param_io)
    return fit, length(test_rows), length(train_rows)
end

function nn_fold(model_xyz, model_cov, Xxyz, Xcov, fold::FoldRef, table, ids, y_te, cens_te, pred_io)
    ctx = "$(fold.deposit) $(fold.target) hole $(fold.test_hole)"
    Xtr_xyz = take_cols(Xxyz, fold.fit_cols)
    Xva_xyz = take_cols(Xxyz, fold.val_cols)
    Xte_xyz = take_cols(Xxyz, fold.test_cols)
    Xtr_cov = take_cols(Xcov, fold.fit_cols)
    Xva_cov = take_cols(Xcov, fold.val_cols)
    Xte_cov = take_cols(Xcov, fold.test_cols)
    members_xyz = Tuple{Any,Any}[]
    members_cov = Tuple{Any,Any}[]
    epochs_xyz = Int[]
    epochs_cov = Int[]
    for seed in NN_SEEDS
        ps, st, best_ep, last_ep, _ = train_member(
            model_xyz, seed, Xtr_xyz, fold.y_fit, Xva_xyz, fold.y_val, ctx * " nn_xyz")
        push!(members_xyz, (ps, st))
        push!(epochs_xyz, best_ep)
        last_ep == NN_EPOCHS &&
            logmsg("$ctx nn_xyz seed $seed ran all $(NN_EPOCHS) epochs (best $best_ep)")
        ps, st, best_ep, last_ep, _ = train_member(
            model_cov, seed, Xtr_cov, fold.y_fit, Xva_cov, fold.y_val, ctx * " nn_cov")
        push!(members_cov, (ps, st))
        push!(epochs_cov, best_ep)
        last_ep == NN_EPOCHS &&
            logmsg("$ctx nn_cov seed $seed ran all $(NN_EPOCHS) epochs (best $best_ep)")
    end
    μx, σx = ensemble_predict(model_xyz, members_xyz, Xte_xyz,
                              fold.y_shift, fold.y_scale, ctx * " nn_xyz")
    μc, σc = ensemble_predict(model_cov, members_cov, Xte_cov,
                              fold.y_shift, fold.y_scale, ctx * " nn_cov")
    rows = Pred[]
    append!(rows, rows_for(fold.deposit, fold.target, "nn_xyz", table, ids,
                           fold.test_rows, y_te, cens_te, μx, σx))
    append!(rows, rows_for(fold.deposit, fold.target, "nn_cov", table, ids,
                           fold.test_rows, y_te, cens_te, μc, σc))
    append_preds(pred_io, rows)
    return epochs_xyz, epochs_cov
end

function prepare_fold(table, col_of, y_all, eligible, test_hole, deposit, target)
    test_rows = [i for i in eligible if table.hole[i] == test_hole]
    train_rows = [i for i in eligible if table.hole[i] != test_hole]
    y_tr = y_all[train_rows]
    m = mean(y_tr)
    s = std(y_tr)
    (isfinite(m) && isfinite(s) && s > 0) || fail(
        "$deposit $target hole $test_hole: training std is not positive")
    holes = sort!(unique(table.hole[train_rows]))
    tag = "val|$(deposit)|$(target)|$(test_hole)"
    fit_holes, val_holes = split_train_val(holes, rng_for(tag))
    fit_set = Set(fit_holes)
    val_set = Set(val_holes)
    fit_rows = [i for i in train_rows if table.hole[i] in fit_set]
    val_rows = [i for i in train_rows if table.hole[i] in val_set]
    (isempty(fit_rows) || isempty(val_rows)) && fail(
        "$deposit $target hole $test_hole: empty neural-net split")
    y_fit = (y_all[fit_rows] .- m) ./ s
    y_val = (y_all[val_rows] .- m) ./ s
    return FoldRef(deposit, string(target), test_hole, test_rows, m, s,
                   columns_of(col_of, fit_rows), columns_of(col_of, val_rows),
                   columns_of(col_of, test_rows), y_fit, y_val, length(val_holes)),
           y_all[test_rows], table.censored[target][test_rows]
end

function write_summaries(preds::Vector{Pred})
    summary_path = joinpath(WORK, "summary.tsv")
    paired_path = joinpath(WORK, "paired.tsv")
    open(summary_path, "w") do io
        println(io, "deposit\ttarget\tmethod\tsubset\tn\trmse\tmae\tr2\tpearson\tcoverage90")
        for d in DEPOSITS, prop in TARGETS, method in METHODS
            sub = [r for r in preds if r.deposit == d.key && r.target == string(prop) &&
                   r.method == method]
            for (subset, rows) in (("all", sub), ("uncensored", [r for r in sub if r.censored == 0]))
                want = method in UNCERTAIN
                met = pooled_metrics([r.obs for r in rows], [r.pred for r in rows],
                                     [r.sd for r in rows], want && !isempty(rows))
                cov = want ? tsv_num(met.coverage) : ""
                println(io, join((
                    d.key, string(prop), method, subset, string(met.n),
                    tsv_num(met.rmse), tsv_num(met.mae), tsv_num(met.r2),
                    tsv_num(met.pearson), cov,
                ), '\t'))
            end
        end
    end
    open(paired_path, "w") do io
        println(io, "deposit\ttarget\tnn_method\tsubset\tn_holes\tmean_rmse_diff\tci95_low\tci95_high\tn_win\tn_total")
        for d in DEPOSITS, prop in TARGETS, nn in ("nn_xyz", "nn_cov")
            for subset in ("all", "uncensored")
                function select(method)
                    rows = [r for r in preds if r.deposit == d.key && r.target == string(prop) &&
                            r.method == method]
                    return subset == "all" ? rows : [r for r in rows if r.censored == 0]
                end
                h_nn, rmse_nn = hole_rmses(select(nn))
                h_ok, rmse_ok = hole_rmses(select("kriging"))
                h_nn == h_ok || fail(
                    "paired holes differ for $(d.key) $prop $nn $subset")
                isempty(h_nn) && fail("no holes for paired comparison $(d.key) $prop $nn $subset")
                diffs = [rmse_nn[h] - rmse_ok[h] for h in h_nn]
                tag = BOOT_SEED_TAG * "|$(d.key)|$(prop)|$(nn)|$(subset)"
                μ, lo, hi = bootstrap_mean_ci(diffs, tag)
                n_win = count(<(0), diffs)
                println(io, join((
                    d.key, string(prop), nn, subset, string(length(h_nn)),
                    tsv_num(μ), tsv_num(lo), tsv_num(hi),
                    string(n_win), string(length(h_nn)),
                ), '\t'))
            end
        end
    end
    logmsg("wrote $summary_path")
    logmsg("wrote $paired_path")
    return nothing
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

function main()
    mkpath(WORK)
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("feasibility LOHO started")
    logmsg("julia " * string(VERSION))
    logmsg("fixed: lags=$N_LAGS range_min=$(RANGE_MIN_M)m dip=$(HORIZONTAL_DIP_DEG) " *
           "min_pairs=$MIN_PAIRS neighbors=$N_NEIGHBORS idw_power=$IDW_POWER")
    logmsg("fixed NN: fourier=$FOURIER_BANDS scales=[$(FOURIER_SCALE_MIN), $(FOURIER_SCALE_MAX)] " *
           "width=$MLP_WIDTH depth=$MLP_DEPTH lr=$NN_LR decay=$NN_WEIGHT_DECAY " *
           "epochs=$NN_EPOCHS patience=$NN_PATIENCE seeds=$(NN_SEEDS) " *
           "val_fraction=$VAL_HOLE_FRACTION sigma_floor=$SIGMA_FLOOR")

    root = dataset_root()
    logmsg("dataset $root")
    samples = load_cloncurry_samples(root)
    paths = ensure_sites(samples)

    pred_path = joinpath(WORK, "predictions.tsv")
    param_path = joinpath(WORK, "kriging_params.tsv")
    pred_io = open(pred_path, "w")
    param_io = open(param_path, "w")
    println(pred_io, "deposit\ttarget\tmethod\thole\tsample\tx\ty\tz\tobs\tcensored\tpred\tsd")
    println(param_io, "deposit\ttarget\ttest_hole\tmodel\tnugget\tpartial_sill\ttotal_sill\t" *
            "range_horizontal_m\trange_vertical_m\tanisotropy_ratio\tfit_rmse\t" *
            "n_pairs_horizontal\tn_pairs_downhole\tn_bins_horizontal\tn_bins_downhole\t" *
            "degenerate\tn_merged_locations")
    flush(pred_io)
    flush(param_io)

    # Phase 1: covariates, variograms, mean, IDW, kriging. Neural nets after,
    # so a variogram or kriging failure stops before the long training loop.
    prepared = Tuple{FoldRef,Any,Vector{String},Matrix{Float64},Matrix{Float64},Vector{Float64},BitVector}[]
    models = Dict{String,Any}()

    for d in DEPOSITS
        logmsg("load_site $(paths[d.key])")
        table, covs, cfg = load_site(paths[d.key])
        String(cfg["deposit"]) == d.label || fail("site deposit label mismatch for $(d.key)")
        any(c -> !(c isa CoordinateCovariate || c isa DepthCovariate ||
                   c isa StructureDistance || c isa SurfaceGeology), covs) &&
            fail("$(d.key) has a covariate outside the allowed set")
        ids = sample_ids_for(samples, table, d.label)
        logmsg(@sprintf("%s: %d rows, depth log_mean=%.6g log_std=%.6g",
                        d.key, nsamples(table),
                        (c = covs[findfirst(c -> c isa DepthCovariate, covs)]; c.log_mean),
                        (c = covs[findfirst(c -> c isa DepthCovariate, covs)]; c.log_std)))
        t_cov = time()
        Xxyz, Xcov, col_of, names = finite_features(covs, table)
        logmsg(@sprintf("%s features: xyz %d, cov %d (%s) in %.1f s",
                        d.key, size(Xxyz, 1), size(Xcov, 1), join(names, ","), time() - t_cov))
        if !haskey(models, "xyz")
            models["xyz"] = build_mlp(size(Xxyz, 1))
            models["cov"] = build_mlp(size(Xcov, 1))
            models["nin_xyz"] = size(Xxyz, 1)
            models["nin_cov"] = size(Xcov, 1)
            logmsg("MLP xyz nin=$(size(Xxyz, 1)) cov nin=$(size(Xcov, 1)) " *
                   "(depth $MLP_DEPTH = $MLP_DEPTH gelu layers plus a linear head)")
        else
            models["nin_xyz"] == size(Xxyz, 1) || fail(
                "$(d.key): xyz feature width $(size(Xxyz, 1)) ≠ $(models["nin_xyz"])")
            models["nin_cov"] == size(Xcov, 1) || fail(
                "$(d.key): cov feature width $(size(Xcov, 1)) ≠ $(models["nin_cov"])")
        end

        for prop in TARGETS
            spec = spec_of(table, prop)
            spec.kind === :continuous || fail("$prop is not continuous")
            eligible = findall(training_mask(table) .& observed_mask(table, prop))
            isempty(eligible) && fail("$(d.key) $prop has no located observations")
            y_all = fill(NaN, nsamples(table))
            y_all[eligible] = transformed(spec, table.values[prop][eligible])
            cens = table.censored[prop]
            n_cens = count(i -> cens[i], eligible)
            holes = sort!(unique(table.hole[eligible]))
            logmsg("$(d.key) $prop: $(length(eligible)) samples, $(length(holes)) holes, " *
                   "$n_cens censored, transform=$(spec.transform)")
            for hole in holes
                t_fold = time()
                fit, n_te, n_tr = baseline_fold(table, ids, d.key, string(prop), hole,
                                                eligible, y_all, cens, pred_io, param_io)
                fold, y_te, cens_te = prepare_fold(table, col_of, y_all, eligible, hole,
                                                   d.key, prop)
                push!(prepared, (fold, table, ids, Xxyz, Xcov, y_te, cens_te))
                logmsg(@sprintf("%s %s hole %s: train %d test %d val_holes %d | %s nugget %.4g partial %.4g ranges %.1f / %.1f m degenerate %d | %.1f s",
                                d.key, prop, hole, n_tr, n_te, fold.n_val_holes,
                                fit.model, fit.nugget, fit.partial_sill,
                                fit.range_h, fit.range_v, fit.degenerate ? 1 : 0,
                                time() - t_fold))
            end
        end
    end

    logmsg("baselines finished in $(round(time() - t0, digits = 1)) s; starting neural nets ($(length(prepared)) folds × $(length(NN_SEEDS)) seeds × 2)")

    model_xyz = models["xyz"]
    model_cov = models["cov"]
    for (k, item) in enumerate(prepared)
        fold, table, ids, Xxyz, Xcov, y_te, cens_te = item
        t_fold = time()
        epx, epc = nn_fold(model_xyz, model_cov, Xxyz, Xcov, fold, table, ids, y_te, cens_te, pred_io)
        logmsg(@sprintf("NN %d/%d %s %s hole %s: xyz best epochs %s | cov best epochs %s | %.1f s",
                        k, length(prepared), fold.deposit, fold.target, fold.test_hole,
                        join(string.(epx), ","), join(string.(epc), ","), time() - t_fold))
    end
    close(pred_io)
    close(param_io)

    preds = load_predictions(pred_path)
    logmsg("predictions rows $(length(preds))")
    write_summaries(preds)
    elapsed = time() - t0
    logmsg(@sprintf("finished in %.1f s (%.2f h)", elapsed, elapsed / 3600))
    logmsg("outputs: $pred_path")
    logmsg("outputs: $param_path")
    close(LOG[])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
