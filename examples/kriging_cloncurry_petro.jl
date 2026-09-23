# Ordinary-kriging baseline via GeoStats.jl (density & log10 susceptibility).
#
# This script (main --project=.):
#   1. Loads today's variogram summary + recovers major-axis nugget / sill
#   2. Rebuilds the Z=100 m drillhole hold-out (split_seed=2026, 84/18/18)
#   3. Writes train/test point TSVs (property values only — no geochem/lith)
#   4. Runs examples/kriging_env/run_ordinary_kriging.jl under the main project
#
# Run:
#   julia --project=. examples/kriging_cloncurry_petro.jl
#
# Env: CLONCURRY_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_SPLIT_SEED,
#      SMARTPRIOR_VARIO_TSV, SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z

using SmartPrior
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK",
                 joinpath(ROOT, "tmp_cloncurry_variogram"))
const VARIO_TSV = get(ENV, "SMARTPRIOR_VARIO_TSV",
                      joinpath(ROOT, "tmp_cloncurry_variogram",
                               "cloncurry_variogram_summary.tsv"))
const KRIG_ENV = joinpath(ROOT, "examples", "kriging_env")
const KRIG_RUNNER = joinpath(KRIG_ENV, "run_ordinary_kriging.jl")
const SPLIT_SEED = parse(Int, get(ENV, "SMARTPRIOR_SPLIT_SEED", "2026"))
const TARGET_CELLS = parse(Int, get(ENV, "SMARTPRIOR_TARGET_CELLS", "50000"))
const CELL_ENV = strip(get(ENV, "SMARTPRIOR_CELL_M", ""))
const CELL_Z_ENV = strip(get(ENV, "SMARTPRIOR_CELL_Z", ""))
const FRACTIONS = (0.70, 0.15, 0.15)
const SPLIT_TARGETS = (84, 18, 18)
const TOLERANCE = 22.5
const N_LAGS = 15
const SAMPLE_SIZE = 2500
const SEED = 42
const NN_REF = Dict(
    "density" => (rmse = 0.600, r2 = -0.742),
    "susceptibility" => (rmse = 1.340, r2 = 0.034),
)
mkpath(WORK)

function default_cloncurry_root()
    env = get(ENV, "CLONCURRY_ROOT", "")
    !isempty(env) && return env
    for c in (
        joinpath(homedir(), "Desktop", "datasets4HY",
                 "Cloncurry_integrated_2026-09-17"),
        "/Users/hayrunnisayildiz/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17",
    )
        isdir(joinpath(c, "derived")) && return c
    end
    error("set CLONCURRY_ROOT to Cloncurry_integrated_2026-09-17")
end

#---------- variogram helpers (same convention as variogram_cloncurry.jl) ----------

function spherical_gamma(h, a, sill, nugget)
    a_safe = max(a, oftype(a, 1e-6))
    if h <= a
        r = h / a_safe
        return nugget + sill * (oftype(r, 1.5) * r - oftype(r, 0.5) * r^3)
    end
    return nugget + sill
end

function exponential_gamma(h, a, sill, nugget)
    return nugget + sill * (1 - exp(-3 * h / max(a, oftype(a, 1e-6))))
end

spherical_model(h, p) = spherical_gamma.(h, p[1], p[2], p[3])
exponential_model(h, p) = exponential_gamma.(h, p[1], p[2], p[3])

function fit_bounded(model, x, y, p0; lower, upper, n_restarts = 8)
    lo = collect(Float64, lower)
    hi = collect(Float64, upper)
    best_p = clamp.(collect(Float64, p0), lo, hi)
    best_s = sum(abs2, model(x, best_p) .- y)
    spans = hi .- lo
    rng = Xoshiro(1)
    for k in 0:n_restarts
        p = k == 0 ? copy(best_p) : clamp.(lo .+ rand(rng, 3) .* spans, lo, hi)
        step = 0.25
        for _ in 1:80
            improved = false
            for j in 1:3
                for s in (-step, step)
                    cand = copy(p)
                    cand[j] = j == 1 ?
                        clamp(cand[j] * exp(s), lo[j], hi[j]) :
                        clamp(cand[j] * (1 + s), lo[j], hi[j])
                    ss = sum(abs2, model(x, cand) .- y)
                    if ss + 1e-15 < best_s
                        best_s = ss
                        p = cand
                        improved = true
                    end
                end
            end
            improved || (step *= 0.5)
            step < 1e-4 && break
        end
        ss = sum(abs2, model(x, p) .- y)
        if ss < best_s
            best_s = ss
            best_p = copy(p)
        end
    end
    return best_p
end

function angular_difference_deg(angle_deg::Real, target_deg::Real)
    diff = abs(mod(angle_deg - target_deg + 180.0, 360.0) - 180.0)
    return min(diff, 180.0 - diff)
end

function directional_pair_stats(coords, values; azimuth, tolerance, maxlag, min_lag_m)
    n = length(values)
    distances = Float64[]
    gammas = Float64[]
    pair_count = 0
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        dx = coords[j, 1] - coords[i, 1]
        dy = coords[j, 2] - coords[i, 2]
        h = hypot(dx, dy)
        h < min_lag_m && continue
        h > maxlag && continue
        az = mod(atand(dy, dx), 180.0)
        angular_difference_deg(az, azimuth) > tolerance && continue
        push!(distances, h)
        push!(gammas, 0.5 * abs2(values[j] - values[i]))
        pair_count += 1
    end
    pair_count < 10 && return nothing
    return (distances = distances, gammas = gammas)
end

function experimental_variogram(distances, gammas; n_lags, maxlag, min_lag_m)
    edges = range(Float64(min_lag_m), Float64(maxlag); length = n_lags + 1)
    centers = Float64[]; experimental = Float64[]
    for k in 1:n_lags
        lo, hi = edges[k], edges[k + 1]
        idx = findall(d -> lo <= d < hi, distances)
        length(idx) < 3 && continue
        push!(centers, 0.5 * (lo + hi))
        push!(experimental, mean(gammas[idx]))
    end
    length(centers) < 3 && return nothing
    return (bins = centers, experimental = experimental)
end

function fit_variogram_model(model_name, bins, experimental, value_variance;
                             range_min = 10.0)
    value_var = max(Float64(value_variance), 1e-6)
    sill_cap = max(value_var * 2.0, maximum(experimental), 1e-6)
    range_upper = max(bins[end] * 2.0, Float64(range_min) * 4.0)
    p0 = [max(Float64(range_min), min(bins[end] * 0.35, range_upper)),
          sill_cap * 0.7, sill_cap * 0.1]
    lower = [Float64(range_min), 1e-8, 0.0]
    upper = [range_upper, sill_cap * 2.0, sill_cap]
    fit_bins = length(bins) > 4 ? bins[2:end] : bins
    fit_exp = length(experimental) > 4 ? experimental[2:end] : experimental
    model = model_name == "exponential" ? exponential_model : spherical_model
    params = fit_bounded(model, fit_bins, fit_exp, p0; lower = lower, upper = upper)
    rmse = sqrt(mean(abs2.(model(bins, params) .- experimental)))
    return params, rmse
end

function recover_major_nugget_sill(coords, values; azimuth, model_name,
                                   maxlag, min_lag_m)
    sv = var(values; corrected = false)
    stats = directional_pair_stats(coords, values; azimuth = Float64(azimuth),
                                   tolerance = TOLERANCE, maxlag = Float64(maxlag),
                                   min_lag_m = Float64(min_lag_m))
    stats === nothing && error("no directional pairs at azimuth=$azimuth")
    expv = experimental_variogram(stats.distances, stats.gammas;
                                  n_lags = N_LAGS, maxlag = maxlag,
                                  min_lag_m = min_lag_m)
    expv === nothing && error("experimental variogram failed at az=$azimuth")
    params, rmse = fit_variogram_model(model_name, expv.bins, expv.experimental, sv;
                                       range_min = min_lag_m)
    nugget = params[3]
    partial = params[2]
    note = "curve_fit"
    # Floor hit → structured sill is effectively zero; OK becomes train-mean.
    # Keep the fitted numbers (today's major-axis parameters) and flag it.
    if partial < 1e-4 * max(sv, 1e-12)
        note = "curve_fit_pure_nugget"
    end
    return (range = params[1], partial_sill = partial, nugget = nugget,
            total_sill = partial + nugget, rmse = rmse, note = note,
            sample_variance = sv)
end

function median_nearest_neighbour(coords::AbstractMatrix)
    n = size(coords, 1)
    n < 2 && return NaN
    nn = fill(Inf, n)
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        d = hypot(coords[j, 1] - coords[i, 1],
                  coords[j, 2] - coords[i, 2],
                  coords[j, 3] - coords[i, 3])
        d < nn[i] && (nn[i] = d)
    end
    return median(nn)
end

function maybe_subsample(coords, values, sample_size; rng)
    n = length(values)
    n <= sample_size && return coords, values
    idx = randperm(rng, n)[1:sample_size]
    return coords[idx, :], values[idx]
end

function property_points(s::CloncurrySamples, name::AbstractString, keep::BitVector)
    xs = Float64[]; ys = Float64[]; zs = Float64[]; vs = Float64[]
    for i in 1:length(s)
        keep[i] || continue
        e, no, el = s.east[i], s.north[i], s.elev[i]
        (isfinite(e) && isfinite(no) && isfinite(el)) || continue
        val = if name == "density"
            d = s.density_g_cm3[i]
            isfinite(d) && d > 0 ? Float64(d) : NaN
        elseif name == "susceptibility"
            sus = s.susceptibility_SI[i]
            isfinite(sus) && sus > 0 ? log10(sus) : NaN
        else
            error("only density / susceptibility")
        end
        isfinite(val) || continue
        push!(xs, e); push!(ys, no); push!(zs, el); push!(vs, val)
    end
    return hcat(xs, ys, zs), vs
end

function named_hole_mask(drillhole)
    m = falses(length(drillhole))
    @inbounds for i in eachindex(drillhole)
        m[i] = !isempty(strip(String(drillhole[i])))
    end
    return m
end

function load_variogram_rows(path)
    rows = Dict{String,Dict{String,Any}}()
    open(path) do io
        header = split(readline(io), '\t')
        col = Dict(h => i for (i, h) in enumerate(header))
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, '\t')
            name = parts[col["property"]]
            rows[name] = Dict{String,Any}(
                "min_lag_m" => parse(Float64, parts[col["min_lag_m"]]),
                "maxlag_m" => parse(Float64, parts[col["maxlag_m"]]),
                "major_azimuth_deg" => parse(Float64, parts[col["major_azimuth_deg"]]),
                "major_range_m" => parse(Float64, parts[col["major_range_m"]]),
                "minor_range_m" => parse(Float64, parts[col["minor_range_m"]]),
                "vertical_range_m" => parse(Float64, parts[col["vertical_range_m"]]),
                "major_model" => parts[col["major_model"]],
            )
        end
    end
    return rows
end

function write_points_tsv(path, coords, values)
    open(path, "w") do io
        println(io, "east\tnorth\telev\tvalue")
        for i in 1:length(values)
            @printf(io, "%.6f\t%.6f\t%.6f\t%.10f\n",
                    coords[i, 1], coords[i, 2], coords[i, 3], values[i])
        end
    end
end

#---------- main ----------

println(repeat("=", 72))
println("  CLONCURRY — GeoStats ordinary kriging (density, susceptibility)")
println(repeat("=", 72))
@printf("  GeoStats runner  %s\n", KRIG_RUNNER)
isfile(KRIG_RUNNER) || error("missing $KRIG_RUNNER")

DATASET = default_cloncurry_root()
samples = load_cloncurry_samples(DATASET)
aabb = cloncurry_sample_bounds(samples)
spacing = cloncurry_district_spacing(aabb; target = TARGET_CELLS)
CELL = isempty(CELL_ENV) ? spacing.cell : parse(Float64, CELL_ENV)
CELL_Z = isempty(CELL_Z_ENV) ? 100.0 : parse(Float64, CELL_Z_ENV)
bounds = cloncurry_sample_bounds(samples; pad_xy = CELL / 2, pad_z = CELL_Z / 2)
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)

vario = load_variogram_rows(VARIO_TSV)
@printf("  dataset   %s\n", DATASET)
@printf("  grid      %s  XY=%.0f  Z=%.0f\n", string(size(grid)), CELL, CELL_Z)
@printf("  variogram %s\n", VARIO_TSV)
@printf("  split     seed=%d  targets=%s\n", SPLIT_SEED, string(SPLIT_TARGETS))

gkeys = cloncurry_group_keys(samples.drillhole)
inside = cloncurry_inside_mask(grid, samples)
eligible = inside .& named_hole_mask(samples.drillhole)
holdout = group_holdout(gkeys, eligible; unit = :groups, targets = SPLIT_TARGETS,
                        fractions = FRACTIONS, rng = Xoshiro(SPLIT_SEED))
train_keep = group_mask(gkeys, holdout.groups_train) .& inside
test_keep = group_mask(gkeys, holdout.groups_test) .& inside
@printf("  holes     %d / %d / %d\n",
        holdout.n_train_groups, holdout.n_val_groups, holdout.n_test_groups)

rng = Random.Xoshiro(SEED)
params_path = joinpath(WORK, "cloncurry_kriging_params.tsv")
open(params_path, "w") do io
    println(io, "property\tmajor_azimuth_deg\tmajor_range_m\tminor_range_m\t",
            "vertical_range_m\tmajor_model\tnugget\tpartial_sill\ttotal_sill\t",
            "train_tsv\ttest_tsv\tpred_tsv")
end

for prop in ("density", "susceptibility")
    v = vario[prop]
    println()
    println(repeat("-", 72))
    @printf("  PROPERTY: %s\n", prop)

    keep_all = trues(length(samples))
    coords_all, values_all = property_points(samples, prop, keep_all)
    prop_nn = median_nearest_neighbour(coords_all)
    min_lag = max(v["min_lag_m"], isfinite(prop_nn) ? prop_nn : v["min_lag_m"])
    coords_fit, values_fit = maybe_subsample(coords_all, values_all, SAMPLE_SIZE;
                                             rng = rng)
    ns = recover_major_nugget_sill(coords_fit, values_fit;
                                   azimuth = v["major_azimuth_deg"],
                                   model_name = v["major_model"],
                                   maxlag = v["maxlag_m"],
                                   min_lag_m = min_lag)
    @printf("  ranges    maj=%.1f min=%.1f vert=%.1f  az=%.0f°  model=%s\n",
            v["major_range_m"], v["minor_range_m"], v["vertical_range_m"],
            v["major_azimuth_deg"], v["major_model"])
    @printf("  nugget=%.6f  partial_sill=%.6f  total_sill=%.6f  (re-fit RMSE=%.4f)\n",
            ns.nugget, ns.partial_sill, ns.total_sill, ns.rmse)
    ns.note != "curve_fit" && println("  sill note: ", ns.note)

    coords_tr, values_tr = property_points(samples, prop, train_keep)
    coords_te, values_te = property_points(samples, prop, test_keep)
    train_tsv = joinpath(WORK, "cloncurry_kriging_$(prop)_train.tsv")
    test_tsv = joinpath(WORK, "cloncurry_kriging_$(prop)_test.tsv")
    pred_tsv = joinpath(WORK, "cloncurry_kriging_$(prop)_pred.tsv")
    write_points_tsv(train_tsv, coords_tr, values_tr)
    write_points_tsv(test_tsv, coords_te, values_te)
    @printf("  wrote     train n=%d  test n=%d\n",
            length(values_tr), length(values_te))

    open(params_path, "a") do io
        @printf(io,
                "%s\t%.1f\t%.4f\t%.4f\t%.4f\t%s\t%.8f\t%.8f\t%.8f\t%s\t%s\t%s\n",
                prop, v["major_azimuth_deg"],
                v["major_range_m"], v["minor_range_m"], v["vertical_range_m"],
                v["major_model"], ns.nugget, ns.partial_sill, ns.total_sill,
                train_tsv, test_tsv, pred_tsv)
    end
end
println("  params → ", params_path)

println()
println("  launching GeoStats OrdinaryKriging …")
cmd = `$(Base.julia_cmd()) --project=$(ROOT) $(KRIG_RUNNER) $(params_path) $(WORK)`
println("  ", cmd)
run(cmd)

#---------- comparison table ----------

report = joinpath(WORK, "cloncurry_kriging_holdout_report.tsv")
println()
println(repeat("=", 88))
println("  COMPARISON — neural net (Z=100 m) vs GeoStats ordinary kriging")
println(repeat("=", 88))
@printf("%-16s %12s %12s %12s %12s\n",
        "property", "NN RMSE", "NN R²", "OK RMSE", "OK R²")
if isfile(report)
    open(report) do io
        hdr = Base.split(readline(io), '\t')
        col = Dict(h => i for (i, h) in enumerate(hdr))
        for line in eachline(io)
            parts = Base.split(line, '\t')
            prop = parts[col["property"]]
            nn = NN_REF[prop]
            @printf("%-16s %12.3f %12.3f %12.3f %12.3f\n",
                    prop, nn.rmse, nn.r2,
                    parse(Float64, parts[col["kriging_rmse"]]),
                    parse(Float64, parts[col["kriging_r2"]]))
        end
    end
else
    println("  (report missing: $report)")
end
println(repeat("=", 88))
println("  NN: Z=100 m calibration. OK: GeoStats.jl via examples/kriging_env.")
println("Done.")
