# Directional variogram + anisotropy for Cloncurry district petrophysics.
#
# Ports the *method* of NISAI `variogram_analysis.py` (azimuth scan, directional
# pair filter, Spherical/Exponential fit, degenerate-range fallback) — not a
# line-by-line copy. No scikit-gstat / geostats package; LsqFit.jl for curve_fit.
#
# Composite support is not used: samples are point specimens. Minimum lag is the
# median nearest-neighbour distance among finite-xyz points (~9.5 m).
#
# Targets: grade (log10 Cu), density, log10 susceptibility, log10
# conductivity_100kHz. Compares fitted ranges to the district prior grid
# (2300 m XY / 200 m Z). Grade and conductivity also re-fit at 2× and 3× the
# default maxlag to check whether the major range sits on the optimizer ceiling.
#
# Run:  julia --project=. examples/variogram_cloncurry.jl
# Env:  CLONCURRY_ROOT, SMARTPRIOR_WORK, SMARTPRIOR_SAMPLE_SIZE,
#       SMARTPRIOR_CELL_M, SMARTPRIOR_CELL_Z, SMARTPRIOR_TARGET_CELLS,
#       SMARTPRIOR_MAXLAG_MULTS (comma list, default 1,2,3 for grade/cond)

using SmartPriorMT
using LinearAlgebra
using LsqFit
using Printf
using Random
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK",
                 joinpath(ROOT, "tmp_cloncurry_variogram"))
const TARGET_CELLS = parse(Int, get(ENV, "SMARTPRIOR_TARGET_CELLS", "50000"))
const CELL_ENV = strip(get(ENV, "SMARTPRIOR_CELL_M", ""))
const CELL_Z_ENV = strip(get(ENV, "SMARTPRIOR_CELL_Z", ""))
const SAMPLE_SIZE = parse(Int, get(ENV, "SMARTPRIOR_SAMPLE_SIZE", "2500"))
const N_LAGS = parse(Int, get(ENV, "SMARTPRIOR_VARIO_LAGS", "15"))
const TOLERANCE = parse(Float64, get(ENV, "SMARTPRIOR_VARIO_TOL", "22.5"))
const MAXLAG_MULTS = let raw = strip(get(ENV, "SMARTPRIOR_MAXLAG_MULTS", "1,2,3"))
    [parse(Float64, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end
const SEED = 42
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

#---------- models ----------

# Dual-safe (LsqFit ForwardDiff); keep arithmetic generic in `a,sill,nugget`.
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

function model_gamma(name::AbstractString, h, a, sill, nugget)
    return name == "exponential" ?
           exponential_gamma(h, a, sill, nugget) :
           spherical_gamma(h, a, sill, nugget)
end

function spherical_model(h, p)
    a, sill, nugget = p[1], p[2], p[3]
    return spherical_gamma.(h, a, sill, nugget)
end

function exponential_model(h, p)
    a, sill, nugget = p[1], p[2], p[3]
    return exponential_gamma.(h, a, sill, nugget)
end

#---------- geometry helpers ----------

"""Smallest angular distance on a 180° period (undirected azimuth)."""
function angular_difference_deg(angle_deg::Real, target_deg::Real)
    diff = abs(mod(angle_deg - target_deg + 180.0, 360.0) - 180.0)
    return min(diff, 180.0 - diff)
end

function default_maxlag(coords::AbstractMatrix; is_vertical::Bool = false)
    if is_vertical && size(coords, 2) >= 3
        span = maximum(coords[:, 3]) - minimum(coords[:, 3])
    else
        span = max(maximum(coords[:, 1]) - minimum(coords[:, 1]),
                   maximum(coords[:, 2]) - minimum(coords[:, 2]))
    end
    return max(10.0, 0.5 * span)
end

"""Median nearest-neighbour Euclidean distance (3-D) among rows of `coords`."""
function median_nearest_neighbour(coords::AbstractMatrix{<:Real})
    n = size(coords, 1)
    n < 2 && return NaN
    nn = fill(Inf, n)
    @inbounds for i in 1:n
        xi, yi, zi = coords[i, 1], coords[i, 2], coords[i, 3]
        best = Inf
        for j in 1:n
            i == j && continue
            d = hypot(hypot(coords[j, 1] - xi, coords[j, 2] - yi),
                      coords[j, 3] - zi)
            d < best && (best = d)
        end
        nn[i] = best
    end
    return median(nn)
end

function percentile(v::AbstractVector{<:Real}, pct::Real)
    a = filter(isfinite, Float64.(v))
    isempty(a) && return NaN
    return quantile(a, pct / 100)
end

#---------- pairwise / binning / fit ----------

struct FittedVariogram
    model_name::String
    parameters::Vector{Float64}   # [range, partial_sill, nugget]
    rmse::Float64
    bins::Vector{Float64}
    experimental::Vector{Float64}
end

function directional_pair_stats(coords::AbstractMatrix{<:Real},
                                values::AbstractVector{<:Real};
                                azimuth::Float64 = 0.0,
                                tolerance::Float64 = 22.5,
                                maxlag::Float64,
                                is_vertical::Bool = false,
                                min_lag_m::Float64 = 0.0)
    n = length(values)
    n < 3 && return nothing
    dist = Float64[]
    semiv = Float64[]
    sizehint!(dist, n * (n - 1) ÷ 4)
    sizehint!(semiv, n * (n - 1) ÷ 4)
    az_t = mod(azimuth, 180.0)
    @inbounds for j in 2:n
        xj, yj, zj = coords[j, 1], coords[j, 2], coords[j, 3]
        vj = values[j]
        for i in 1:(j - 1)
            dx = xj - coords[i, 1]
            dy = yj - coords[i, 2]
            dz = zj - coords[i, 3]
            if is_vertical
                horiz = hypot(dx, dy)
                dip = rad2deg(atan(abs(dz), max(horiz, 1e-9)))
                angular_difference_deg(dip, 90.0) <= tolerance || continue
                d = hypot(horiz, dz)
            else
                d = hypot(dx, dy)
                ang = mod(rad2deg(atan(dy, dx)), 180.0)
                angular_difference_deg(ang, az_t) <= tolerance || continue
            end
            (isfinite(d) && d > 0.0 && d <= maxlag && d >= min_lag_m) || continue
            push!(dist, d)
            push!(semiv, 0.5 * (values[i] - vj)^2)
        end
    end
    isempty(dist) && return nothing
    return (distances = dist, semivariances = semiv, pair_count = length(dist))
end

function bin_experimental(distances, semivariances, n_lags, maxlag, min_lag_m)
    lag_floor = max(0.0, Float64(min_lag_m))
    maxlag <= lag_floor + 1e-6 && return nothing
    edges = range(lag_floor, maxlag; length = n_lags + 1)
    centers = Float64[]
    experimental = Float64[]
    for k in 1:n_lags
        lo, hi = edges[k], edges[k + 1]
        s = 0.0
        c = 0
        @inbounds for t in eachindex(distances)
            d = distances[t]
            if d > lo && d <= hi
                s += semivariances[t]
                c += 1
            end
        end
        if c > 0
            push!(centers, 0.5 * (lo + hi))
            push!(experimental, s / c)
        end
    end
    length(centers) < 3 && return nothing
    return (bins = centers, experimental = experimental)
end

function pairwise_lag_diagnostics(coords, values; azimuth, tolerance, maxlag,
                                  min_lag_m, is_vertical)
    stats = directional_pair_stats(coords, values; azimuth, tolerance, maxlag,
                                   is_vertical, min_lag_m)
    if stats === nothing
        return Dict{String,Float64}(
            "n_directional_pairs" => 0.0,
            "min_lag_m" => Float64(min_lag_m),
            "max_lag_m" => Float64(maxlag),
        )
    end
    d = stats.distances
    return Dict{String,Float64}(
        "n_directional_pairs" => Float64(stats.pair_count),
        "min_lag_m" => Float64(min_lag_m),
        "max_lag_m" => Float64(maxlag),
        "pair_distance_min_m" => percentile(d, 0),
        "pair_distance_p10_m" => percentile(d, 10),
        "pair_distance_median_m" => percentile(d, 50),
        "pair_distance_p90_m" => percentile(d, 90),
        "pair_distance_max_m" => percentile(d, 100),
    )
end

function clamp_range(r; range_min=nothing, range_max=nothing)
    out = Float64(r)
    range_min !== nothing && (out = max(out, Float64(range_min)))
    range_max !== nothing && (out = min(out, Float64(range_max)))
    return out
end

function is_degenerate_range_fit(params; range_min, model_name)
    fitted_range = params[1]
    partial_sill = length(params) > 1 ? params[2] : 0.0
    at_floor = fitted_range <= Float64(range_min) * 1.05
    !at_floor && return false
    model_name == "exponential" && return true
    return partial_sill <= 1e-6
end

function finalize_variogram_range(fitted_range; range_min, range_max, lag_diag,
                                  experimental=nothing, sample_variance=nothing)
    resolved = clamp_range(fitted_range; range_min, range_max)
    p90 = get(lag_diag, "pair_distance_p90_m", NaN)
    med = get(lag_diag, "pair_distance_median_m", NaN)

    if resolved <= Float64(range_min) * 1.05 && isfinite(p90) && p90 > range_min * 2
        return clamp_range(p90; range_min, range_max),
               @sprintf("degenerate_curve_fit; using pairwise p90 lag %.1f m", p90)
    end

    if isfinite(med) && med > range_min * 2 && resolved < med * 0.25
        if isfinite(p90) && p90 > resolved
            return clamp_range(p90; range_min, range_max),
                   @sprintf("fitted range %.1f m << median pair lag %.1f m; using pairwise p90 lag %.1f m",
                            resolved, med, p90)
        end
    end

    if experimental !== nothing && sample_variance !== nothing
        expv = filter(isfinite, Float64.(experimental))
        sv = max(Float64(sample_variance), 1e-6)
        if length(expv) >= 3 && (maximum(expv) - minimum(expv)) / sv < 0.35
            if isfinite(p90) && p90 > resolved * 1.5
                return clamp_range(p90; range_min, range_max),
                       @sprintf("flat experimental variogram; using pairwise p90 lag %.1f m", p90)
            end
        end
    end

    return resolved, "curve_fit"
end

function range_upper_bound(bins, range_min, range_max)
    ru = max(bins[end] * 2.0, Float64(range_min) * 4.0)
    range_max !== nothing && (ru = min(ru, Float64(range_max)))
    return ru
end

function fit_variogram_model(model_name, bins, experimental, value_variance;
                             range_min=10.0, range_max=nothing)
    value_var = max(Float64(value_variance), 1e-6)
    sill_cap = max(value_var * 2.0, maximum(experimental), 1e-6)
    range_upper = range_upper_bound(bins, range_min, range_max)

    p0 = [
        max(Float64(range_min), min(bins[end] * 0.35, range_upper)),
        sill_cap * 0.7,
        sill_cap * 0.1,
    ]
    lower = [Float64(range_min), 1e-8, 0.0]
    upper = [range_upper, sill_cap * 2.0, sill_cap]

    fit_bins = length(bins) > 4 ? bins[2:end] : bins
    fit_exp = length(experimental) > 4 ? experimental[2:end] : experimental
    model = model_name == "exponential" ? exponential_model : spherical_model

    fit = curve_fit(model, fit_bins, fit_exp, p0; lower=lower, upper=upper)
    params = Float64.(fit.param)
    fitted = model(bins, params)
    rmse = sqrt(mean(abs2.(fitted .- experimental)))
    return params, rmse, range_upper
end

function find_best_model(coords, values; azimuth=0.0, tolerance=22.5, n_lags=20,
                         maxlag=nothing, is_vertical=false, range_min=10.0,
                         range_max=nothing, min_lag_m=0.0)
    n = length(values)
    n < 3 && return nothing, "", Inf, NaN

    ml = maxlag === nothing ?
         default_maxlag(coords; is_vertical=is_vertical) : Float64(maxlag)

    stats = directional_pair_stats(coords, values; azimuth=Float64(azimuth),
                                   tolerance=Float64(tolerance), maxlag=ml,
                                   is_vertical=is_vertical,
                                   min_lag_m=Float64(min_lag_m))
    (stats === nothing || stats.pair_count < 30) && return nothing, "", Inf, NaN

    binned = bin_experimental(stats.distances, stats.semivariances, n_lags, ml,
                              min_lag_m)
    binned === nothing && return nothing, "", Inf, NaN

    best_rmse = Inf
    best_vario = nothing
    best_name = ""
    value_variance = var(values; corrected=false)

    best_range_upper = NaN
    for model_name in ("spherical", "exponential")
        try
            params, rmse, ru = fit_variogram_model(model_name, binned.bins,
                                                   binned.experimental,
                                                   value_variance;
                                                   range_min=range_min,
                                                   range_max=range_max)
            (!isfinite(params[1])) && continue
            params[1] = clamp_range(params[1]; range_min=range_min,
                                    range_max=range_max)
            score = Float64(rmse)
            if is_degenerate_range_fit(params; range_min=range_min,
                                       model_name=model_name)
                score += 1.0
            end
            if score < best_rmse
                best_rmse = score
                best_name = model_name
                best_range_upper = ru
                best_vario = FittedVariogram(model_name, params, rmse,
                                             binned.bins, binned.experimental)
            end
        catch
            # LsqFit / bounds failure — try the other model
        end
    end

    return best_vario, best_name, best_rmse, best_range_upper
end

function apply_finalized_range(vario::FittedVariogram; range_min, range_max,
                               lag_diag, sample_variance)
    raw = vario.parameters[1]
    final_r, note = finalize_variogram_range(raw; range_min, range_max, lag_diag,
                                             experimental=vario.experimental,
                                             sample_variance=sample_variance)
    if abs(final_r - raw) > 1e-6
        p = copy(vario.parameters)
        p[1] = final_r
        vario = FittedVariogram(vario.model_name, p, vario.rmse, vario.bins,
                                vario.experimental)
    end
    return vario, final_r, note
end

#---------- property extraction ----------

function property_points(s::CloncurrySamples, name::AbstractString)
    n = length(s)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    vs = Float64[]
    sizehint!(xs, n)
    for i in 1:n
        e, no, el = s.east[i], s.north[i], s.elev[i]
        (isfinite(e) && isfinite(no) && isfinite(el)) || continue
        val = if name == "grade"
            cu_log10(s.cu_ppm[i]; detection_limit = CLONCURRY_CU_DL_PPM)
        elseif name == "density"
            d = s.density_g_cm3[i]
            isfinite(d) && d > 0 ? Float64(d) : NaN
        elseif name == "susceptibility"
            sus = s.susceptibility_SI[i]
            isfinite(sus) && sus > 0 ? log10(sus) : NaN
        elseif name == "conductivity_100kHz"
            c = s.conductivity_S_m_100kHz[i]
            isfinite(c) ? log10(max(c, CLONCURRY_COND_FLOOR_S_M)) : NaN
        else
            error("unknown property $name")
        end
        isfinite(val) || continue
        push!(xs, e); push!(ys, no); push!(zs, el); push!(vs, val)
    end
    coords = hcat(xs, ys, zs)
    return coords, vs
end

function maybe_subsample(coords, values, sample_size; rng)
    n = length(values)
    n <= sample_size && return coords, values
    idx = randperm(rng, n)[1:sample_size]
    return coords[idx, :], values[idx]
end

#---------- full analysis for one property ----------

function analyse_property(coords0, values0; name, tolerance, n_lags, maxlag,
                          min_lag_m, range_min, range_max, sample_size, rng,
                          label = nothing)
    tag = label === nothing ? String(name) : String(label)
    println()
    println(repeat("=", 72))
    println("  PROPERTY: ", tag)
    println(repeat("=", 72))
    n0 = length(values0)
    sample_variance = var(values0; corrected=false)
    @printf("  n_points (finite)     %d\n", n0)
    @printf("  sample variance       %.6f\n", sample_variance)
    @printf("  min_lag / range_min   %.3f m  (median NN)\n", min_lag_m)

    coords, values = maybe_subsample(coords0, values0, sample_size; rng=rng)
    if length(values) < n0
        @printf("  subsampled to         %d  (seed=%d)\n", length(values), SEED)
    end

    ml_default = default_maxlag(coords; is_vertical=false)
    ml = maxlag === nothing ? ml_default : Float64(maxlag)
    @printf("  maxlag (horizontal)   %.1f m  (default 0.5·span = %.1f m)\n",
            ml, ml_default)

    # NISAI uses az=0 pairwise diag for major-axis finalize
    major_lag_diag = pairwise_lag_diagnostics(
        coords, values; azimuth=0.0, tolerance, maxlag=ml,
        min_lag_m, is_vertical=false)

    @printf("  pairwise (az=0)       n=%d  median=%.1f  p90=%.1f m\n",
            Int(get(major_lag_diag, "n_directional_pairs", 0)),
            get(major_lag_diag, "pair_distance_median_m", NaN),
            get(major_lag_diag, "pair_distance_p90_m", NaN))

    println("  azimuth scan 0–165° (step 15°)...")
    best_az = 0.0
    best_score = Inf
    max_range = -1.0
    major_vario = nothing
    major_ru = NaN

    for az in 0.0:15.0:165.0
        vario, _, rmse, ru = find_best_model(
            coords, values; azimuth=az, tolerance, n_lags, maxlag=ml,
            is_vertical=false, range_min, range_max, min_lag_m)
        vario === nothing && continue
        v_range = vario.parameters[1]
        score = Float64(rmse)
        if score < best_score || (abs(score - best_score) < 1e-6 && v_range > max_range)
            best_score = score
            max_range = v_range
            best_az = az
            major_vario = vario
            major_ru = ru
        end
        @printf("    az=%5.1f°  model=%-11s  R=%10.1f  RMSE=%.5f\n",
                az, vario.model_name, v_range, rmse)
    end

    major_vario === nothing && error("$tag: major-axis fit failed")
    # Prefer range_upper from the winning fit; fall back from bins
    if !isfinite(major_ru)
        major_ru = range_upper_bound(major_vario.bins, range_min, range_max)
    end
    major_vario, max_range, major_note = apply_finalized_range(
        major_vario; range_min, range_max, lag_diag=major_lag_diag,
        sample_variance=sample_variance)
    major_note != "curve_fit" && println("  => major range fix: ", major_note)
    at_ceiling = isfinite(major_ru) && max_range >= 0.95 * major_ru
    @printf("  => MAJOR  az=%.0f°  R=%.2f m  (%s, RMSE=%.4f)  range_upper=%.1f%s\n",
            best_az, max_range, major_vario.model_name, major_vario.rmse, major_ru,
            at_ceiling ? "  [AT CEILING]" : "")

    minor_az = mod(best_az + 90.0, 180.0)
    minor_lag_diag = pairwise_lag_diagnostics(
        coords, values; azimuth=minor_az, tolerance, maxlag=ml,
        min_lag_m, is_vertical=false)
    minor_vario, _, _, _ = find_best_model(
        coords, values; azimuth=minor_az, tolerance, n_lags, maxlag=ml,
        is_vertical=false, range_min, range_max, min_lag_m)
    minor_vario === nothing && error("$tag: minor-axis fit failed (az=$minor_az)")
    minor_vario, minor_range, minor_note = apply_finalized_range(
        minor_vario; range_min, range_max, lag_diag=minor_lag_diag,
        sample_variance=sample_variance)
    minor_note != "curve_fit" && println("  => minor range fix: ", minor_note)
    @printf("  => MINOR  az=%.0f°  R=%.2f m  (%s, RMSE=%.4f)\n",
            minor_az, minor_range, minor_vario.model_name, minor_vario.rmse)

    ml_v = maxlag === nothing ? default_maxlag(coords; is_vertical=true) : maxlag
    # Vertical maxlag stays at the default 0.5·Δz unless explicitly set; for
    # horizontal maxlag sweeps, keep vertical on the geometric default.
    if maxlag !== nothing
        ml_v = default_maxlag(coords; is_vertical=true)
    end
    vert_lag_diag = pairwise_lag_diagnostics(
        coords, values; azimuth=best_az, tolerance, maxlag=ml_v,
        min_lag_m, is_vertical=true)
    vert_vario, _, _, _ = find_best_model(
        coords, values; azimuth=best_az, tolerance, n_lags, maxlag=ml_v,
        is_vertical=true, range_min, range_max, min_lag_m)
    vert_vario === nothing && error("$tag: vertical-axis fit failed")
    vert_vario, vert_range, vert_note = apply_finalized_range(
        vert_vario; range_min, range_max, lag_diag=vert_lag_diag,
        sample_variance=sample_variance)
    vert_note != "curve_fit" && println("  => vertical range fix: ", vert_note)
    @printf("  => VERT               R=%.2f m  (%s, RMSE=%.4f)\n",
            vert_range, vert_vario.model_name, vert_vario.rmse)

    ani_hm = max_range / minor_range
    ani_hv = max_range / vert_range
    @printf("  anisotropy major/minor     %.3f\n", ani_hm)
    @printf("  anisotropy major/vertical  %.3f\n", ani_hv)

    return (
        name = String(name),
        label = tag,
        n = n0,
        sample_variance = sample_variance,
        min_lag_m = min_lag_m,
        maxlag_m = ml,
        maxlag_default_m = ml_default,
        range_upper_m = major_ru,
        at_ceiling = at_ceiling,
        major_azimuth = best_az,
        minor_azimuth = minor_az,
        major_range = max_range,
        minor_range = minor_range,
        vertical_range = vert_range,
        major_model = major_vario.model_name,
        minor_model = minor_vario.model_name,
        vertical_model = vert_vario.model_name,
        major_rmse = major_vario.rmse,
        minor_rmse = minor_vario.rmse,
        vertical_rmse = vert_vario.rmse,
        major_note = major_note,
        minor_note = minor_note,
        vertical_note = vert_note,
        ani_major_minor = ani_hm,
        ani_major_vertical = ani_hv,
        major_nugget = major_vario.parameters[3],
        major_partial_sill = major_vario.parameters[2],
    )
end

#---------- main ----------

const DATASET = default_cloncurry_root()
samples = load_cloncurry_samples(DATASET)
aabb = cloncurry_sample_bounds(samples)
spacing = cloncurry_district_spacing(aabb; target = TARGET_CELLS)
CELL = isempty(CELL_ENV) ? spacing.cell : parse(Float64, CELL_ENV)
CELL_Z = isempty(CELL_Z_ENV) ? spacing.cell_z : parse(Float64, CELL_Z_ENV)
bounds = cloncurry_sample_bounds(samples; pad_xy = CELL / 2, pad_z = CELL_Z / 2)
grid = cloncurry_grid(bounds; cell = CELL, cell_z = CELL_Z)

println(repeat("=", 72))
println("  CLONCURRY DISTRICT — DIRECTIONAL VARIOGRAM (NISAI method)")
println(repeat("=", 72))
@printf("  dataset   %s\n", DATASET)
@printf("  samples   %d  (finite-xyz AABB n=%d)\n",
        length(samples), aabb.n_samples)
@printf("  grid      %s  cells=%d  XY=%.0f m  Z=%.0f m\n",
        string(size(grid)), ncells(grid), CELL, CELL_Z)
@printf("  work dir  %s\n", WORK)
println("  azimuth convention: atan2(dN,dE) mod 180° (0°=east, 90°=north)")
println("  composite_length_m: skipped (point specimens)")

# Global NN floor from all finite-xyz rows (property-independent support scale)
all_xyz = begin
    keep = BitVector(undef, length(samples))
    for i in 1:length(samples)
        keep[i] = isfinite(samples.east[i]) && isfinite(samples.north[i]) &&
                  isfinite(samples.elev[i])
    end
    hcat(samples.east[keep], samples.north[keep], samples.elev[keep])
end
nn_med = median_nearest_neighbour(all_xyz)
min_lag_m = max(0.5, nn_med)
range_min = min_lag_m
@printf("  median NN distance  %.3f m  → min_lag = range_min = %.3f m\n",
        nn_med, min_lag_m)

rng = Random.Xoshiro(SEED)
properties = ("grade", "density", "susceptibility", "conductivity_100kHz")
results = NamedTuple[]
prop_cache = Dict{String,Tuple{Matrix{Float64},Vector{Float64},Float64}}()

for prop in properties
    coords, values = property_points(samples, prop)
    isempty(values) && error("no finite samples for $prop")
    prop_nn = median_nearest_neighbour(coords)
    prop_min = max(min_lag_m, isfinite(prop_nn) ? prop_nn : min_lag_m)
    prop_cache[prop] = (coords, values, prop_min)
    r = analyse_property(coords, values;
                         name=prop, tolerance=TOLERANCE, n_lags=N_LAGS,
                         maxlag=nothing, min_lag_m=prop_min, range_min=prop_min,
                         range_max=nothing, sample_size=SAMPLE_SIZE, rng=rng)
    push!(results, r)
end

#---------- maxlag sensitivity (grade + conductivity) ----------

println()
println(repeat("=", 96))
println("  MAXLAG SENSITIVITY — grade & conductivity (multipliers of default 0.5·span)")
println(repeat("=", 96))
sens = NamedTuple[]
for prop in ("grade", "conductivity_100kHz")
    coords, values, prop_min = prop_cache[prop]
    ml0 = default_maxlag(coords; is_vertical=false)
    for mult in MAXLAG_MULTS
        ml = ml0 * mult
        label = @sprintf("%s maxlag×%.0f", prop, mult)
        r = analyse_property(coords, values;
                             name=prop, label=label, tolerance=TOLERANCE,
                             n_lags=N_LAGS, maxlag=ml, min_lag_m=prop_min,
                             range_min=prop_min, range_max=nothing,
                             sample_size=SAMPLE_SIZE, rng=rng)
        push!(sens, merge(r, (maxlag_mult = mult,)))
    end
end

println()
println(repeat("-", 96))
@printf("%-22s %6s %12s %12s %12s %10s %8s\n",
        "property", "×maxlag", "maxlag_m", "R_major", "range_upper",
        "at_ceiling", "maj_az")
for r in sens
    @printf("%-22s %6.0f %12.1f %12.1f %12.1f %10s %8.0f\n",
            r.name, r.maxlag_mult, r.maxlag_m, r.major_range, r.range_upper_m,
            r.at_ceiling ? "YES" : "no", r.major_azimuth)
end
println(repeat("-", 96))
println("  YES at ceiling → fitted range ≥ 95% of optimizer upper bound;",
        " sill not resolved within scanned lags.")

#---------- summary table ----------

println()
println(repeat("=", 96))
println("  SUMMARY — ranges vs district grid (XY=$(round(Int,CELL)) m, Z=$(round(Int,CELL_Z)) m)")
println(repeat("=", 96))
@printf("%-22s %8s %10s %10s %10s %8s %8s %10s %10s %8s\n",
        "property", "maj_az", "R_major", "R_minor", "R_vert",
        "maj/min", "maj/Z", "Rmaj/dXY", "Rvert/dZ", "ceiling")
for r in results
    @printf("%-22s %8.0f %10.1f %10.1f %10.1f %8.2f %8.2f %10.2f %10.2f %8s\n",
            r.name, r.major_azimuth, r.major_range, r.minor_range, r.vertical_range,
            r.ani_major_minor, r.ani_major_vertical,
            r.major_range / CELL, r.vertical_range / CELL_Z,
            r.at_ceiling ? "YES" : "no")
end
println()
@printf("%-22s %12s %12s %12s %10s\n",
        "property", "model_maj", "model_min", "model_vert", "n")
for r in results
    @printf("%-22s %12s %12s %12s %10d\n",
            r.name, r.major_model, r.minor_model, r.vertical_model, r.n)
end
println()
println("  Interpretation notes:")
@printf("  • district cell XY = %.0f m, Z = %.0f m\n", CELL, CELL_Z)
println("  • R / cell_size ≪ 1  → structure is sub-cell; grid cannot resolve it")
println("  • R / cell_size ≫ 1  → structure spans many cells (smooth at this mesh)")
@printf("  • grade R_vert=%.0f m vs Z cell → %.2f× cells; Z=100 m would be %.2f×\n",
        results[findfirst(r -> r.name == "grade", results)].vertical_range,
        results[findfirst(r -> r.name == "grade", results)].vertical_range / CELL_Z,
        results[findfirst(r -> r.name == "grade", results)].vertical_range / 100.0)
println(repeat("=", 96))

# write TSV
tsv = joinpath(WORK, "cloncurry_variogram_summary.tsv")
open(tsv, "w") do io
    println(io, "property\tn\tmin_lag_m\tmaxlag_m\trange_upper_m\tat_ceiling\t",
            "major_azimuth_deg\tminor_azimuth_deg\t",
            "major_range_m\tminor_range_m\tvertical_range_m\t",
            "ani_major_minor\tani_major_vertical\t",
            "major_model\tminor_model\tvertical_model\t",
            "major_rmse\tminor_rmse\tvertical_rmse\t",
            "cell_xy_m\tcell_z_m\trange_major_over_cell_xy\trange_vert_over_cell_z\t",
            "major_note\tminor_note\tvertical_note")
    for r in results
        @printf(io,
                "%s\t%d\t%.4f\t%.4f\t%.4f\t%s\t%.1f\t%.1f\t%.4f\t%.4f\t%.4f\t%.6f\t%.6f\t%s\t%s\t%s\t%.6f\t%.6f\t%.6f\t%.1f\t%.1f\t%.6f\t%.6f\t%s\t%s\t%s\n",
                r.name, r.n, r.min_lag_m, r.maxlag_m, r.range_upper_m,
                r.at_ceiling ? "yes" : "no",
                r.major_azimuth, r.minor_azimuth,
                r.major_range, r.minor_range, r.vertical_range,
                r.ani_major_minor, r.ani_major_vertical,
                r.major_model, r.minor_model, r.vertical_model,
                r.major_rmse, r.minor_rmse, r.vertical_rmse,
                CELL, CELL_Z, r.major_range / CELL, r.vertical_range / CELL_Z,
                r.major_note, r.minor_note, r.vertical_note)
    end
end
println("  wrote ", tsv)

tsv_s = joinpath(WORK, "cloncurry_variogram_maxlag_sensitivity.tsv")
open(tsv_s, "w") do io
    println(io, "property\tmaxlag_mult\tmaxlag_m\tmaxlag_default_m\t",
            "major_azimuth_deg\tmajor_range_m\trange_upper_m\tat_ceiling\t",
            "minor_range_m\tvertical_range_m\tmajor_model\tmajor_rmse")
    for r in sens
        @printf(io, "%s\t%.4f\t%.4f\t%.4f\t%.1f\t%.4f\t%.4f\t%s\t%.4f\t%.4f\t%s\t%.6f\n",
                r.name, r.maxlag_mult, r.maxlag_m, r.maxlag_default_m,
                r.major_azimuth, r.major_range, r.range_upper_m,
                r.at_ceiling ? "yes" : "no",
                r.minor_range, r.vertical_range, r.major_model, r.major_rmse)
    end
end
println("  wrote ", tsv_s)
println("Done.")
