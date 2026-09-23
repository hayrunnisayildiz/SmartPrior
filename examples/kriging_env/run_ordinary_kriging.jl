# GeoStats.jl ordinary kriging runner (activated under examples/kriging_env).
#
# Args: <params.tsv> <work_dir>
# Reads train/test point TSVs + variogram params written by
# examples/kriging_cloncurry_petro.jl; writes predictions and a hold-out report.
#
# No geochemistry / lithology / gravity / MT / magnetics / VFSA.

using GeoStats
using DelimitedFiles
using Printf
using Statistics

length(ARGS) >= 2 || error("usage: run_ordinary_kriging.jl <params.tsv> <work_dir>")
const PARAMS = ARGS[1]
const WORK = ARGS[2]

function r_squared(obs, pred)
    n = length(obs)
    n < 2 && return NaN
    μ = mean(obs)
    ss_tot = sum(abs2, obs .- μ)
    ss_tot == 0 && return NaN
    return 1 - sum(abs2, pred .- obs) / ss_tot
end

function read_points(path)
    data, hdr = readdlm(path, '\t', header = true)
    # east north elev value
    east = Float64.(data[:, 1])
    north = Float64.(data[:, 2])
    elev = Float64.(data[:, 3])
    value = Float64.(data[:, 4])
    return east, north, elev, value
end

"""Average coincident XYZ (duplicate locations → singular OK system)."""
function dedupe(east, north, elev, value; tol = 1e-3)
    n = length(value)
    used = falses(n)
    e2 = Float64[]; n2 = Float64[]; z2 = Float64[]; v2 = Float64[]
    for i in 1:n
        used[i] && continue
        acc = value[i]; cnt = 1; used[i] = true
        for j in (i + 1):n
            used[j] && continue
            if hypot(east[j] - east[i], north[j] - north[i], elev[j] - elev[i]) <= tol
                acc += value[j]; cnt += 1; used[j] = true
            end
        end
        push!(e2, east[i]); push!(n2, north[i]); push!(z2, elev[i])
        push!(v2, acc / cnt)
    end
    return e2, n2, z2, v2
end

function make_variogram(model, r_maj, r_min, r_vert, az_deg, total_sill, nugget)
    # ranges: major / minor / vertical; RotZ aligns major with az from +E
    # (atan2(dN,dE) convention used in variogram_cloncurry.jl).
    # GeoStats sill = total sill; nugget is the discontinuity contribution.
    rot = RotZ(deg2rad(az_deg))
    kwargs = (ranges = (r_maj, r_min, r_vert), rotation = rot,
              sill = total_sill, nugget = nugget)
    return model == "exponential" ?
           ExponentialVariogram(; kwargs...) :
           SphericalVariogram(; kwargs...)
end

function ordinary_krige(train_path, test_path; model, az_deg,
                        r_maj, r_min, r_vert, total_sill, nugget)
    te, tn, tz, tv = read_points(train_path)
    te, tn, tz, tv = dedupe(te, tn, tz, tv)
    qe, qn, qz, qv = read_points(test_path)

    gtb = georef((value = tv,), Point.(te, tn, tz))
    γ = make_variogram(model, r_maj, r_min, r_vert, az_deg, total_sill, nugget)
    ok = Kriging(γ)   # → OrdinaryKriging when mean unspecified
    queries = PointSet(Point.(qe, qn, qz))
    @printf("  GeoStats OK  train=%d unique  test=%d  model=%s\n",
            length(tv), length(qv), typeof(ok))
    @printf("  γ            %s\n", γ)

    out = gtb |> Interpolate(queries; model = ok)
    pred = collect(out.value)
    return pred, qv, length(tv)
end

println(repeat("=", 72))
println("  GeoStats.jl OrdinaryKriging")
println(repeat("=", 72))
@printf("  GeoStats %s\n", pkgversion(GeoStats))
@printf("  params   %s\n", PARAMS)

NN_REF = Dict(
    "density" => (rmse = 0.600, r2 = -0.742),
    "susceptibility" => (rmse = 1.340, r2 = 0.034),
)

rows = NamedTuple[]
open(PARAMS) do io
    hdr = split(readline(io), '\t')
    col = Dict(h => i for (i, h) in enumerate(hdr))
    for line in eachline(io)
        isempty(strip(line)) && continue
        p = split(line, '\t')
        prop = p[col["property"]]
        println()
        println(repeat("-", 72))
        @printf("  PROPERTY: %s\n", prop)
        pred, obs, n_train = ordinary_krige(
            p[col["train_tsv"]], p[col["test_tsv"]];
            model = p[col["major_model"]],
            az_deg = parse(Float64, p[col["major_azimuth_deg"]]),
            r_maj = parse(Float64, p[col["major_range_m"]]),
            r_min = parse(Float64, p[col["minor_range_m"]]),
            r_vert = parse(Float64, p[col["vertical_range_m"]]),
            total_sill = parse(Float64, p[col["total_sill"]]),
            nugget = parse(Float64, p[col["nugget"]]),
        )
        rmse = sqrt(mean(abs2, pred .- obs))
        r2 = r_squared(obs, pred)
        @printf("  test RMSE  %.6f\n", rmse)
        @printf("  test R²    %.6f\n", r2)

        pred_path = p[col["pred_tsv"]]
        open(pred_path, "w") do po
            println(po, "obs\tpred")
            for i in eachindex(obs)
                @printf(po, "%.10f\t%.10f\n", obs[i], pred[i])
            end
        end
        @printf("  wrote      %s\n", pred_path)

        push!(rows, (
            property = prop,
            n_train = n_train,
            n_test = length(obs),
            major_azimuth_deg = parse(Float64, p[col["major_azimuth_deg"]]),
            major_range_m = parse(Float64, p[col["major_range_m"]]),
            minor_range_m = parse(Float64, p[col["minor_range_m"]]),
            vertical_range_m = parse(Float64, p[col["vertical_range_m"]]),
            major_model = p[col["major_model"]],
            nugget = parse(Float64, p[col["nugget"]]),
            partial_sill = parse(Float64, p[col["partial_sill"]]),
            total_sill = parse(Float64, p[col["total_sill"]]),
            kriging_rmse = rmse,
            kriging_r2 = r2,
            nn_rmse = NN_REF[prop].rmse,
            nn_r2 = NN_REF[prop].r2,
            pred_mean = mean(pred),
            obs_mean = mean(obs),
        ))
    end
end

report = joinpath(WORK, "cloncurry_kriging_holdout_report.tsv")
open(report, "w") do io
    println(io, "property\tn_train\tn_test\tmajor_azimuth_deg\t",
            "major_range_m\tminor_range_m\tvertical_range_m\tmajor_model\t",
            "nugget\tpartial_sill\ttotal_sill\t",
            "kriging_rmse\tkriging_r2\tnn_rmse\tnn_r2\t",
            "pred_mean\tobs_mean\tsplit_seed\ttargets\tengine")
    for r in rows
        @printf(io,
                "%s\t%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%s\t%.8f\t%.8f\t%.8f\t%.6f\t%.6f\t%.3f\t%.3f\t%.6f\t%.6f\t%d\t%s\t%s\n",
                r.property, r.n_train, r.n_test, r.major_azimuth_deg,
                r.major_range_m, r.minor_range_m, r.vertical_range_m,
                r.major_model, r.nugget, r.partial_sill, r.total_sill,
                r.kriging_rmse, r.kriging_r2, r.nn_rmse, r.nn_r2,
                r.pred_mean, r.obs_mean, 2026, "84,18,18", "GeoStats.jl")
    end
end
println()
@printf("  wrote report %s\n", report)
println("Done.")
