# Keivitsa data figures for the feasibility write-up.
#
# Writes three 150 dpi PNGs under docs/figures/:
#   data_overview.png, variogram_fold6.png, azimuth_check.png
#
# Run from the repo root (KEIVITSA_ROOT must point at the GTK delivery):
#   julia --project=. examples/figures_data.jl
#
# Numbers (detection limit, KKJ origin, fold id, vertical-range upper bound,
# azimuth metrics) are loaded from site TOML / TSV / existing example source —
# not retyped as bare literals.

const ROOT = dirname(@__DIR__)
ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, "tmp_feasibility_keivitsa_run2")
include(joinpath(@__DIR__, "feasibility_keivitsa.jl"))

using GLMakie
using TOML
using Printf
using Statistics

const FIG_DIR = joinpath(ROOT, "docs", "figures")
const RUN2 = joinpath(ROOT, "tmp_feasibility_keivitsa_run2")
const INSPECT = joinpath(ROOT, "tmp_keivitsa_inspect")
const CHECKS_JL = joinpath(@__DIR__, "keivitsa_run2_checks.jl")
const DPI = 150

#---------- load constants from existing files ----------

function load_site_bounds_and_lod(site_path)
    cfg = TOML.parsefile(site_path)
    bx = cfg["bounds"]["x"]
    by = cfg["bounds"]["y"]
    lod = Float64(cfg["properties"]["cu"]["lod"])
    unit = String(cfg["properties"]["cu"]["unit"])
    return Float64(bx[1]), Float64(by[1]), lod, unit
end

function parse_plot_fold(checks_path)
    # First entry of `const PLOT_FOLDS = (6, 7, 9)` in keivitsa_run2_checks.jl.
    for line in eachline(checks_path)
        m = match(r"const\s+PLOT_FOLDS\s*=\s*\((\d+)", line)
        m === nothing && continue
        return parse(Int, m.captures[1])
    end
    error("PLOT_FOLDS not found in $checks_path")
end

function load_fold_test_holes(fold_split_path, fold_id::Int)
    holes = String[]
    for (i, line) in enumerate(eachline(fold_split_path))
        i == 1 && continue
        p = split(line, '\t')
        length(p) >= 2 || continue
        parse(Int, p[1]) == fold_id || continue
        push!(holes, String(p[2]))
    end
    isempty(holes) && error("no test holes for fold $fold_id in $fold_split_path")
    return holes
end

function load_fold_range_vertical(kriging_params_path, fold_id::Int)
    label = "fold$(fold_id)"
    header = nothing
    col = Dict{String,Int}()
    for (i, line) in enumerate(eachline(kriging_params_path))
        if i == 1
            header = split(line, '\t')
            col = Dict(String(h) => j for (j, h) in enumerate(header))
            continue
        end
        p = split(line, '\t')
        String(p[col["test_hole"]]) == label || continue
        return parse(Float64, p[col["range_vertical_m"]])
    end
    error("fold $fold_id not found in $kriging_params_path")
end

function load_azimuth_metrics(path)
    # Stage 4 output: tmp_keivitsa_inspect/azimuth_crosshole.tsv
    rows = NamedTuple{(:convention, :corr, :mean_abs_delta, :n_pairs),
                      Tuple{String,Float64,Float64,Int}}[]
    for line in eachline(path)
        startswith(line, "clockwise") || startswith(line, "counterclockwise") || continue
        p = split(line, '\t')
        length(p) >= 4 || continue
        push!(rows, (convention = String(p[1]),
                     corr = parse(Float64, p[2]),
                     mean_abs_delta = parse(Float64, p[3]),
                     n_pairs = parse(Int, p[4])))
    end
    length(rows) == 2 || error("expected clockwise and counterclockwise rows in $path")
    return rows
end

#---------- figure helpers ----------

function save_fig(path, fig)
    GLMakie.save(path, fig; dpi = DPI)
    return path
end

function experimental_and_fit(table, y_all, train_rows)
    # Same experimental construction as examples/keivitsa_run2_checks.jl
    # (variogram_curves); fit_variogram is called read-only.
    y_tr = y_all[train_rows]
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    hspan = max(maximum(dx) - minimum(dx), maximum(dy) - minimum(dy))
    vspan = maximum(dz) - minimum(dz)
    maxlag_h = max(10.0, 0.5 * hspan)
    maxlag_v = max(10.0, 0.5 * vspan)
    dist_h, semi_h, dist_v, semi_v = collect_pairs(dx, dy, dz, dv, dh;
                                                    maxlag_h = maxlag_h, maxlag_v = maxlag_v,
                                                    dip_deg = HORIZONTAL_DIP_DEG)
    bins_h, exp_h = bin_experimental(dist_h, semi_h, N_LAGS, maxlag_h)
    bins_v, exp_v = bin_experimental(dist_v, semi_v, N_LAGS, maxlag_v)
    vvar = var(dv; corrected = false)
    rmax_h = max(bins_h[end] * 2, RANGE_MIN_M * 4)
    rmax_v = max(bins_v[end] * 2, RANGE_MIN_M * 4)
    sill_cap = max(2 * max(vvar, 1.0e-8), maximum(exp_h), maximum(exp_v), 1.0e-6)
    hi_v = max(rmax_v, RANGE_MIN_M * 1.01)
    fit = fit_variogram(dx, dy, dz, dv, dh; context = "figures_data fold train")
    return (; bins_h, exp_h, bins_v, exp_v, fit, maxlag_h, maxlag_v, hi_v,
            n_merged, n_train = length(train_rows), n_deduped = length(dv))
end

function figure_data_overview(table, eligible, x0, y0, lod, unit)
    mask = eligible
    # Prefer real drillhole rows only (excludes non-drillhole / placeholder types).
    real = real_hole_mask(table)
    idx = [i for i in mask if real[i]]
    xloc = table.x[idx] .- x0
    yloc = table.y[idx] .- y0
    cu = table.values[TARGET][idx]
    logcu = log10.(Float64.(cu))
    lod_log = log10(lod)

    fig = Figure(size = (1100, 480), backgroundcolor = :white,
               fontsize = 14)
    Label(fig[0, 1:3], "Keivitsa Cu data overview";
          fontsize = 18, font = :bold, tellwidth = false)

    axp = Axis(fig[1, 1]; xlabel = "Easting local (m)", ylabel = "Northing local (m)",
               title = "Plan view of desurveyed traces", aspect = DataAspect())
    # Draw each hole as a polyline in local metres, coloured by sample log10 Cu.
    by_hole = Dict{String,Vector{Int}}()
    for (k, i) in enumerate(idx)
        push!(get!(by_hole, table.hole[i], Int[]), k)
    end
    for (_, ks) in by_hole
        order = sortperm(collect(ks); by = k -> table.z[idx[k]])
        ks_ord = ks[order]
        if length(ks_ord) >= 2
            lines!(axp, xloc[ks_ord], yloc[ks_ord]; color = (:gray55, 0.35), linewidth = 0.6)
        end
    end
    sc = scatter!(axp, xloc, yloc; color = logcu, colormap = :plasma,
                  colorrange = extrema(logcu), markersize = 4)
    Colorbar(fig[1, 2], sc, label = "log₁₀ Cu ($unit)", width = 16)

    axh = Axis(fig[1, 3]; xlabel = "log₁₀ Cu ($unit)", ylabel = "count",
               title = "Cu grade histogram")
    hist!(axh, logcu; bins = 40, color = (:steelblue, 0.75))
    vlines!(axh, [lod_log]; color = :firebrick, linewidth = 2,
            label = @sprintf("%.0f %s DL (log₁₀=%.2f)", lod, unit, lod_log))
    axislegend(axh; position = :rt)

    colsize!(fig.layout, 1, Relative(1.15))
    colsize!(fig.layout, 3, Relative(0.85))
    return fig
end

function figure_variogram(fold_id, curves, range_v_from_params)
    fit = curves.fit
    hs = range(0.0, curves.maxlag_h; length = 200)
    vs = range(0.0, curves.maxlag_v; length = 200)
    gh = [model_gamma(fit.model, h, fit.range_h, fit.partial_sill, fit.nugget) for h in hs]
    gv = [model_gamma(fit.model, h, fit.range_v, fit.partial_sill, fit.nugget) for h in vs]
    # Upper bound of the vertical range: optimizer hi_v from the same formula as
    # keivitsa_run2_checks; kriging_params reports the fitted range (hits hi_v).
    v_bound = curves.hi_v

    fig = Figure(size = (1000, 420), backgroundcolor = :white, fontsize = 14)
    Label(fig[0, 1:2],
          @sprintf("Fold %d training variogram (n_train=%d, deduped=%d)",
                   fold_id, curves.n_train, curves.n_deduped);
          fontsize = 18, font = :bold, tellwidth = false)

    axh = Axis(fig[1, 1]; xlabel = "horizontal lag (m)", ylabel = "semivariance",
               title = "Horizontal")
    scatter!(axh, curves.bins_h, curves.exp_h; label = "experimental", markersize = 10)
    lines!(axh, collect(hs), gh; label = fit.model, linewidth = 2)
    axislegend(axh; position = :rb)

    axv = Axis(fig[1, 2]; xlabel = "downhole lag (m)", ylabel = "semivariance",
               title = "Downhole")
    scatter!(axv, curves.bins_v, curves.exp_v; label = "experimental", markersize = 10)
    lines!(axv, collect(vs), gv; label = fit.model, linewidth = 2)
    vlines!(axv, [v_bound]; color = :firebrick, linewidth = 2, linestyle = :dash,
            label = @sprintf("vertical range upper bound %.0f m", v_bound))
    axislegend(axv; position = :rb)

    @info "variogram fold" fold_id range_h=fit.range_h range_v=fit.range_v hi_v=v_bound params_range_v=range_v_from_params
    return fig
end

function figure_azimuth(rows)
    labels = [r.convention for r in rows]
    corrs = [r.corr for r in rows]
    mads = [r.mean_abs_delta for r in rows]
    xs = 1:length(rows)

    fig = Figure(size = (900, 420), backgroundcolor = :white, fontsize = 14)
    Label(fig[0, 1:2], "Stage 4 azimuth cross-hole agreement (log₁₀ Cu)";
          fontsize = 18, font = :bold, tellwidth = false)

    axc = Axis(fig[1, 1]; title = "Correlation", ylabel = "correlation",
               xticks = (xs, labels))
    barplot!(axc, xs, corrs; color = (:steelblue, 0.85), width = 0.55)
    ylims!(axc, 0, max(1.0, maximum(corrs) * 1.15))
    for (x, v) in zip(xs, corrs)
        text!(axc, x, v; text = @sprintf("%.3f", v), align = (:center, :bottom),
              offset = (0, 4), fontsize = 13)
    end

    axm = Axis(fig[1, 2]; title = "Mean absolute difference",
               ylabel = "mean |Δ| log₁₀ Cu", xticks = (xs, labels))
    barplot!(axm, xs, mads; color = (:darkorange, 0.85), width = 0.55)
    ylims!(axm, 0, maximum(mads) * 1.25)
    for (x, v) in zip(xs, mads)
        text!(axm, x, v; text = @sprintf("%.3f", v), align = (:center, :bottom),
              offset = (0, 4), fontsize = 13)
    end
    return fig
end

#---------- main ----------

function main()
    mkpath(FIG_DIR)
    x0, y0, lod, unit = load_site_bounds_and_lod(SITE_PATH)
    fold_id = parse_plot_fold(CHECKS_JL)
    az_path = joinpath(INSPECT, "azimuth_crosshole.tsv")
    isfile(az_path) || error("Stage 4 azimuth file missing: $az_path")
    az_rows = load_azimuth_metrics(az_path)
    range_v_params = load_fold_range_vertical(joinpath(RUN2, "kriging_params.tsv"), fold_id)

    @info "loaded constants" lod_ppm=lod unit fold_id range_v_params

    table, _, _ = load_site(SITE_PATH)
    spec = spec_of(table, TARGET)
    eligible = findall(
        training_mask(table) .& real_hole_mask(table) .& observed_mask(table, TARGET))
    y_all = fill(NaN, nsamples(table))
    y_all[eligible] = transformed(spec, table.values[TARGET][eligible])

    test_holes = load_fold_test_holes(joinpath(RUN2, "fold_split.tsv"), fold_id)
    test_set = Set(test_holes)
    train_rows = [i for i in eligible if table.hole[i] ∉ test_set]

    GLMakie.activate!(; visible = false)

    fig1 = figure_data_overview(table, eligible, x0, y0, lod, unit)
    p1 = joinpath(FIG_DIR, "data_overview.png")
    save_fig(p1, fig1)
    @info "wrote" path=p1

    curves = experimental_and_fit(table, y_all, train_rows)
    fig2 = figure_variogram(fold_id, curves, range_v_params)
    p2 = joinpath(FIG_DIR, "variogram_fold$(fold_id).png")
    save_fig(p2, fig2)
    @info "wrote" path=p2 n_train=curves.n_train

    fig3 = figure_azimuth(az_rows)
    p3 = joinpath(FIG_DIR, "azimuth_check.png")
    save_fig(p3, fig3)
    @info "wrote" path=p3 azimuth=az_rows

    for p in (p1, p2, p3)
        isfile(p) && filesize(p) > 0 || error("missing or empty figure: $p")
        println("OK ", p, " (", filesize(p), " bytes)")
    end
    return nothing
end

main()
