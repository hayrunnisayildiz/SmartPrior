# 3D / section figures from tmp_keivitsa_blockmodel/ (no re-fit).
#
# Run:  julia --project=. examples/figures_3d.jl
# Out:  docs/figures/{3d_mu,3d_sigma,3d_shell,section_mu_sigma}.png

using GLMakie
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const WORK = joinpath(ROOT, "tmp_keivitsa_blockmodel")
const FIGDIR = joinpath(ROOT, "docs", "figures")
const DPI = 150
const PX_PER_UNIT = DPI / 72

# Method identity colours (scalar fields use a shared sequential map).
const COL_KRIGING = RGBf(0.15, 0.35, 0.85)
const COL_NEURAL = RGBf(0.85, 0.15, 0.20)
const COL_MEAN = RGBf(0.45, 0.45, 0.45)
const COL_SHELL = RGBf(0.55, 0.55, 0.55)

const TSV = joinpath(WORK, "keivitsa_blockmodel.tsv")
const SAMPLES = joinpath(WORK, "samples_local.tsv")
const REPORT = joinpath(WORK, "report.tsv")

# Shared 3D camera:
# - azimuth −π/2 looks toward −y, so +x is to the right (increases L→R)
# - yreversed maps increasing y into the away-from-viewer direction
const CAM_AZIMUTH = -0.5π
const CAM_ELEVATION = 0.18π
const CAM_XREVERSED = false
const CAM_YREVERSED = true

const MU_SHELL_THRESH = 2.5  # log10 Cu filter (≈ 300 ppm); stated in shell title

struct BlockRow
    x::Float64
    y::Float64
    z::Float64
    nearest::Float64
    μ_kr::Float64
    σ_kr::Float64
    μ_nn::Float64
    σ_nn::Float64
end

struct SampleRow
    hole::String
    x::Float64
    y::Float64
    z::Float64
    log10_cu::Float64
end

function fail(msg)
    error(msg)
end

function read_report_map(path)
    isfile(path) || fail("missing $path — run examples/keivitsa_blockmodel.jl first")
    d = Dict{String,String}()
    open(path) do io
        readline(io)
        for line in eachline(io)
            parts = split(line, '\t')
            length(parts) >= 2 || continue
            d[parts[1]] = parts[2]
        end
    end
    return d
end

function read_blocks(path)
    isfile(path) || fail("missing $path")
    rows = BlockRow[]
    open(path) do io
        header = split(readline(io), '\t')
        col = Dict(h => i for (i, h) in enumerate(header))
        for need in ("x_local_m", "y_local_m", "z_m", "nearest_sample_m",
                     "mu_kriging_v1", "sigma_kriging_v1",
                     "mu_neural_field", "sigma_neural_field")
            haskey(col, need) || fail("block TSV missing column $need")
        end
        for line in eachline(io)
            p = split(line, '\t')
            push!(rows, BlockRow(
                parse(Float64, p[col["x_local_m"]]),
                parse(Float64, p[col["y_local_m"]]),
                parse(Float64, p[col["z_m"]]),
                parse(Float64, p[col["nearest_sample_m"]]),
                parse(Float64, p[col["mu_kriging_v1"]]),
                parse(Float64, p[col["sigma_kriging_v1"]]),
                parse(Float64, p[col["mu_neural_field"]]),
                parse(Float64, p[col["sigma_neural_field"]]),
            ))
        end
    end
    isempty(rows) && fail("no kept blocks in $path")
    return rows
end

function read_samples(path)
    isfile(path) || fail("missing $path")
    rows = SampleRow[]
    open(path) do io
        header = split(readline(io), '\t')
        col = Dict(h => i for (i, h) in enumerate(header))
        for line in eachline(io)
            p = split(line, '\t')
            push!(rows, SampleRow(
                String(p[col["hole"]]),
                parse(Float64, p[col["x_local_m"]]),
                parse(Float64, p[col["y_local_m"]]),
                parse(Float64, p[col["z_m"]]),
                parse(Float64, p[col["log10_cu"]]),
            ))
        end
    end
    return rows
end

function hole_traces(samples::Vector{SampleRow})
    by = Dict{String,Vector{SampleRow}}()
    for s in samples
        push!(get!(Vector{SampleRow}, by, s.hole), s)
    end
    traces = Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}[]
    for (_, recs) in by
        ordered = sort(recs; by = r -> -r.z)
        push!(traces, ([r.x for r in ordered], [r.y for r in ordered], [r.z for r in ordered]))
    end
    return traces
end

function finite_extrema(vals)
    f = filter(isfinite, vals)
    isempty(f) && return (0.0, 1.0)
    lo, hi = extrema(f)
    lo == hi && return (lo - 0.1, hi + 0.1)
    return (lo, hi)
end

function subsample_blocks(blocks::Vector{BlockRow}; max_n::Int = 80_000)
    n = length(blocks)
    n <= max_n && return blocks
    step = cld(n, max_n)
    return blocks[1:step:n]
end

function add_traces!(ax, traces; color = RGBf(0.05, 0.05, 0.05), linewidth = 0.9)
    for (hx, hy, hz) in traces
        length(hx) >= 2 || continue
        lines!(ax, hx, hy, hz; color = color, linewidth = linewidth)
    end
    return nothing
end

function activate_backend!()
    try
        GLMakie.activate!(; visible = false)
        return :GLMakie
    catch e
        @warn "GLMakie off-screen activate failed; trying CairoMakie" exception = e
        try
            @eval using CairoMakie
            CairoMakie.activate!()
            return :CairoMakie
        catch e2
            error("Neither GLMakie nor CairoMakie could activate: $e2")
        end
    end
end

function save_fig(path, fig)
    mkpath(dirname(path))
    save(path, fig; px_per_unit = PX_PER_UNIT)
    return path
end

"Colorbar ticks for μ log10 Cu with ppm annotations at 1, 2, 3."
function mu_tick_labels(lo::Real, hi::Real)
    ppm = Dict(1.0 => "10 ppm", 2.0 => "100 ppm", 3.0 => "1,000 ppm")
    vals = Float64[]
    for v in (1.0, 2.0, 3.0)
        lo - 1e-9 <= v <= hi + 1e-9 && push!(vals, v)
    end
    # Keep endpoints if they are distinct from the ppm ticks.
    for v in (Float64(lo), Float64(hi))
        any(abs(v - t) < 1e-9 for t in vals) || push!(vals, v)
    end
    sort!(vals)
    labels = String[]
    for v in vals
        if haskey(ppm, v)
            push!(labels, @sprintf("%.0f  (%s)", v, ppm[v]))
        else
            push!(labels, @sprintf("%.2g", v))
        end
    end
    return vals, labels
end

function apply_shared_camera!(ax1, ax2, xs, ys, zs)
    xlo, xhi = extrema(xs)
    ylo, yhi = extrema(ys)
    zlo, zhi = extrema(zs)
    # Sparse y ticks so labels do not stack when y runs into the page.
    y_ticks = round.(collect(range(ylo, yhi; length = 4)); digits = 0)
    for ax in (ax1, ax2)
        ax.xreversed[] = CAM_XREVERSED
        ax.yreversed[] = CAM_YREVERSED
        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)
        zlims!(ax, zlo, zhi)
        ax.azimuth[] = CAM_AZIMUTH
        ax.elevation[] = CAM_ELEVATION
        ax.yticks[] = y_ticks
        ax.ylabelrotation[] = 0.0
        ax.yticklabelsize[] = 11
        ax.xticklabelsize[] = 11
        ax.zticklabelsize[] = 11
        ax.xlabeloffset[] = 45
        ax.ylabeloffset[] = 60
        ax.zlabeloffset[] = 45
        ax.yticklabelpad[] = 6
    end
    return nothing
end

function plot_3d_pair(blocks, traces, field_kr, field_nn, path, title;
                      colormap = :viridis, mu_colorbar::Bool = false,
                      sigma_label = "σ log10 Cu")
    lo, hi = finite_extrema(vcat(field_kr, field_nn))
    fig = Figure(size = (1500, 720), backgroundcolor = :white, fontsize = 14)
    Label(fig[0, 1:2], title; fontsize = 18, font = :bold, tellwidth = false)
    ax1 = Axis3(fig[1, 1];
                xlabel = "x local (m)", ylabel = "y local (m)", zlabel = "z (m)",
                title = "kriging (v1)", titlecolor = COL_KRIGING,
                perspectiveness = 0.0, protrusions = (60, 60, 40, 40))
    ax2 = Axis3(fig[1, 2];
                xlabel = "x local (m)", ylabel = "y local (m)", zlabel = "z (m)",
                title = "neural field", titlecolor = COL_NEURAL,
                perspectiveness = 0.0, protrusions = (60, 60, 40, 40))
    xs = [b.x for b in blocks]
    ys = [b.y for b in blocks]
    zs = [b.z for b in blocks]
    ms = Vec3f(14, 14, 7)
    meshscatter!(ax1, xs, ys, zs; color = field_kr, colormap = colormap,
                 colorrange = (lo, hi), markersize = ms,
                 marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
                 transparency = true, alpha = 0.55)
    meshscatter!(ax2, xs, ys, zs; color = field_nn, colormap = colormap,
                 colorrange = (lo, hi), markersize = ms,
                 marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
                 transparency = true, alpha = 0.55)
    add_traces!(ax1, traces)
    add_traces!(ax2, traces)
    apply_shared_camera!(ax1, ax2, xs, ys, zs)
    if mu_colorbar
        ticks, labs = mu_tick_labels(lo, hi)
        Colorbar(fig[1, 3], limits = (lo, hi), colormap = colormap,
                 label = "μ log10 Cu", ticks = (ticks, labs))
    else
        Colorbar(fig[1, 3], limits = (lo, hi), colormap = colormap, label = sigma_label)
    end
    save_fig(path, fig)
    return fig
end

function plot_3d_shell(blocks, traces, path, title, n_kr, n_nn, n_kept;
                       thresh = MU_SHELL_THRESH)
    μ_hi = vcat([b.μ_kr for b in blocks if isfinite(b.μ_kr) && b.μ_kr >= thresh],
                [b.μ_nn for b in blocks if isfinite(b.μ_nn) && b.μ_nn >= thresh])
    lo, hi = isempty(μ_hi) ? (thresh, thresh + 1) : finite_extrema(μ_hi)
    lo = min(lo, thresh)
    fig = Figure(size = (1500, 720), backgroundcolor = :white, fontsize = 14)
    Label(fig[0, 1:2], title; fontsize = 17, font = :bold, tellwidth = false)
    ax1 = Axis3(fig[1, 1];
                xlabel = "x local (m)", ylabel = "y local (m)", zlabel = "z (m)",
                title = @sprintf("kriging (v1)  —  %d / %d blocks ≥ %.1f", n_kr, n_kept, thresh),
                titlecolor = COL_KRIGING,
                perspectiveness = 0.0, protrusions = (60, 60, 40, 40))
    ax2 = Axis3(fig[1, 2];
                xlabel = "x local (m)", ylabel = "y local (m)", zlabel = "z (m)",
                title = @sprintf("neural field  —  %d / %d blocks ≥ %.1f", n_nn, n_kept, thresh),
                titlecolor = COL_NEURAL,
                perspectiveness = 0.0, protrusions = (60, 60, 40, 40))

    xs_all = [b.x for b in blocks]
    ys_all = [b.y for b in blocks]
    zs_all = [b.z for b in blocks]
    ms = Vec3f(14, 14, 7)

    function draw_shell!(ax, μs)
        low = findall(i -> !(isfinite(μs[i]) && μs[i] >= thresh), eachindex(μs))
        high = findall(i -> isfinite(μs[i]) && μs[i] >= thresh, eachindex(μs))
        if !isempty(low)
            meshscatter!(ax, xs_all[low], ys_all[low], zs_all[low];
                         color = COL_SHELL, markersize = ms,
                         marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
                         transparency = true, alpha = 0.05)
        end
        if !isempty(high)
            meshscatter!(ax, xs_all[high], ys_all[high], zs_all[high];
                         color = μs[high], colormap = :viridis, colorrange = (lo, hi),
                         markersize = ms, marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
                         transparency = true, alpha = 0.75)
        end
        return nothing
    end

    draw_shell!(ax1, [b.μ_kr for b in blocks])
    draw_shell!(ax2, [b.μ_nn for b in blocks])
    add_traces!(ax1, traces)
    add_traces!(ax2, traces)
    apply_shared_camera!(ax1, ax2, xs_all, ys_all, zs_all)
    ticks, labs = mu_tick_labels(lo, hi)
    Colorbar(fig[1, 3], limits = (lo, hi), colormap = :viridis,
             label = @sprintf("μ log10 Cu  (≥ %.1f coloured)", thresh),
             ticks = (ticks, labs))
    save_fig(path, fig)
    return fig
end

function plot_section(blocks, samples, path, title; half_width = 20.0)
    y0 = mean(s.y for s in samples)
    near = [b for b in blocks if abs(b.y - y0) <= half_width]
    samp_near = [s for s in samples if abs(s.y - y0) <= half_width]
    isempty(near) && fail("no kept blocks within ±$(half_width) m of section y=$(y0)")

    μ_lo, μ_hi = finite_extrema(vcat([b.μ_kr for b in near], [b.μ_nn for b in near]))
    σ_lo, σ_hi = finite_extrema(vcat([b.σ_kr for b in near], [b.σ_nn for b in near]))

    fig = Figure(size = (1200, 980), backgroundcolor = :white, fontsize = 13)
    Label(fig[0, 1:2], title; fontsize = 17, font = :bold, tellwidth = false)

    function panel_mu!(row, col, vals, panel_title, titlecolor)
        ax = Axis(fig[row, col];
                  xlabel = "x local (m)", ylabel = "z (m)",
                  title = panel_title, titlecolor = titlecolor)
        scatter!(ax, [b.x for b in near], [b.z for b in near];
                 color = vals, colormap = :viridis, colorrange = (μ_lo, μ_hi),
                 markersize = 6, marker = :rect)
        if !isempty(samp_near)
            scatter!(ax, [s.x for s in samp_near], [s.z for s in samp_near];
                     color = [s.log10_cu for s in samp_near], colormap = :viridis,
                     colorrange = (μ_lo, μ_hi), markersize = 11,
                     strokewidth = 0.9, strokecolor = :black)
        end
        return ax
    end

    function panel_sigma!(row, col, vals, panel_title, titlecolor)
        ax = Axis(fig[row, col];
                  xlabel = "x local (m)", ylabel = "z (m)",
                  title = panel_title, titlecolor = titlecolor)
        scatter!(ax, [b.x for b in near], [b.z for b in near];
                 color = vals, colormap = :plasma, colorrange = (σ_lo, σ_hi),
                 markersize = 6, marker = :rect)
        if !isempty(samp_near)
            scatter!(ax, [s.x for s in samp_near], [s.z for s in samp_near];
                     color = :black, markersize = 3.5)
        end
        return ax
    end

    panel_mu!(1, 1, [b.μ_kr for b in near], "μ — kriging (v1)", COL_KRIGING)
    panel_mu!(1, 2, [b.μ_nn for b in near], "μ — neural field", COL_NEURAL)
    panel_sigma!(2, 1, [b.σ_kr for b in near], "σ — kriging (v1)", COL_KRIGING)
    panel_sigma!(2, 2, [b.σ_nn for b in near], "σ — neural field", COL_NEURAL)

    ticks, labs = mu_tick_labels(μ_lo, μ_hi)
    Colorbar(fig[1, 3], limits = (μ_lo, μ_hi), colormap = :viridis,
             label = "μ log10 Cu", ticks = (ticks, labs))
    Colorbar(fig[2, 3], limits = (σ_lo, σ_hi), colormap = :plasma, label = "σ log10 Cu")
    Label(fig[3, 1:2],
          @sprintf("E–W section at y_local = %.1f m (mean drilling northing); samples |Δy| ≤ %.0f m",
                   y0, half_width);
          fontsize = 11, tellwidth = false)
    save_fig(path, fig)
    return fig
end

function main()
    backend = activate_backend!()
    report = read_report_map(REPORT)
    blocks_all = read_blocks(TSV)
    samples = read_samples(SAMPLES)
    traces = hole_traces(samples)
    blocks = subsample_blocks(blocks_all)

    n_kept = length(blocks_all)
    n_kept_report = parse(Int, get(report, "kept_blocks", string(n_kept)))
    n_holes = get(report, "cu_holes", "?")

    n_shell_kr = count(b -> isfinite(b.μ_kr) && b.μ_kr >= MU_SHELL_THRESH, blocks_all)
    n_shell_nn = count(b -> isfinite(b.μ_nn) && b.μ_nn >= MU_SHELL_THRESH, blocks_all)

    mkpath(FIGDIR)
    open(joinpath(FIGDIR, "figures_3d_report.txt"), "w") do io
        println(io, "backend\t", backend)
        println(io, "kept_blocks_read\t", n_kept)
        println(io, "kept_blocks_report\t", n_kept_report)
        println(io, "cu_holes\t", n_holes)
        println(io, "blocks_plotted_3d\t", length(blocks))
        println(io, "shell_thresh_log10_cu\t", MU_SHELL_THRESH)
        println(io, "shell_blocks_kriging_v1\t", n_shell_kr)
        println(io, "shell_blocks_neural_field\t", n_shell_nn)
        println(io, "dpi\t", DPI)
        println(io, "camera_azimuth\t", CAM_AZIMUTH)
        println(io, "camera_elevation\t", CAM_ELEVATION)
        println(io, "source_tsv\t", TSV)
        println(io, "source_samples\t", SAMPLES)
    end

    title_mu = @sprintf("Keivitsa Cu — μ (log10 Cu); kept blocks ≤60 m (%s holes)", n_holes)
    title_sg = @sprintf("Keivitsa Cu — σ (log10 Cu); kept blocks ≤60 m (%s holes)", n_holes)
    title_sec = "Keivitsa Cu — E–W section μ / σ; kriging (v1) vs neural field"
    title_shell = @sprintf(
        "Keivitsa Cu — μ ≥ %.1f (≈ 300 ppm) coloured; other kept blocks grey shell (%s holes)",
        MU_SHELL_THRESH, n_holes)

    plot_3d_pair(blocks, traces,
                 [b.μ_kr for b in blocks], [b.μ_nn for b in blocks],
                 joinpath(FIGDIR, "3d_mu.png"), title_mu;
                 colormap = :viridis, mu_colorbar = true)
    plot_3d_pair(blocks, traces,
                 [b.σ_kr for b in blocks], [b.σ_nn for b in blocks],
                 joinpath(FIGDIR, "3d_sigma.png"), title_sg;
                 colormap = :plasma, mu_colorbar = false)
    plot_3d_shell(blocks, traces, joinpath(FIGDIR, "3d_shell.png"), title_shell,
                  n_shell_kr, n_shell_nn, n_kept)
    plot_section(blocks_all, samples, joinpath(FIGDIR, "section_mu_sigma.png"), title_sec)

    println("backend=", backend)
    println("shell_thresh=", MU_SHELL_THRESH)
    println("shell_kriging_v1=", n_shell_kr, " / ", n_kept)
    println("shell_neural_field=", n_shell_nn, " / ", n_kept)
    println("wrote ", joinpath(FIGDIR, "3d_mu.png"))
    println("wrote ", joinpath(FIGDIR, "3d_sigma.png"))
    println("wrote ", joinpath(FIGDIR, "3d_shell.png"))
    println("wrote ", joinpath(FIGDIR, "section_mu_sigma.png"))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
