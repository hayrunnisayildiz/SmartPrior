# 2×2 comparison of Musgrave VFSA starts vs results (coarse mesh).
#
# One-off copy of examples/musgrave_plot_result.jl. Reads
# tmp_musgrave_coarse/inv_half and inv_prior, uses the same
# y_core_cell = 20000.0 mesh as the coarse training/VFSA scripts, and
# writes tmp_musgrave_coarse/musgrave_comparison_coarse.png.
#
# MTGeophysics' VFSA2DMT wrapper plots TE-only NaN TM data maps and crashes
# after the inversion has already finished. This script reads the on-disk
# models with `from_mt2d` and draws them with Plots.jl (GR). It does not
# call `plot_mt2d_model` / `plot_model_mean` / `plot_model_median`, and it
# does not run VFSA.
#
# Run:  julia --project=. examples/musgrave_plot_result_coarse.jl
#
# Optional env: SMARTPRIOR_WORK (default tmp_musgrave_coarse next to this repo).

using SmartPriorMT
using MTGeophysics
using Printf
using Plots
gr()

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_musgrave_coarse"))
const CLIMS = (0.5, 4.5)

const MODELS = (
    (joinpath(WORK, "inv_half", "model.start"),  "Half-space start"),
    (joinpath(WORK, "inv_half", "model.c2best"), "Half-space result (rms=2.299)"),
    (joinpath(WORK, "inv_prior", "model.start"), "Prior start"),
    (joinpath(WORK, "inv_prior", "model.c1best"), "Prior result (rms=2.497)"),
)

for (path, _) in MODELS
    isfile(path) || error("missing $path; run examples/musgrave_vfsa_compare_coarse.jl first")
end

# Same as build_musgrave_mesh, except y_core_cell = 20000.0.
sites = build_musgrave_profile()
frequencies = collect(1 ./ sites.periods)
issorted(frequencies) || reverse!(frequencies)
mesh = BuildMesh2D(;
    frequencies = frequencies,
    y_core_range = (-50000.0, 494727.4),
    y_core_cell = 20000.0,
    y_padding = 100000.0,
    air_cells = 6,
    ground_layers = vcat(fill(200.0, 10), fill(1000.0, 10),
                         fill(5000.0, 10), fill(20000.0, 10)),
    receiver_positions = sites.y,
)
grid = grid_from_mt2dmesh(mesh)

y_km = grid.cy ./ 1000
z_km = grid.cz ./ 1000
site_km = sites.y ./ 1000

load_log_rho(path) = from_mt2d(load_model2d(path).resistivity, mesh)

fields = [load_log_rho(path) for (path, _) in MODELS]

println("log10 ρ extrema  (finite cells)")
for ((_, title), A) in zip(MODELS, fields)
    lo, hi = extrema(filter(isfinite, vec(A)))
    @printf("  %-32s  min %7.4f  max %7.4f\n", title, lo, hi)
end

as_heatmap(A) = permutedims(A[1, :, :], (2, 1))   # [nz, ny] for heatmap(y, z, Z)

function panel(A, title; colorbar)
    p = heatmap(y_km, z_km, as_heatmap(A);
                clims = CLIMS,
                c = :Spectral,
                yflip = true,
                ylims = (0, maximum(z_km)),
                xlims = (minimum(y_km), maximum(y_km)),
                colorbar = colorbar,
                colorbar_title = colorbar ? "log₁₀ ρ" : "",
                title = title,
                xlabel = "y (km)",
                ylabel = "depth (km)",
                framestyle = :box,
                legend = false)
    scatter!(p, site_km, fill(0.0, length(site_km));
             markershape = :dtriangle,
             markersize = 6,
             markercolor = :black,
             markerstrokewidth = 0)
    return p
end

plt = plot(panel(fields[1], MODELS[1][2]; colorbar = false),
           panel(fields[2], MODELS[2][2]; colorbar = true),
           panel(fields[3], MODELS[3][2]; colorbar = false),
           panel(fields[4], MODELS[4][2]; colorbar = true);
           layout = (2, 2),
           size = (1400, 900),
           left_margin = 6Plots.mm,
           bottom_margin = 6Plots.mm,
           plot_title = "Musgrave VFSA coarse  (shared log₁₀ ρ scale $(CLIMS[1])–$(CLIMS[2]))")

out = joinpath(WORK, "musgrave_comparison_coarse.png")
savefig(plt, out)
@info "wrote" out bytes = filesize(out)
