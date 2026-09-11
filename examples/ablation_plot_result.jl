# Ablasyonun 4 kolu için yeraltı özdirenç alanlarını çiziyor: truth, ve
# A(yarı-uzay)/B(NB-bedava)/C(NB+doğrusal)/D(tam prior) için hem başlangıç hem
# VFSA sonucu. examples/musgrave_plot_result.jl ile aynı desen (from_mt2d + GR
# heatmap); VFSA2DMT'nin kendi çizim fonksiyonunu çağırmıyor.
#
# Önce examples/ablation_prior_2d.jl'i çalıştırmış olmanız gerekir; bu script
# onun WORK dizinindeki dosyaları okur.
#
# Run:  julia --project=. examples/ablation_plot_result.jl
# Optional env: SMARTPRIOR_WORK (ablation_prior_2d.jl'in yazdırdığı WORK yolu)

using SmartPriorMT
using MTGeophysics
using Printf
using Plots

# Headless GR. Without this the second savefig can hang when stdout is a pty
# (`send: No buffer space available`).
get!(ENV, "GKSwstype", "100")
gr()

const WORK = get(ENV, "SMARTPRIOR_WORK", "")
isempty(WORK) && error("SMARTPRIOR_WORK ortam değişkenini ablation_prior_2d.jl'in " *
                        "yazdırdığı WORK dizinine ayarlayın")

const CLIMS = (0.0, 4.0)   # ablation_prior_2d.jl'deki LOG_RHO_BOUNDS ile aynı

const ARMS = (
    (key = :A, label = "A yarı-uzay"),
    (key = :B, label = "B NB-bedava"),
    (key = :C, label = "C NB+doğrusal"),
    (key = :D, label = "D tam prior"),
)

# mesh'i yeniden kurmak yerine, ablation_prior_2d.jl'in yazdığı truth modelinden
# geometriyi okuyoruz -- mesh parametreleri script'te sabit olduğu için
# yeniden inşa etmek de güvenli, ama burada dosyadan okumak script'ler arası
# senkron kalmayı garanti eder.
mesh = BuildMesh2D(
    frequencies = reverse(1 ./ (10.0 .^ range(-2, 2.5; length = 14))),
    y_core_range = (-8000.0, 8000.0),
    y_core_cell = 400.0,
    y_padding = 10_000.0,
    air_cells = 6,
    ground_layers = vcat(fill(200.0, 8), fill(400.0, 8), fill(800.0, 6)),
    receiver_positions = collect(-7000.0:1000.0:7000.0),
)
grid = grid_from_mt2dmesh(mesh)

load_log_rho(path) = from_mt2d(load_model2d(path).resistivity, mesh)

truth_field = load_log_rho(joinpath(WORK, "true.rho"))

starts  = Dict{Symbol,Array{Float64,3}}()
results = Dict{Symbol,Array{Float64,3}}()
for arm in ARMS
    start_path  = joinpath(WORK, "start_$(arm.key).rho")
    result_path = joinpath(WORK, "inv_$(arm.key)", "model.c2best")
    isfile(start_path)  || error("missing $start_path")
    isfile(result_path) || error("missing $result_path -- run ablation_prior_2d.jl first")
    starts[arm.key]  = load_log_rho(start_path)
    results[arm.key] = load_log_rho(result_path)
end

y_km = grid.cy ./ 1000
z_km = grid.cz ./ 1000

as_heatmap(A) = permutedims(A[1, :, :], (2, 1))   # [nz, ny] for heatmap(y, z, Z)

# Colorbar on the last panel of each row; that column is wider so D's heatmap
# stays the same width as A–C. Do not use a 7%-width dummy colorbar subplot:
# GR then emits invalid viewports and the strip steals a data slot.
#
# Top-row x-labels must be off — they collide with the bottom-row titles.
# `plot_title` needs an explicit gap or it sits on the first-row titles.
function panel(A, title; colorbar = false, ylabel = "depth (km)", xlabel = "y (km)")
    heatmap(y_km, z_km, as_heatmap(A);
            clims = CLIMS, c = :Spectral, yflip = true,
            ylims = (0, maximum(z_km)), xlims = (minimum(y_km), maximum(y_km)),
            colorbar = colorbar,
            colorbar_title = colorbar ? "log₁₀ ρ" : "",
            title = title, titlefontsize = 12,
            xlabel = xlabel, ylabel = ylabel,
            guidefontsize = 9, tickfontsize = 8,
            framestyle = :box, legend = false)
end

lyt = @layout [a{0.175w} b{0.175w} c{0.175w} d{0.175w} e{0.30w}
               f{0.175w} g{0.175w} h{0.175w} i{0.175w} j{0.30w}]

plt = plot(
    panel(truth_field, "TRUTH"; xlabel = ""),
    panel(starts[:A], "A start"; ylabel = "", xlabel = ""),
    panel(starts[:B], "B start"; ylabel = "", xlabel = ""),
    panel(starts[:C], "C start"; ylabel = "", xlabel = ""),
    panel(starts[:D], "D start"; colorbar = true, ylabel = "", xlabel = ""),
    panel(truth_field, "TRUTH"),
    panel(results[:A], "A VFSA"; ylabel = ""),
    panel(results[:B], "B VFSA"; ylabel = ""),
    panel(results[:C], "C VFSA"; ylabel = ""),
    panel(results[:D], "D VFSA"; colorbar = true, ylabel = "");
    layout = lyt,
    size = (2600, 920),
    left_margin = 8Plots.mm,
    right_margin = 4Plots.mm,
    top_margin = 8Plots.mm,
    bottom_margin = 6Plots.mm,
    plot_title = "Ablasyon: A yarı-uzay · B NB · C NB+doğrusal · D tam prior   (üst=başlangıç, alt=VFSA, log₁₀ ρ $(CLIMS[1])–$(CLIMS[2]))",
    plot_titlefontsize = 14,
    plot_titlegap = 12,
)

out = joinpath(WORK, "ablation_comparison.png")
savefig(plt, out)
@info "wrote" out bytes = filesize(out)

lyt2 = @layout [a{0.175w} b{0.175w} c{0.175w} d{0.175w} e{0.30w}]
plt2 = plot(
    panel(truth_field, "TRUTH"),
    panel(results[:A], "A VFSA"; ylabel = ""),
    panel(results[:B], "B VFSA"; ylabel = ""),
    panel(results[:C], "C VFSA"; ylabel = ""),
    panel(results[:D], "D VFSA"; colorbar = true, ylabel = "");
    layout = lyt2,
    size = (2600, 500),
    left_margin = 8Plots.mm,
    right_margin = 4Plots.mm,
    top_margin = 4Plots.mm,
    bottom_margin = 8Plots.mm,
)
out2 = joinpath(WORK, "ablation_results_only.png")
savefig(plt2, out2)
@info "wrote" out2 bytes = filesize(out2)