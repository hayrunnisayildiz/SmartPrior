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

# Her veri paneli colorbar=false: renk skalasını hiçbir panele gömmüyoruz, o
# yüzden 5 panel de (truth, A, B, C, D) tam eşit genişlikte kalıyor -- önceki
# sürümde D'nin "ezik" görünmesinin sebebi tam olarak buydu (colorbar o
# panelin kendi genişliğinden yer çalıyordu).
function panel(A, title)
    heatmap(y_km, z_km, as_heatmap(A);
            clims = CLIMS, c = :Spectral, yflip = true,
            ylims = (0, maximum(z_km)), xlims = (minimum(y_km), maximum(y_km)),
            colorbar = false,
            title = title, titlefontsize = 9,
            xlabel = "y (km)", ylabel = "depth (km)",
            framestyle = :box, legend = false)
end

# ayrı, ince bir "sadece renk skalası" paneli -- veri panellerinin genişliğini
# etkilemesin diye kendi dar sütununda duruyor
function colorbar_only()
    strip = reshape(range(CLIMS[1], CLIMS[2]; length = 256), :, 1)
    heatmap(strip; c = :Spectral, clims = CLIMS, colorbar = true,
            colorbar_title = "log₁₀ ρ", xaxis = false, yaxis = false,
            legend = false, framestyle = :none)
end

top_panels = Any[panel(truth_field, "TRUTH")]
for arm in ARMS
    push!(top_panels, panel(starts[arm.key], "$(arm.label) — başlangıç"))
end
push!(top_panels, colorbar_only())

bottom_panels = Any[panel(truth_field, "TRUTH")]
for arm in ARMS
    push!(bottom_panels, panel(results[arm.key], "$(arm.label) — VFSA sonucu"))
end
push!(bottom_panels, colorbar_only())

plt = plot(top_panels[1:5]..., bottom_panels[1:5]..., colorbar_only();
           layout = (2, 6),
           size = (2200, 700),
           left_margin = 5Plots.mm,
           bottom_margin = 5Plots.mm,
           plot_title = "Ablasyon: TRUTH vs A/B/C/D  (üst=başlangıç, alt=VFSA sonucu, ortak skala $(CLIMS[1])-$(CLIMS[2]))")

out = joinpath(WORK, "ablation_comparison.png")
savefig(plt, out)
@info "wrote" out bytes = filesize(out)

# tek başına, sadece VFSA sonuçlarının yan yana karşılaştırması (rapor için)
plt2 = plot(panel(truth_field, "TRUTH"),
            [panel(results[arm.key], "$(arm.label)") for arm in ARMS]...,
            colorbar_only();
            layout = (1, 6),
            size = (2200, 400),
            left_margin = 5Plots.mm, bottom_margin = 8Plots.mm,
            plot_title = "VFSA sonuçları: TRUTH vs A/B/C/D")
out2 = joinpath(WORK, "ablation_results_only.png")
savefig(plt2, out2)
@info "wrote" out2 bytes = filesize(out2)