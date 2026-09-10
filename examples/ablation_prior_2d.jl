# Ablasyon: sinir ağı gerçekten bir şey katıyor mu?
#
# examples/compare_prior_2d.jl'nin aynısı (aynı truth, aynı gürültülü veri, aynı
# VFSA ayarları) ama 2 yerine 4 başlangıç kolu var:
#
#   A. yarı-uzay              -- hiçbir veri kullanmıyor (mevcut taban çizgi)
#   B. NB-bedava              -- sadece Niblett-Bostick, gravite YOK, eğitim YOK
#   C. NB + doğrusal gravite  -- kapalı-form damped least-squares gravite
#                                ters çözümü + TEK SABİT petrofizik eğim.
#                                Ağ yok, gradyan inişi yok.
#   D. tam akıllı prior       -- mevcut PriorNet + eğitim (compare_prior_2d.jl'deki
#                                ile birebir aynı)
#
# Soru: D, C'ye göre anlamlı bir kazanç sağlıyor mu? Sağlamıyorsa, ağın
# (eğitim, topluluk, Fourier-kodlama gibi karmaşıklığın tamamı) katkısı NB +
# tek-parametreli doğrusal düzeltmeden fazla değil demektir.
#
# Çalıştırma:  julia --project=. examples/ablation_prior_2d.jl
# compare_prior_2d.jl'den daha uzun sürer (2 yerine 4 VFSA koşusu var).

using SmartPriorMT
using MTGeophysics
using LinearAlgebra
using Printf
using Random
using Statistics

const WORK = get(ENV, "SMARTPRIOR_WORK", mktempdir(; prefix = "smartprior_ablation_"))
const MT_SEED = parse(Int, get(ENV, "SMARTPRIOR_MT_SEED", "20260827"))
const TRAIN_SEED = parse(Int, get(ENV, "SMARTPRIOR_TRAIN_SEED", "2026"))
const VFSA_SEED = parse(Int, get(ENV, "SMARTPRIOR_VFSA_SEED", "4242"))
const GRAV_SEED = parse(Int, get(ENV, "SMARTPRIOR_GRAV_SEED", "11"))
mkpath(WORK)
@info "working directory" WORK
@info "seeds" MT_SEED TRAIN_SEED VFSA_SEED GRAV_SEED

#---------- 1. mesh ve truth (compare_prior_2d.jl ile birebir aynı) ----------

periods = 10.0 .^ range(-2, 2.5; length = 14)

mesh = BuildMesh2D(
    frequencies = reverse(1 ./ periods),
    y_core_range = (-8000.0, 8000.0),
    y_core_cell = 400.0,
    y_padding = 10_000.0,
    air_cells = 6,
    ground_layers = vcat(fill(200.0, 8), fill(400.0, 8), fill(800.0, 6)),
    receiver_positions = collect(-7000.0:1000.0:7000.0),
)

grid = grid_from_mt2dmesh(mesh)
@info "prior grid" size(grid) ncells(grid)

const HOST_LOG_RHO = 2.6
const SLAB_LOG_RHO = 0.6
const LOG_RHO_BOUNDS = (0.0, 4.0)
# aynı işaret sınırı network'ün eğitiminde kullandığı slope_bounds ile aynı:
# iletken cisim aynı zamanda yoğun, bu yüzden eğim negatif olmak zorunda.
const SLOPE_BOUNDS = (-4.0, -0.2)

truth = add_dipping_slab(truth_halfspace(grid; log_rho = HOST_LOG_RHO);
                         x0 = -1000.0, z0 = 1800.0,
                         dip_deg = 35.0, thickness = 700.0,
                         log_rho = SLAB_LOG_RHO, density = 350.0,
                         strike = :x)
@info "truth" truth

#---------- 2. MT gözlemleri (2B çözücüden, compare_prior_2d.jl ile aynı) ----------

rho_true_2d = to_mt2d(truth, mesh)
true_model_path = write_model2d(joinpath(WORK, "true.rho"), mesh, rho_true_2d;
                                title = "dipping conductive slab")

template_path = write_mt2d_data_template(joinpath(WORK, "template.ref"), mesh;
                                         impedance_error_fraction = 0.05)

obs_path = ForwardSolve2D(true_model_path, template_path;
                          add_noise = true, rng_seed = MT_SEED,
                          output_path = joinpath(WORK, "data.obs"))

obs = load_data2d(obs_path)
sites = profile_sites(mesh, obs)

#---------- 3. gravite gözlemleri (compare_prior_2d.jl ile aynı) ----------

grav_x = collect(-9000.0:500.0:9000.0)
gravity = synth_gravity(truth, zeros(length(grav_x)), grav_x;
                        noise = 0.1, rng = Xoshiro(GRAV_SEED))
@printf("gravity anomaly: %.2f to %.2f mGal, error %.2f mGal\n",
        minimum(gravity.value), maximum(gravity.value), gravity.err[1])

A_grav = gravity_matrix(grid, gravity.x, gravity.y, gravity.z)

#---------- 4. Kol A -- yarı-uzay ----------

start_A = fill(HOST_LOG_RHO, size(grid))

#---------- 5. Kol B -- NB-bedava (gravite YOK, eğitim YOK) ----------

baseline = nb_baseline(grid, sites)
start_B = clamp.(baseline, LOG_RHO_BOUNDS...)
@printf("NB baseline vs truth: rmse %.3f, correlation %.3f\n",
        rmse(truth.log_rho, baseline), anomaly_correlation(truth.log_rho, baseline))

#---------- 6. Kol C -- NB + doğrusal gravite (ağ YOK, kapalı-form) ----------
#
# Klasik Tikhonov-damped en küçük kareler gravite ters çözümü:
#   rho_density = (A'A + lambda*I) \ (A' * obs_detrended)
# Ardından bu yoğunluk düzeltmesi, TEK SABİT bir petrofizik eğimle (network'ün
# öğrendiği eğim değil, SLOPE_BOUNDS'ın orta noktası -- elle, bir kez seçilmiş)
# log10-özdirenç düzeltmesine çevrilip NB temeline eklenir. Hiçbir gradyan inişi
# veya sinir ağı yok; tek bir doğrusal cebir çözümü.

w = 1 ./ abs2.(gravity.err)
obs_mean = sum(w .* gravity.value) / sum(w)
obs_detrended = gravity.value .- obs_mean

lambda = 0.1 * mean(diag(A_grav' * A_grav))   # tek düzenlileştirme sabiti, elle
density_lin = (A_grav' * A_grav + lambda * I) \ (A_grav' * obs_detrended)

slope_fixed = mean(SLOPE_BOUNDS)   # -2.1, elle seçilmiş tek sabit
mu_correction = density_lin ./ (SmartPriorMT.DENSITY_SCALE * slope_fixed)

start_C = clamp.(vec(baseline) .+ mu_correction, LOG_RHO_BOUNDS...)
start_C = reshape(start_C, size(grid))

@printf("NB+linear vs truth:   rmse %.3f, correlation %.3f\n",
        rmse(truth.log_rho, start_C), anomaly_correlation(truth.log_rho, start_C))

#---------- 7. Kol D -- tam akıllı prior (compare_prior_2d.jl ile aynı) ----------

stack = build_features(grid; gravity = gravity, sites = sites, baseline = baseline)
X = encode_features(stack; n_bands = 4)

site_cells = [(1, clamp(searchsortedlast(grid.y, y), 1, size(grid, 2)))
              for y in sites.y]

targets = PriorTargets(
    gravity = (A_grav, gravity.value, gravity.err),
    mt = (sites, site_cells),
    reference = vec(baseline),
    sigma_target = 0.35,
    sigma_drive = SigmaDriveConfig(),
)

net = PriorNet(size(X, 1);
               width = 96, depth = 4,
               log_rho_bounds = LOG_RHO_BOUNDS,
               residual_span = 2.5,
               sigma_bounds = (0.05, 0.9))

config = TrainConfig(
    epochs = 3000,
    learning_rate = 3.0e-3,
    log_every = 250,
    seed = TRAIN_SEED,
    slope_bounds = SLOPE_BOUNDS,
    weights = LossWeights(gravity = 1.0, mt = 1.0, smooth = 10.0, sigma = 0.1,
                          reference = 0.1),
    checkpoint_path = joinpath(WORK, "prior.jld2"),
    checkpoint_every = 1000,
)

ensemble, results = train_ensemble(net, X, grid, targets;
                                   nmembers = 5, config = config,
                                   offset = vec(baseline))

bundle_D = prior_from_ensemble(grid, ensemble, X;
                               offset = vec(baseline), k = 2.0,
                               log_rho_bounds = LOG_RHO_BOUNDS)
start_D = bundle_D.mu

@printf("full prior (D) vs truth: rmse %.3f, correlation %.3f  (öğrenen slope %+.3f)\n",
        rmse(truth.log_rho, start_D), anomaly_correlation(truth.log_rho, start_D),
        results[1].history[end].slope)

#---------- 8. dört kolu da aynı VFSA ayarlarıyla ters çöz ----------

inv_config = VFSA2DMTConfig(
    n_chains = 2, n_ctrl = 250, max_iter = 1200,
    log_bounds = LOG_RHO_BOUNDS, step_scale = 0.11, cool_ratio = 1.0e-3,
    target_rms = 1.0, seed = VFSA_SEED, keep_models = false, output_root = WORK,
)

arms = (
    A = (label = "yari-uzay",         mu = start_A),
    B = (label = "NB-bedava",         mu = start_B),
    C = (label = "NB+dogrusal",       mu = start_C),
    D = (label = "tam akilli prior",  mu = start_D),
)

results_vfsa = Dict{Symbol,Any}()
for (key, arm) in pairs(arms)
    @info "inverting" arm.label
    path = write_model2d(joinpath(WORK, "start_$(key).rho"), mesh,
                         to_mt2d(reshape(arm.mu, size(grid)), mesh);
                         title = arm.label)
    run = VFSA2DMT(path, obs_path;
                   run_dir = joinpath(WORK, "inv_$(key)"),
                   true_model_path = true_model_path,
                   config = inv_config)
    result = from_mt2d(run.best_chain.best_resistivity, mesh)
    results_vfsa[key] = (run = run, result = result,
                         cmp = compare_starts(truth.log_rho, arm.mu, result))
end

#---------- 9. özet tablo ----------

println()
println(repeat("=", 78))
println("  ablasyon sonucu: ağ gerçekten bir şey katıyor mu?")
println(repeat("=", 78))
@printf("%-20s %14s %14s %14s %14s\n", "", "A yari-uzay", "B NB-bedava", "C NB+dogrusal", "D tam prior")
@printf("%-20s %14.4f %14.4f %14.4f %14.4f\n", "baslangic rmse",
        rmse(truth.log_rho, start_A), rmse(truth.log_rho, start_B),
        rmse(truth.log_rho, start_C), rmse(truth.log_rho, start_D))
@printf("%-20s %14.4f %14.4f %14.4f %14.4f\n", "VFSA sonrasi rmse",
        results_vfsa[:A].cmp.rmse_final, results_vfsa[:B].cmp.rmse_final,
        results_vfsa[:C].cmp.rmse_final, results_vfsa[:D].cmp.rmse_final)
@printf("%-20s %14.3f %14.3f %14.3f %14.3f\n", "iyilesme orani",
        results_vfsa[:A].cmp.improvement, results_vfsa[:B].cmp.improvement,
        results_vfsa[:C].cmp.improvement, results_vfsa[:D].cmp.improvement)
@printf("%-20s %14.3f %14.3f %14.3f %14.3f\n", "korelasyon (son)",
        results_vfsa[:A].cmp.correlation_final, results_vfsa[:B].cmp.correlation_final,
        results_vfsa[:C].cmp.correlation_final, results_vfsa[:D].cmp.correlation_final)
println(repeat("=", 78))

gain_C_over_B = results_vfsa[:B].cmp.rmse_final - results_vfsa[:C].cmp.rmse_final
gain_D_over_C = results_vfsa[:C].cmp.rmse_final - results_vfsa[:D].cmp.rmse_final
println()
@printf("gravitenin (agsiz, dogrusal) NB'ye katkisi (B -> C):  rmse degisimi %+.4f\n", gain_C_over_B)
@printf("agin dogrusal duzeltmeye katkisi         (C -> D):  rmse degisimi %+.4f\n", gain_D_over_C)
if gain_D_over_C < 0.02 * results_vfsa[:C].cmp.rmse_final
    println("\n-> AG, doğrusal düzeltmeye göre ölçülebilir bir kazanç sağlamıyor.")
    println("   Bu senaryoda NB + tek-parametreli doğrusal gravite düzeltmesi yeterli görünüyor.")
else
    println("\n-> AG, doğrusal düzeltmeye göre ölçülebilir bir kazanç sağlıyor.")
end

open(joinpath(WORK, "ablation_metrics.txt"), "w") do io
    for key in (:A, :B, :C, :D)
        r = results_vfsa[key]
        println(io, key, "\t", arms[key].label, "\trmse_start=", r.cmp.rmse_start,
                "\trmse_final=", r.cmp.rmse_final, "\timprovement=", r.cmp.improvement,
                "\tcorrelation_final=", r.cmp.correlation_final)
    end
end
@info "all outputs kept" WORK
