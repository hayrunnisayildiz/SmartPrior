# SmartPriorMT

2B manyetotellürik (MT) ters çözümü için **warm-start prior**. Homojen yarı-uzay
yerine, gravite + gözlenen MT’nin Niblett–Bostick (NB) dönüşümü (+ isteğe bağlı
topoğrafya) ile hücre-bazlı bir nominal log₁₀ ρ (`μ`) üretir ve bunu
MTGeophysics VFSA’sına başlangıç modeli olarak verir.

**Bu bir joint inversion değildir.** Gravite VFSA χ²’sine girmez; füzyon yalnızca
prior üretiminde olur. Çözücünün ileri fiziği, pertürbasyonu ve soğuma çizelgesi
değiştirilmez. `σ` (`prior.std`, `prior.lo` / `prior.hi`) yazılır ama mevcut
`VFSA2DMT` global `log_bounds` kullanır — hücre aralığı inversiyona girmez.

Sürüm 0.1.0 · Julia 1.10 · MTGeophysics 0.5.0. Tam tablo, figür ve sunum sırası:
[`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md).

## Kurulum

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Yerel yol:

```julia
Pkg.add(path="path/to/SmartPriorModel")
```

| Paket | Compat |
|---|---|
| ArchGDAL | 0.10.12 |
| ComponentArrays | 0.15.47 |
| ForwardDiff | 1.4.5 |
| Interpolations | 0.15.1 |
| JLD2 | 0.6.6 |
| Lux | 1.31.4 |
| MTGeophysics | 0.5.0 |
| Optimisers | 0.4.9 |
| Plots | 1.41.7 |
| Zygote | 0.7.12 |
| julia | 1.10 |

Stdlib (compat yok): `LinearAlgebra`, `Printf`, `Random`, `Statistics`.

## Hızlı örnek

```bash
julia --project=. examples/compare_prior_2d.jl
```

Sentetik eğimli iletken slab; 2B sonlu-hacim MT + bağımsız yoğunluktan gravite.
Prior truth’u görmez. Aynı VFSA ayarı ve seed ile iki koşu: yarı-uzay vs prior.
Sürenin çoğu inversiyonda geçer.

Truth-bağımsız hiperparametre (eğitim only, VFSA yok):

```bash
SMARTPRIOR_PROTOCOL=blind SMARTPRIOR_WORK=tmp_blind_t1 \
  julia --project=. examples/train_prior_blind.jl
```

## Mimari

```
FeatureStack → Lux koordinat-ağı (NB’ye residual) → PriorBundle (μ, σ)
            → start_prior.rho → VFSA2DMT
```

2B profilde `x_norm` ve `gravity_dx` tanım gereği sıfır → **efektif 12 kanal**
(14 değil). Sentetik yolda topo kanalı yok; Musgrave DEM’i `surface_z` olarak
verir. VFSA ileri çözümü düz-datum varsayar.

Kayıp etiketsiz ve AD-güvenli: prizma gravite, 1B MT kolon, pürüzlülük, `σ`
hedefi (`SigmaDriveConfig`; likelihood değil). `slope` işareti dışsal varsayım
(sentetikte yoğun=iletken); büyüklük kör protokolde geniş tutucu bant.

Bu sürücü sonlu bütçede start-bağımlıdır (`max_iter=400`, `step_scale=0.11`,
geometrik soğuma). Kanonik küresel VFSA (Ingber / Sen & Stoffa) iddiası yoktur.

## Doğrulama

Kaynak koşular ve std: [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md) §4–§6.
Tek seed iddia sayılmaz. Data RMS, model RMSE değildir.

### Sentetik — 3 seed (VFSA, chain 1)

| | yarı-uzay | prior |
|---|---:|---:|
| data RMS, iter 1 | 30.260 ± 0.019 | **4.558 ± 0.080** |
| data RMS, iter 400 | 4.995 ± 0.347 | **3.744 ± 0.340** |

Iter 1 farkı stokastik değil (iki başlangıcın ileri çözümü).

| | NB | prior |
|---|---:|---:|
| RMSE (log₁₀ ρ) | 0.551 ± 0.004 | 0.546 ± 0.008 |
| korelasyon | 0.429 ± 0.002 | **0.443 ± 0.008** |

Prior, ham RMSE’de NB’yi kesin geçmez. Kazanç korelasyonda. 3-seed VFSA tablosu
düzgün-σ koşularından; kod artık `SigmaDriveConfig` kullanıyor.

**Blind** (`residual_span` = half_band, `slope_bounds = (-10, −0.1)`, eğitim
only): RMSE **0.518 ± 0.003**, korelasyon **0.501 ± 0.007**. Oracle span=2.5
kazancı taşımıyor. İşaret hâlâ dışsal varsayım.

`reference = 0.1` en dengeli kalibrasyon noktası — **tek seed**.

### Ablasyon

Aynı sentetik + VFSA, dört başlangıç (`examples/ablation_prior_2d.jl`): A
yarı-uzay, B NB, C NB+doğrusal gravite, D tam prior. **Tek seed.**

![Ablasyon: TRUTH vs A/B/C/D](docs/figures/ablation_comparison.png)

![Yalnızca VFSA sonuçları](docs/figures/ablation_results_only.png)

### Musgrave (AusLAMP) — truth yok

Nihai yeraltı kalitesi iddia edilemez. Ölçülen şey veriye uyum hızı.

Tam bütçe (`max_iter=400`, erken durma kapalı): prior 5. iterasyonda RMS 1.0
altına iner, plato **0.911**; yarı-uzay final **1.040**. Plato arama tıkanıklığı
değil: `model_err_frac=0.4`, N=414, χ²/datum=1 tam RMS=1. Kaba ağda (20 km)
prior **daha kötü** (best RMS 2.497 vs 2.299).

## Sınırlamalar

- Tek küresel gravite–özdirenç eğimi → tek baskın litoloji. Karışık litolojide
  (Musgrave: dirençli Giles + iletken zon) zorlanır.
- TE-only. TM mevcut ama istasyon-bazlı model hatası tek `model_err_frac` ile
  kalibre edilemiyor (v0.2).
- `σ` inversiyona girmez; anchor yokken NLL yok. `sigma_penalty` AD-dışı hedef
  haritayı izler. `corr(σ, |hata|)` sigma_drive ile yeniden ölçülmedi.
- Gravite: operatör-düzeyinde ters suç (bilerek); petrofizik yok. MT’de yok.
- Gravite her kata sönümsüz kopyalanır (derinlik körü).
- `RealDataIO.jl` Musgrave’e özgü.
- `BoundedVFSA.jl` yazıldı, 2B/3B yolda kullanılmadı, kanıtsız.
- Slab eğimi iddiası **geri çekildi** (işaret seed’e göre değişiyor).


## Lisans

MIT, bkz. [LICENSE](LICENSE).
