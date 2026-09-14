# SmartPriorMT

2B manyetotellürik (MT) ters çözümü için **warm-start prior**. Homojen yarı-uzay
yerine, gravite + gözlenen MT’nin Niblett–Bostick (NB) dönüşümü (+ isteğe bağlı
topoğrafya) ile hücre-bazlı bir nominal log₁₀ ρ (`μ`) ve `σ` üretir; `μ`’yü
MTGeophysics VFSA’sına başlangıç modeli olarak verir.

**Bu bir joint inversion değildir.** Gravite VFSA χ²’sine girmez; füzyon yalnızca
prior aşamasındadır (prior-aşaması füzyonu). Çözücünün ileri fiziği, pertürbasyonu
ve soğuma çizelgesi değiştirilmez. `σ` (`prior.std`, `prior.lo` / `prior.hi`)
yazılır ama mevcut `VFSA2DMT` global `log_bounds` kullanır — hücre aralığı
inversiyona girmez.

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

Ablasyon (A yarı-uzay, B NB, C NB+doğrusal gravite, D tam prior):

```bash
julia --project=. examples/ablation_prior_2d.jl
```

İsteğe bağlı derinlik-duyarlı gravite kanalı (varsayılan kapalı; geriye uyumlu):

```bash
SMARTPRIOR_GRAVITY_SENSITIVITY=true julia --project=. examples/ablation_prior_2d.jl
```

Truth-bağımsız hiperparametre (eğitim only, VFSA yok):

```bash
SMARTPRIOR_PROTOCOL=blind SMARTPRIOR_WORK=tmp_blind_t1 \
  julia --project=. examples/train_prior_blind.jl
```

Jeoloji sweep'i (eğim açısı + kontrast işareti, VFSA yok — bkz.
[`docs/ARA_RAPOR.md` §10](docs/ARA_RAPOR.md#10-genelleştirme-planı-sıradaki-adımlar)):

```bash
julia --project=. examples/scenario_sweep_prior_2d.jl
```

## Mimari

```
FeatureStack → Lux koordinat-ağı (NB’ye residual) → PriorBundle (μ, σ)
            → start_prior.rho → VFSA2DMT
```

2B özellikler: 3 koordinat + 2 derinlik + 7 gravite + 0 topo + 1 coverage +
1 baseline = 14 kanal. `x_norm` ve `gravity_dx` tanım gereği sıfır → **efektif 12**.
“14 kanal” demek yanıltıcıdır. Sentetik yolda topo kanalı yok; Musgrave DEM’i
`surface_z` olarak verir. VFSA ileri çözümü düz-datum varsayar.

Mevcut 7 gravite kanalı yüzey haritasının her kata sönümsüz kopyasıdır (extrude;
derinlik körü). `build_features(...; gravity_sensitivity=false)` varsayılanı bunu
korur. `true` iken bir ek kanal eklenir — aşağıda.

Kayıp etiketsiz ve AD-güvenli: prizma gravite, 1B MT kolon, pürüzlülük, `σ`
hedefi (`SigmaDriveConfig`; likelihood değil). `slope` işareti dışsal varsayım
(sentetikte yoğun=iletken); büyüklük kör protokolde geniş tutucu bant.

Bu sürücü sonlu bütçede start-bağımlıdır (`max_iter=400`, `step_scale=0.11`,
geometrik soğuma). Kanonik küresel VFSA (Ingber / Sen & Stoffa) iddiası yoktur.

### `gravity_sensitivity` (isteğe bağlı, varsayılan kapalı)

Hücre-bazlı `|bu hücredeki birim yoğunluk kontrastının yüzey gravitesine katkısı|`:
`prism_gz` / `gravity_matrix` sütun L2 normu, hücre hacmine bölünür, yüzey
anomalisi ile çarpılır. Yerel 3B (extrude değil); derinlikle sönümlenir. Mevcut
extrude kanallar durur.

İlk ablasyon (kol D, aynı seed’ler: MT=20260827, TRAIN=2026, VFSA=4242, GRAV=11)
bu kanalın y≈−3 km’deki düşey iletken şeridi kaldırmadığını gösterdi. Prior vs
truth, **tek seed**: RMSE 0.547 → 0.552, `anomaly_correlation` 0.438 → 0.433
(hafif kötüleşme). Derinlik-smearing şeridini düzeltti iddiası yoktur; isteğe
bağlı derinlik-duyarlı gravite özelliğidir.

### `σ`

Üretim yolu (`compare_prior_2d.jl`, `ablation_prior_2d.jl`) `SigmaDriveConfig()`
verir. `σ` likelihood değildir: NLL yalnızca `heteroscedastic_nll` ile **anchor**
hücrelerinde; anchor yokken `σ` üzerinde NLL gradyanı yoktur. `sigma_penalty` tüm
hücrelerde, AD dışında üretilen hedef haritayı izler (MT kolon RMS artığı +
gravite kolon-normu). Kalibre belirsizlik haritası iddiası yoktur.

| koşu | `corr(σ, |residual|)` |
|---|---:|
| düzgün-σ (3-seed tabloları) | ≈ +0.09 |
| `sigma_drive`, ablasyon kol D | ≈ +0.27 (`tmp_ablation_gsens`: +0.269) |

## Doğrulama

Kaynak koşular ve std: [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md) §4–§6.
Tek seed iddia sayılmaz. Data RMS, model RMSE değildir.

### Sentetik — 3 seed (VFSA, chain 1)

| | yarı-uzay | prior |
|---|---:|---:|
| data RMS, iter 1 | 30.260 ± 0.019 | **4.558 ± 0.080** |
| data RMS, iter 400 | 4.995 ± 0.347 | **3.744 ± 0.340** |

Iter 1 farkı stokastik değil (iki başlangıcın ileri çözümü). Ayakta duran
warm-start iddiası buradadır.

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

Kol D başlangıç RMSE **0.547**, `anomaly_correlation` **0.438** (`gravity_sensitivity`
kapalı). Kanal açıkken aynı seed’lerle 0.552 / 0.433 — yukarıdaki gibi şerit
kalkmadı.

### Musgrave (AusLAMP) — truth yok

Nihai yeraltı kalitesi iddia edilemez. Ölçülen şey veriye uyum hızı.

Tam bütçe (`max_iter=400`, erken durma kapalı): prior 5. iterasyonda RMS 1.0
altına iner, plato **0.911**; yarı-uzay final **1.040**. Plato arama tıkanıklığı
değil: `model_err_frac=0.4`, N=414, χ²/datum=1 tam RMS=1. Kaba ağda (20 km)
prior **daha kötü** (best RMS 2.497 vs 2.299).

## Sınırlamalar

- Tek küresel gravite–özdirenç eğimi → tek baskın litoloji varsayımı.
  **İyi:** jeotermal kil örtüsü, sedimanter havza, sülfid.
  **Kötü:** karışık litoloji (grafitli şeyl / mafik intrüzyon / tuz bir arada).
  Her sahada çalışır iddiası yoktur. Musgrave (dirençli Giles + iletken zon)
  karışık litoloji örneğidir. Şu ana kadarki sentetik sonuçların tümü (§Doğrulama)
  tek bir eğim açısı ve tek bir kontrast işareti üzerinde; bunu genişletme planı
  [`docs/ARA_RAPOR.md` §10](docs/ARA_RAPOR.md#10-genelleştirme-planı-sıradaki-adımlar).
- TE-only. TM mevcut ama istasyon-bazlı model hatası tek `model_err_frac` ile
  kalibre edilemiyor (v0.2).
- `σ` inversiyona girmez ve kalibre belirsizlik haritası değildir (yukarıdaki
  tablodaki korelasyonlar).
- Gravite: `synth_gravity` aynı `gravity_matrix`’i çağırır — **operatör** ters
  suçu VAR. Petrofizik ters suçu YOK (slab yoğunluğu `density_from_mu`’dan
  geçmez). Ağ A’nın serbest tersini değil, A ∘ (lineer μ-kalıntısı) bileşimini
  uydurur; eğim `(-4, −0.2)` ile sınırlı. MT’de ters suç yok (obs 2B sonlu hacim,
  eğitim terimi 1B resürsiyon).
- Mevcut gravite kanalları extrude (derinlik körü). `gravity_sensitivity` isteğe
  bağlı ve varsayılan kapalı; ilk D-kolu ablasyonu şeridi kaldırmadı.
- `RealDataIO.jl` Musgrave’e özgü.
- `BoundedVFSA.jl` 3B için hazırlandı, çalıştırılmadı; 2B/3B yolda kullanılmadı,
  kanıtsız.
- Slab eğimi iddiası **geri çekildi** (işaret seed’e göre değişiyor).

## Lisans

MIT, bkz. [LICENSE](LICENSE).
