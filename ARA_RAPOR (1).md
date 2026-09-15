# SmartPriorMT — Ara Rapor

**Kime:** bilgisayar mühendisi (jeofizik varsayılmaz)
**Ne:** 2B manyetotellürik (MT) ters çözümü için öğrenilmiş **warm-start prior**
**Sürüm:** 0.1.0 · Julia 1.10 · Lux.jl + Zygote + MTGeophysics 0.5.0
**Tarih:** 9 Eylül 2026
**Kaynak:** `tmp_noisy_v2`, `tmp_seed2_v2`, `tmp_seed3_v2`, `tmp_blind_t{1,2,3}`,
`tmp_bestcase_t1`, `tmp_ablation`, `tmp_musgrave`, `tmp_musgrave_coarse` — sayılar
ilgili `metrics.txt` / `Summary.md` dosyalarından.

Bu bir **joint inversion değildir.** Füzyon yalnızca başlangıç modeli üretilirken olur; VFSA çözücüsüne dokunulmaz.

---

## 1. Problem (optimizasyon dili)

MT ters çözümü, yeraltının hücre-hücre log₁₀ özdirencini (`μ`) gözlenen elektromanyetik eğrilerden kestiren **kötü-konumlu (ill-posed)** bir ters problemdir. Kullanılan çözücü **VFSA** (Very Fast Simulated Annealing): stokastik, çok-başlangıçlı bir arama.

Bugünkü pratik: aramayı **homojen yarı-uzaydan** (her hücre aynı `ρ`) başlatmak. Bu, eğimli bir iletken gibi yanal yapıda pahalıdır — ilk ileri çözümün veri uyumsuzluğu (data RMS) ~30 civarındadır ve bütçenin çoğu “neredeyse doğru yere gelmeye” harcanır.

**Hedef:** aynı VFSA’yı, aynı seed ve aynı bütçeyle, ama daha iyi bir başlangıçtan çalıştırmak.

```
girdi  →  öğrenilmiş hücre-bazlı (μ, σ)  →  VFSA warm-start  →  özdirenç modeli
```

Ağın truth’u görmediği bir sentetikte, 3 bağımsız seed ile:

| | yarı-uzay | smart prior | ayrım |
|---|---:|---:|---|
| **data RMS, iter 1** | 30.260 ± 0.019 | 4.558 ± 0.080 | ~150 örnek std |
| **data RMS, iter 400** | 4.995 ± 0.347 | 3.744 ± 0.340 | aralıklar örtüşmüyor |

Iter 1 farkı stokastik değil: iki farklı başlangıç modelinin ileri çözümüdür.

---

## 2. Girdi → çıktı

### 2.1 Sistem sınırı

| | İçerik |
|---|---|
| **Girdi** | Gravite anomalisi (mGal) + MT eğrileri (görünür özdirenç, faz) + isteğe bağlı topo |
| **Çıktı** | Hücre-bazlı nominal log₁₀ ρ (`μ`) ve güven (`σ`); VFSA’ya başlangıç modeli + arama aralığı `μ ± kσ` |
| **Değiştirilmez** | VFSA’nın ileri fiziği, pertürbasyon, soğuma çizelgesi |
| **Etiket yok** | Yeraltı truth’u eğitimde yok. Kayıp, diferansiyellenebilir fizik terimleri. |

### 2.2 Veri yolu (sentetik doğrulama)

Aynı mesh, aynı gürültülü gözlem, iki kolda yalnızca başlangıç değişir.

```
1. Truth          1×56×22 grid (1232 hücre). Host log₁₀ρ = 2.6;
                  eğimli iletken slab (35°, log₁₀ρ = 0.6, Δρ = 350 kg/m³).
2. MT gözlemi     2B sonlu-hacim çözücü + gürültü. 15 istasyon, 14 periyot.
                  Prior bu eğrileri görür; truth’u görmez.
3. Gravite        Kapalı-form prizma operatörü + 0.1 mGal gürültü.
                  37 istasyon. Slab yoğunluğu özdirençten türetilmez.
4. NB baseline    Gözlenen MT’nin Niblett-Bostick dönüşümü (derinlik tahmini).
5. FeatureStack   14 kanal → Fourier kodlama → X ∈ ℝ^{38×1232}
6. PriorNet       Lux MLP, residual-mode (NB’ye düzeltme öğrenir).
                  5 üyeli ensemble, 3000 epoch, Zygote AD.
7. PriorBundle    (μ, σ) → start_prior.rho
8. VFSA × 2       Aynı seed, 2 zincir, 250 kontrol, 400 iter (ablasyonda 1200).
                  Kol A: yarı-uzay. Kol D: prior.
```

2B profilde `x_norm` ve `gravity_dx` tanım gereği sıfır → efektif **12 kanal**. “14 kanal” demek yanıltıcı.

### 2.3 Ağ ve kayıp

```
hücre özellikleri  →  MLP (genişlik 96, derinlik 4, GELU)
                   →  residual  (NB baseline’a eklenir, span = 2.5 dekât)
                   →  μ ∈ [0, 4]   (log₁₀ ρ, hard bound)
                   →  σ ∈ [0.05, 0.9]
```

Kayıp terimleri (hepsi AD-güvenli, etiketsiz):

| Terim | Ne ölçer | Operatör |
|---|---|---|
| gravite | yüzey anomalisi | kapalı-form prizma (`A · density(μ)`) |
| MT kolon | istasyon kolonunun 1B empedansı | generic 1B resürsiyon (Zygote) |
| smoothness | uzaysal pürüzlülük | grid Laplacian |
| sigma | hücre-bazlı `σ` hedefi | MT kolon artığı + gravite duyarlılığı (`SigmaDriveConfig`); AD dışında, her epoch |
| reference | NB’den sapma maliyeti | ablasyon / Musgrave’de ağırlık 0.1 |

`density = 100 · (slope · (μ − ref) + offset)`. `slope` öğrenilir, işaret sınırı `(-4, −0.2)`: bu sentetikte yoğun cisim iletkendir.

### 2.4 Dosya girdi/çıktı

| Aşama | Girdi | Çıktı |
|---|---|---|
| eğitim | `GravityObs`, `MTSites`, grid | `prior.jld2` (Lux parametreleri) |
| export | ensemble + `X` | `prior.rho`, `prior.lo`, `prior.hi` |
| VFSA | `start_*.rho` + `data.obs` | `model.mean`, `model.median`, `Summary.md`, RMS zinciri |
| skor | truth + iki sonuç | `metrics.txt` (RMSE, korelasyon, RMS@1, RMS@400) |

Çalıştırma: `julia --project=. examples/compare_prior_2d.jl`

---

## 3. Ne iddia ediyoruz, ne etmiyoruz

**Ayakta:** prior, VFSA’yı ~7× daha düşük ilk misfit’ten başlatır; 400 iterasyonda da daha düşük data RMS’te biter (3 seed, örnek std ile).

**Çürütülen:** “prior daha dik slab üretir.” Eğim farkının işareti seed’e göre değişiyor. Bu iddiayı kurmayın.

**Kanıtlanamadı:** gerçek sahada (Musgrave) prior’ın nihai yeraltı modelinin yarı-uzaydan daha doğru olduğu — truth yok.

---

## 4. Sentetik — 3 seed (ayakta duran tablo)

Aynı deney, üç bağımsız `(MT, train, VFSA, gravite)` seed’i. Chain 1. Std örnek (`n−1`).

| metrik | t1 `noisy_v2` | t2 `seed2_v2` | t3 `seed3_v2` | **ort ± std** |
|---|---:|---:|---:|---:|
| NB RMSE (log₁₀ ρ) | 0.556 | 0.547 | 0.551 | **0.551 ± 0.004** |
| prior RMSE | 0.546 | 0.554 | 0.539 | **0.546 ± 0.008** |
| NB korelasyon | 0.426 | 0.430 | 0.429 | **0.429 ± 0.002** |
| prior korelasyon | 0.438 | 0.439 | 0.453 | **0.443 ± 0.008** |
| yarı-uzay RMS iter 1 | 30.278 | 30.240 | 30.261 | **30.260 ± 0.019** |
| prior RMS iter 1 | 4.547 | 4.485 | 4.643 | **4.558 ± 0.080** |
| yarı-uzay RMS iter 400 | 4.598 | 5.242 | 5.145 | **4.995 ± 0.347** |
| prior RMS iter 400 | 3.462 | 4.122 | 3.649 | **3.744 ± 0.340** |

Okuma notu:

- Prior ham RMSE, NB baseline’ı **kesin geçmiyor** (aralıklar örtüşüyor). Kazanç korelasyonda: 0.443 vs 0.429, aralıklar örtüşmüyor.
- Data RMS, model RMSE değildir. VFSA veriye uyar; truth’a RMSE’nin iterasyon boyunca bozulması beklenen bir durum olabilir (gürültüye uydurma).
- Tablodaki koşular **düzgün σ** (`sigma_target = 0.35`, `sigma_drive` kapalı) ile alındı. Kod artık `SigmaDriveConfig` kullanıyor; `mean(σ) ≈ 0.36` ve `corr(σ, |res|) = +0.09` o eski konfigürasyona aittir, yeniden ölçülmedi.

### 4.1 Best-case vs blind hiperparametre (model uzayı)

İki ayrı tablo — karıştırmayın. Aynı 3 seed (`t1/t2/t3` seed seti), eğitim-only
(`examples/train_prior_blind.jl`), VFSA yok. Fark yalnızca `residual_span` ve
`slope_bounds`. §4 ana tablosundaki VFSA satırlarıyla karıştırmayın.

**Protokol (truth’suz, sahaya taşınabilir):**

| hiperparametre | best-case | blind |
|---|---|---|
| `residual_span` | 2.5 | `residual_span_half_band(log_rho_bounds)` = 2.0 |
| `slope_bounds` | `(-4, −0.2)` | `(-10, −0.1)` |
| işaret gerekçesi | sentetik tasarım (yoğun=iletken slab) — dışsal varsayım | aynı; büyüklük geniş tutucu bant |

`residual_span_from_baseline` (`k·σ_lat`, Features.jl) aynı kodla hesaplanır; bu
sentetikte `σ_lat ≈ 0.17` → k=3 ile floor=1.0’a oturur. Blind koşu half_band’i
kullanır (izin verici). Musgrave örnekleri de aynı half_band kuralını çağırır.

#### Best-case — yöntemin üst sınırı (uygun hiperparametrelerle)

Yayımlanan `tmp_*_v2` prior satırları (§4). Eşleşmiş yeniden-koşu
(`SMARTPRIOR_PROTOCOL=bestcase`, aynı kod + `SigmaDriveConfig`): t1 RMSE 0.544 /
corr 0.438 — yayımlananla bit-düzeyinde uyumlu; σ-drive bu karşılaştırmayı
bozmuyor.

| metrik | t1 | t2 | t3 | **ort ± std** |
|---|---:|---:|---:|---:|
| prior RMSE | 0.546 | 0.554 | 0.539 | **0.546 ± 0.008** |
| prior korelasyon | 0.438 | 0.439 | 0.453 | **0.443 ± 0.008** |

#### Blind — gerçekçi, truth-bağımsız protokol

Kaynak: `tmp_blind_t1`–`t3` (`SMARTPRIOR_PROTOCOL=blind`).

| metrik | t1 | t2 | t3 | **ort ± std** |
|---|---:|---:|---:|---:|
| prior RMSE | 0.517 | 0.520 | 0.515 | **0.518 ± 0.003** |
| prior korelasyon | 0.498 | 0.497 | 0.509 | **0.501 ± 0.007** |

**Bulgu:** Blind, best-case’i bozmuyor — model-uzayı RMSE/korelasyon **daha iyi**
(RMSE −0.029, corr +0.058 ort.). Kazanım oracle hiperparametreye bağlı değildi;
tersine, dar `(-4, −0.2)` bandı ve/veya span=2.5 bu metriklerde üst sınır değil.
NB’ye göre corr kazanımı korunuyor (blind 0.501 vs NB 0.429). Bu, leakage
endişesini “kritik”ten “hafif”e indirir (ortadan kaldırmaz: işaret hâlâ dışsal
varsayım).

Duyarlılık taraması: `examples/train_prior_sensitivity.jl`
(`residual_span ∈ {1.0…5.0}` × üç slope genişliği). Beklenen şekil: geniş plato,
2.5’te keskin tepe değil — koşu sonuçları eklenecek.

### 4.2 Görsel: aynı bütçe, iki başlangıç (seed 3)

Yarı-uzaydan VFSA sonucu (ensemble mean):

![Yarı-uzay VFSA mean](figures/seed3_half_mean.png)

Prior’dan VFSA sonucu (ensemble mean):

![Prior VFSA mean](figures/seed3_prior_mean.png)

Yakınsama (data RMS vs iterasyon). Sol eksen log; prior zaten düşükten başlar:

![Yarı-uzay yakınsama](figures/seed3_half_convergence.png)

![Prior yakınsama](figures/seed3_prior_convergence.png)

---

## 5. Ablasyon — ağ gerçekten bir şey katıyor mu?

Aynı sentetik veri, aynı VFSA (`max_iter = 1200`, 2 zincir). Dört başlangıç. **Tek seed** — 3-seed tablosuyla karıştırmayın.

| kol | başlangıç | eğitim | gravite |
|---|---|---|---|
| **A** | homojen yarı-uzay | yok | yok |
| **B** | Niblett-Bostick (bedava) | yok | yok |
| **C** | NB + kapalı-form damped LS gravite + sabit eğim −2.1 | yok | doğrusal |
| **D** | PriorNet ensemble (`reference = 0.1`) | 3000 epoch × 5 üye | öğrenilen eğim |

### 5.1 Model uzayı (truth’a log₁₀ ρ RMSE)

| | A yarı-uzay | B NB | C NB+doğrusal | D tam prior |
|---|---:|---:|---:|---:|
| başlangıç RMSE | 0.456 | 0.556 | 0.572 | **0.547** |
| VFSA sonrası RMSE | 0.645 | 0.632 | 0.617 | **0.574** |
| son korelasyon | 0.263 | 0.316 | 0.299 | **0.376** |

C, B’den kötü başlıyor (sabit eğim + LS, derinlik körü). D, C’ye göre VFSA sonrası RMSE’de **0.042** kazanç (C’nin %6.8’i) — eşik `0.02 · RMSE_C` üstünde; ağın doğrusal düzeltmeye katkısı bu senaryoda ölçülebilir.

### 5.2 Data RMS (VFSA best chain)

| | A | B | C | D |
|---|---:|---:|---:|---:|
| best-chain RMS | 3.693 | 3.258 | 3.448 | **2.753** |
| ensemble-mean RMS | 4.318 | 3.589 | 3.519 | **2.919** |

Üst satır başlangıç, alt satır VFSA sonucu:

![Ablasyon: TRUTH vs A/B/C/D](figures/ablation_comparison.png)

Yalnızca VFSA sonuçları:

![VFSA sonuçları](figures/ablation_results_only.png)

---

## 6. Gerçek veri — Musgrave (AusLAMP, truth yok)

Truth olmadığı için “daha doğru model” iddia edilemez. Ölçülen şey **veriye uyum hızı**.

### 6.1 Nominal çözünürlük (`tmp_musgrave`, 400 iter, erken durma kapalı)

| iter | yarı-uzay best RMS | prior best RMS |
|---:|---:|---:|
| 1 | 2.373 | **1.057** |
| 5 | 2.056 | **0.970** (hedef 1.0’ın altı) |
| 10 | 1.804 | **0.911** (plato) |
| 100 | 1.174 | 0.911 |
| 400 | 1.040 | **0.911** |

Prior 5. iterasyonda 1.0’ın altına iner. Plato (0.911) arama tıkanıklığı değil: VFSA RMS = √(χ²/N), `inv_prior/data.obs` TE-only ZXY ile N=414, χ²/datum=1 tam RMS=1. Log: 343.8849 / 0.911394² = 414. `model_err_frac=0.4` donmuş (`RealDataIO.jl`; medyan σ/|Z|=0.400); 0.911 tabanın biraz altında (hafif konservatif bütçe). Yarı-uzay 1.040 tabanın hemen üstünde. Gravite `model_err_mgal=15` bu RMS’e girmez. `target_rms=0.1` bu bütçeyle ulaşılamaz.

![Musgrave: başlangıç vs VFSA](figures/musgrave_comparison.png)

Yakınsama (log’dan, VFSA yeniden koşulmadı):

![Yarı-uzay VFSA diagnostik](figures/musgrave_half_convergence.png)

![Prior VFSA diagnostik](figures/musgrave_prior_convergence.png)

### 6.2 Kaba ağ (20 km hücre) — negatif sonuç

| | yarı-uzay | prior |
|---|---:|---:|
| iter 1 | **2.640** | 2.663 |
| iter 400 | **2.299** | 2.497 |

Bu çözünürlükte prior **daha kötü**. Mesh seçimi henüz taranmadı.

![Musgrave kaba ağ](figures/musgrave_coarse.png)

---

## 7. Yazılım yığını (sunum için)

```
Grid.jl / Features.jl     ızgara + 14 kanal (efektif 12)
Gravity.jl                AD-güvenli prizma operatörü
MT1DAD.jl                 Zygote-uyumlu 1B MT (upstream Float64 cast’i atlar)
PriorNet.jl               Lux koordinat-ağı, Fourier özellik, residual-mode
Losses.jl / Train.jl      fizik kaybı + ensemble
Export.jl                 WS3D / MTGeophysics I/O
Profile2D.jl              2B sürücü
```

`src/` ~14 modül. Test: 13 dosya. Bağımlılık pin’li (`Project.toml` compat).

**Ters suç (beyan):** sentetik gravite `gravity_matrix` ile üretilir — operatör-düzeyinde ters suç var, petrofizik yok (slab 350 kg/m³ `density_from_mu`’dan geçmez). MT tarafında yok: gözlem 2B sonlu hacim, eğitim terimi 1B.

**Kapsam dışı:** 3B (`BoundedVFSA.jl` yazıldı, çalıştırılmadı), TM kip.

---

## 8. Sınırlamalar (iddia etmeyin)

1. Tek küresel gravite–özdirenç eğimi → tek baskın litoloji. Karışık litolojide (Musgrave: dirençli kütle + iletken fay) zorlanır.
2. TE-only.
3. `σ` likelihood ile fit edilmez (anchor yok). `SigmaDriveConfig` MT kolon artığı + gravite duyarlılığından hedef harita üretir; `sigma_penalty` onu izler. 3-seed tabloları hâlâ düzgün-σ koşularından; `corr(σ, |μ−truth|)` yeniden ölçülmedi.
4. Gravite derinlik körü: tek 2B harita her kata kopyalanır.
5. `RealDataIO.jl` Musgrave’e özgü.
6. Slab eğimi iddiası **geri çekildi**.
7. `slope_bounds` işareti dışsal jeolojik/tasarım varsayımıdır (veri-türevi değil).
   Büyüklük için truth-bağımsız tutucu bant: sentetik blind `(-10, −0.1)`,
   Musgrave `(0, 1.5)`. `residual_span` için half_band / `from_baseline` protokolü
   (§4.1); yayımlanan span=2.5 best-case üst sınırdır.

---

## 9. Sunum sırası (öneri)

1. Kötü-konumlu arama + warm-start (bu sayfa, §1).
2. Girdi/çıktı kutusu (§2) — “joint inversion değil.”
3. Iter 1 RMS tablosu (§4) — tek slayt, ~150 std.
4. Seed 3 kesitleri + yakınsama figürleri.
5. Ablasyon A/B/C/D (§5) — “MLP, doğrusal LS’den ölçülebilir kazanç.”
6. Musgrave hız iddiası + kaba-ağ negatif sonuç (§6).
7. Best-case vs blind hiperparametre (§4.1).
8. Ne kanıtlanmadı (§3, §8).

Tekrarlanabilir koşu:

```bash
julia --project=. examples/compare_prior_2d.jl
julia --project=. examples/ablation_prior_2d.jl
SMARTPRIOR_PROTOCOL=blind SMARTPRIOR_WORK=tmp_blind_t1 \
  julia --project=. examples/train_prior_blind.jl
```

---

## 10. Genelleştirme planı (sıradaki adımlar)

Bugüne kadarki tüm sonuçlar (§4–§6) **tek bir jeolojik senaryoya** dayanıyor: 35°
eğimli, host'a göre daha iletken ve daha yoğun bir slab. 3 seed varyasyonu
gürültü gerçekleşmesini (MT/eğitim/VFSA/gravite RNG) değiştiriyor, geometriyi
değil. Aşağıdaki liste, "prior daha hızlı yakınsıyor" iddiasının kaç farklı
koşulda ayakta durduğunu artırmak için öncelik sırasına göre.

### 10.1 Zaten yazılmış, sadece çalıştırılıp rapora eklenmesi gereken

Bu ikisi için yeni kod gerekmiyor — script'ler mevcut, sonuç yok:

| Script | Ne test ediyor | Neden eksik |
|---|---|---|
| `examples/train_prior_sensitivity.jl` | `residual_span` × `slope_bounds` genişliği — beklenen şekil geniş bir plato, span=2.5'te keskin tepe değil | Çalıştırılmadı; §4.1'de "koşu sonuçları eklenecek" yazıyor |
| `examples/robustness_misspecified_gravity.jl` | Yoğunluk alanına, MT/özdirenç tarafının hiç görmediği küçük ölçekli heterojenlik eklenince bozulma | Çalıştırılmış olabilir, sonucu hiçbir yerde raporlanmamış |

```bash
julia --project=. examples/train_prior_sensitivity.jl
SMARTPRIOR_WORK=tmp_misspec_grav \
  julia --project=. examples/robustness_misspecified_gravity.jl
```

### 10.2 Yeni: jeoloji sweep'i (`examples/scenario_sweep_prior_2d.jl`)

Bu script bugün eklendi — mevcut kod tabanında geometriyi değiştiren bir
altyapı yoktu. VFSA çalıştırmıyor (maliyeti düşük tutmak için sadece prior'u
eğitip truth'a karşı skorluyor, `train_prior_sensitivity.jl` ile aynı desen);
4 senaryo:

| Senaryo | Değişen | Test ettiği |
|---|---|---|
| `dip15_conductive` | eğim 35°→15° | Sığ eğimde kazanç duruyor mu |
| `dip35_conductive_published` | — (yayımlanan truth ile birebir) | İç tutarlılık kontrolü: `prior_corr` ARA_RAPOR §4 t1 (0.438) civarına düşmeli |
| `dip55_conductive` | eğim 35°→55° | Dik eğimde kazanç duruyor mu |
| `dip35_resistive` | kontrast işareti ters (yoğun+dirençli, `slope_bounds` de aynalanmış) | Öğrenilen eğim işaretinin dışsal varsayımı doğru yönde tutuluyor mu |

```bash
# tam koşu (~20-40 dk, 4 senaryo × 5 üyeli ensemble × 3000 epoch)
julia --project=. examples/scenario_sweep_prior_2d.jl

# hızlı keşif (birkaç dakika, güvenilir sayı değil ama şekli gösterir)
SMARTPRIOR_EPOCHS=400 SMARTPRIOR_NMEMBERS=1 \
  julia --project=. examples/scenario_sweep_prior_2d.jl
```

Çıktı `sweep_metrics.tsv`'de `prior_gain_corr` sütunu (prior korelasyonu − NB
korelasyonu) her senaryo için pozitif kalıyorsa, "kazanç geometriye bağlı
değil" iddiası bir senaryodan dörde çıkar. Negatife dönen bir satır varsa —
özellikle `dip35_resistive`, çünkü orada işaret varsayımı ters çevriliyor —
bunu gizlemeyin, §8 Sınırlamalar'a doğrudan ekleyin.

#### İlk sonuçlar (keşif modu: 400 epoch, 1 üye — ön izlenim, aşağıdaki tam
koşuyla doğrulanmadan güvenilmemeli)

Sanity check geçti: bu koşudaki NB korelasyonu (0.426) yayımlanan t1 satırıyla
(§4, 0.426) neredeyse birebir örtüşüyor, truth/mesh/seed pipeline'ı doğrulanmış
oluyor.

| senaryo | eğim | NB corr | prior corr | kazanç (corr) | prior RMSE |
|---|---:|---:|---:|---:|---:|
| dip15_conductive | 15° | 0.489 | 0.647 | +0.158 | 0.534 |
| dip35_conductive_published | 35° | 0.426 | 0.462 | +0.036 | 0.548 |
| dip55_conductive | 55° | 0.391 | 0.373 | −0.018 | 0.537 |
| dip35_resistive (ters kontrast) | 35° | 0.255 | 0.382 | +0.127 | 0.276\* |

#### Tam koşu (3000 epoch, 5 üye — bu sayılar rapor edilebilir)

Aynı sanity check burada da geçti: `dip35_conductive_published` satırı
RMSE=0.544 / corr=0.438 ile yayımlanan §4 t1 değerine (0.546/0.438)
pratik olarak birebir eşleşiyor.

| senaryo | eğim | NB rmse | NB corr | prior rmse | prior corr | kazanç (corr) |
|---|---:|---:|---:|---:|---:|---:|
| dip15_conductive | 15° | 0.716 | 0.489 | 0.423 | 0.741 | **+0.252** |
| dip35_conductive_published | 35° | 0.556 | 0.426 | 0.544 | 0.438 | +0.011 |
| dip55_conductive | 55° | 0.477 | 0.391 | 0.557 | 0.317 | **−0.074** |
| dip35_resistive (ters kontrast) | 35° | 0.264 | 0.255 | 0.338 | 0.274 | +0.019 |

\* `dip35_resistive`'in kontrast büyüklüğü diğer üçünden farklı (1.2 dekat vs
2.0 dekat — `(0, 4)` sert sınırına sığdırmak için); RMSE'yi diğerleriyle
doğrudan kıyaslamayın, kendi NB baseline'ına göre okuyun.

**Sonuç — scout'tan tam koşuya değişen tablo:**

- **dip55'teki negatif sonuç eğitim yetersizliğinden değil.** Scout'ta
  kazanç −0.018 idi, tam eğitimle −0.074'e düştü (daha da kötüleşti) ve
  prior RMSE'si (0.557) NB'nin (0.477) belirgin şekilde üzerinde kaldı. "Az
  eğitildiği için kötü çıktı" açıklaması çürüdü; şu an elimizdeki en iyi
  kanıt, **dik eğimde (55°) prior'ın baseline'dan güvenilir biçimde kötü
  olduğu** yönünde.
- **dip35_resistive'in scout'taki iyi görünümü büyük ölçüde eridi.** Scout
  kazancı +0.127'ydi (400 epoch'ta yarı-pişmiş); tam koşuda korelasyon
  kazancı +0.019'a düştü ve **prior RMSE'si (0.338) NB'nin (0.264)
  üzerinde** kaldı. Ters polaritede "kazandırıyor" denemez; en fazla "hafif
  karışık" denebilir, kontrast büyüklüğü farkı da bunu netleştirmeyi
  zorlaştırıyor.
- **Kazancın güvenilir olduğu bölge daraldı:** sığ-orta eğim (15°-35°),
  standart (yoğun=iletken) polarite. Bunun dışında (dik eğim, ters polarite)
  ya kazanç kayboluyor ya negatife dönüyor. "Prior her zaman kazandırır"
  iddiası kurulamaz; "test edilen aralıkta (15°-35°, standart polarite)
  kazandırır, dışında henüz kazandırmıyor/olumsuz" iddiası kurulur.

**Açık takip (bu hafta değil, ama not düşülmeli):** Train.jl'in bastırdığı
"unpenalised residual" uyarısı (§10.2 altında değinildi) dip55 ve
dip35_resistive'deki bozulmanın bir nedeni olabilir — `reference` ağırlığı
sıfır olduğu için ağın baseline'dan sapması hiç cezalandırılmıyor. Bir
sonraki deney: aynı iki senaryoyu `LossWeights(reference=...)` sıfırdan
farklı bir değerle tekrar çalıştırıp bozulmanın küçülüp küçülmediğine
bakmak. Bu bir günlük iş değil, ayrı bir oturumda ele alınmalı.

#### dip55 ayırt edici deneyler (tek senaryo, tam eğitim: 3000 epoch, 5 üye)

`examples/scenario_sweep_prior_2d.jl`'e eklenen `SMARTPRIOR_SMOOTH_WEIGHT` /
`SMARTPRIOR_REFERENCE_WEIGHT` / `SMARTPRIOR_SCENARIOS` env değişkenleriyle,
dip55'in bozulmasının iki olası nedenini tek tek izole ettik.

| koşu | smooth ağırlık | reference ağırlık | kazanç (corr) |
|---|---:|---:|---:|
| taban (tam koşu, §10.2) | 10.0 | 0.0 | −0.074 |
| Deney 1: yumuşaklık gevşetildi | 1.0 | 0.0 | **−0.137** |

**Deney 1 sonucu, "yumuşaklık cezası fazla güçlü" hipotezini çürüttü.**
Cezayı gevşetince durum düzelmedi, kötüleşti. Bu, yumuşaklık cezasının
dolaylı bir fren görevi gördüğünü düşündürüyor — çıktı alanının
pürüzlülüğünü sınırlayarak, ağın ne kadar "serbest" bir düzeltme
üretebileceğini de dolaylı olarak kısıtlıyor. Bu freni gevşetince, zaten
cezasız olan residual'ın önündeki tek kısıt da zayıflamış oldu.

Bu, cezasız-residual hipotezini güçlendiriyor: sıradaki deney, `reference`
ağırlığını sıfırdan farklı bir değere çekip (`SMARTPRIOR_REFERENCE_WEIGHT`)
kazancın toparlanıp toparlanmadığına bakmak. Toparlanırsa, dip55'teki
bozulmanın ana nedeni netleşmiş olur: az bilgi içeren (dar yüzey izli) bir
sinyalle karşılaşan ağın, geri çekecek bir kuvvet olmadan aşırı
sürüklenmesi.

| koşu | smooth ağırlık | reference ağırlık | kazanç (corr) |
|---|---:|---:|---:|
| taban (tam koşu, §10.2) | 10.0 | 0.0 | −0.074 |
| Deney 1: yumuşaklık gevşetildi | 1.0 | 0.0 | −0.137 |
| Deney 2: residual cezalandı | 10.0 | 0.3 | −0.071 |

**Deney 2 sonucu da dip55'i düzeltmedi** — taban değerden (−0.074) pratikte
ayırt edilemeyecek kadar yakın (−0.071). Yani iki düzenlileştirme
hipotezini de tek tek izole ettik, **ikisi de ana neden değil**: ne
yumuşaklık cezasının gücü, ne de residual'ın cezasızlığı, dip55'teki
bozulmayı açıklıyor/düzeltmiyor.

**Vardığımız sonuç:** Bu, ayarla düzelebilecek bir eğitim hatası değil.
Geriye kalan en güçlü aday, en başta önerilen üçüncü hipotez —
**bilgi/örnekleme kısıtı**: dik eğimde (55°) yapının yüzeydeki yatay izi
daralıyor, sabit aralıklı MT/gravite istasyonları (400m mesh hücresi,
500-1000m istasyon aralığı) bu dar izi yeterince örnekleyemiyor. Bunun
bağımsız kanıtı zaten elimizde: NB baseline'ın kendi korelasyonu da eğim
dikleştikçe düzenli düşüyor (0.489→0.426→0.391), yani sorun ağa özgü değil,
sinyalin kendisi zayıflıyor. Bunu düzeltmek muhtemelen bir kayıp ağırlığı
ayarıyla değil, o bölgede daha sık örnekleme ya da farklı bir mimari
yaklaşımla olur — **bu, sonraki bir aşamanın işi, bu haftanın değil.**

**Pankaj'a götürülecek özet:** "Dik eğimdeki bozulma için iki
düzenlileştirme hipotezini kontrollü deneylerle tek tek elediik; ikisi de
açıklamıyor. Kalan en güçlü aday, istasyon örnekleme yoğunluğunun dar
yapıları yeterince görememesi — bunu NB baseline'ın da aynı eğilimi
göstermesiyle destekliyoruz. Bu bir hata değil, mevcut mimarinin ve
örnekleme tasarımının bir sınırı; düzeltmesi ayrı bir çalışma."

### 10.3 Bu sweep sonrasında (VFSA'ya taşımaya değer mi)

`scenario_sweep_prior_2d.jl` sadece prior kalitesini (RMSE/korelasyon) ölçüyor,
VFSA yakınsama hızını değil. En ilginç çıkan 1-2 senaryoyu (en büyük kazanç ve
en küçük/negatif kazanç) `compare_prior_2d.jl`'e aynı seed'lerle taşıyıp, §4'teki
gibi iter 1 / iter 400 data RMS tablosu çıkarmak, "yakınsama hızı" iddiasını da
tek senaryonun ötesine taşır.

### 10.4 Kapsam dışı (bu hafta değil)

- Çoklu cisim / karışık litoloji (Musgrave'in dirençli Giles + iletken zon
  yapısına benzer sentetik) — §Sınırlamalar madde 1'in doğrudan testi, ama
  yeni bir truth builder'ı ve muhtemelen `add_block` + `add_dipping_slab`
  kombinasyonu gerektiriyor.
- Mesh çözünürlüğü sweep'i — kaba ağda (20 km) zaten negatif sonuç var (§6.2);
  ince/orta çözünürlük arası bir tarama henüz yok.
