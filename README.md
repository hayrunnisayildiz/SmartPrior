# SmartPriorMT

**Tek cümle:** 2B MT (manyetotellürik) ters çözümüne düz bir yarı-uzayla değil,
gravite + MT verisinden üretilmiş "akıllı" bir başlangıç modeliyle (prior)
başlıyoruz; bunun inversiyonu hızlandırıp hızlandırmadığını test ediyoruz.

Sürüm 0.1.0 · Julia 1.10 · MTGeophysics 0.5.0
Tüm ayrıntılı tablolar, ablasyonlar ve figürler: [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md)

---

## Özet (TL;DR)

| Soru | Cevap |
|---|---|
| Prior, veriye daha hızlı uyuyor mu? | **Evet** — 3/3 seed'de, başlangıçtan itibaren tutarlı şekilde |
| Prior, gerçek modelle daha çok örtüşüyor mu? | **Kısmen** — yapısal korelasyon iyileşiyor, mutlak genlik (RMSE) belirsiz |
| Her jeolojik senaryoda çalışıyor mu? | **Hayır** — dik eğimli / dirençli-kontrastlı yapılarda geride kalabiliyor |
| VFSA arama hedefine (RMS=1.0) ulaşıyor mu? | **Hayır** — bütçe artışı (400→3000) kazancın yarısından azını getirdi; darboğaz artık `max_iter` değil, RBF parametrizasyonu |

Bu proje bir "joint inversion" değildir. Gravite yalnızca prior üretiminde
kullanılır; VFSA'nın kendi arama sürecine (χ²) girmez.

---

## Proje Nedir?

```
Gravite + MT verisi (Niblett–Bostick) → Lux ağı → Prior model (μ, σ)
                                                  → VFSA2DMT (başlangıç modeli olarak)
```

Amaç: prior'ı başlangıç noktası verip VFSA aramasının (a) veriye daha hızlı
uyup uymadığını, (b) sonucun gerçek modele daha çok benzeyip benzemediğini
görmek. Fiziksel çözücü, pertürbasyon ve soğuma çizelgesi değişmiyor — tek
değişen başlangıç noktası.

---

## Ana Bulgular

### 1) Prior, veriye daha hızlı uyuyor

3 farklı seed'de, VFSA'nın 400 iterasyon sonundaki veri uyumu (data RMS):

| Seed | Yarı-uzay | Prior | İyileşme |
|---|---:|---:|---:|
| t1 | 4.60 | 3.47 | %25 |
| t2 | 5.02 | 3.64 | %27 |
| t3 | 5.15 | 3.82 | %26 |

Bu fark büyük ölçüde başlangıç noktasından geliyor: prior, ilk iterasyondan
itibaren yarı-uzaydan çok daha az hata veriyor (~4.6 vs ~30).

**Örnek (t1): yakınsama ve veri uyumu, yarı-uzay vs prior**

| Yarı-uzay | Prior |
|---|---|
| ![t1 half convergence](docs/assets/t1_convergence_half.png) | ![t1 prior convergence](docs/assets/t1_convergence_prior.png) |
| ![t1 half data fit](docs/assets/t1_data_fit_half.png) | ![t1 prior data fit](docs/assets/t1_data_fit_prior.png) |

### 2) Gerçek modelle örtüşme: yapı evet, genlik belirsiz

| Ölçüt | Sonuç |
|---|---|
| Korelasyon (gerçek modelle) | Prior 3/3 seed'de daha yüksek (örn. 0.30 vs 0.21) |
| RMSE (gerçek modelle) | Tutarsız — bazı seed'lerde prior daha iyi, bazılarında değil |

Yorum: prior, yapıyı (nerede iletken/dirençli) daha doğru yakalıyor ama
mutlak direnç değerlerinde garanti bir kazanç yok.

**Örnek (t1): inversiyon sonrası ortalama model, gerçek modelle karşılaştırma**

| Yarı-uzay | Prior |
|---|---|
| ![t1 half model mean](docs/assets/t1_model_mean_half.png) | ![t1 prior model mean](docs/assets/t1_model_mean_prior.png) |

### 3) Her jeolojide işe yaramıyor

4 farklı sentetik jeoloji senaryosunda (VFSA'sız, doğrudan prior vs NB
karşılaştırması): sığ/orta eğimli iletken yapılarda prior belirgin şekilde
iyi, dik eğimli veya dirençli-kontrastlı yapılarda geride kalıyor.

**Tam tablolar, ablasyon sonuçları, sigma/belirsizlik analizi ve Musgrave
(gerçek veri) sonuçları için:** [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md)

---

## Sonuç: VFSA Bütçesi Değil, Parametrizasyon Darboğaz

Yukarıdaki sonuçlar `max_iter=400` ile alındı. Hedef veri uyumu
(`target_rms=1.0`) — mevcut sonuçlar bunun üzerinde kalıyor. Bunu çözmek için
bütçeyi kademeli artırıp test ettik: 400 → 800 → 3000 iterasyon.

**t1'de bütçe artışının etkisi (en iyi zincir, final RMS):**

| bütçe | prior | yarı-uzay |
|---|---:|---:|
| 400 | 3.47 | 4.60 |
| 800 | 2.90 | 3.87 |
| 3000 | **2.64** | **3.50** |

**Sonuç netleşti:**

- **Eski 400-iter teşhisi doğru çıktı** — o "plato" değil, kısa bütçeydi.
- **Ama 3000 de yetmiyor.** 800→3000 arası 5.5× daha fazla iterasyon
  harcandı, kazancın yarısından azı geldi. Son 200 iterasyonda eğim
  ~−0.0003/iter, kabul oranı %4–9 — soğuk uçta arama neredeyse durmuş.
  Kalan fark (~1.6 RMS) artık iterasyonla kapanmıyor.
- **Sebep muhtemelen RBF parametrizasyonu.** 250 kontrol noktası ve
  ~800–1000 m çekirdek genişliği, %5 gürültülü 2B veriyi RMS=1'e kadar temsil
  etmeye yetmiyor olabilir. `step_scale` darboğaz değildi.
- **Prior'a "kilitlenme" değil.** Prior, veriye yarı-uzaydan tutarlı şekilde
  daha iyi oturuyor (2.64 vs 3.50) — bu bir arama artefaktı değil, gerçek bir
  başlangıç avantajı. Yarı-uzay bütçe arttıkça yaklaşıyor ama geçmiyor.
- **Truth RMSE ayrı bir konu olarak kalıyor.** VFSA, gürültülü veriye uyuyor;
  data RMS düşmesi otomatik olarak gerçek modele yaklaşmak anlamına gelmiyor
  (bkz. "Gerçek modelle örtüşme" bulgusu).

**Pratik sonuç:** yukarıdaki 400-iter tablo start-bağımlı, tam yakınsamamış
bir aramadan geliyor — ama bu, prior'ın veri uyumunu hızlandırdığı bulgusunu
geçersiz kılmıyor. RMS=1 hedefleniyorsa sıradaki kaldıraç `max_iter` değil;
`n_ctrl` artırımı, daha dar RBF çekirdeği veya farklı bir parametrizasyon.

---

## Sonraki Adımlar

1. RBF parametrizasyonunu ayarla: `n_ctrl` artırımı ve/veya daha dar çekirdek
   (`rbf_sigma_scale` küçültme) — t1'de tek seed sonda
2. Sonda RMS=1 hedefine yaklaşırsa 3-seed doğrulamasını yeni parametrizasyonla
   tekrar çalıştır
3. `max_iter` artık kaldıraç değil — bütçe 400'de sabitlenebilir, kazanılan
   zaman parametrizasyon aramasına aktarılabilir

---

## Kurulum ve Hızlı Başlangıç

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

```bash
julia --project=. examples/compare_prior_2d.jl
```

Sentetik eğimli iletken slab üzerinde yarı-uzay vs prior karşılaştırması
çalıştırır (aynı seed, aynı VFSA ayarları). Diğer örnek komutlar (ablasyon,
blind protokol, jeoloji sweep'i) için [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md).

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

---

## Bilinen Sınırlamalar (kısa)

- **Tek litoloji varsayımı:** tek bir gravite–özdirenç eğimi kullanılıyor;
  karışık litolojili sahalarda (örn. Musgrave) güvenilirlik düşer.
- **TE-only:** TM modu var ama istasyon-bazlı hata kalibrasyonu henüz yok.
- **`σ` kalibre bir belirsizlik haritası değil** — inversiyona da girmiyor.
- **Kısmi ters suç:** gravite operatöründe var (aynı ileri model hem sentetik
  veri hem inversiyon için), petrofizik ve MT tarafında yok.
- **3B warm-start kapsam dışı** — upstream (`MTGeophysics`) hücre-bazlı sınır
  desteklemiyor.
- **VFSA bütçesi henüz hedefe ulaşmıyor** (yukarıda detaylı).

Tam liste ve gerekçeler: [`docs/ARA_RAPOR.md`](docs/ARA_RAPOR.md)

---

## Lisans

MIT, bkz. [LICENSE](LICENSE).