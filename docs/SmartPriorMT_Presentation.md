# SmartPriorMT — Presentation

**Length:** ~12–15 min + questions  
**Language:** simple English  
**Version:** 0.1.0 · Julia 1.10 · Lux.jl + Zygote · MTGeophysics 0.5.0  
**Sources:** `docs/ARA_RAPOR.md`, 3-seed runs, ablation, Musgrave

**One-line pitch:** A better first guess for 2D MT inversion — not a new solver, not joint inversion.

Suggested order is fixed. Do not rearrange under time pressure.

---

## Slide 0 — Title (30 s)

On screen: title only.

> SmartPriorMT is a warm start for 2D MT inversion. Same VFSA solver, same
> settings, same seed — only the starting model changes. Gravity helps build
> that start. Gravity does **not** enter the VFSA misfit.

Do not say “joint inversion.”

---

## Slide 1 — The problem (1 min)

> We want cell-by-cell resistivity under the ground. The search is hard.
> People start VFSA from a flat half-space. First data RMS is about **30**.
> Most of the budget is spent climbing out of a bad start.

**Pipeline (one line on screen):**
```
input → smart first guess (μ, σ) → VFSA starts here → final model
```

**Goal:** same VFSA, same budget, better first guess.

---

## Slide 2 — What I built / I/O (1.5 min) — say this early

On screen: this box first.

| | What it is |
|---|---|
| **Input** | Gravity (mGal) + MT (ρ_a, phase) + optional topography |
| **Output** | Per-cell first guess `μ` (log₁₀ ρ) + confidence `σ` |
| **Unchanged** | VFSA physics, proposals, cooling schedule |
| **No labels** | Network never sees the true earth during training |

> I take gravity + MT. I build a free NB (Niblett–Bostick) baseline from MT.
> A small Lux MLP refines that baseline with physics losses — gravity prism,
> 1D MT column, smoothness. Five-network ensemble, 3000 steps each.
> Features: 14 channels listed, **12 effective** in 2D (`x_norm` and
> `gravity_dx` are zero by construction).

If asked — stack: `Grid` / `Features` → `Gravity` + `MT1DAD` → `PriorNet` /
`Losses` / `Train` → `Export` → VFSA. ~14 modules, 13 tests, pinned deps.

---

## Slide 3 — Strongest result: first-step RMS (2 min)

On screen: one table only.

| | flat start | smart start |
|---|---:|---:|
| data RMS, step 1 | 30.260 ± 0.019 | **4.558 ± 0.080** |
| data RMS, step 400 | 4.995 ± 0.347 | **3.744 ± 0.340** |

3 independent seeds · chain 1 · mean ± sample std

> At step 1, smart start is about **7×** lower misfit. That gap is **not**
> stochastic luck — it is the forward response of two different start models.
> After 400 steps the smart start is still lower; intervals do not overlap.

**This is the claim that stands.** Lead with it; return to it if discussion drifts.

**Caveat (one sentence, if needed):** Data RMS ≠ model RMSE vs truth. VFSA
fits data; truth RMSE can get worse while data fit improves.

---

## Slide 4 — Pictures: seed 3 (1.5 min)

On screen, side-by-side:

| Flat | Smart |
|---|---|
| `docs/figures/seed3_half_mean.png` | `docs/figures/seed3_prior_mean.png` |
| `docs/figures/seed3_half_convergence.png` | `docs/figures/seed3_prior_convergence.png` |

> Same budget, two starts. Convergence: smart line begins much lower.
> Cross-sections: visual support only — the numeric claim is the RMS table.

---

## Slide 5 — Ablation A/B/C/D (2 min)

On screen: method table, then numbers.

| method | start | train? | gravity? |
|---|---|---|---|
| A | flat half-space | no | no |
| B | NB only (free) | no | no |
| C | NB + linear gravity fit | no | yes (fixed slope) |
| D | full PriorNet ensemble | yes | yes (learned) |

Same synthetic, same VFSA (`max_iter=1200`). **One seed** — do not mix with the 3-seed table.

| | A flat | B NB | C NB+line | D full |
|---|---:|---:|---:|---:|
| start RMSE | 0.456 | 0.556 | 0.572 | **0.547** |
| RMSE after VFSA | 0.645 | 0.632 | 0.617 | **0.574** |
| final correlation | 0.263 | 0.316 | 0.299 | **0.376** |

If time: `docs/figures/ablation_comparison.png` or `ablation_results_only.png`

> C is worse than B at the start — a fixed slope is not smart.
> After VFSA, D beats C by **0.042** RMSE (~6.8% of C). That is above our
> noise threshold. The network adds measurable value over a linear gravity fit.

If asked — data RMS (best chain): A 3.693 · B 3.258 · C 3.448 · D **2.753**

---

## Slide 6 — Musgrave real data (2 min)

On screen: `docs/figures/musgrave_comparison.png`  
optional: `musgrave_half_convergence.png` / `musgrave_prior_convergence.png`

**Nominal grid (truth unknown — claim is fit speed only):**

| iter | flat best RMS | smart best RMS |
|---:|---:|---:|
| 1 | 2.373 | **1.057** |
| 5 | 2.056 | **0.970** (below 1.0) |
| 10 | 1.804 | **0.911** (plateau) |
| 400 | 1.040 | **0.911** |

> Smart start is below RMS 1.0 by step 5. Flat never reaches that by step 400
> (ends at 1.040; mine at 0.911).
>
> Important: the 0.911 plateau is **not** a search stuck. With frozen
> `model_err_frac = 0.4`, χ²/datum = 1 means RMS = 1. We sit slightly under
> that error floor. Gravity error budget does not enter this RMS.

Do not say “Prior got stuck at 0.911.”

**Negative result — coarse 20 km grid** (`docs/figures/musgrave_coarse.png`):

| | flat | smart |
|---|---:|---:|
| step 1 | **2.640** | 2.663 |
| step 400 | **2.299** | 2.497 |

> At this mesh, smart is **worse**. I have not diagnosed why yet. Say it
> honestly.

---

## Slide 7 — Fair vs tuned training (1 min)

Training only, no VFSA. Same 3 seeds. Best-case (tuned span/slope) vs blind
(truth-independent hyperparameters).

| score | best-case (tuned) | blind (fair) |
|---|---:|---:|
| RMSE | 0.546 ± 0.008 | **0.518 ± 0.003** |
| correlation | 0.443 ± 0.008 | **0.501 ± 0.007** |

> Blind is not worse — it is a bit better. The good result is not an artifact
> of oracle hyperparameter tuning. Sign of the gravity–resistivity slope is
> still an external assumption.

---

## Slide 8 — What I do NOT claim (2 min) — end here on purpose

Say each bullet once; do not defend soft claims.

1. **Slab angle** — retracted. Sign flips across seeds. Will not claim again.
2. **Depth-aware gravity feature** — did not fix the vertical stripe; slightly worse.
3. **Coarse Musgrave mesh** — smart loses (Slide 6). Cause unknown.
4. **`σ`** — not a calibrated uncertainty map; weak link to residual
   (~+0.09 smooth-σ runs; ~+0.27 with sigma_drive on one ablation). Does not
   enter VFSA χ² today.
5. **One global gravity–ρ slope** → one dominant lithology. Hard on mixed geology
   (Musgrave).
6. **TE-only** well calibrated; TM exists in code, not ready.
7. **No Musgrave truth** — faster data fit ≠ proven better earth model.
8. **Synthetic gravity inverse crime (disclosed):** same prism operator as training.
   No petrophysical inverse crime; none on MT (obs is 2D FH; train term is 1D).
9. **3D** written, never run — out of scope.
10. **Prior vs NB on raw RMSE** — prior does **not** clearly beat NB
    (0.546 ± 0.008 vs 0.551 ± 0.004). Gain is correlation (0.443 vs 0.429).

---

## Closing line (15 s)

> Standing claim: same VFSA, ~7× better first misfit, still better at 400
> iterations on synthetic; faster data fit on Musgrave nominal; network beats
> linear gravity in ablation. Everything else I will not oversell.

---

## Live demo (only if asked)

```bash
julia --project=. examples/compare_prior_2d.jl
julia --project=. examples/ablation_prior_2d.jl
SMARTPRIOR_PROTOCOL=blind SMARTPRIOR_WORK=tmp_blind_t1 \
  julia --project=. examples/train_prior_blind.jl
```

Full numbers and figures: `docs/ARA_RAPOR.md`.

---

## Timing cheat sheet

| Min | Slide |
|---:|---|
| 0:00 | Title + not joint inversion |
| 0:30 | Problem |
| 1:30 | I/O / what I built |
| 3:00 | RMS table (strongest claim) |
| 5:00 | Seed-3 figures |
| 6:30 | Ablation |
| 8:30 | Musgrave + coarse failure |
| 10:30 | Blind vs best-case |
| 11:30 | What I do not claim |
| 13:30 | Stop · questions |
