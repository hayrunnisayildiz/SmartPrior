# Training / architecture figures for Keivitsa feasibility runs.
#
# Run:  julia --project=. examples/figures_training.jl
# Out:  docs/figures/{network_architecture,loss_curves,val_rmse_curves,nn_diagnosis}.png
#
# Reads numbers from TSVs only. Architecture labels are verified against
# examples/feasibility_loho.jl constants (FOURIER_BANDS/MLP_*/NN_SEEDS).

using DelimitedFiles
using Statistics
using Printf
using GLMakie

const MAKIE_BACKEND = :GLMakie

const ROOT = dirname(@__DIR__)
const FIGDIR = joinpath(ROOT, "docs", "figures")
const LOSS_TSV = joinpath(ROOT, "tmp_feasibility_keivitsa_run2", "loss_curves.tsv")
const DIAG_TSV = joinpath(ROOT, "tmp_keivitsa_diag", "diagnosis_summary.tsv")
const LOHO = joinpath(ROOT, "examples", "feasibility_loho.jl")

const PX_PER_UNIT = 150 / 72   # 150 dpi

const COL_KRIG = RGBf(31 / 255, 119 / 255, 180 / 255)
const COL_XYZ = RGBf(214 / 255, 39 / 255, 40 / 255)
const COL_COV = RGBf(255 / 255, 127 / 255, 14 / 255)
const COL_GREY = RGBf(127 / 255, 127 / 255, 127 / 255)
const COL_XYZ_FAINT = (COL_XYZ, 0.18)
const COL_COV_FAINT = (COL_COV, 0.18)
const COL_GREY_FAINT = (COL_GREY, 0.22)

#---------- architecture constants from feasibility_loho.jl ----------

function read_loho_const(name::AbstractString)
    for line in eachline(LOHO)
        m = match(Regex("^const\\s+" * name * "\\s*=\\s*(.+)"), line)
        m === nothing && continue
        return strip(m.captures[1])
    end
    error("constant $name not found in $LOHO")
end

function architecture_spec()
    bands = parse(Int, read_loho_const("FOURIER_BANDS"))
    smin = parse(Float64, read_loho_const("FOURIER_SCALE_MIN"))
    smax = parse(Float64, read_loho_const("FOURIER_SCALE_MAX"))
    width = parse(Int, read_loho_const("MLP_WIDTH"))
    depth = parse(Int, read_loho_const("MLP_DEPTH"))
    seeds_txt = read_loho_const("NN_SEEDS")
    n_ens = count(c -> c == ',', seeds_txt) + 1
    n_fourier = 2 * 3 * bands   # sin+cos × 3 axes × bands
    return (; bands, smin, smax, width, depth, n_ens, n_fourier,
            nin_xyz = 3 + n_fourier)
end

#---------- loss-curve IO ----------

struct LossRow
    fold::Int
    seed::Int
    method::String
    step::Float64
    train_loss::Float64
    val_loss::Float64
    val_rmse::Float64
end

function load_loss_curves(path::AbstractString)
    raw, hdr = readdlm(path, '\t', header = true)
    names = String.(vec(hdr))
    need = ["fold", "seed", "method", "step", "train_loss", "val_loss", "val_rmse"]
    idx = Dict(n => findfirst(==(n), names) for n in need)
    any(isnothing, values(idx)) && error("unexpected header in $path: $names")
    rows = LossRow[]
    for i in 1:size(raw, 1)
        push!(rows, LossRow(
            Int(raw[i, idx["fold"]]),
            Int(raw[i, idx["seed"]]),
            String(raw[i, idx["method"]]),
            Float64(raw[i, idx["step"]]),
            Float64(raw[i, idx["train_loss"]]),
            Float64(raw[i, idx["val_loss"]]),
            Float64(raw[i, idx["val_rmse"]]),
        ))
    end
    return rows
end

"One training run = one (fold, seed) for a method."
function member_keys(rows::Vector{LossRow}, method::AbstractString)
    ks = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for r in rows
        r.method == method || continue
        k = (r.fold, r.seed)
        if k ∉ seen
            push!(seen, k)
            push!(ks, k)
        end
    end
    return ks
end

function member_series(rows::Vector{LossRow}, method::AbstractString, fold::Int, seed::Int)
    steps = Float64[]; train = Float64[]; vall = Float64[]; rmse = Float64[]
    for r in rows
        (r.method == method && r.fold == fold && r.seed == seed) || continue
        push!(steps, r.step); push!(train, r.train_loss)
        push!(vall, r.val_loss); push!(rmse, r.val_rmse)
    end
    perm = sortperm(steps)
    return steps[perm], train[perm], vall[perm], rmse[perm]
end

"""
Median best step (preferred definition):
  for each (fold, seed) member, take the logged step that minimises validation RMSE;
  return the median of those per-member best steps.
"""
function median_best_step(rows::Vector{LossRow}, method::AbstractString)
    bests = Float64[]
    for (fold, seed) in member_keys(rows, method)
        steps, _, _, rmse = member_series(rows, method, fold, seed)
        isempty(rmse) && continue
        push!(bests, steps[argmin(rmse)])
    end
    isempty(bests) && error("no members for $method")
    return median(bests), bests
end

function lin_interp(xq::Float64, x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n == 0 && return NaN
    xq <= x[1] && return y[1]
    xq >= x[n] && return y[n]
    j = searchsortedlast(x, xq)
    j < 1 && return y[1]
    j >= n && return y[n]
    x0, x1 = x[j], x[j + 1]
    y0, y1 = y[j], y[j + 1]
    x1 == x0 && return y0
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
end

"""Median curve on a common step grid via linear interpolation of each member."""
function median_curve(rows::Vector{LossRow}, method::AbstractString, field::Symbol)
    keys = member_keys(rows, method)
    all_steps = Float64[]
    series = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    for (fold, seed) in keys
        steps, train, vall, rmse = member_series(rows, method, fold, seed)
        y = field === :train_loss ? train :
            field === :val_loss ? vall :
            field === :val_rmse ? rmse :
            error("unknown field $field")
        append!(all_steps, steps)
        push!(series, (steps, y))
    end
    grid = sort!(unique(all_steps))
    med = Vector{Float64}(undef, length(grid))
    for (i, s) in enumerate(grid)
        vals = Float64[]
        for (sx, sy) in series
            s < sx[1] && continue
            s > sx[end] && continue
            push!(vals, lin_interp(s, sx, sy))
        end
        med[i] = isempty(vals) ? NaN : median(vals)
    end
    return grid, med
end

"""
Y-limits so the early loss valley is readable.
Window = max(3 × median best step, 150).
Use [min, q90] of train/val in that window, padded 8%, so late explosions
inside the window do not hide the early minimum.
"""
function loss_clip_limits(rows::Vector{LossRow}, method::AbstractString, med_best::Real)
    window = max(3 * Float64(med_best), 150.0)
    ys = Float64[]
    for r in rows
        r.method == method || continue
        r.step <= window || continue
        push!(ys, r.train_loss)
        push!(ys, r.val_loss)
    end
    isempty(ys) && return (-1.0, 1.0), window
    lo = minimum(ys)
    hi = quantile(ys, 0.90)
    hi < lo && (hi = maximum(ys))
    pad = 0.08 * (hi - lo + 1e-9)
    return (lo - pad, hi + pad), window
end

#---------- diagnosis IO ----------

function load_diagnosis(path::AbstractString)
    raw, hdr = readdlm(path, '\t', header = true)
    names = String.(vec(hdr))
    col(n) = findfirst(==(n), names)
    rows = NamedTuple[]
    for i in 1:size(raw, 1)
        push!(rows, (
            variant = String(raw[i, col("variant")]),
            synthetic_test_skill = Float64(raw[i, col("synthetic_test_skill")]),
            kriging_test_skill = Float64(raw[i, col("kriging_test_skill")]),
            median_best_step = Float64(raw[i, col("median_best_step")]),
        ))
    end
    return rows
end

#---------- save helper ----------

function save_fig(path, fig)
    mkpath(dirname(path))
    save(path, fig; px_per_unit = PX_PER_UNIT)
    println("wrote $path")
end

#---------- figure 1: architecture ----------

function fig_architecture(_backend::Symbol)
    spec = architecture_spec()
    fig = Figure(size = (1000, 420), fontsize = 16,
                backgroundcolor = :white)

    ax = Axis(fig[1, 1];
              title = "Neural-field architecture (Keivitsa feasibility)",
              aspect = DataAspect(),
              limits = (0, 22, 0, 8))
    hidedecorations!(ax)
    hidespines!(ax)

    function box!(x, y, w, h, label; color = :white, tsize = 13)
        poly!(ax, Rect(x, y, w, h);
              color = color, strokecolor = :black, strokewidth = 1.5)
        text!(ax, x + w / 2, y + h / 2; text = label, align = (:center, :center),
              fontsize = tsize)
    end
    function arrow_h!(x0, x1, y)
        lines!(ax, [x0, x1], [y, y]; color = :black, linewidth = 1.5)
        poly!(ax, Point2f[(x1, y), (x1 - 0.25, y + 0.18), (x1 - 0.25, y - 0.18)];
              color = :black)
    end

    box!(0.4, 3.0, 2.2, 2.0, "x, y, z"; color = :gray90)
    arrow_h!(2.7, 3.5, 4.0)

    fourier_lbl = @sprintf("Fourier features\n%d + %d channels\n%d log-spaced\nscales %.0f–%.0f / axis",
                           3, spec.n_fourier, spec.bands, spec.smin, spec.smax)
    box!(3.5, 2.2, 4.0, 3.6, fourier_lbl; color = :gray95, tsize = 12)
    arrow_h!(7.6, 8.4, 4.0)

    mlp_lbl = @sprintf("%d × %d GELU\n(+ linear head → 2)",
                       spec.depth, spec.width)
    box!(8.4, 2.6, 3.6, 2.8, mlp_lbl; color = (:red, 0.12), tsize = 13)
    arrow_h!(12.1, 12.9, 4.0)

    box!(12.9, 3.0, 2.4, 2.0, "(μ, σ)"; color = (COL_KRIG, 0.15))

    box!(16.0, 5.2, 5.4, 1.8,
         @sprintf("Deep ensemble of %d\n(independent seeds)", spec.n_ens);
         color = (COL_COV, 0.15), tsize = 13)
    box!(16.0, 2.8, 5.4, 1.8, "Loss: Gaussian NLL"; color = :gray92, tsize = 13)
    box!(16.0, 0.5, 5.4, 1.8, "Early stopping on\nvalidation RMSE";
         color = :gray92, tsize = 13)

    text!(ax, 0.4, 0.35;
          text = @sprintf("Constants from feasibility_loho.jl: FOURIER_BANDS=%d, scales [%.0f,%.0f], MLP_WIDTH=%d, MLP_DEPTH=%d, |NN_SEEDS|=%d  → nin(xyz)=%d",
                          spec.bands, spec.smin, spec.smax, spec.width, spec.depth,
                          spec.n_ens, spec.nin_xyz),
          align = (:left, :center), fontsize = 10, color = COL_GREY)

    return fig
end

#---------- figure 2 & 3: loss / RMSE panels ----------

function style_for(method::AbstractString)
    if method == "nn_xyz"
        return (bold = COL_XYZ, faint = COL_XYZ_FAINT, title = "nn_xyz")
    elseif method == "nn_cov"
        return (bold = COL_COV, faint = COL_COV_FAINT, title = "nn_cov")
    else
        return (bold = COL_GREY, faint = COL_GREY_FAINT, title = method)
    end
end

function add_member_traces!(ax, rows, method, yfield::Symbol; faint)
    for (fold, seed) in member_keys(rows, method)
        steps, train, vall, rmse = member_series(rows, method, fold, seed)
        y = yfield === :train_loss ? train :
            yfield === :val_loss ? vall : rmse
        lines!(ax, steps, y; color = faint, linewidth = 0.7)
    end
end

function fig_loss_curves(rows::Vector{LossRow})
    fig = Figure(size = (1100, 520), fontsize = 14, backgroundcolor = :white)
    methods = ("nn_xyz", "nn_cov")
    clip_notes = String[]

    for (j, method) in enumerate(methods)
        st = style_for(method)
        med_best, _ = median_best_step(rows, method)
        (ylo, yhi), window = loss_clip_limits(rows, method, med_best)
        push!(clip_notes,
              @sprintf("%s: y ∈ [%.3f, %.3f] (q90 of train/val at steps ≤ %.0f)",
                       method, ylo, yhi, window))

        ax = Axis(fig[1, j];
                  title = @sprintf("%s — train / validation loss", st.title),
                  xlabel = "step",
                  ylabel = j == 1 ? "loss (Gaussian NLL)" : "",
                  limits = (nothing, nothing, ylo, yhi))

        for (fold, seed) in member_keys(rows, method)
            steps, train, vall, _ = member_series(rows, method, fold, seed)
            lines!(ax, steps, train; color = st.faint, linewidth = 0.6)
            lines!(ax, steps, vall; color = st.faint, linewidth = 0.6)
        end

        g_tr, m_tr = median_curve(rows, method, :train_loss)
        g_va, m_va = median_curve(rows, method, :val_loss)
        lines!(ax, g_tr, m_tr; color = st.bold, linewidth = 2.5,
               label = "median train", linestyle = :solid)
        lines!(ax, g_va, m_va; color = st.bold, linewidth = 2.5,
               label = "median validation", linestyle = :dash)
        vlines!(ax, [med_best]; color = :black, linestyle = :dot, linewidth = 1.5,
                label = @sprintf("median best step = %g", med_best))

        axislegend(ax; position = :rt, framevisible = true, labelsize = 11)
        text!(ax, 0.02, 0.02;
              text = @sprintf("y clipped to [%.3f, %.3f]", ylo, yhi),
              space = :relative, align = (:left, :bottom), fontsize = 11,
              color = COL_GREY)
    end

    Label(fig[0, :], "Keivitsa feasibility — ensemble loss curves";
          fontsize = 18, font = :bold, tellwidth = false)
    Label(fig[2, :], join(clip_notes, "   |   ");
          fontsize = 11, color = COL_GREY, tellwidth = false)

    return fig
end

function fig_val_rmse(rows::Vector{LossRow})
    fig = Figure(size = (1100, 480), fontsize = 14, backgroundcolor = :white)
    methods = ("nn_xyz", "nn_cov")

    for (j, method) in enumerate(methods)
        st = style_for(method)
        med_best, _ = median_best_step(rows, method)

        ax = Axis(fig[1, j];
                  title = @sprintf("%s — validation RMSE", st.title),
                  xlabel = "step",
                  ylabel = j == 1 ? "validation RMSE" : "")

        add_member_traces!(ax, rows, method, :val_rmse; faint = st.faint)
        g, m = median_curve(rows, method, :val_rmse)
        lines!(ax, g, m; color = st.bold, linewidth = 2.5, label = "median val RMSE")
        vlines!(ax, [med_best]; color = :black, linestyle = :dot, linewidth = 1.5,
                label = @sprintf("median best step = %g", med_best))
        axislegend(ax; position = :rt, framevisible = true, labelsize = 11)
    end

    Label(fig[0, :], "Keivitsa feasibility — validation RMSE curves";
          fontsize = 18, font = :bold, tellwidth = false)
    Label(fig[2, :], "No y-axis clipping (RMSE range already readable)";
          fontsize = 11, color = COL_GREY, tellwidth = false)

    return fig
end

#---------- figure 4: diagnosis bars ----------

function fig_diagnosis(diag_rows)
    want = ["E0", "E1", "E2", "E3"]
    by = Dict(r.variant => r for r in diag_rows)
    variants = [v for v in want if haskey(by, v)]
    isempty(variants) && error("no E0–E3 rows in diagnosis summary")

    skills = [by[v].synthetic_test_skill for v in variants]
    steps = [by[v].median_best_step for v in variants]
    krig = by[variants[1]].kriging_test_skill

    fig = Figure(size = (900, 480), fontsize = 14, backgroundcolor = :white)
    ax = Axis(fig[1, 1];
              title = "Keivitsa NN diagnosis — synthetic test skill (E0–E3)",
              xlabel = "experiment",
              ylabel = "synthetic test skill",
              xticks = (1:length(variants), variants))

    barplot!(ax, 1:length(variants), skills;
             color = COL_XYZ, strokecolor = :black, strokewidth = 1)

    hlines!(ax, [krig]; color = COL_KRIG, linestyle = :dash, linewidth = 2.5,
            label = @sprintf("kriging (%.3f)", krig))

    yspan = maximum(skills) - minimum(vcat(skills, [krig]))
    pad = max(0.03, 0.08 * abs(yspan))
    for (i, (sk, st)) in enumerate(zip(skills, steps))
        text!(ax, i, sk + pad;
              text = @sprintf("step %g", st),
              align = (:center, :bottom), fontsize = 12, color = :black)
    end

    axislegend(ax; position = :rb, framevisible = true)
    Label(fig[2, 1],
          "Bars: nn_xyz variants (red). Annotations: median_best_step from diagnosis_summary.tsv.";
          fontsize = 11, color = COL_GREY, tellwidth = false)

    return fig
end

#---------- main ----------

function main()
    isfile(LOSS_TSV) || error("missing $LOSS_TSV")
    isfile(DIAG_TSV) || error("missing $DIAG_TSV")
    isfile(LOHO) || error("missing $LOHO")

    println("Makie backend: $MAKIE_BACKEND")

    spec = architecture_spec()
    println(@sprintf("architecture: %d+%d Fourier (bands=%d, scales %.0f–%.0f), %d×%d GELU, ensemble=%d",
                     3, spec.n_fourier, spec.bands, spec.smin, spec.smax,
                     spec.depth, spec.width, spec.n_ens))

    rows = load_loss_curves(LOSS_TSV)
    for method in ("nn_xyz", "nn_cov")
        med, bests = median_best_step(rows, method)
        println(@sprintf("%s median best step = %g  (n=%d members = fold×seed; metric=min val_rmse)",
                         method, med, length(bests)))
    end

    diag = load_diagnosis(DIAG_TSV)
    for r in diag
        println(@sprintf("%s median_best_step=%g synthetic_test_skill=%g krig=%g",
                         r.variant, r.median_best_step, r.synthetic_test_skill,
                         r.kriging_test_skill))
    end

    mkpath(FIGDIR)

    save_fig(joinpath(FIGDIR, "network_architecture.png"), fig_architecture(MAKIE_BACKEND))
    save_fig(joinpath(FIGDIR, "loss_curves.png"), fig_loss_curves(rows))
    save_fig(joinpath(FIGDIR, "val_rmse_curves.png"), fig_val_rmse(rows))
    save_fig(joinpath(FIGDIR, "nn_diagnosis.png"), fig_diagnosis(diag))

    println("done.")
end

main()
