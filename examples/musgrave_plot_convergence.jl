# VFSA diagnostics for the Musgrave run, from on-disk chain logs.
#
# VFSA2DMT's Makie wrapper crashes on TE-only NaN TM, so musgrave_vfsa_compare.jl
# called run_mt2d_vfsa and never wrote plot_convergence.png. This script reads
# chain_0{1,2}/0vfsa2DMT.log and redraws the same four panels MTGeophysics uses
# (Best χ², Current RMS, Temperature, rolling acceptance of width 15). It does
# not run VFSA.
#
# Run:  julia --project=. examples/musgrave_plot_convergence.jl
#
# Optional env: SMARTPRIOR_WORK (default tmp_musgrave next to this repo).
# Copies the PNGs to docs/figures/ as well.

using Printf
using Statistics
using Plots
gr()

const ROOT = dirname(@__DIR__)
const WORK = get(ENV, "SMARTPRIOR_WORK", joinpath(ROOT, "tmp_musgrave"))
const DOCS = joinpath(ROOT, "docs", "figures")

struct ChainTrace
    chain_id::Int
    iter::Vector{Int}
    temp::Vector{Float64}
    chi2best::Vector{Float64}
    rmscurr::Vector{Float64}
    accepted::Vector{Float64}
end

function parse_vfsa_log(path::AbstractString)
    isfile(path) || error("missing $path")
    iter = Int[]
    temp = Float64[]
    chi2best = Float64[]
    rmscurr = Float64[]
    accepted = Float64[]
    chain_id = 0
    for line in eachline(path)
        startswith(line, "#") && continue
        startswith(line, "-") && continue
        startswith(strip(line), "Chain") && continue
        isempty(strip(line)) && continue
        parts = split(line)
        length(parts) >= 10 || continue
        cid = parse(Int, parts[1])
        chain_id == 0 && (chain_id = cid)
        push!(iter, parse(Int, parts[2]))
        push!(temp, parse(Float64, parts[3]))
        push!(rmscurr, parse(Float64, parts[6]))
        push!(chi2best, parse(Float64, parts[8]))
        push!(accepted, parse(Float64, parts[10]))
    end
    isempty(iter) && error("no iterations in $path")
    return ChainTrace(chain_id, iter, temp, chi2best, rmscurr, accepted)
end

rolling_accept(accepted) = [mean(accepted[max(1, i - 14):i]) for i in eachindex(accepted)]

function load_side(side::AbstractString)
    traces = ChainTrace[]
    for k in (1, 2)
        path = joinpath(WORK, side, "chain_0$k", "0vfsa2DMT.log")
        push!(traces, parse_vfsa_log(path))
    end
    return traces
end

function plot_convergence(traces::Vector{ChainTrace}; title)
    p_chi = plot(; xlabel = "VFSA iteration", ylabel = "Best χ²",
                 yscale = :log10, title = "Best objective", legend = :topright)
    p_rms = plot(; xlabel = "VFSA iteration", ylabel = "Current RMS",
                 title = "Current fit", legend = false)
    p_tmp = plot(; xlabel = "VFSA iteration", ylabel = "Temperature",
                 yscale = :log10, title = "Annealing schedule", legend = false)
    p_acc = plot(; xlabel = "VFSA iteration", ylabel = "Rolling acceptance",
                 title = "Acceptance", legend = false, ylims = (0, 1.05))

    offset = 0
    for tr in traces
        x = offset .+ tr.iter
        label = "Chain $(tr.chain_id)"
        plot!(p_chi, x, tr.chi2best; lw = 3, label = label)
        plot!(p_rms, x, tr.rmscurr; lw = 3, label = label)
        plot!(p_tmp, x, tr.temp; lw = 3, label = label)
        plot!(p_acc, x, rolling_accept(tr.accepted); lw = 3, label = label)
        offset += length(tr.iter)
    end

    return plot(p_chi, p_rms, p_tmp, p_acc;
                layout = (2, 2),
                size = (1400, 950),
                left_margin = 6Plots.mm,
                bottom_margin = 6Plots.mm,
                plot_title = title)
end

function write_side(side::AbstractString, stem::AbstractString, title::AbstractString)
    traces = load_side(side)
    plt = plot_convergence(traces; title = title)
    mkpath(joinpath(WORK, side))
    mkpath(DOCS)
    local_path = joinpath(WORK, side, "plot_convergence.png")
    docs_path = joinpath(DOCS, stem)
    savefig(plt, local_path)
    cp(local_path, docs_path; force = true)
    @info "wrote" local_path docs_path bytes = filesize(docs_path)
    for tr in traces
        @printf("  %s chain %d  n=%d  RMSCurr start %.4f  Chi2Best end %.4f\n",
                side, tr.chain_id, length(tr.iter), first(tr.rmscurr), last(tr.chi2best))
    end
end

write_side("inv_half", "musgrave_half_convergence.png",
           "Musgrave VFSA — half-space start")
write_side("inv_prior", "musgrave_prior_convergence.png",
           "Musgrave VFSA — smart prior start")
