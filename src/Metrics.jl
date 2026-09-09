# Scoring a prior against a known truth.
#
# Two numbers decide whether a smart prior is worth using, and they pull against
# each other:
#
#   coverage  -- does the search interval actually contain the truth
#   volume    -- how much smaller is that interval than the scalar one
#
# Either alone is trivial to win. A prior with bounds spanning the full physical
# range has perfect coverage and buys nothing; a prior with bounds a hundredth of
# a decade wide reduces the volume enormously and excludes the answer, which is
# worse than the half-space it replaced because the inversion can no longer
# recover from it. Any claim about a prior has to report both.
#
# Calibration is the diagnostic that explains a bad pair. It asks whether sigma
# means what it says, by looking at the standardised residual z = (truth - mu)/
# sigma. A well-calibrated prior has |z| below 1 in about 68 per cent of cells and
# an rms z near 1. An rms z above 1 is overconfidence and shows up as lost
# coverage; below 1 is timidity and shows up as an unreduced volume. Which of the
# two is happening tells you whether to reweight the sigma penalty or add
# constraints, so it is worth computing before touching either.

# only cells where every quantity is defined; air is NaN throughout the pipeline
function _live_mask(arrays...)
    mask = trues(size(first(arrays)))
    for a in arrays
        size(a) == size(mask) || throw(DimensionMismatch(
            "metrics: array sizes differ, $(size(a)) vs $(size(mask))"))
        mask .&= isfinite.(a)
    end
    return mask
end

"""
    coverage(truth, lo, hi) -> Float64

Fraction of cells whose true value lies inside the search interval.

Cells where any of the three is non-finite are skipped, so air does not inflate
the score. This is the number a narrowed search has to protect: whatever the
inversion does afterwards, a cell whose truth is outside its bounds cannot be
recovered.
"""
function coverage(truth::AbstractArray{<:Real},
                  lo::AbstractArray{<:Real},
                  hi::AbstractArray{<:Real})
    mask = _live_mask(truth, lo, hi)
    n = count(mask)
    n > 0 || throw(ArgumentError("coverage: no cells where truth and bounds are all finite"))
    inside = 0
    @inbounds for idx in eachindex(mask)
        mask[idx] || continue
        (lo[idx] <= truth[idx] <= hi[idx]) && (inside += 1)
    end
    return inside / n
end

"""
    volume_reduction(lo, hi, reference) -> Float64

Geometric-mean interval width relative to the scalar interval `reference`.

`reference` is a `(lo, hi)` pair, normally the `cfg.log_bounds` the run would
otherwise have used. A value of 0.2 means the typical cell is searched over a
fifth of the range.

The geometric mean is the right average because the search space is a product
over cells: halving the width in every one of `n` cells shrinks the space by
`2^n`, and only a multiplicative average tracks that. An arithmetic mean would
let a handful of wide cells hide a prior that is tight and wrong everywhere else.
"""
function volume_reduction(lo::AbstractArray{<:Real}, hi::AbstractArray{<:Real},
                          reference::Tuple{Real,Real})
    reference[1] < reference[2] ||
        throw(ArgumentError("volume_reduction: reference must be increasing"))
    mask = _live_mask(lo, hi)
    count(mask) > 0 || throw(ArgumentError("volume_reduction: no cells with finite bounds"))

    w = Float64[]
    @inbounds for idx in eachindex(mask)
        mask[idx] || continue
        width = hi[idx] - lo[idx]
        width > 0 || throw(ArgumentError("volume_reduction: non-positive width at cell $idx"))
        push!(w, width)
    end
    return exp(mean(log.(w))) / (Float64(reference[2]) - Float64(reference[1]))
end

"""
    rmse(a, b) -> Float64

Root-mean-square difference over cells where both are finite.

In log10 resistivity, so 0.3 is roughly a factor of two.
"""
function rmse(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    mask = _live_mask(a, b)
    n = count(mask)
    n > 0 || throw(ArgumentError("rmse: no cells where both arrays are finite"))
    s = 0.0
    @inbounds for idx in eachindex(mask)
        mask[idx] && (s += abs2(a[idx] - b[idx]))
    end
    return sqrt(s / n)
end

"""
    mae(a, b) -> Float64

Mean absolute difference over cells where both are finite. Less sensitive than
[`rmse`](@ref) to a few badly wrong cells, so reporting both separates "wrong
everywhere by a little" from "right except for one blown-up region".
"""
function mae(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    mask = _live_mask(a, b)
    n = count(mask)
    n > 0 || throw(ArgumentError("mae: no cells where both arrays are finite"))
    s = 0.0
    @inbounds for idx in eachindex(mask)
        mask[idx] && (s += abs(a[idx] - b[idx]))
    end
    return s / n
end

"""
    anomaly_correlation(a, b) -> Float64

Correlation of the two fields after removing each one's own mean.

Complements [`rmse`](@ref) for the question this package is about. A prior can
point at the right structure and still score a poor rmse because it damped the
amplitude -- which is the expected and desirable behaviour of an uncertain model.
That prior has done its job as a starting point; one with the right amplitude in
the wrong place has not, and only the correlation separates them.

Returns `NaN` if either field is constant over the live cells.
"""
function anomaly_correlation(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    mask = _live_mask(a, b)
    count(mask) > 1 || throw(ArgumentError("anomaly_correlation: need at least two live cells"))
    av = [a[idx] for idx in eachindex(mask) if mask[idx]]
    bv = [b[idx] for idx in eachindex(mask) if mask[idx]]
    (std(av) > 0 && std(bv) > 0) || return NaN
    return cor(av, bv)
end

"""
    calibration(truth, mu, sigma) -> NamedTuple

How honest the predicted uncertainty is.

Returns the empirical fractions of cells within one and two sigma (`within1`,
`within2`), the rms standardised residual (`zrms`), the mean signed residual
(`bias`, in decades) and the live cell count.

Read it as: `zrms` near 1 with `within1` near 0.68 and `within2` near 0.95 is
calibrated. `zrms` above 1 means sigma is too small and the bounds will exclude
the truth. Below 1 means sigma is too large and the search was not really
narrowed. A large `bias` alongside a reasonable `zrms` means the field is
systematically offset -- a coupling or reference problem, not an uncertainty one.
"""
function calibration(truth::AbstractArray{<:Real},
                     mu::AbstractArray{<:Real},
                     sigma::AbstractArray{<:Real})
    mask = _live_mask(truth, mu, sigma)
    n = count(mask)
    n > 0 || throw(ArgumentError("calibration: no cells where all three are finite"))

    z = Float64[]
    resid = Float64[]
    @inbounds for idx in eachindex(mask)
        mask[idx] || continue
        sigma[idx] > 0 || throw(ArgumentError("calibration: non-positive sigma at cell $idx"))
        r = truth[idx] - mu[idx]
        push!(resid, r)
        push!(z, r / sigma[idx])
    end

    return (n = n,
            within1 = count(<=(1.0), abs.(z)) / n,
            within2 = count(<=(2.0), abs.(z)) / n,
            zrms = sqrt(mean(abs2, z)),
            bias = mean(resid))
end

"""
    prior_report(truth, b::PriorBundle; reference=nothing) -> NamedTuple

Every metric for one prior against one truth.

`truth` is a log10 resistivity array or a [`SyntheticModel`](@ref). `reference`
is the scalar interval to compare the volume against, defaulting to the bundle's
own `log_rho_bounds`, which is what a run without a smart prior would have used.

Report `coverage` and `volume_ratio` together; see [`print_report`](@ref).
"""
function prior_report(truth::AbstractArray{<:Real}, b::PriorBundle;
                      reference::Union{Nothing,Tuple{Real,Real}} = nothing)
    size(truth) == size(b) || throw(DimensionMismatch(
        "prior_report: truth is $(size(truth)) but the bundle is $(size(b))"))

    lo, hi = prior_bounds(b)
    ref = reference === nothing ? b.log_rho_bounds : reference
    cal = calibration(truth, b.mu, b.sigma)

    return (rmse = rmse(truth, b.mu),
            mae = mae(truth, b.mu),
            correlation = anomaly_correlation(truth, b.mu),
            coverage = coverage(truth, lo, hi),
            volume_ratio = volume_reduction(lo, hi, ref),
            k = b.k,
            reference = (Float64(ref[1]), Float64(ref[2])),
            calibration = cal)
end

prior_report(truth::SyntheticModel, b::PriorBundle; kwargs...) =
    prior_report(truth.log_rho, b; kwargs...)

"""
    print_report(io=stdout, r; label="prior")

Print a [`prior_report`](@ref) as a short block.
"""
function print_report(io::IO, r; label::AbstractString = "prior")
    println(io, "--- ", label, " ---")
    @printf(io, "  rmse            %.3f decades\n", r.rmse)
    @printf(io, "  mae             %.3f decades\n", r.mae)
    @printf(io, "  correlation     %.3f\n", r.correlation)
    @printf(io, "  coverage        %.3f  (truth inside +-%.1f sigma)\n", r.coverage, r.k)
    @printf(io, "  volume ratio    %.3f  (vs %.1f-%.1f log10 ohm-m)\n",
            r.volume_ratio, r.reference[1], r.reference[2])
    c = r.calibration
    @printf(io, "  calibration     zrms %.2f, within1 %.2f, within2 %.2f, bias %+.3f\n",
            c.zrms, c.within1, c.within2, c.bias)
    return nothing
end

print_report(r; kwargs...) = print_report(stdout, r; kwargs...)

"""
    compare_starts(truth, baseline, result) -> NamedTuple

Score an inversion result against the truth and against where it started.

`baseline` is the starting model, `result` what the inversion returned. Reports
both rmse values and `improvement`, the fractional reduction: 0.3 means the
inversion closed 30 per cent of the gap it began with. Negative means it moved
away from the truth, which a poor starting model can cause and is exactly the
failure a smart prior is meant to prevent.
"""
function compare_starts(truth::AbstractArray{<:Real},
                        baseline::AbstractArray{<:Real},
                        result::AbstractArray{<:Real})
    r0 = rmse(truth, baseline)
    r1 = rmse(truth, result)
    return (rmse_start = r0,
            rmse_final = r1,
            improvement = r0 > 0 ? (r0 - r1) / r0 : NaN,
            correlation_start = anomaly_correlation(truth, baseline),
            correlation_final = anomaly_correlation(truth, result))
end
