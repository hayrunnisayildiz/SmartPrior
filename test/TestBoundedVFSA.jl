using Test
using SmartPriorMT
import MTGeophysics
using MTGeophysics: build_rbf_map, load_ws3d_model
using Random
using Statistics

# a uniform core with graded padding, so core_ranges detects a core the way it
# would on a real mesh; built through the writer rather than the struct, since
# WS3DModel carries derived fields that are easy to get wrong by hand
function _bvfsa_loaded(; nz = 4, fill_value = 2.0)
    dx = vcat([4000.0, 2000.0], fill(500.0, 6), [2000.0, 4000.0])
    dy = vcat([4000.0, 2000.0], fill(500.0, 5), [2000.0, 4000.0])
    dz = [100.0 * 1.4^(k - 1) for k in 1:nz]
    A = fill(fill_value, length(dx), length(dy), nz)
    path = joinpath(mktempdir(), "m.rho")
    MTGeophysics.write_ws3d_model(path, dx, dy, dz, A, [0.0, 0.0, 0.0]; rotation = 0.0)
    return load_ws3d_model(path)
end

@testset "vfsa_step matches MTGeophysics' own generating function" begin
    for T in (0.5, 1.0, 5.0, 100.0), u in (0.0, 0.1, 0.4999, 0.5, 0.75, 1.0)
        @test SmartPriorMT.vfsa_step(u, T) ≈ MTGeophysics.vfsa_y(u, T)
    end
    # antisymmetric about u = 0.5 and bounded by +-1 in units of the interval
    @test SmartPriorMT.vfsa_step(0.5, 2.0) ≈ 0.0
    @test SmartPriorMT.vfsa_step(0.75, 2.0) ≈ -SmartPriorMT.vfsa_step(0.25, 2.0)
    @test abs(SmartPriorMT.vfsa_step(1.0, 3.0)) ≈ 1.0
end

@testset "BoundedCore validation" begin
    lo = fill(1.0, 3, 3, 2)
    hi = fill(3.0, 3, 3, 2)
    bc = SmartPriorMT.BoundedCore(lo, hi, 1:3, 1:3, 1:2)
    @test size(bc) == (3, 3, 2)

    @test_throws DimensionMismatch SmartPriorMT.BoundedCore(fill(1.0, 2, 2, 2), hi, 1:3, 1:3, 1:2)
    @test_throws DimensionMismatch SmartPriorMT.BoundedCore(lo, fill(3.0, 2, 2, 2), 1:3, 1:3, 1:2)

    # a half-open interval would let the clamp write NaN into the model
    half = copy(lo); half[1, 1, 1] = NaN
    @test_throws ArgumentError SmartPriorMT.BoundedCore(half, hi, 1:3, 1:3, 1:2)

    bad = copy(hi); bad[2, 2, 1] = 0.5
    @test_throws ArgumentError SmartPriorMT.BoundedCore(lo, bad, 1:3, 1:3, 1:2)

    # both bounds non-finite together is fine: that is a frozen cell
    fl = copy(lo); fl[1, 1, 1] = NaN
    fh = copy(hi); fh[1, 1, 1] = NaN
    @test SmartPriorMT.BoundedCore(fl, fh, 1:3, 1:3, 1:2) isa SmartPriorMT.BoundedCore
end

@testset "core_bounds slices to the core" begin
    lo = [Float64(i) for i in 1:6, j in 1:5, k in 1:3]
    hi = lo .+ 2.0
    bc = SmartPriorMT.core_bounds(lo, hi, 2:5, 2:4, 1:2)
    @test size(bc) == (4, 3, 2)
    @test bc.lo[1, 1, 1] == 2.0
    @test bc.hi[1, 1, 1] == 4.0
    @test bc.ix == 2:5
end

@testset "core_bounds from a PriorBundle" begin
    g = PriorGrid(fill(500.0, 6), fill(500.0, 5), fill(200.0, 3))
    mu = fill(2.0, size(g))
    mu[:, :, 1] .= NaN                        # air layer
    sigma = fill(0.25, size(g))
    sigma[:, :, 1] .= NaN
    b = PriorBundle(g, mu, sigma; k = 2.0, log_rho_bounds = (0.0, 5.0))

    bc = SmartPriorMT.core_bounds(b, 2:5, 2:4, 1:3)
    @test size(bc) == (4, 3, 3)
    # the air layer arrives unbounded, and both bounds are non-finite together
    @test all(!isfinite, bc.lo[:, :, 1])
    @test all(!isfinite, bc.hi[:, :, 1])
    @test all(bc.lo[:, :, 2] .≈ 1.5)
    @test all(bc.hi[:, :, 2] .≈ 2.5)
end

@testset "frozen_from_bounds" begin
    lo = fill(1.0, 3, 3, 2); lo[:, :, 1] .= NaN
    hi = fill(3.0, 3, 3, 2); hi[:, :, 1] .= NaN
    bc = SmartPriorMT.BoundedCore(lo, hi, 1:3, 1:3, 1:2)

    mask = SmartPriorMT.frozen_from_bounds(bc)
    @test mask isa BitArray{3}
    @test all(mask[:, :, 1])
    @test !any(mask[:, :, 2])
end

@testset "control_bounds looks up each control's own cell" begin
    m = _bvfsa_loaded()
    ix, iy = MTGeophysics.core_ranges(m; tol = 0.2)
    kz = 1:m.nz
    nxc, nyc, nzc = length(ix), length(iy), length(kz)

    # depth-varying bounds, so a lookup that ignored ck would show up
    lo = [1.0 + 0.5 * k for i in 1:nxc, j in 1:nyc, k in 1:nzc]
    hi = lo .+ 1.0
    bc = SmartPriorMT.BoundedCore(lo, hi, ix, iy, kz)

    rbf = build_rbf_map(m, ix, iy, 20, MersenneTwister(1); kz = kz, sigma_scale = 1.5)
    lo_c, hi_c = SmartPriorMT.control_bounds(rbf, bc)

    @test length(lo_c) == length(rbf.ci)
    @test length(hi_c) == length(rbf.ci)
    for q in eachindex(lo_c)
        @test lo_c[q] ≈ 1.0 + 0.5 * rbf.ck[q]
        @test hi_c[q] ≈ lo_c[q] + 1.0
    end
end

@testset "control_bounds rejects a control in an unbounded cell" begin
    m = _bvfsa_loaded()
    ix, iy = MTGeophysics.core_ranges(m; tol = 0.2)
    kz = 1:m.nz
    dims = (length(ix), length(iy), length(kz))

    lo = fill(1.0, dims); lo[:, :, 1] .= NaN
    hi = fill(3.0, dims); hi[:, :, 1] .= NaN
    bc = SmartPriorMT.BoundedCore(lo, hi, ix, iy, kz)

    # without the exclude mask, controls land in the unbounded layer and the
    # error has to point at the missing mask rather than fail later as NaN
    unmasked = build_rbf_map(m, ix, iy, 60, MersenneTwister(2); kz = kz, sigma_scale = 1.5)
    @test_throws ErrorException SmartPriorMT.control_bounds(unmasked, bc)

    masked = build_rbf_map(m, ix, iy, 30, MersenneTwister(2);
                           kz = kz, sigma_scale = 1.5,
                           exclude = SmartPriorMT.frozen_from_bounds(bc))
    lo_c, hi_c = SmartPriorMT.control_bounds(masked, bc)
    @test all(isfinite, lo_c)
    @test all(isfinite, hi_c)
    @test all(rbf_k -> rbf_k > 1, masked.ck)
end

@testset "propose_controls_bounded! reproduces the scalar version exactly" begin
    # the whole adaptor rests on this: with constant bounds it must consume the
    # random stream in the same order and land on the same numbers as upstream
    M = 40
    v0 = collect(range(1.0, 3.0; length = M))
    lo_s, hi_s = 0.5, 4.0

    for (T, step_scale, nsel) in ((5.0, 1.0, 10), (0.5, 0.3, 40), (50.0, 2.0, 1))
        d_ref = fill(0.0, M)
        d_new = fill(0.0, M)

        i_ref = MTGeophysics.propose_controls!(d_ref, T, lo_s, hi_s, v0, nsel,
                                              MersenneTwister(7); step_scale = step_scale)
        i_new = SmartPriorMT.propose_controls_bounded!(d_new, T, fill(lo_s, M), fill(hi_s, M), v0, nsel,
                                          MersenneTwister(7); step_scale = step_scale)

        @test i_new == i_ref
        @test d_new == d_ref
    end
end

@testset "propose_controls_bounded! respects per-control intervals" begin
    M = 60
    v0 = fill(2.0, M)
    # alternating tight and loose intervals around the current value
    lo = [isodd(q) ? 1.95 : 0.5 for q in 1:M]
    hi = [isodd(q) ? 2.05 : 4.5 for q in 1:M]

    d = fill(0.0, M)
    SmartPriorMT.propose_controls_bounded!(d, 5.0, lo, hi, v0, M, MersenneTwister(11))
    v = v0 .+ d

    @test all(lo .<= v .<= hi)

    # tight cells must move less than loose ones, which is the second thing
    # per-cell bounds buy: the step width follows the prior's confidence.
    # compared on medians, since a single loose draw can land anywhere in its
    # interval and happen to be tiny
    tight = abs.(d[1:2:M])
    loose = abs.(d[2:2:M])
    @test all(tight .<= 0.05 + 1e-12)      # deterministic: half the tight interval
    @test median(loose) > 10 * median(tight)

    # narrow bounds must not raise the rejection rate, since the step scales with
    # the interval; a control stuck at its midpoint is the fallback signature
    mids = (lo .+ hi) ./ 2
    @test count(q -> v[q] ≈ mids[q], 1:M) <= 2
end

@testset "propose_controls_bounded! recovers a control outside its interval" begin
    M = 8
    v0 = fill(9.0, M)          # far above the allowed range
    lo = fill(1.0, M)
    hi = fill(3.0, M)
    d = fill(0.0, M)
    SmartPriorMT.propose_controls_bounded!(d, 1.0, lo, hi, v0, M, MersenneTwister(3))
    v = v0 .+ d
    # the resampling loop cannot succeed, so every control resets to the middle
    @test all(v .≈ 2.0)
end

@testset "propose_controls_bounded! validation" begin
    d = zeros(5)
    @test_throws DimensionMismatch SmartPriorMT.propose_controls_bounded!(d, 1.0, zeros(4), ones(5),
                                                             zeros(5), 2, MersenneTwister(1))
    @test_throws DimensionMismatch SmartPriorMT.propose_controls_bounded!(d, 1.0, zeros(5), ones(4),
                                                             zeros(5), 2, MersenneTwister(1))
    @test_throws ArgumentError SmartPriorMT.propose_controls_bounded!(d, 1.0, zeros(5), ones(5),
                                                         zeros(5), 0, MersenneTwister(1))
    @test_throws ArgumentError SmartPriorMT.propose_controls_bounded!(d, 1.0, zeros(5), ones(5),
                                                         zeros(5), 6, MersenneTwister(1))
end

@testset "clamp_to_bounds" begin
    lo = fill(1.0, 3, 3, 2)
    hi = fill(3.0, 3, 3, 2)
    bc = SmartPriorMT.BoundedCore(lo, hi, 1:3, 1:3, 1:2)

    v = fill(2.0, 3, 3, 2)
    v[1, 1, 1] = -5.0
    v[2, 2, 1] = 99.0
    out = SmartPriorMT.clamp_to_bounds(v, bc)
    @test out[1, 1, 1] == 1.0
    @test out[2, 2, 1] == 3.0
    @test out[3, 3, 2] == 2.0
    @test v[1, 1, 1] == -5.0        # input untouched

    # with constant bounds it must agree with the upstream broadcast clamp
    @test out == clamp.(v, 1.0, 3.0)

    @test_throws DimensionMismatch SmartPriorMT.clamp_to_bounds(zeros(2, 2, 2), bc)
end

@testset "clamp_to_bounds passes unbounded cells through" begin
    # clamp(x, NaN, NaN) is NaN, which would poison the model sent to the solver
    lo = fill(1.0, 3, 3, 2); lo[:, :, 1] .= NaN
    hi = fill(3.0, 3, 3, 2); hi[:, :, 1] .= NaN
    bc = SmartPriorMT.BoundedCore(lo, hi, 1:3, 1:3, 1:2)

    v = fill(50.0, 3, 3, 2)
    out = SmartPriorMT.clamp_to_bounds(v, bc)
    @test all(out[:, :, 1] .== 50.0)
    @test all(out[:, :, 2] .== 3.0)
end

@testset "bounded_trial! end to end" begin
    m = _bvfsa_loaded()
    ix, iy = MTGeophysics.core_ranges(m; tol = 0.2)
    kz = 1:m.nz
    dims = (length(ix), length(iy), length(kz))

    v0_core = Array(m.A[ix, iy, kz])
    # a laterally varying prior, tight at the top and loose at depth
    lo = [v0_core[i, j, k] - 0.1 * k for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    hi = [v0_core[i, j, k] + 0.1 * k for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    bc = SmartPriorMT.BoundedCore(lo, hi, ix, iy, kz)

    rbf = build_rbf_map(m, ix, iy, 25, MersenneTwister(4); kz = kz, sigma_scale = 1.5)
    M = length(rbf.ci)
    lo_c, hi_c = SmartPriorMT.control_bounds(rbf, bc)
    v0_ctrl = [v0_core[rbf.ci[q], rbf.cj[q], rbf.ck[q]] for q in 1:M]

    delta = zeros(M)
    buffer = zeros(dims)
    v_trial, idxs = SmartPriorMT.bounded_trial!(delta, buffer, rbf, bc, 5.0, lo_c, hi_c,
                                   v0_ctrl, v0_core, max(1, M ÷ 2), MersenneTwister(5))

    @test size(v_trial) == dims
    @test length(idxs) == max(1, M ÷ 2)
    @test all(isfinite, v_trial)
    # every cell inside its own interval: the guarantee the inversion relies on
    @test all(bc.lo .<= v_trial .<= bc.hi)
    # and it actually moved, otherwise the test proves nothing
    @test v_trial != v0_core
    # the clamp is per layer, not global: no layer may exceed its own half-width.
    # excursions sit well inside it because apply_rbf_map! normalises the kernel
    # weights, so each cell gets a weighted average of nearby control deltas and
    # unperturbed neighbours dilute it -- the bound is a cap reached over
    # iterations, not in one trial
    for k in 1:dims[3]
        @test maximum(abs.(v_trial[:, :, k] .- v0_core[:, :, k])) <= 0.1 * k + 1e-12
    end
end

@testset "bounded_trial! restores frozen cells the kernels bleed into" begin
    m = _bvfsa_loaded()
    ix, iy = MTGeophysics.core_ranges(m; tol = 0.2)
    kz = 1:m.nz
    dims = (length(ix), length(iy), length(kz))

    v0_core = Array(m.A[ix, iy, kz])
    lo = fill(1.0, dims); hi = fill(3.0, dims)
    lo[:, :, 1] .= NaN; hi[:, :, 1] .= NaN
    bc = SmartPriorMT.BoundedCore(lo, hi, ix, iy, kz)
    frozen = SmartPriorMT.frozen_from_bounds(bc)

    rbf = build_rbf_map(m, ix, iy, 20, MersenneTwister(6);
                        kz = kz, sigma_scale = 2.0, exclude = frozen)
    M = length(rbf.ci)
    lo_c, hi_c = SmartPriorMT.control_bounds(rbf, bc)
    v0_ctrl = [v0_core[rbf.ci[q], rbf.cj[q], rbf.ck[q]] for q in 1:M]

    v_trial, _ = SmartPriorMT.bounded_trial!(zeros(M), zeros(dims), rbf, bc, 10.0, lo_c, hi_c,
                                v0_ctrl, v0_core, M, MersenneTwister(9);
                                frozen = frozen)

    @test v_trial[:, :, 1] == v0_core[:, :, 1]
    @test v_trial[:, :, 2] != v0_core[:, :, 2]

    # without the mask the kernels would have written into that layer, which is
    # exactly why upstream re-imposes it
    v_unfrozen, _ = SmartPriorMT.bounded_trial!(zeros(M), zeros(dims), rbf, bc, 10.0, lo_c, hi_c,
                                   v0_ctrl, v0_core, M, MersenneTwister(9))
    @test v_unfrozen[:, :, 1] != v0_core[:, :, 1]
end

@testset "bounded_trial! validation" begin
    m = _bvfsa_loaded()
    ix, iy = MTGeophysics.core_ranges(m; tol = 0.2)
    kz = 1:m.nz
    dims = (length(ix), length(iy), length(kz))
    bc = SmartPriorMT.BoundedCore(fill(1.0, dims), fill(3.0, dims), ix, iy, kz)
    rbf = build_rbf_map(m, ix, iy, 10, MersenneTwister(1); kz = kz)
    M = length(rbf.ci)

    @test_throws DimensionMismatch SmartPriorMT.bounded_trial!(zeros(M), zeros(dims), rbf, bc, 1.0,
                                                   fill(1.0, M), fill(3.0, M),
                                                   fill(2.0, M), zeros(2, 2, 2), 1,
                                                   MersenneTwister(1))
end

@testset "bound_report" begin
    lo = fill(1.5, 4, 4, 2)
    hi = fill(2.5, 4, 4, 2)
    lo[:, :, 1] .= NaN
    hi[:, :, 1] .= NaN
    bc = SmartPriorMT.BoundedCore(lo, hi, 1:4, 1:4, 1:2)

    r = SmartPriorMT.bound_report(bc; reference = (0.0, 5.0))
    @test r.nbounded == 16
    @test r.nfrozen == 16
    @test r.width_min ≈ 1.0
    @test r.width_max ≈ 1.0
    @test r.width_geomean ≈ 1.0
    # one decade out of the five a scalar run would have searched
    @test r.volume_ratio ≈ 0.2
    @test isnan(SmartPriorMT.bound_report(bc).volume_ratio)

    # the geometric mean is the honest average: a prior that is tight nearly
    # everywhere but loose in a few cells must not be flattered
    lo2 = fill(1.99, 10, 10, 1); hi2 = fill(2.01, 10, 10, 1)
    lo2[1, 1, 1] = 0.0; hi2[1, 1, 1] = 5.0
    bc2 = SmartPriorMT.BoundedCore(lo2, hi2, 1:10, 1:10, 1:1)
    r2 = SmartPriorMT.bound_report(bc2)
    @test r2.width_geomean < mean(vec(hi2 .- lo2))

    all_nan = fill(NaN, 2, 2, 1)
    @test_throws ArgumentError SmartPriorMT.bound_report(SmartPriorMT.BoundedCore(all_nan, all_nan, 1:2, 1:2, 1:1))
end
