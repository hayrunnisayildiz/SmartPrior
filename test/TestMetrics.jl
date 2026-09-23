using Test
using SmartPrior
using Random
using Statistics

_metrics_grid() = PriorGrid(fill(500.0, 4), fill(500.0, 4), fill(200.0, 3))

@testset "coverage" begin
    truth = fill(2.0, 3, 3, 2)
    @test coverage(truth, fill(1.0, 3, 3, 2), fill(3.0, 3, 3, 2)) == 1.0
    @test coverage(truth, fill(2.5, 3, 3, 2), fill(3.0, 3, 3, 2)) == 0.0

    # exactly on a bound counts as covered
    @test coverage(truth, fill(2.0, 3, 3, 2), fill(3.0, 3, 3, 2)) == 1.0

    lo = fill(1.0, 3, 3, 2)
    hi = fill(3.0, 3, 3, 2)
    hi[1, 1, 1] = 1.5
    hi[2, 1, 1] = 1.5
    @test coverage(truth, lo, hi) ≈ 16 / 18

    # air must not inflate the score: excluded cells are not counted as covered
    t_air = copy(truth); t_air[:, :, 1] .= NaN
    lo_air = copy(lo); lo_air[:, :, 1] .= NaN
    hi_air = copy(hi); hi_air[:, :, 1] .= NaN
    @test coverage(t_air, lo_air, hi_air) == 1.0

    all_nan = fill(NaN, 2, 2, 1)
    @test_throws ArgumentError coverage(all_nan, all_nan, all_nan)
    @test_throws DimensionMismatch coverage(truth, zeros(2, 2, 2), hi)
end

@testset "volume_reduction" begin
    lo = fill(2.0, 4, 4, 2)
    hi = fill(3.0, 4, 4, 2)
    @test volume_reduction(lo, hi, (0.0, 5.0)) ≈ 0.2
    @test volume_reduction(lo, hi, (0.0, 1.0)) ≈ 1.0

    # the geometric mean is the honest average, because the search space is a
    # product over cells: an arithmetic mean would let a few wide cells hide a
    # prior that is tight and wrong everywhere else
    mixed_lo = fill(2.0, 10, 10, 1)
    mixed_hi = fill(2.02, 10, 10, 1)
    mixed_hi[1, 1, 1] = 7.0
    geo = volume_reduction(mixed_lo, mixed_hi, (0.0, 5.0))
    arith = mean(vec(mixed_hi .- mixed_lo)) / 5.0
    @test geo < arith

    @test_throws ArgumentError volume_reduction(lo, hi, (5.0, 0.0))
    bad = copy(hi); bad[1, 1, 1] = 1.0
    @test_throws ArgumentError volume_reduction(lo, bad, (0.0, 5.0))
end

@testset "rmse and mae" begin
    a = fill(2.0, 3, 3, 2)
    b = copy(a)
    @test rmse(a, b) == 0.0
    @test mae(a, b) == 0.0

    b[1, 1, 1] = 2.3
    b[2, 1, 1] = 1.7
    @test rmse(a, b) ≈ sqrt(2 * 0.09 / 18)
    @test mae(a, b) ≈ 0.6 / 18

    # rmse punishes a single bad cell harder than mae, which is why both are
    # reported: they separate "wrong everywhere a little" from "one blown-up cell"
    spread = fill(2.0, 10, 10, 1) .+ 0.1
    concentrated = fill(2.0, 10, 10, 1); concentrated[1, 1, 1] = 12.0
    base = fill(2.0, 10, 10, 1)
    @test mae(base, spread) ≈ mae(base, concentrated) rtol = 1e-12
    @test rmse(base, concentrated) > rmse(base, spread)

    # air is skipped
    a_air = copy(a); a_air[:, :, 1] .= NaN
    @test rmse(a_air, b) == 0.0

    all_nan = fill(NaN, 2, 2, 1)
    @test_throws ArgumentError rmse(all_nan, all_nan)
    @test_throws ArgumentError mae(all_nan, all_nan)
end

@testset "anomaly_correlation" begin
    truth = [Float64(i) for i in 1:5, j in 1:3, k in 1:2]

    @test anomaly_correlation(truth, truth) ≈ 1.0
    # a constant offset does not change the pattern
    @test anomaly_correlation(truth, truth .+ 3.0) ≈ 1.0
    # nor does a damped amplitude, which is the whole point of the metric: an
    # uncertain prior should damp, and rmse punishes it for doing the right thing
    damped = 0.2 .* (truth .- mean(truth)) .+ mean(truth)
    @test anomaly_correlation(truth, damped) ≈ 1.0
    @test rmse(truth, damped) > 0
    @test anomaly_correlation(truth, -truth) ≈ -1.0

    @test isnan(anomaly_correlation(truth, fill(2.0, 5, 3, 2)))
    @test_throws ArgumentError anomaly_correlation(fill(1.0, 1, 1, 1), fill(1.0, 1, 1, 1))
end

@testset "calibration" begin
    # residuals drawn at exactly one sigma: zrms must be 1 and within1 must be 1
    truth = fill(2.5, 10, 10, 2)
    mu = fill(2.0, 10, 10, 2)
    sigma = fill(0.5, 10, 10, 2)
    c = calibration(truth, mu, sigma)
    @test c.n == 200
    @test c.zrms ≈ 1.0
    @test c.within1 == 1.0
    @test c.within2 == 1.0
    @test c.bias ≈ 0.5

    # halving sigma is overconfidence and must show up as zrms above 1
    over = calibration(truth, mu, fill(0.25, 10, 10, 2))
    @test over.zrms ≈ 2.0
    @test over.within1 == 0.0
    @test over.within2 == 1.0

    # doubling it is timidity, zrms below 1
    @test calibration(truth, mu, fill(1.0, 10, 10, 2)).zrms ≈ 0.5

    # a genuinely calibrated field: residuals drawn from N(0, sigma)
    rng = Xoshiro(4)
    s = 0.4
    t2 = mu .+ s .* randn(rng, size(mu))
    c2 = calibration(t2, mu, fill(s, size(mu)))
    @test isapprox(c2.zrms, 1.0; atol = 0.15)
    @test isapprox(c2.within1, 0.68; atol = 0.08)
    @test isapprox(c2.within2, 0.95; atol = 0.05)
    @test abs(c2.bias) < 0.1

    @test_throws ArgumentError calibration(truth, mu, fill(0.0, 10, 10, 2))
    all_nan = fill(NaN, 2, 2, 1)
    @test_throws ArgumentError calibration(all_nan, all_nan, all_nan)
end

@testset "coverage agrees with calibration at the same k" begin
    g = PriorGrid(fill(500.0, 10), fill(500.0, 10), fill(200.0, 2))
    rng = Xoshiro(7)
    mu = fill(2.0, size(g))
    sigma = fill(0.4, size(g))
    truth = mu .+ 0.4 .* randn(rng, size(g))
    c = calibration(truth, mu, sigma)
    @test coverage(truth, mu .- sigma, mu .+ sigma) ≈ c.within1
    @test coverage(truth, mu .- 2 .* sigma, mu .+ 2 .* sigma) ≈ c.within2
end
