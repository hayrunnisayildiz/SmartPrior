using Test
using SmartPriorMT
using Lux
using Random
using Statistics
using Zygote

_net_grid() = PriorGrid(fill(500.0, 5), fill(500.0, 4), [80.0, 120.0, 180.0];
                        origin = [-1250.0, -1000.0, 0.0])

@testset "fourier_encode" begin
    X = [0.0 0.5 1.0; 2.0 2.0 2.0]

    # no bands is a no-op
    @test fourier_encode(X, [1], 0) == X
    @test fourier_encode(X, Int[], 4) == X

    E = fourier_encode(X, [1], 2)
    @test size(E) == (2 + 2 * 1 * 2, 3)
    @test E[1:2, :] == X
    # band 0 is sin/cos of pi*x, band 1 doubles the frequency
    @test E[3, :] ≈ sin.(π .* X[1, :])
    @test E[4, :] ≈ cos.(π .* X[1, :])
    @test E[5, :] ≈ sin.(2π .* X[1, :])
    @test E[6, :] ≈ cos.(2π .* X[1, :])

    # encoding two rows doubles the added block
    @test size(fourier_encode(X, [1, 2], 3), 1) == 2 + 2 * 2 * 3

    # the encoding separates coordinates a linear map would not: two cells whose
    # raw coordinates differ slightly get distinguishable high-band values
    E8 = fourier_encode([0.0 0.01], [1], 8)
    @test abs(E8[end-1, 1] - E8[end-1, 2]) > 0.5

    @test_throws ArgumentError fourier_encode(X, [1], -1)
    @test_throws ArgumentError fourier_encode(X, [1], 2; base = 1.0)
    @test_throws ArgumentError fourier_encode(X, [3], 2)
end

@testset "encode_features" begin
    g = _net_grid()
    s = build_features(g; coordinates = true)
    nc = nchannels(s)

    X = encode_features(s; n_bands = 4)
    @test size(X) == (nc + 2 * 3 * 4, ncells(g))
    @test X[1:nc, :] == feature_matrix(s)

    # absent channels are skipped, so the same call works for a transferable
    # model where the coordinate channels were deliberately left out
    s2 = build_features(g; coordinates = false)
    X2 = encode_features(s2; n_bands = 4)
    @test size(X2, 1) == nchannels(s2)
end

@testset "PriorNet: construction and validation" begin
    net = PriorNet(12; width = 16, depth = 3)
    @test net.nin == 12
    @test Lux.parameterlength(net) > 0

    @test_throws ArgumentError PriorNet(0)
    @test_throws ArgumentError PriorNet(4; width = 0)
    @test_throws ArgumentError PriorNet(4; depth = 0)
    @test_throws ArgumentError PriorNet(4; log_rho_bounds = (5.0, 0.0))
    @test_throws ArgumentError PriorNet(4; sigma_bounds = (0.0, 1.0))
    @test_throws ArgumentError PriorNet(4; sigma_bounds = (1.0, 0.5))
    @test_throws ArgumentError PriorNet(4; residual_span = 0.0)
end

@testset "setup_prior precision" begin
    net = PriorNet(6; width = 8, depth = 2)
    ps64, st = setup_prior(Xoshiro(1), net)
    @test eltype(ps64.layer_1.weight) === Float64

    ps32, _ = setup_prior(Xoshiro(1), net; precision = Float32)
    @test eltype(ps32.layer_1.weight) === Float32

    @test_throws ArgumentError setup_prior(Xoshiro(1), net; precision = Float16)
end

@testset "predict: outputs respect their hard bounds" begin
    net = PriorNet(6; width = 32, depth = 3,
                   log_rho_bounds = (0.5, 4.5), sigma_bounds = (0.1, 0.9))
    ps, st = setup_prior(Xoshiro(7), net)

    # deliberately extreme inputs, to try to push the outputs past their limits
    X = 50 .* randn(Xoshiro(3), 6, 200)
    (mu, sigma), _ = predict(net, X, ps, st)

    @test length(mu) == 200
    @test length(sigma) == 200
    @test all(0.5 .<= mu .<= 4.5)
    @test all(0.1 .<= sigma .<= 0.9)
    @test all(isfinite, mu)
    @test all(isfinite, sigma)
end

@testset "predict: residual mode tracks the baseline" begin
    net = PriorNet(6; width = 32, depth = 3, residual_span = 1.0,
                   log_rho_bounds = (0.0, 5.0))
    ps, st = setup_prior(Xoshiro(11), net)
    X = randn(Xoshiro(4), 6, 50)

    offset = fill(2.0, 50)
    (mu, _), _ = predict(net, X, ps, st; offset = offset)
    @test all(1.0 .<= mu .<= 3.0)

    # shifting the baseline shifts the prediction by exactly the same amount:
    # this is the property that makes a trained net reusable on another survey.
    # it holds only while both bands stay clear of the physical bounds, which
    # the offsets here do
    (mu2, _), _ = predict(net, X, ps, st; offset = offset .+ 0.7)
    @test mu2 ≈ mu .+ 0.7 rtol = 1e-12

    @test_throws DimensionMismatch predict(net, X, ps, st; offset = fill(2.0, 49))
end

@testset "predict: residual mode stays inside the physical range" begin
    # mu is written out as the inversion's starting model, so a residual span
    # wide enough to leave log_rho_bounds would produce an unphysical start model
    # that no later step can repair
    net = PriorNet(6; width = 32, depth = 3, residual_span = 3.0,
                   log_rho_bounds = (1.0, 3.0))
    ps, st = setup_prior(Xoshiro(5), net)
    X = 50 .* randn(Xoshiro(6), 6, 300)

    (mu, _), _ = predict(net, X, ps, st; offset = fill(2.0, 300))
    @test all(1.0 .<= mu .<= 3.0)

    # an offset outside the range is pulled in rather than followed
    (mu_hi, _), _ = predict(net, X, ps, st; offset = fill(9.0, 300))
    @test all(1.0 .<= mu_hi .<= 3.0)
    (mu_lo, _), _ = predict(net, X, ps, st; offset = fill(-9.0, 300))
    @test all(1.0 .<= mu_lo .<= 3.0)
end

@testset "predict: the residual band is asymmetric near a bound" begin
    # a baseline close to a physical bound gets less room on that side, which is
    # the intended consequence of intersecting the two intervals
    net = PriorNet(6; width = 32, depth = 3, residual_span = 1.0,
                   log_rho_bounds = (0.0, 5.0))
    ps, st = setup_prior(Xoshiro(9), net)
    X = 50 .* randn(Xoshiro(10), 6, 400)

    (centred, _), _ = predict(net, X, ps, st; offset = fill(2.5, 400))
    @test maximum(centred) - minimum(centred) > 1.5      # nearly the full 2.0 band

    (near_edge, _), _ = predict(net, X, ps, st; offset = fill(0.3, 400))
    @test all(0.0 .<= near_edge .<= 1.3)
    # the band is [0.0, 1.3], so it is narrower than the centred one
    @test maximum(near_edge) - minimum(near_edge) <
          maximum(centred) - minimum(centred)
end

@testset "predict: dimension checks" begin
    net = PriorNet(6; width = 8, depth = 2)
    ps, st = setup_prior(Xoshiro(1), net)
    @test_throws DimensionMismatch predict(net, randn(5, 10), ps, st)
end

@testset "predict_grid reshapes back onto the grid" begin
    g = _net_grid()
    s = build_features(g)
    X = encode_features(s; n_bands = 3)

    net = PriorNet(size(X, 1); width = 16, depth = 2)
    ps, st = setup_prior(Xoshiro(5), net)

    mu, sigma, _ = predict_grid(net, X, ps, st, size(g))
    @test size(mu) == size(g)
    @test size(sigma) == size(g)

    (muv, sigmav), _ = predict(net, X, ps, st)
    @test vec(mu) == muv
    @test vec(sigma) == sigmav

    @test_throws DimensionMismatch predict_grid(net, X, ps, st, (2, 2, 2))
end

@testset "predict is differentiable through both outputs" begin
    net = PriorNet(8; width = 16, depth = 2)
    ps, st = setup_prior(Xoshiro(9), net)
    X = randn(Xoshiro(2), 8, 30)
    target = randn(Xoshiro(6), 30)

    # a loss touching mu and sigma together, as the heteroscedastic likelihood does
    function loss(p)
        (mu, sigma), _ = predict(net, X, p, st)
        return mean(@. 0.5 * ((target - mu) / sigma)^2 + log(sigma))
    end

    l0 = loss(ps)
    @test isfinite(l0)

    grad = Zygote.gradient(loss, ps)[1]
    @test grad !== nothing

    gw = grad.layer_1.weight
    @test size(gw) == size(ps.layer_1.weight)
    @test all(isfinite, gw)
    @test any(!iszero, gw)

    # the last layer feeds both outputs, so its gradient must be non-trivial in
    # both rows; a bug that dropped sigma would leave row 2 at zero
    gout = grad.layer_3.weight
    @test any(!iszero, gout[1, :])
    @test any(!iszero, gout[2, :])
end
