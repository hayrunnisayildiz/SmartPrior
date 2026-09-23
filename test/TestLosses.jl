using Test
using SmartPrior
using Random
using Statistics
using Zygote

_loss_grid() = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0, 225.0];
                         origin = [-1000.0, -1000.0, 0.0])

@testset "heteroscedastic_nll" begin
    target = [1.0, 2.0, 3.0]

    exact = heteroscedastic_nll(target, fill(0.5, 3), target)
    @test exact ≈ log(0.5)

    res = 0.4
    mu = target .- res
    losses = [heteroscedastic_nll(mu, fill(s, 3), target) for s in 0.1:0.02:1.5]
    best = (0.1:0.02:1.5)[argmin(losses)]
    @test best ≈ res atol = 0.03

    @test heteroscedastic_nll(mu, fill(0.05, 3), target) >
          heteroscedastic_nll(mu, fill(res, 3), target)
    @test heteroscedastic_nll(mu, fill(5.0, 3), target) >
          heteroscedastic_nll(mu, fill(res, 3), target)

    w = [1.0, 0.0, 0.0]
    mixed = [1.0, 99.0, 99.0]
    @test heteroscedastic_nll(mixed, fill(0.5, 3), target; weight = w) ≈ log(0.5)

    @test_throws DimensionMismatch heteroscedastic_nll([1.0], [1.0], [1.0, 2.0])
    @test_throws DimensionMismatch heteroscedastic_nll(mu, fill(0.5, 3), target; weight = [1.0])
    @test_throws ArgumentError heteroscedastic_nll(Float64[], Float64[], Float64[])
end

@testset "smoothness" begin
    g = _loss_grid()

    @test smoothness(fill(2.0, size(g)), g) ≈ 0.0

    nx, ny, nz = size(g)
    rough = [iseven(i + j + k) ? 1.0 : -1.0 for i in 1:nx, j in 1:ny, k in 1:nz]
    ramp = [0.1 * i for i in 1:nx, j in 1:ny, k in 1:nz]
    @test smoothness(rough, g) > smoothness(ramp, g)

    layered = [0.0 + 1.0 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    @test smoothness(layered, g; vertical_weight = 0.1) <
          smoothness(layered, g; vertical_weight = 1.0)

    g1 = PriorGrid([100.0], [100.0], [100.0])
    @test smoothness(fill(3.0, 1, 1, 1), g1) ≈ 0.0

    @test_throws DimensionMismatch smoothness(zeros(2, 2, 2), g)
end

@testset "reference_penalty" begin
    @test reference_penalty(fill(2.0, 4), fill(2.0, 4)) ≈ 0.0
    @test reference_penalty(fill(2.5, 4), fill(2.0, 4)) ≈ 0.25
    @test reference_penalty([2.5, 9.0, 9.0], fill(2.0, 3); weight = [1.0, 0.0, 0.0]) ≈ 0.25

    @test_throws DimensionMismatch reference_penalty(fill(2.0, 4), fill(2.0, 3))
    @test_throws DimensionMismatch reference_penalty(fill(2.0, 4), fill(2.0, 4); weight = [1.0])
    @test_throws ArgumentError reference_penalty(fill(2.0, 4), fill(2.0, 4); weight = [1.0, -1.0, 1.0, 1.0])
    @test_throws ArgumentError reference_penalty(fill(2.0, 4), fill(2.0, 4); weight = zeros(4))
end

@testset "sigma_penalty" begin
    @test sigma_penalty(fill(0.5, 4); target = 0.5) ≈ 0.0
    @test sigma_penalty(fill(0.7, 4); target = 0.5) ≈ 0.04
    @test_throws ArgumentError sigma_penalty([0.5]; target = 0.0)

    vec_t = [0.3, 0.5, 0.4, 0.6]
    @test sigma_penalty(vec_t; target = vec_t) ≈ 0.0
    @test sigma_penalty(fill(0.5, 4); target = vec_t) ≈ mean(abs2.([0.2, 0.0, 0.1, -0.1]))
    @test sigma_penalty(fill(0.7, 4); target = 0.5) ≈
          sigma_penalty(fill(0.7, 4); target = fill(0.5, 4))

    @test_throws DimensionMismatch sigma_penalty([0.5, 0.5]; target = [0.3])
    @test_throws ArgumentError sigma_penalty([0.5]; target = [0.0])
    @test_throws ArgumentError sigma_penalty([0.5]; target = [-0.1])
end

@testset "four independent named anchor groups" begin
    g = _loss_grid()
    n = ncells(g)
    mus = fill(2.0, 4, n)
    sigmas = fill(0.5, 4, n)

    t = PriorTargets(
        anchors_grade = ([1, 2], [2.0, 2.0], [1.0, 1.0]),
        anchors_density = ([3], [2.0], [1.0]),
        anchors_susceptibility = ([4], [2.0], [1.0]),
        anchors_resistivity = ([5], [2.0], [1.0]),
        property_names = ["grade", "density", "susceptibility", "resistivity"],
        sigma_target = 0.5,
    )
    r = loss_report(mus, sigmas, g, t)
    @test r.grade ≈ 2 * log(0.5)
    @test r.density ≈ 2 * log(0.5)
    @test r.susceptibility ≈ 2 * log(0.5)
    @test r.resistivity ≈ 2 * log(0.5)
    @test isnan(r.anchor)
    @test r.grade ≈ r.density

    w2 = LossWeights(density = 2.0, grade = 1.0, susceptibility = 0.0,
                     resistivity = 0.0, smooth = 0.0, sigma = 0.0)
    r2 = loss_report(mus, sigmas, g, t, w2)
    @test r2.total ≈ 1.0 * r2.grade + 2.0 * r2.density

    @test_throws ArgumentError prior_loss(mus[1, :], sigmas[1, :], g, t)
    @test_throws ArgumentError prior_loss(mus, sigmas, g,
        PriorTargets(anchors = ([1], [2.0], [1.0])))
end

@testset "conductivity_100kHz is a named fourth head, not resistivity" begin
    g = _loss_grid()
    n = ncells(g)
    mus = fill(2.0, 4, n)
    sigmas = fill(0.5, 4, n)
    t = PriorTargets(
        anchors_grade = ([1], [2.0], [1.0]),
        anchors_density = ([2], [2.0], [1.0]),
        anchors_susceptibility = ([3], [2.0], [1.0]),
        anchors_conductivity = ([4], [2.0], [1.0]),
        property_names = ["grade", "density", "susceptibility", "conductivity_100kHz"],
        sigma_target = 0.5,
    )
    r = loss_report(mus, sigmas, g, t)
    @test r.conductivity_100kHz ≈ 2 * log(0.5)
    @test isnan(r.resistivity)
    @test isfinite(r.grade)
    net = PriorNet(3; width = 8, depth = 2, nproperties = 4,
                   property_names = ["grade", "density", "susceptibility",
                                     "conductivity_100kHz"])
    @test net.property_names[4] == "conductivity_100kHz"
end

@testset "named-anchor NLL is calibrated to each group's own std" begin
    g = _loss_grid()
    n = ncells(g)
    mus = zeros(2, n)
    sigmas = ones(2, n)
    mus[1, 1:2] .= 1.0
    mus[2, 1:2] .= 10.0
    sigmas[1, 1:2] .= 1.0
    sigmas[2, 1:2] .= 10.0
    t = PriorTargets(
        anchors_grade = ([1, 2], [0.0, 2.0], [1.0, 1.0]),
        anchors_density = ([1, 2], [0.0, 20.0], [1.0, 1.0]),
        property_names = ["grade", "density"],
    )
    r = loss_report(mus, sigmas, g, t)
    @test r.grade ≈ 1.0 atol = 1e-10
    @test r.density ≈ 1.0 atol = 1e-10
    @test r.grade ≈ r.density

    raw_g = heteroscedastic_nll(mus[1, 1:2], sigmas[1, 1:2], [0.0, 2.0])
    raw_d = heteroscedastic_nll(mus[2, 1:2], sigmas[2, 1:2], [0.0, 20.0])
    @test raw_d > raw_g + 1.0
end

@testset "sigma_bounds_from_anchors follows group std" begin
    anchors = ([1, 2, 3], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    s = sqrt(2 / 3)
    lo, hi = sigma_bounds_from_anchors(anchors; fraction = 1.0, hi = 1.2)
    @test lo ≈ s
    @test hi ≈ 2 * s
    lo2, hi2 = sigma_bounds_from_anchors(anchors; fraction = 0.5, hi = 0.1)
    @test lo2 ≈ 0.5 * s
    @test hi2 ≈ 2 * s
    @test_throws ArgumentError sigma_bounds_from_anchors(anchors; fraction = 0.0)
end

@testset "single-property prior_loss uses anchors + regularisers" begin
    g = _loss_grid()
    n = ncells(g)
    mu = fill(2.0, n)
    sigma = fill(0.5, n)
    t = PriorTargets(anchors = ([1, 2], [2.0, 2.0], [1.0, 1.0]),
                     sigma_target = 0.5)
    r = loss_report(mu, sigma, g, t)
    @test isfinite(r.total)
    @test isfinite(r.anchor)
    @test isfinite(r.smooth)
    @test isfinite(r.sigma)
    @test isnan(r.reference)
end
