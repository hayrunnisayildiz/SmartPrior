using Test
using SmartPrior
using Statistics
using Random

@testset "GaussianField reproducibility and moments" begin
    L_h = 100.0
    field = gaussian_field(; μ = 0.0, σ = 1.0, η = 0.0, L_h = L_h, seed = 7)
    @test field.L_v ≈ L_h / 2
    @test size(field.ω) == (3, SYNTHETIC_RFF_M)

    rng = Xoshiro(11)
    n = 8000
    xyz = Matrix{Float64}(undef, 3, n)
    @inbounds for j in 1:n
        xyz[1, j] = 2000 * randn(rng)
        xyz[2, j] = 2000 * randn(rng)
        xyz[3, j] = 1000 * randn(rng)
    end

    # (a) same point twice → same value
    p = reshape([10.0, 20.0, -5.0], 3, 1)
    @test evaluate(field, p) == evaluate(field, p)

    # (b) sample variance ≈ 1 (η = 0)
    v = evaluate(field, xyz)
    @test isapprox(var(v; corrected = false), 1.0; rtol = 0.1)

    # (d) subset equivalence
    @test evaluate(field, xyz[:, 1:10]) == evaluate(field, xyz)[1:10]

    # nugget path is also pure / subset-stable
    noisy = gaussian_field(; μ = 1.0, σ = 2.0, η = 0.5, L_h = L_h, seed = 3)
    @test evaluate(noisy, xyz[:, 1:10]) == evaluate(noisy, xyz)[1:10]
    @test evaluate(noisy, p) == evaluate(noisy, p)
end

@testset "GaussianField correlation at L_h" begin
    L_h = 250.0
    field = gaussian_field(; μ = 0.0, σ = 1.0, η = 0.0, L_h = L_h, seed = 19)
    rng = Xoshiro(23)
    n_pairs = 20000
    a = Matrix{Float64}(undef, 3, n_pairs)
    b = Matrix{Float64}(undef, 3, n_pairs)
    @inbounds for j in 1:n_pairs
        x = 5000 * rand(rng) - 2500
        y = 5000 * rand(rng) - 2500
        z = 500 * rand(rng) - 250
        # horizontal separation of exactly L_h
        θ = 2π * rand(rng)
        a[1, j] = x
        a[2, j] = y
        a[3, j] = z
        b[1, j] = x + L_h * cos(θ)
        b[2, j] = y + L_h * sin(θ)
        b[3, j] = z
    end
    za = evaluate(field, a)
    zb = evaluate(field, b)
    # (c) empirical correlation ≈ exp(−1)
    ρ = cor(za, zb)
    @test isapprox(ρ, exp(-1); atol = 0.1)
end
