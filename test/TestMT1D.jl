using Test
using SmartPriorMT
using SmartPriorMT: MU0
using MTGeophysics: solve_mt1d_analytical
using ForwardDiff
using Zygote

@testset "mt1d_impedance: half-space matches the closed form" begin
    f = [0.01, 0.1, 1.0, 10.0]
    ρ = 100.0
    Z = mt1d_impedance(f, [ρ], Float64[])

    for (i, fi) in enumerate(f)
        ω = 2π * fi
        k = sqrt(-1im * ω * MU0 / ρ)
        @test Z[i] ≈ (ω * MU0) / k rtol = 1e-12
    end

    rho_a, phase = mt1d_apparent(f, [ρ], Float64[])
    @test all(≈(ρ; rtol = 1e-10), rho_a)
    @test all(≈(45.0; rtol = 1e-10), phase)
end

@testset "mt1d_apparent: agrees with MTGeophysics.solve_mt1d_analytical" begin
    f = 10 .^ range(-3, 2; length = 25)

    cases = [
        ([100.0], Float64[]),
        ([100.0, 10.0], [500.0]),
        ([50.0, 500.0, 20.0], [200.0, 800.0]),
        ([100.0, 20.0, 350.0, 40.0, 800.0], [120.0, 280.0, 650.0, 1400.0]),
        ([1.0, 10_000.0, 3.0], [50.0, 5000.0]),
    ]

    for (ρ, t) in cases
        ref = solve_mt1d_analytical(f, ρ, t)
        rho_a, phase = mt1d_apparent(f, ρ, t)
        Z = mt1d_impedance(f, ρ, t)

        @test Z ≈ ref.impedance rtol = 1e-10
        @test rho_a ≈ ref.apparent_resistivity rtol = 1e-10
        @test phase ≈ ref.phase rtol = 1e-10
    end
end

@testset "mt1d_impedance: differentiable with respect to resistivity" begin
    f = [0.05, 0.5, 5.0]
    ρ0 = [80.0, 15.0, 400.0]
    t = [300.0, 900.0]

    loss(ρ) = sum(abs2, first(mt1d_apparent(f, ρ, t)))

    grad = Zygote.gradient(loss, ρ0)[1]
    @test grad !== nothing
    @test length(grad) == 3
    @test all(isfinite, grad)

    # central finite differences on a relative step, since the scales differ
    for i in 1:3
        h = 1e-5 * ρ0[i]
        ρp = copy(ρ0); ρp[i] += h
        ρm = copy(ρ0); ρm[i] -= h
        fd = (loss(ρp) - loss(ρm)) / (2h)
        @test grad[i] ≈ fd rtol = 1e-5
    end
end

@testset "mt1d_impedance: forward mode works here but not on the reference" begin
    # records why MT1DAD.jl exists. The `Float64.(rho)` casts in
    # solve_mt1d_analytical are transparent to reverse mode, which tracks
    # pullbacks over plain Float64 values, but they cannot accept a Dual, so
    # forward mode fails on the reference and succeeds here.
    f = [0.5]
    ρ0 = [80.0, 15.0]
    t = [300.0]
    ref_loss(ρ) = sum(abs2, solve_mt1d_analytical(f, ρ, t).apparent_resistivity)
    our_loss(ρ) = sum(abs2, first(mt1d_apparent(f, ρ, t)))

    # both agree in reverse mode, so the two implementations really are the same
    # function and not merely close
    @test Zygote.gradient(our_loss, ρ0)[1] ≈ Zygote.gradient(ref_loss, ρ0)[1] rtol = 1e-10

    @test_throws MethodError ForwardDiff.gradient(ref_loss, ρ0)

    fwd = ForwardDiff.gradient(our_loss, ρ0)
    @test fwd ≈ Zygote.gradient(our_loss, ρ0)[1] rtol = 1e-10
end

@testset "mt1d_column_response" begin
    f = [0.1, 1.0]
    rho_cells = [100.0, 100.0, 10.0, 10.0, 500.0]
    dz_cells = [100.0, 150.0, 200.0, 300.0, 1000.0]

    rho_a, phase = mt1d_column_response(f, rho_cells, dz_cells)
    # the bottom cell is the basement, so its thickness is ignored
    ref_a, ref_p = mt1d_apparent(f, rho_cells, dz_cells[1:end-1])
    @test rho_a ≈ ref_a
    @test phase ≈ ref_p

    # a uniform column reduces to a half-space at every period
    uniform = mt1d_column_response(f, fill(250.0, 5), dz_cells)
    @test all(≈(250.0; rtol = 1e-10), uniform[1])

    @test_throws ArgumentError mt1d_column_response(f, [1.0, 2.0], [1.0])
end

@testset "skin depth and Niblett-Bostick" begin
    # the textbook form is 503 * sqrt(rho * T) metres
    @test skin_depth(1.0, 1.0) ≈ 503.292 rtol = 1e-5
    @test skin_depth(100.0, 1.0) ≈ 503.292 * sqrt(100.0) rtol = 1e-5
    @test skin_depth(100.0, 100.0) ≈ 10 * skin_depth(100.0, 1.0) rtol = 1e-12

    # a 45 degree phase means a uniform earth, so the transform returns rho_a
    @test bostick_resistivity(100.0, 45.0) ≈ 100.0 rtol = 1e-12
    # steeper phase means a conductor below, shallower means a resistor
    @test bostick_resistivity(100.0, 60.0) < 100.0
    @test bostick_resistivity(100.0, 30.0) > 100.0
    # non-physical phases are clamped rather than returning garbage
    @test isfinite(bostick_resistivity(100.0, 0.0))
    @test isfinite(bostick_resistivity(100.0, 90.0))

    # on a half-space the transform should recover the true resistivity at all
    # depths, which is the property the residual baseline relies on
    T = 10 .^ range(-2, 3; length = 20)
    f = 1 ./ T
    rho_a, phase = mt1d_apparent(f, [77.0], Float64[])
    depth, rho = niblett_bostick(T, rho_a, phase)
    @test issorted(depth)
    @test all(≈(77.0; rtol = 1e-6), rho)

    @test_throws ArgumentError niblett_bostick([1.0, 2.0], [1.0], [45.0])
end
