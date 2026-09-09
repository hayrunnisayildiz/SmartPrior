using Test
using SmartPriorMT
using SmartPriorMT: G_GRAV, M_S2_TO_MGAL

@testset "prism_gz: sign and symmetry" begin
    # a prism directly below the station must attract downward, i.e. add to g
    g_below = prism_gz(-500.0, 500.0, -500.0, 500.0, 100.0, 600.0)
    @test g_below > 0

    # shifting the station laterally weakens the vertical component
    g_offset = prism_gz(1500.0, 2500.0, -500.0, 500.0, 100.0, 600.0)
    @test 0 < g_offset < g_below

    # symmetric about the prism centre
    g_left = prism_gz(-2500.0, -1500.0, -500.0, 500.0, 100.0, 600.0)
    @test g_left ≈ g_offset rtol = 1e-12

    # and it falls off with depth
    g_deep = prism_gz(-500.0, 500.0, -500.0, 500.0, 5100.0, 5600.0)
    @test g_deep < g_below
end

@testset "prism_gz: converges to a point mass in the far field" begin
    # a small prism far away should look like its total mass concentrated at its
    # centre: gz = G*M*z/r^3
    a = 50.0
    depth = 20_000.0
    x1, x2 = -a, a
    y1, y2 = -a, a
    z1, z2 = depth - a, depth + a

    numeric = prism_gz(x1, x2, y1, y2, z1, z2)

    mass = (x2 - x1) * (y2 - y1) * (z2 - z1)     # unit density
    r = depth
    analytic = G_GRAV * mass * depth / r^3

    @test numeric ≈ analytic rtol = 1e-4
end

@testset "prism_gz: additive over a split" begin
    whole = prism_gz(-1000.0, 1000.0, -800.0, 800.0, 200.0, 1200.0)
    lower = prism_gz(-1000.0, 1000.0, -800.0, 800.0, 200.0, 700.0)
    upper = prism_gz(-1000.0, 1000.0, -800.0, 800.0, 700.0, 1200.0)
    @test whole ≈ lower + upper rtol = 1e-10
end

@testset "gravity_matrix and forward_gravity" begin
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(250.0, 4);
                  origin = [-1500.0, -1500.0, 0.0])

    sx = [0.0, 750.0, -750.0]
    sy = [0.0, 0.0, 500.0]
    sz = fill(0.0, 3)

    A = gravity_matrix(g, sx, sy, sz)
    @test size(A) == (3, ncells(g))
    @test all(isfinite, A)

    # a uniform positive contrast raises gravity everywhere, most at the station
    # over the middle of the block
    rho = fill(200.0, size(g))
    d = forward_gravity(A, rho)
    @test length(d) == 3
    @test all(d .> 0)
    @test d[1] == maximum(d)

    # linearity: doubling the contrast doubles the response
    @test forward_gravity(A, 2 .* rho) ≈ 2 .* d rtol = 1e-12

    # the flat vector ordering must match vec() of the [nx, ny, nz] array
    @test forward_gravity(A, vec(rho)) ≈ d

    # a single anomalous cell is picked up by the nearest station
    single = zeros(size(g))
    single[2, 2, 1] = 500.0
    ds = forward_gravity(A, single)
    @test ds[3] > ds[2]

    # mGal is 1e5 times the SI value
    A_si = gravity_matrix(g, sx, sy, sz; units = :si)
    @test A ≈ M_S2_TO_MGAL .* A_si rtol = 1e-12
end

@testset "gravity_matrix: validation" begin
    g = PriorGrid([100.0], [100.0], [100.0])
    @test_throws ArgumentError gravity_matrix(g, [0.0, 1.0], [0.0], [0.0])
    @test_throws ArgumentError gravity_matrix(g, [0.0], [0.0], [0.0]; units = :furlongs)
    A = gravity_matrix(g, [0.0], [0.0], [0.0])
    @test_throws DimensionMismatch forward_gravity(A, zeros(2, 2, 2))
end

@testset "gravity_matrix: station on a cell face stays finite" begin
    # the closed form has log and atan singularities on prism faces; the epsilon
    # guard has to keep a station sitting exactly on the grid top usable
    g = PriorGrid([100.0, 100.0], [100.0, 100.0], [100.0]; origin = [-100.0, -100.0, 0.0])
    A = gravity_matrix(g, [0.0], [0.0], [0.0])
    @test all(isfinite, A)
end
