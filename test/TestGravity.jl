using Test
using LinearAlgebra
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

@testset "gravity_cell_sensitivity: finite, non-negative, matches prism_gz" begin
    g = PriorGrid(fill(400.0, 5), fill(400.0, 5), fill(200.0, 6);
                  origin = [-1000.0, -1000.0, 0.0])
    # station on the surface, directly above the centre column
    sx = [g.cx[3]]
    sy = [g.cy[3]]
    sz = [g.z[1] - 1.0]

    S = gravity_cell_sensitivity(g, sx, sy, sz)
    @test size(S) == size(g)
    @test all(isfinite, S)
    @test all(>=(0), S)

    A = gravity_matrix(g, sx, sy, sz)
    @test vec(S) ≈ vec(sqrt.(sum(abs2, A; dims = 1))) rtol = 1e-12

    # a single station: column norm is |prism_gz| in mGal
    i, j, k = 3, 3, 2
    expected = M_S2_TO_MGAL * abs(prism_gz(g.x[i] - sx[1], g.x[i+1] - sx[1],
                                           g.y[j] - sy[1], g.y[j+1] - sy[1],
                                           g.z[k] - sz[1], g.z[k+1] - sz[1]))
    @test S[i, j, k] ≈ expected rtol = 1e-12
end

@testset "gravity_cell_sensitivity: strictly decreasing with depth" begin
    # equal-size cells, observer on the surface above one column
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), fill(250.0, 8);
                  origin = [-1000.0, -1000.0, 0.0])
    sx = [g.cx[2]]
    sy = [g.cy[2]]
    sz = [g.z[1] - 1.0]
    S = gravity_cell_sensitivity(g, sx, sy, sz)
    col = S[2, 2, :]
    @test all(>(0), col)
    @test all(diff(col) .< 0)

    # off-axis: a point-mass gz = z/(r_h²+z²)^{3/2} peaks at z = r_h/√2, so a
    # corner column is *not* monotone. The property we need is that it still
    # varies with depth (unlike an extruded map).
    col_edge = S[1, 1, :]
    @test all(isfinite, col_edge)
    @test maximum(col_edge) > minimum(col_edge)
end

@testset "gravity_cell_sensitivity: validation" begin
    g = PriorGrid([100.0], [100.0], [100.0])
    @test_throws ArgumentError gravity_cell_sensitivity(g, [0.0, 1.0], [0.0], [0.0])
    @test_throws ArgumentError gravity_cell_sensitivity(g, Float64[], Float64[], Float64[])
    @test_throws ArgumentError gravity_cell_sensitivity(g, [0.0], [0.0], [0.0]; units = :furlongs)
end

@testset "gravity_cell_sensitivity: matches gravity_matrix column norms" begin
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(250.0, 4);
                  origin = [-1500.0, -1500.0, 0.0])
    sx = [0.0, 750.0, -750.0]
    sy = [0.0, 0.0, 500.0]
    sz = fill(0.0, 3)

    A = gravity_matrix(g, sx, sy, sz)
    sens = gravity_cell_sensitivity(g, sx, sy, sz)
    @test size(sens) == size(g)
    @test all(isfinite, sens)
    @test all(>=(0), sens)
    @test vec(sens) ≈ [norm(view(A, :, c)) for c in 1:ncells(g)] rtol = 1e-12

    A_si = gravity_matrix(g, sx, sy, sz; units = :si)
    sens_si = gravity_cell_sensitivity(g, sx, sy, sz; units = :si)
    @test vec(sens_si) ≈ [norm(view(A_si, :, c)) for c in 1:ncells(g)] rtol = 1e-12
    @test sens ≈ M_S2_TO_MGAL .* sens_si rtol = 1e-12
end

@testset "gravity_cell_sensitivity: deeper cells less sensitive" begin
    # equal-volume layers: the Plouff kernel must decay with depth under a
    # station. A graded mesh can offset that via cell thickening, so this uses
    # uniform dz.
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(250.0, 4);
                  origin = [-1500.0, -1500.0, 0.0])
    sens = gravity_cell_sensitivity(g, [0.0], [0.0], [0.0])
    @test all(isfinite, sens)
    # cells straddling the station (cx/cy nearest 0)
    @test sens[3, 3, 1] > sens[3, 3, 2] > sens[3, 3, 3] > sens[3, 3, 4]
    @test sens[4, 4, 1] > sens[4, 4, 4]
    @test maximum(sens[3, 3, :]) > minimum(sens[3, 3, :])

    @test_throws ArgumentError gravity_cell_sensitivity(g, [0.0, 1.0], [0.0], [0.0])
    @test_throws ArgumentError gravity_cell_sensitivity(g, Float64[], Float64[], Float64[])
    @test_throws ArgumentError gravity_cell_sensitivity(g, [0.0], [0.0], [0.0]; units = :furlongs)
end

@testset "gravity_cell_sensitivity: station on a cell face stays finite" begin
    g = PriorGrid([100.0, 100.0], [100.0, 100.0], [100.0]; origin = [-100.0, -100.0, 0.0])
    sens = gravity_cell_sensitivity(g, [0.0], [0.0], [0.0])
    @test all(isfinite, sens)
end
