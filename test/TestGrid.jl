using Test
using SmartPriorMT
using MTGeophysics: write_ws3d_model, load_ws3d_model

@testset "PriorGrid" begin
    dx = [100.0, 200.0, 400.0]
    dy = [50.0, 50.0]
    dz = [10.0, 20.0, 40.0, 80.0]
    g = PriorGrid(dx, dy, dz; origin = [-350.0, -50.0, 0.0])

    @test size(g) == (3, 2, 4)
    @test ncells(g) == 24
    @test g.x[1] ≈ -350.0
    @test g.x[end] ≈ 350.0
    @test g.cx ≈ [-300.0, -150.0, 150.0]
    @test g.cz ≈ [5.0, 20.0, 50.0, 110.0]

    @test containing_cell(g, -300.0, -25.0, 5.0) == LinearIndices(size(g))[1, 1, 1]
    @test containing_cell(g, 349.9, 49.9, 149.9) == ncells(g)
    @test containing_cell(g, -999.0, 0.0, 5.0) == 0
    @test containing_cell(g, 0.0, 0.0, NaN) == 0

    V = cell_volumes(g)
    @test size(V) == (3, 2, 4)
    @test V[3, 1, 4] ≈ 400.0 * 50.0 * 80.0
    @test sum(V) ≈ sum(dx) * sum(dy) * sum(dz)

    X, Y, Z = cell_centers(g)
    @test X[2, 1, 1] ≈ -150.0
    @test Y[1, 2, 1] ≈ 25.0
    @test Z[1, 1, 3] ≈ 50.0

    D = depth_below_top(g)
    @test D[1, 1, 1] ≈ 5.0
    @test D[1, 1, 4] ≈ 110.0

    Xn, Yn, Zn = normalized_centers(g)
    @test all(-1 .<= Xn .<= 1)
    @test all(-1 .<= Zn .<= 1)
    @test Xn[1, 1, 1] < Xn[3, 1, 1]

    # cached because `median` sorts in place, which Zygote rejects inside a
    # differentiated function
    @test g.h_median == 100.0     # median of [100, 200, 400, 50, 50]
end

@testset "PriorGrid: origin offset does not leak into depth" begin
    # a grid whose datum sits 500 m above the ground surface, as on topography
    # models where the top layers are air
    g = PriorGrid([100.0], [100.0], [250.0, 250.0]; origin = [0.0, 0.0, -500.0])
    @test g.cz ≈ [-375.0, -125.0]
    @test depth_below_top(g)[1, 1, :] ≈ [125.0, 375.0]
end

@testset "PriorGrid: validation" begin
    @test_throws ArgumentError PriorGrid(Float64[], [1.0], [1.0])
    @test_throws ArgumentError PriorGrid([-1.0], [1.0], [1.0])
    @test_throws ArgumentError PriorGrid([1.0], [1.0], [0.0])
    @test_throws ArgumentError PriorGrid([1.0], [1.0], [1.0]; origin = [0.0, 0.0])
end

@testset "PriorGrid: round trip through a WS3D model" begin
    dx = [500.0, 500.0, 500.0]
    dy = [400.0, 400.0]
    dz = [100.0, 200.0]
    origin = [-750.0, -400.0, 0.0]
    A = fill(2.0, 3, 2, 2)

    path = joinpath(mktempdir(), "grid_roundtrip.rho")
    write_ws3d_model(path, dx, dy, dz, A, origin)
    m = load_ws3d_model(path)
    g = PriorGrid(m)

    @test size(g) == (3, 2, 2)
    @test g.dx ≈ dx
    @test g.dy ≈ dy
    @test g.dz ≈ dz
    @test g.origin ≈ origin
    @test g.cx ≈ m.cx
    @test g.cy ≈ m.cy
    @test g.cz ≈ m.cz
end
