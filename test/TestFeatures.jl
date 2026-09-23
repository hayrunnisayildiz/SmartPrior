using Test
using SmartPrior
using Statistics

function _test_grid()
    dx = [400.0, 400.0, 400.0, 400.0, 600.0, 900.0]
    dy = [400.0, 400.0, 400.0, 400.0, 600.0]
    dz = [50.0, 75.0, 110.0, 165.0, 250.0, 375.0]
    return PriorGrid(dx, dy, dz; origin = [-1550.0, -1100.0, 0.0])
end

@testset "standardize" begin
    A = [1.0 2.0; 3.0 4.0]
    S = standardize(A)
    @test mean(S) ≈ 0 atol = 1e-12
    @test std(S) ≈ 1 rtol = 1e-12
    @test all(iszero, standardize(fill(7.0, 3, 3)))
    @test_throws ArgumentError standardize([1.0, NaN])
end

@testset "extrude" begin
    M = [1.0 2.0; 3.0 4.0]
    E = extrude(M, 3)
    @test size(E) == (2, 2, 3)
    @test E[:, :, 1] == M
    @test E[:, :, 3] == M
end

@testset "idw_to_grid" begin
    g = _test_grid()
    sx = [-1000.0, 1000.0]
    sy = [0.0, 0.0]
    v = [10.0, 20.0]

    M = idw_to_grid(g, sx, sy, v)
    @test size(M) == (6, 5)
    @test all(10.0 .<= M .<= 20.0)
    @test M[1, 3] < M[6, 3]
    @test all(≈(5.0), idw_to_grid(g, [0.0], [0.0], [5.0]))
    @test all(≈(3.0), idw_to_grid(g, sx, sy, [3.0, 3.0]))
    @test_throws ArgumentError idw_to_grid(g, sx, sy, [1.0])
    @test_throws ArgumentError idw_to_grid(g, sx, sy, v; power = 0.0)
    @test_throws ArgumentError idw_to_grid(g, sx, sy, v; smoothing = -1.0)
end

@testset "gaussian_smooth_xy preserves constants on a graded mesh" begin
    g = _test_grid()
    nx, ny, _ = size(g)
    @test all(≈(2.5), gaussian_smooth_xy(g, fill(2.5, nx, ny), 1000.0))
    @test all(≈(-7.0), gaussian_smooth_xy(g, fill(-7.0, nx, ny), 200.0))
end

@testset "gaussian_smooth_xy spreads a spike" begin
    g = PriorGrid(fill(400.0, 7), fill(400.0, 7), [100.0])
    spike = zeros(7, 7)
    spike[4, 4] = 1.0

    sm = gaussian_smooth_xy(g, spike, 800.0)
    @test sm[4, 4] < 1.0
    @test sm[4, 4] == maximum(sm)
    @test sm[3, 4] ≈ sm[5, 4] rtol = 1e-12
    @test sm[4, 3] ≈ sm[4, 5] rtol = 1e-12
    @test sm[3, 4] > sm[2, 4] > 0

    wider = gaussian_smooth_xy(g, spike, 2000.0)
    @test wider[4, 4] < sm[4, 4]
    @test (maximum(wider) - minimum(wider)) < (maximum(sm) - minimum(sm))
end

@testset "gaussian_smooth_xy validation" begin
    g = _test_grid()
    nx, ny, _ = size(g)
    @test_throws DimensionMismatch gaussian_smooth_xy(g, zeros(2, 2), 100.0)
    @test_throws ArgumentError gaussian_smooth_xy(g, zeros(nx, ny), 0.0)
end

@testset "gradient_xy" begin
    g = _test_grid()
    nx, ny, _ = size(g)

    slope = 0.003
    M = [slope * g.cx[i] for i in 1:nx, j in 1:ny]
    dx, dy = gradient_xy(g, M)
    @test all(≈(slope; rtol = 1e-10), dx)
    @test all(≈(0.0; atol = 1e-14), dy)

    M2 = [-0.002 * g.cy[j] for i in 1:nx, j in 1:ny]
    dx2, dy2 = gradient_xy(g, M2)
    @test all(≈(0.0; atol = 1e-14), dx2)
    @test all(≈(-0.002; rtol = 1e-10), dy2)

    g1 = PriorGrid([100.0], [100.0, 100.0], [100.0])
    d1x, d1y = gradient_xy(g1, [1.0 2.0])
    @test all(iszero, d1x)
    @test size(d1y) == (1, 2)
end

@testset "channel builders" begin
    g = _test_grid()
    nx, ny, nz = size(g)

    surface = fill(0.0, nx, ny)
    surface[1, :] .= 120.0
    tc, tn = topography_channels(g, surface)
    @test tn == ["surface_z", "depth_below_surface"]
    @test all(c -> size(c) == (nx, ny, nz), tc)
    @test tc[2][1, 1, 2] < tc[2][2, 1, 2]
    @test_throws DimensionMismatch topography_channels(g, zeros(2, 2))

    dc, dn = depth_channels(g; rho_ref = 100.0, period_max = 100.0)
    @test dn == ["log_depth", "depth_over_skin"]
    @test issorted(vec(dc[2][1, 1, :]))
    @test dc[2][1, 1, 1] > 0
    @test_throws ArgumentError depth_channels(g; rho_ref = 0.0)
    @test_throws ArgumentError depth_channels(g; period_max = -1.0)

    cc, cn = coverage_channels(g, [-600.0, 0.0, 600.0], [-300.0, 0.0, 300.0])
    @test cn == ["sample_distance"]
    @test all(cc[1] .>= 0)
    @test cc[1][6, 5, 1] > cc[1][3, 3, 1]
    @test_throws ArgumentError coverage_channels(g, [0.0], Float64[])
end

@testset "nearest-sample geochemistry and lithology channels" begin
    g = PriorGrid(fill(100.0, 4), fill(100.0, 4), fill(50.0, 3);
                  origin = [0.0, 0.0, 0.0])
    nx, ny, nz = size(g)

    samples = PointSamples([50.0, 350.0], [200.0, 200.0],
                           [10.0 100.0; 1000.0 10.0],
                           ["CU", "NI"])
    gc, gn = geochemistry_channels(g, samples)
    @test gn == ["geochem_CU", "geochem_NI"]
    @test all(c -> size(c) == (nx, ny, nz), gc)
    @test all(c -> all(isfinite, c), gc)
    @test mean(gc[1][1, :, :]) < mean(gc[1][4, :, :])

    sparse = PointSamples([50.0, 350.0], [200.0, 200.0],
                          [10.0 NaN; 1000.0 5.0],
                          ["CU", "NI"])
    sc, sn = nearest_sample_channels(g, sparse; standardize_values = false)
    @test sn == ["CU", "NI"]
    @test all(sc[2] .== 5.0)

    lith = LabelSamples([50.0, 50.0, 350.0, 350.0, 50.0],
                        [50.0, 350.0, 50.0, 350.0, 200.0],
                        ["oliviini", "oliviini", "maata", "maata", "rare_dyke"];
                        z = [25.0, 25.0, 25.0, 25.0, 25.0])
    lc, ln = lithology_channels(g, lith; min_frequency = 0.25)
    @test "lith_OLIVIINI" in ln
    @test "lith_MAATA" in ln
    @test "lith_OTHER" in ln
    @test !("lith_RARE_DYKE" in ln)
    @test all(c -> size(c) == (nx, ny, nz), lc)
    stacked = reduce((a, b) -> a .+ b, lc)
    @test all(x -> x ≈ 1.0, stacked)

    latin1_maa = String(UInt8[0x6d, 0x61, 0x61, 0xc4])
    lith_latin = LabelSamples([50.0, 50.0, 350.0, 350.0, 50.0],
                              [50.0, 350.0, 50.0, 350.0, 200.0],
                              ["oliviini", "oliviini", latin1_maa, latin1_maa, "rare_dyke"];
                              z = [25.0, 25.0, 25.0, 25.0, 25.0])
    lc_l, ln_l = lithology_channels(g, lith_latin; min_frequency = 0.25)
    @test "lith_MAA" in ln_l
    @test "lith_OLIVIINI" in ln_l
    @test "lith_OTHER" in ln_l

    cc3, cn3 = coverage_channels(g, lith.x, lith.y, lith.z; name = "sample_distance")
    @test cn3 == ["sample_distance"]
    @test size(cc3[1]) == (nx, ny, nz)
    @test std(cc3[1][2, 2, :]) > 0

    @test_throws ArgumentError PointSamples([0.0], [0.0], zeros(1, 1), ["a", "b"])
    @test_throws ArgumentError LabelSamples([0.0], [0.0, 1.0], ["a"])
end

@testset "build_features assembles only what it is given" begin
    g = _test_grid()
    nx, ny, nz = size(g)
    surface = zeros(nx, ny)

    minimal = build_features(g; coordinates = true)
    @test minimal.names == ["x_norm", "y_norm", "z_norm", "log_depth", "depth_over_skin"]
    @test size(minimal) == (nx, ny, nz, 5)

    with_topo = build_features(g; surface_z = surface)
    @test "surface_z" in with_topo.names
    @test "depth_below_surface" in with_topo.names
    @test nchannels(with_topo) == length(with_topo.names)
    @test all(isfinite, with_topo.data)

    no_coords = build_features(g; surface_z = surface, coordinates = false)
    @test !("x_norm" in no_coords.names)
    @test nchannels(no_coords) == 4  # log_depth, depth_over_skin, surface_z, depth_below_surface

    @test with_topo["surface_z"] isa AbstractArray{Float64,3}
    @test_throws KeyError with_topo["not_a_channel"]
end

@testset "feature_matrix layout matches vec of the grid" begin
    g = _test_grid()
    nx, ny, nz = size(g)
    s = build_features(g; coordinates = true)

    M = feature_matrix(s)
    @test size(M) == (nchannels(s), nx * ny * nz)
    for k in 1:nchannels(s)
        @test M[k, :] == vec(s.data[:, :, :, k])
    end
end

@testset "FeatureStack validation" begin
    @test_throws ArgumentError FeatureStack(zeros(2, 2, 2, 2), ["a"])
    @test_throws ArgumentError FeatureStack(zeros(2, 2, 2, 2), ["a", "a"])
end

@testset "append_channels" begin
    g = PriorGrid(fill(100.0, 2), fill(100.0, 2), [50.0])
    s = build_features(g; coordinates = false)
    extra = [ones(size(g)...)]
    s2 = append_channels(s, extra, ["extra"])
    @test "extra" in s2.names
    @test nchannels(s2) == nchannels(s) + 1
    @test_throws ArgumentError append_channels(s, extra, ["log_depth"])
end
