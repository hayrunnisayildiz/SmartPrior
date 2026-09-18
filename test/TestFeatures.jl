using Test
using SmartPriorMT
using Statistics

# a small but non-uniform grid, so mesh grading bugs have somewhere to show up
function _test_grid()
    dx = [400.0, 400.0, 400.0, 400.0, 600.0, 900.0]
    dy = [400.0, 400.0, 400.0, 400.0, 600.0]
    dz = [50.0, 75.0, 110.0, 165.0, 250.0, 375.0]
    return PriorGrid(dx, dy, dz; origin = [-1550.0, -1100.0, 0.0])
end

function _test_sites(g)
    sx = [-600.0, 0.0, 600.0]
    sy = [-300.0, 0.0, 300.0]
    T = 10 .^ range(-2, 2; length = 12)
    f = 1 ./ T
    rho_a = Matrix{Float64}(undef, length(T), 3)
    phase = similar(rho_a)
    for (t, ρ) in enumerate([50.0, 100.0, 200.0])
        a, p = mt1d_apparent(f, [ρ], Float64[])
        rho_a[:, t] .= a
        phase[:, t] .= p
    end
    return MTSites(sx, sy, T, rho_a, phase)
end

@testset "GravityObs and MTSites validation" begin
    @test_throws ArgumentError GravityObs([0.0], [0.0], [0.0], [1.0], [1.0, 2.0])
    @test_throws ArgumentError GravityObs(Float64[], Float64[], Float64[], Float64[], Float64[])

    o = GravityObs([0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 2.0], [0.1, 0.1])
    @test length(o) == 2

    T = [1.0, 10.0]
    @test_throws ArgumentError MTSites([0.0], [0.0], T, zeros(3, 1), zeros(2, 1))
    @test_throws ArgumentError MTSites([0.0], [0.0, 1.0], T, zeros(2, 1), zeros(2, 1))
    s = MTSites([0.0], [0.0], T, fill(100.0, 2, 1), fill(45.0, 2, 1))
    @test nsites(s) == 1
end

@testset "standardize" begin
    A = [1.0 2.0; 3.0 4.0]
    S = standardize(A)
    @test mean(S) ≈ 0 atol = 1e-12
    @test std(S) ≈ 1 rtol = 1e-12

    # a constant field has no information and must not divide by zero
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
    @test all(10.0 .<= M .<= 20.0)          # interpolation, never extrapolation in value
    # cells near the first station lean towards its value
    @test M[1, 3] < M[6, 3]

    # a single station gives a constant field
    @test all(≈(5.0), idw_to_grid(g, [0.0], [0.0], [5.0]))

    # identical values give that value everywhere regardless of layout
    @test all(≈(3.0), idw_to_grid(g, sx, sy, [3.0, 3.0]))

    @test_throws ArgumentError idw_to_grid(g, sx, sy, [1.0])
    @test_throws ArgumentError idw_to_grid(g, sx, sy, v; power = 0.0)
    @test_throws ArgumentError idw_to_grid(g, sx, sy, v; smoothing = -1.0)
end

@testset "gaussian_smooth_xy preserves constants on a graded mesh" begin
    # the property the renormalisation exists for: no edge darkening, and no
    # dependence on how the mesh grades
    g = _test_grid()
    nx, ny, _ = size(g)
    @test all(≈(2.5), gaussian_smooth_xy(g, fill(2.5, nx, ny), 1000.0))
    @test all(≈(-7.0), gaussian_smooth_xy(g, fill(-7.0, nx, ny), 200.0))
end

@testset "gaussian_smooth_xy spreads a spike" begin
    # peak location and symmetry only hold on a uniform mesh; on a graded one the
    # per-cell weight sum differs and the maximum can land a cell off the spike
    g = PriorGrid(fill(400.0, 7), fill(400.0, 7), [100.0])
    spike = zeros(7, 7)
    spike[4, 4] = 1.0

    sm = gaussian_smooth_xy(g, spike, 800.0)
    @test sm[4, 4] < 1.0
    @test sm[4, 4] == maximum(sm)
    @test sm[3, 4] ≈ sm[5, 4] rtol = 1e-12
    @test sm[4, 3] ≈ sm[4, 5] rtol = 1e-12
    @test sm[3, 4] > sm[2, 4] > 0

    # a wider kernel flattens the field: the peak drops and the peak-to-trough
    # range shrinks. Renormalising per cell means the kernel does not conserve
    # mass, so the far-field value is not monotonic in sigma and the range is the
    # property worth asserting.
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

    # an exactly linear field must give the exact slope everywhere, which only
    # holds if the non-uniform spacing is handled properly
    slope = 0.003
    M = [slope * g.cx[i] for i in 1:nx, j in 1:ny]
    dx, dy = gradient_xy(g, M)
    @test all(≈(slope; rtol = 1e-10), dx)
    @test all(≈(0.0; atol = 1e-14), dy)

    M2 = [-0.002 * g.cy[j] for i in 1:nx, j in 1:ny]
    dx2, dy2 = gradient_xy(g, M2)
    @test all(≈(0.0; atol = 1e-14), dx2)
    @test all(≈(-0.002; rtol = 1e-10), dy2)

    # a degenerate axis returns zeros rather than failing
    g1 = PriorGrid([100.0], [100.0, 100.0], [100.0])
    d1x, d1y = gradient_xy(g1, [1.0 2.0])
    @test all(iszero, d1x)
    @test size(d1y) == (1, 2)
end

@testset "nb_baseline recovers a half-space" begin
    g = _test_grid()
    T = 10 .^ range(-2, 3; length = 30)
    f = 1 ./ T
    ρ_true = 120.0
    a, p = mt1d_apparent(f, [ρ_true], Float64[])

    sites = MTSites([-500.0, 500.0], [0.0, 0.0], T,
                    hcat(a, a), hcat(p, p))

    base = nb_baseline(g, sites)
    @test size(base) == size(g)
    # both sites see the same uniform earth, so every cell must land on it
    @test all(≈(log10(ρ_true); rtol = 1e-3), base)
end

@testset "nb_baseline blends laterally between differing sites" begin
    g = _test_grid()
    T = 10 .^ range(-2, 3; length = 30)
    f = 1 ./ T

    a1, p1 = mt1d_apparent(f, [10.0], Float64[])
    a2, p2 = mt1d_apparent(f, [1000.0], Float64[])
    sites = MTSites([-1200.0, 1200.0], [0.0, 0.0], T,
                    hcat(a1, a2), hcat(p1, p2))

    base = nb_baseline(g, sites)
    # the conductive site is at negative x, so the field must increase with x
    @test base[1, 3, 1] < base[6, 3, 1]
    @test all(log10(10.0) - 0.1 .<= base .<= log10(1000.0) + 0.1)
end

@testset "truth-free residual_span helpers" begin
    @test residual_span_half_band((0.0, 4.0)) == 2.0
    @test residual_span_half_band((0.5, 4.5)) == 2.0
    @test_throws ArgumentError residual_span_half_band((4.0, 0.0))

    uniform = fill(2.0, 2, 4, 3)
    @test nb_baseline_lateral_std(uniform) == 0.0
    @test residual_span_from_baseline(uniform; k = 3.0, floor = 1.0, ceil = 5.0) == 1.0

    varying = zeros(1, 4, 2)
    varying[1, 1, :] .= 1.0
    varying[1, 2, :] .= 2.0
    varying[1, 3, :] .= 3.0
    varying[1, 4, :] .= 4.0
    σ = nb_baseline_lateral_std(varying)
    @test σ ≈ std([1.0, 2.0, 3.0, 4.0])
    @test residual_span_from_baseline(varying; k = 2.0, floor = 0.5, ceil = 10.0) ≈ 2.0 * σ
    @test residual_span_from_baseline(varying; k = 100.0, floor = 1.0, ceil = 2.5) == 2.5
    @test_throws ArgumentError residual_span_from_baseline(varying; k = 0.0)
    @test_throws ArgumentError residual_span_from_baseline(varying; floor = 3.0, ceil = 1.0)
end

@testset "channel builders" begin
    g = _test_grid()
    nx, ny, nz = size(g)
    sites = _test_sites(g)

    obs = GravityObs(collect(range(-1200.0, 1200.0; length = 9)),
                     zeros(9), zeros(9),
                     [0.0, 1.0, 3.0, 6.0, 8.0, 6.0, 3.0, 1.0, 0.0],
                     fill(0.2, 9))

    gc, gn = gravity_channels(g, obs)
    @test length(gc) == length(gn)
    @test gn[1:3] == ["gravity", "gravity_dx", "gravity_dy"]
    @test "gravity_long" in gn
    @test !("gravity_sensitivity" in gn)
    @test all(c -> size(c) == (nx, ny, nz), gc)
    @test all(c -> all(isfinite, c), gc)

    # default-off: optional depth-aware channel does not change the historical count
    gc_off, gn_off = gravity_channels(g, obs; sensitivity = false)
    @test gn_off == gn
    @test length(gc_off) == length(gc)

    gc_on, gn_on = gravity_channels(g, obs; sensitivity = true)
    @test length(gc_on) == length(gc_off) + 1
    @test gn_on[1:(end - 1)] == gn_off
    @test gn_on[end] == "gravity_sensitivity"
    @test all(gc_on[i] == gc_off[i] for i in eachindex(gc_off))
    sens_ch = gc_on[end]
    @test all(isfinite, sens_ch)
    # not an extruded copy: must vary with depth along a column
    @test std(sens_ch[3, 3, :]) > 0
    @test std(gc_off[1][3, 3, :]) == 0  # extruded gravity is z-constant
    @test gravity_sensitivity_channel(g, obs) == sens_ch
    @test std(gravity_cell_sensitivity(g, obs)[3, 3, :]) > 0
    @test all(isfinite, gravity_cell_sensitivity(g, obs))
    @test all(>=(0), gravity_cell_sensitivity(g, obs))

    surface = fill(0.0, nx, ny)
    surface[1, :] .= 120.0
    tc, tn = topography_channels(g, surface)
    @test tn == ["surface_z", "depth_below_surface"]
    @test all(c -> size(c) == (nx, ny, nz), tc)
    # a cell under a higher surface is shallower below it
    @test tc[2][1, 1, 2] < tc[2][2, 1, 2]
    @test_throws DimensionMismatch topography_channels(g, zeros(2, 2))

    dc, dn = depth_channels(g; rho_ref = 100.0, period_max = 100.0)
    @test dn == ["log_depth", "depth_over_skin"]
    @test issorted(vec(dc[2][1, 1, :]))
    @test dc[2][1, 1, 1] > 0
    @test_throws ArgumentError depth_channels(g; rho_ref = 0.0)
    @test_throws ArgumentError depth_channels(g; period_max = -1.0)

    cc, cn = coverage_channels(g, sites.x, sites.y)
    @test cn == ["site_distance"]
    @test all(cc[1] .>= 0)
    # the far corner is further from every site than the middle
    @test cc[1][6, 5, 1] > cc[1][3, 3, 1]
    @test_throws ArgumentError coverage_channels(g, [0.0], Float64[])
end

@testset "nearest-sample geochemistry and lithology channels" begin
    g = PriorGrid(fill(100.0, 4), fill(100.0, 4), fill(50.0, 3);
                  origin = [0.0, 0.0, 0.0])
    nx, ny, nz = size(g)

    # two plan-view samples: left cell should inherit the left value
    samples = PointSamples([50.0, 350.0], [200.0, 200.0],
                           [10.0 100.0; 1000.0 10.0],
                           ["CU", "NI"])
    gc, gn = geochemistry_channels(g, samples)
    @test gn == ["geochem_CU", "geochem_NI"]
    @test all(c -> size(c) == (nx, ny, nz), gc)
    @test all(c -> all(isfinite, c), gc)
    # log10 then standardise: left side of CU is the smaller concentration
    # so after standardise the left column mean is below the right
    @test mean(gc[1][1, :, :]) < mean(gc[1][4, :, :])

    # missing nickel on the left sample must not paint from the copper-only row
    sparse = PointSamples([50.0, 350.0], [200.0, 200.0],
                          [10.0 NaN; 1000.0 5.0],
                          ["CU", "NI"])
    sc, sn = nearest_sample_channels(g, sparse; standardize_values = false)
    @test sn == ["CU", "NI"]
    @test all(sc[2] .== 5.0)   # only the right sample has Ni, so it wins everywhere

    lith = LabelSamples([50.0, 50.0, 350.0, 350.0, 50.0],
                        [50.0, 350.0, 50.0, 350.0, 200.0],
                        ["oliviini", "oliviini", "maata", "maata", "rare_dyke"];
                        z = [25.0, 25.0, 25.0, 25.0, 25.0])
    lc, ln = lithology_channels(g, lith; min_frequency = 0.25)
    @test "lith_OLIVIINI" in ln
    @test "lith_MAATA" in ln
    @test "lith_OTHER" in ln          # rare_dyke collapsed
    @test !("lith_RARE_DYKE" in ln)
    @test all(c -> size(c) == (nx, ny, nz), lc)
    # one-hot: each cell's channels sum to 1
    stacked = reduce((a, b) -> a .+ b, lc)
    @test all(x -> x ≈ 1.0, stacked)

    # GTK dBase Latin-1 Ä (0xC4) must not throw in the tokenizer
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
    # 3-D coverage varies with depth, unlike the extruded MT channel
    @test std(cc3[1][2, 2, :]) > 0

    @test_throws ArgumentError PointSamples([0.0], [0.0], zeros(1, 1), ["a", "b"])
    @test_throws ArgumentError LabelSamples([0.0], [0.0, 1.0], ["a"])
end

@testset "build_features assembles only what it is given" begin
    g = _test_grid()
    nx, ny, nz = size(g)
    sites = _test_sites(g)
    obs = GravityObs([-500.0, 500.0], [0.0, 0.0], [0.0, 0.0], [2.0, -1.0], [0.1, 0.1])
    surface = zeros(nx, ny)

    minimal = build_features(g; coordinates = true)
    @test minimal.names == ["x_norm", "y_norm", "z_norm", "log_depth", "depth_over_skin"]
    @test size(minimal) == (nx, ny, nz, 5)

    full = build_features(g; gravity = obs, surface_z = surface, sites = sites)
    @test "gravity" in full.names
    @test !("gravity_sensitivity" in full.names)
    @test "surface_z" in full.names
    @test "site_distance" in full.names
    @test "baseline" in full.names
    @test nchannels(full) == length(full.names)
    @test all(isfinite, full.data)

    # gravity_sensitivity defaults OFF: same channel count as the historical stack
    full_off = build_features(g; gravity = obs, surface_z = surface, sites = sites,
                              gravity_sensitivity = false)
    @test nchannels(full_off) == nchannels(full)
    @test full_off.names == full.names

    full_on = build_features(g; gravity = obs, surface_z = surface, sites = sites,
                             gravity_sensitivity = true)
    @test nchannels(full_on) == nchannels(full) + 1
    @test "gravity_sensitivity" in full_on.names
    @test all(isfinite, full_on.data)
    col = full_on["gravity_sensitivity"][3, 3, :]
    @test all(isfinite, col)
    @test std(col) > 0
    # without gravity observations the flag is a no-op
    no_grav = build_features(g; coordinates = true, gravity_sensitivity = true)
    @test nchannels(no_grav) == nchannels(minimal)
    @test !("gravity_sensitivity" in no_grav.names)

    # dropping the coordinate channels is how a transferable model is trained
    no_coords = build_features(g; gravity = obs, sites = sites, coordinates = false)
    @test !("x_norm" in no_coords.names)
    @test nchannels(no_coords) == nchannels(full) - 3 - 2  # no coords, no topography

    # an explicit baseline overrides the automatic Niblett-Bostick one
    custom = fill(2.0, nx, ny, nz)
    withbase = build_features(g; baseline = custom, coordinates = false)
    @test "baseline" in withbase.names
    @test_throws DimensionMismatch build_features(g; baseline = zeros(2, 2, 2))

    @test full["gravity"] isa AbstractArray{Float64,3}
    @test_throws KeyError full["not_a_channel"]

    off = build_features(g; gravity = obs, coordinates = false)
    on = build_features(g; gravity = obs, coordinates = false,
                        gravity_sensitivity = true)
    @test nchannels(on) == nchannels(off) + 1
    @test !("gravity_sensitivity" in off.names)
    @test "gravity_sensitivity" in on.names
    @test off.names == filter(!=("gravity_sensitivity"), on.names)
    for name in off.names
        @test off[name] == on[name]
    end
    @test std(on["gravity_sensitivity"][1, 2, :]) > 0
    @test std(on["gravity"][1, 2, :]) == 0
end

@testset "feature_matrix layout matches vec of the grid" begin
    g = _test_grid()
    nx, ny, nz = size(g)
    s = build_features(g; coordinates = true)

    M = feature_matrix(s)
    @test size(M) == (nchannels(s), nx * ny * nz)

    # row k of the matrix must be vec of channel k, so a network output can be
    # reshaped straight back onto the grid without reordering
    for k in 1:nchannels(s)
        @test M[k, :] == vec(s.data[:, :, :, k])
    end
end

@testset "FeatureStack validation" begin
    @test_throws ArgumentError FeatureStack(zeros(2, 2, 2, 2), ["a"])
    @test_throws ArgumentError FeatureStack(zeros(2, 2, 2, 2), ["a", "a"])
end
