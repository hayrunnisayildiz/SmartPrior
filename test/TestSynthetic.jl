using Test
using SmartPriorMT
using Random
using Statistics

_synth_grid() = PriorGrid(fill(500.0, 8), fill(500.0, 8), fill(200.0, 6);
                          origin = [-2000.0, -2000.0, 0.0])

@testset "truth_halfspace" begin
    g = _synth_grid()
    m = truth_halfspace(g; log_rho = 2.3)
    @test size(m) == size(g)
    @test all(m.log_rho .== 2.3)
    # a homogeneous model must produce no gravity anomaly, or the contrast
    # convention is wrong somewhere
    @test all(m.density .== 0.0)

    obs = synth_gravity(m, [0.0, 500.0], [0.0, 0.0])
    @test all(≈(0.0; atol = 1e-12), obs.value)
end

@testset "SyntheticModel validation" begin
    g = _synth_grid()
    ok = fill(2.0, size(g))
    @test_throws DimensionMismatch SyntheticModel(g, zeros(2, 2, 2), ok)
    @test_throws DimensionMismatch SyntheticModel(g, ok, zeros(2, 2, 2))
end

@testset "add_layer" begin
    g = _synth_grid()
    # layer centres sit at 100, 300, 500, 700, 900, 1100
    m = add_layer(truth_halfspace(g; log_rho = 2.0);
                  ztop = 200.0, zbot = 600.0, log_rho = 1.0, density = 300.0)

    @test all(m.log_rho[:, :, 1] .== 2.0)      # centre 100, above ztop
    @test all(m.log_rho[:, :, 2] .== 1.0)      # centre 300
    @test all(m.log_rho[:, :, 3] .== 1.0)      # centre 500
    @test all(m.log_rho[:, :, 4] .== 2.0)      # centre 700, at or past zbot
    @test all(m.density[:, :, 2] .== 300.0)
    @test all(m.density[:, :, 1] .== 0.0)

    @test_throws ArgumentError add_layer(m; ztop = 600.0, zbot = 200.0, log_rho = 1.0)
end

@testset "add_block" begin
    g = _synth_grid()
    m = add_block(truth_halfspace(g; log_rho = 2.5);
                  x = (-1000.0, 0.0), y = (-500.0, 500.0), z = (300.0, 700.0),
                  log_rho = 0.7, density = -200.0)

    body = m.log_rho .== 0.7
    @test any(body)
    @test !all(body)
    # the body must be a contiguous box, so the count is the product of the
    # per-axis cell counts it spans
    nxb = count(x -> -1000.0 <= x <= 0.0, g.cx)
    nyb = count(y -> -500.0 <= y <= 500.0, g.cy)
    nzb = count(z -> 300.0 <= z <= 700.0, g.cz)
    @test count(body) == nxb * nyb * nzb
    @test all(m.density[body] .== -200.0)
    @test all(m.density[.!body] .== 0.0)

    @test_throws ArgumentError add_block(m; x = (0.0, -1.0), y = (0.0, 1.0),
                                        z = (0.0, 1.0), log_rho = 1.0)
end

@testset "resistivity and density are independent" begin
    # a prior that only works when the two agree has learned nothing, so the
    # generators must allow a conductive-and-light body as readily as a
    # conductive-and-dense one
    g = _synth_grid()
    dense = add_block(truth_halfspace(g); x = (-1000.0, 0.0), y = (-500.0, 500.0),
                      z = (300.0, 700.0), log_rho = 0.7, density = 400.0)
    light = add_block(truth_halfspace(g); x = (-1000.0, 0.0), y = (-500.0, 500.0),
                      z = (300.0, 700.0), log_rho = 0.7, density = -400.0)

    @test dense.log_rho == light.log_rho
    @test dense.density == -light.density

    obs_d = synth_gravity(dense, [-500.0], [0.0])
    obs_l = synth_gravity(light, [-500.0], [0.0])
    @test obs_d.value[1] > 0
    @test obs_l.value[1] < 0
    @test obs_d.value[1] ≈ -obs_l.value[1]
end

@testset "add_dipping_slab" begin
    g = PriorGrid(fill(400.0, 20), fill(400.0, 4), fill(200.0, 10);
                  origin = [-4000.0, -800.0, 0.0])
    m = add_dipping_slab(truth_halfspace(g; log_rho = 2.5);
                         x0 = 0.0, z0 = 1000.0, dip_deg = 30.0, thickness = 400.0,
                         log_rho = 0.5, density = 250.0)

    body = m.log_rho .== 0.5
    @test any(body)

    # a dipping slab must get deeper across the profile; take the mean depth of
    # the body in the first and last populated x columns
    depth_of(i) = mean(g.cz[k] for k in axes(body, 3), j in axes(body, 2) if body[i, j, k])
    cols = [i for i in axes(body, 1) if any(body[i, :, :])]
    @test length(cols) > 3
    @test depth_of(cols[end]) != depth_of(cols[1])

    # invariant along strike, which is what :y means
    @test all(body[:, 1, :] .== body[:, end, :])

    # a vertical slab dips at 90 degrees and must occupy the same columns at
    # every depth. thickness is chosen so no cell centre sits on the slab's own
    # boundary, where the comparison would turn on floating-point noise
    v = add_dipping_slab(truth_halfspace(g); x0 = 0.0, z0 = 1000.0, dip_deg = 90.0,
                         thickness = 900.0, log_rho = 0.5)
    vbody = v.log_rho .== 0.5
    @test any(vbody)
    @test all(vbody[:, 1, 1] .== vbody[:, 1, end])

    @test_throws ArgumentError add_dipping_slab(m; x0 = 0.0, z0 = 0.0, dip_deg = 30.0,
                                               thickness = 0.0, log_rho = 1.0)
    @test_throws ArgumentError add_dipping_slab(m; x0 = 0.0, z0 = 0.0, dip_deg = 0.0,
                                               thickness = 100.0, log_rho = 1.0)
    @test_throws ArgumentError add_dipping_slab(m; x0 = 0.0, z0 = 0.0, dip_deg = 30.0,
                                               thickness = 100.0, log_rho = 1.0,
                                               strike = :z)
end

@testset "add_topography marks air as NaN" begin
    g = _synth_grid()
    nx, ny, _ = size(g)
    # ground at 400 m depth on one half, at the datum on the other
    elev = [i <= nx ÷ 2 ? 400.0 : 0.0 for i in 1:nx, j in 1:ny]
    m = add_topography(truth_halfspace(g; log_rho = 2.0); elevation = elev)

    # centres at 100 and 300 are above 400, so they are air on the raised half
    @test all(isnan, m.log_rho[1:nx÷2, :, 1])
    @test all(isnan, m.log_rho[1:nx÷2, :, 2])
    @test all(isfinite, m.log_rho[1:nx÷2, :, 3])
    @test all(isfinite, m.log_rho[nx÷2+1:end, :, 1])

    # air carries no density, so it must not generate an anomaly
    @test all(m.density[isnan.(m.log_rho)] .== 0.0)

    @test_throws DimensionMismatch add_topography(m; elevation = zeros(2, 2))
end

@testset "synth_gravity" begin
    g = _synth_grid()
    m = add_block(truth_halfspace(g); x = (-500.0, 500.0), y = (-500.0, 500.0),
                  z = (100.0, 500.0), log_rho = 1.0, density = 500.0)

    sx = collect(-1500.0:500.0:1500.0)
    sy = zeros(length(sx))
    obs = synth_gravity(m, sx, sy)

    @test obs isa GravityObs
    @test length(obs.value) == length(sx)
    @test all(obs.z .== g.z[1])                    # stations default to the datum
    # a dense body must give a positive anomaly, peaking above itself
    @test all(obs.value .> 0)
    @test argmax(obs.value) == findfirst(≈(0.0), sx)
    # the default error is a fraction of the anomaly range, never zero
    @test all(obs.err .> 0)

    noisy = synth_gravity(m, sx, sy; noise = 0.05, rng = Xoshiro(1))
    @test noisy.value != obs.value
    @test all(noisy.err .== 0.05)
    @test maximum(abs.(noisy.value .- obs.value)) < 0.3     # ~6 sigma

    # an explicit error overrides both defaults
    @test all(synth_gravity(m, sx, sy; noise = 0.05, err = 0.1).err .== 0.1)
    @test_throws ArgumentError synth_gravity(m, sx, sy; noise = -1.0)

    # stations above the datum see a weaker anomaly
    high = synth_gravity(m, sx, sy, fill(-1000.0, length(sx)))
    @test maximum(high.value) < maximum(obs.value)
end

@testset "synth_mt_sites" begin
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(200.0, 12))
    # the range must straddle the layer tested below: at 0.001 s the skin depth
    # in 100 ohm-m is about 160 m, well above a conductor at 800 m, and at 100 s
    # it is tens of kilometres, well below it
    periods = 10.0 .^ range(-3, 2; length = 10)

    # a half-space must return its own resistivity at every period, with a
    # 45 degree phase
    hs = truth_halfspace(g; log_rho = 2.0)
    sites = synth_mt_sites(hs, [1000.0, 2000.0], [1000.0, 1000.0], periods)
    @test sites isa MTSites
    @test nsites(sites) == 2
    @test size(sites.rho_a) == (10, 2)
    @test sites.periods ≈ periods
    @test all(isapprox.(sites.rho_a, 100.0; rtol = 1e-6))
    @test all(isapprox.(sites.phase, 45.0; atol = 1e-6))

    # a conductive layer at 800-1600 m must be invisible to the shortest period
    # and dominate the longest, which is the depth sensitivity the whole method
    # relies on
    lay = add_layer(hs; ztop = 800.0, zbot = 1600.0, log_rho = 0.5)
    ls = synth_mt_sites(lay, [1000.0], [1000.0], periods)
    @test ls.rho_a[1, 1] ≈ 100.0 rtol = 1e-3
    @test ls.rho_a[end, 1] < 50.0
    @test minimum(ls.rho_a) < 10.0
    # no monotonicity assertion: a conductive layer over a resistive basement
    # overshoots above the half-space value just before the conductor takes over

    noisy = synth_mt_sites(lay, [1000.0], [1000.0], periods; noise = 0.05,
                           rng = Xoshiro(2))
    @test noisy.rho_a != ls.rho_a
    @test all(abs.(noisy.rho_a ./ ls.rho_a .- 1) .< 0.3)
    @test noisy.err_rho_a !== nothing
    @test noisy.err_phase !== nothing
    @test all(isapprox.(noisy.err_rho_a, 0.05 .* noisy.rho_a))
    @test all(isapprox.(noisy.err_phase, 0.05 / 2 * 180 / pi))

    @test sites.err_rho_a === nothing
    @test sites.err_phase === nothing

    @test_throws ArgumentError synth_mt_sites(hs, [0.0], [0.0, 1.0], periods)
    @test_throws ArgumentError synth_mt_sites(hs, [0.0], [0.0], periods; noise = -0.1)
    @test_throws ArgumentError synth_mt_sites(hs, [0.0], [0.0], [-1.0])
end

@testset "synth_mt_sites feeds the feature pipeline directly" begin
    # the returned sites must be usable where real observations would be, which
    # is the reason the generator returns an MTSites rather than loose arrays
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(200.0, 12))
    hs = truth_halfspace(g; log_rho = 2.0)
    sites = synth_mt_sites(hs, [1000.0, 2000.0], [1000.0, 2000.0],
                           10.0 .^ range(-1, 2; length = 10))

    base = nb_baseline(g, sites)
    @test size(base) == size(g)
    # a half-space transforms back to its own resistivity at every depth
    @test all(isapprox.(base, 2.0; atol = 0.15))

    stack = build_features(g; sites = sites)
    @test nchannels(stack) > 0
end

@testset "synth_mt_sites skips air above topography" begin
    g = PriorGrid(fill(500.0, 6), fill(500.0, 6), fill(200.0, 12))
    periods = [1.0, 10.0]
    nx, ny, _ = size(g)

    hs = truth_halfspace(g; log_rho = 2.0)
    topo = add_topography(hs; elevation = fill(400.0, nx, ny))

    # the column starts at the ground, so the response is still the half-space's
    ts = synth_mt_sites(topo, [1000.0], [1000.0], periods)
    @test all(isapprox.(ts.rho_a, 100.0; rtol = 1e-6))
    @test all(isapprox.(ts.phase, 45.0; atol = 1e-6))

    all_air = add_topography(hs; elevation = fill(1e6, nx, ny))
    @test_throws ErrorException synth_mt_sites(all_air, [1000.0], [1000.0], periods)
end

@testset "truth_vector matches the feature-matrix ordering" begin
    g = _synth_grid()
    m = add_block(truth_halfspace(g); x = (-1000.0, 0.0), y = (-500.0, 500.0),
                  z = (300.0, 700.0), log_rho = 0.7)
    v = truth_vector(m)
    @test length(v) == ncells(g)
    # the same flattening the feature stack uses, so mu and truth line up cell
    # for cell without any reordering
    s = build_features(g)
    @test size(feature_matrix(s), 2) == length(v)
    @test reshape(v, size(g)) == m.log_rho
end
