using Test
using SmartPriorMT
using MTGeophysics: load_ws3d_model, air_mask_from_model
using Random

_export_grid() = PriorGrid(fill(500.0, 4), fill(400.0, 3), [100.0, 150.0, 225.0];
                           origin = [-1000.0, -600.0, 0.0])

@testset "PriorBundle validation" begin
    g = _export_grid()
    mu = fill(2.0, size(g))
    sigma = fill(0.3, size(g))

    b = PriorBundle(g, mu, sigma)
    @test size(b) == size(g)
    @test b.k == 2.0

    @test_throws DimensionMismatch PriorBundle(g, zeros(2, 2, 2), sigma)
    @test_throws DimensionMismatch PriorBundle(g, mu, zeros(2, 2, 2))
    @test_throws DimensionMismatch PriorBundle(g, mu, sigma; spread = zeros(2, 2, 2))
    @test_throws ArgumentError PriorBundle(g, mu, sigma; log_rho_bounds = (5.0, 0.0))
    @test_throws ArgumentError PriorBundle(g, mu, sigma; k = 0.0)
    @test_throws ArgumentError PriorBundle(g, mu, sigma; min_width = 0.0)
    @test_throws ArgumentError PriorBundle(g, mu, sigma;
                                          log_rho_bounds = (1.0, 1.5), min_width = 2.0)

    # a non-positive sigma in a live cell makes the interval meaningless
    bad = copy(sigma); bad[1, 1, 1] = 0.0
    @test_throws ArgumentError PriorBundle(g, mu, bad)

    # but air cells are exempt, since they are never perturbed
    air_mu = copy(mu); air_mu[1, 1, 1] = NaN
    air_sigma = copy(sigma); air_sigma[1, 1, 1] = NaN
    @test PriorBundle(g, air_mu, air_sigma) isa PriorBundle
end

@testset "prior_bounds brackets mu and respects the global range" begin
    g = _export_grid()
    mu = fill(2.0, size(g))
    sigma = fill(0.3, size(g))
    b = PriorBundle(g, mu, sigma; k = 2.0, log_rho_bounds = (0.0, 5.0))

    lo, hi = prior_bounds(b)
    @test all(lo .≈ 1.4)
    @test all(hi .≈ 2.6)
    @test all(lo .<= mu .<= hi)

    # a larger sigma widens the interval, which is the whole mechanism
    wide = PriorBundle(g, mu, fill(0.8, size(g)); k = 2.0)
    lo_w, hi_w = prior_bounds(wide)
    @test all(hi_w .- lo_w .> hi .- lo)

    # and k scales it
    b3 = PriorBundle(g, mu, sigma; k = 3.0)
    lo3, hi3 = prior_bounds(b3)
    @test all(hi3 .- lo3 .≈ 3 / 2 .* (hi .- lo))
end

@testset "prior_bounds clamps into the physical range" begin
    g = _export_grid()
    # mu near the top of the range with a wide sigma must not propose resistivity
    # above the global bound
    mu = fill(4.8, size(g))
    b = PriorBundle(g, mu, fill(0.5, size(g)); k = 2.0, log_rho_bounds = (0.0, 5.0))
    lo, hi = prior_bounds(b)
    @test all(hi .≈ 5.0)
    @test all(lo .≈ 3.8)
    @test all(0.0 .<= lo)
end

@testset "prior_bounds never collapses an interval" begin
    # residual mode can put mu outside the physical range; clamping both ends
    # would then give a zero-width interval and silently freeze the cell
    g = _export_grid()
    mu = fill(6.5, size(g))          # above the global upper bound
    b = PriorBundle(g, mu, fill(0.2, size(g)); k = 2.0,
                    log_rho_bounds = (0.0, 5.0), min_width = 0.3)
    lo, hi = prior_bounds(b)
    @test all(hi .- lo .≈ 0.3)
    @test all(hi .≈ 5.0)
    @test all(lo .≈ 4.7)

    # and the same at the bottom
    b2 = PriorBundle(g, fill(-3.0, size(g)), fill(0.2, size(g));
                     k = 2.0, log_rho_bounds = (0.0, 5.0), min_width = 0.3)
    lo2, hi2 = prior_bounds(b2)
    @test all(lo2 .≈ 0.0)
    @test all(hi2 .≈ 0.3)

    # a tiny sigma in the interior is widened symmetrically about mu
    b3 = PriorBundle(g, fill(2.0, size(g)), fill(1e-4, size(g));
                     k = 2.0, log_rho_bounds = (0.0, 5.0), min_width = 0.3)
    lo3, hi3 = prior_bounds(b3)
    @test all(hi3 .- lo3 .≈ 0.3)
    @test all(lo3 .≈ 1.85)
    @test all(hi3 .≈ 2.15)
end

@testset "prior_bounds keeps air as NaN" begin
    g = _export_grid()
    mu = fill(2.0, size(g))
    mu[:, :, 1] .= NaN
    sigma = fill(0.3, size(g))

    b = PriorBundle(g, mu, sigma)
    lo, hi = prior_bounds(b)
    @test all(isnan, lo[:, :, 1])
    @test all(isnan, hi[:, :, 1])
    @test all(isfinite, lo[:, :, 2:end])
    @test all(isfinite, hi[:, :, 2:end])
end

@testset "apply_air_mask" begin
    g = _export_grid()
    mu = fill(2.0, size(g))
    air = falses(size(g))
    air[:, :, 1] .= true

    masked = apply_air_mask(mu, air)
    @test all(isnan, masked[:, :, 1])
    @test all(masked[:, :, 2] .== 2.0)
    @test all(mu .== 2.0)      # the input is not touched

    @test_throws DimensionMismatch apply_air_mask(mu, falses(2, 2, 2))
end

@testset "write_prior round trips through the WS3D format" begin
    g = _export_grid()
    nx, ny, nz = size(g)

    # a laterally varying model, so a transposed or mirrored write shows up
    mu = [1.0 + 0.5 * i + 0.1 * j + 0.05 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    sigma = [0.2 + 0.02 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    spread = fill(0.05, nx, ny, nz)

    b = PriorBundle(g, mu, sigma; spread = spread, k = 2.0, log_rho_bounds = (0.0, 5.0))
    dir = mktempdir()
    paths = write_prior(dir, b)

    @test isfile(paths.rho)
    @test isfile(paths.std)
    @test isfile(paths.lo)
    @test isfile(paths.hi)
    @test paths.spread !== nothing && isfile(paths.spread)

    m = load_ws3d_model(paths.rho)
    @test m.dx ≈ g.dx
    @test m.dy ≈ g.dy
    @test m.dz ≈ g.dz
    @test m.origin ≈ g.origin
    # the WS3D writer reverses the i index per slice; going through it in both
    # directions is the only way to be sure the model is not mirrored
    @test m.A ≈ mu rtol = 1e-5

    mlo = load_ws3d_model(paths.lo)
    mhi = load_ws3d_model(paths.hi)
    lo, hi = prior_bounds(b)
    @test mlo.A ≈ lo rtol = 1e-5
    @test mhi.A ≈ hi rtol = 1e-5
    @test all(mlo.A .<= m.A .<= mhi.A)

    # a custom prefix keeps several priors side by side in one directory
    p2 = write_prior(dir, b; prefix = "variant")
    @test isfile(joinpath(dir, "variant.rho"))
    @test p2.rho != paths.rho
end

@testset "write_prior carries air through as air" begin
    g = _export_grid()
    mu = fill(2.0, size(g))
    mu[:, :, 1] .= NaN
    sigma = fill(0.3, size(g))
    sigma[:, :, 1] .= NaN

    b = PriorBundle(g, mu, sigma)
    paths = write_prior(mktempdir(), b)

    m = load_ws3d_model(paths.rho)
    # the loader re-tags anything above 1e15 ohm-metres as NaN, so the air layer
    # has to come back as air and be seen by MTGeophysics' own mask helper
    @test all(isnan, m.A[:, :, 1])
    @test all(isapprox.(m.A[:, :, 2], 2.0; rtol = 1e-5))
    @test all(air_mask_from_model(m)[:, :, 1])
    @test !any(air_mask_from_model(m)[:, :, 2])
end

@testset "prior_from_ensemble" begin
    g = _export_grid()
    s = build_features(g)
    X = encode_features(s; n_bands = 2)

    net = PriorNet(size(X, 1); width = 8, depth = 2, log_rho_bounds = (0.5, 4.5))
    targets = PriorTargets(anchors = ([1, 4, 9], [2.0, 2.5, 3.0], [1.0, 1.0, 1.0]))
    ens, _ = train_ensemble(net, X, g, targets; nmembers = 2,
                            config = TrainConfig(epochs = 20, log_every = 20, verbose = false))

    b = prior_from_ensemble(g, ens, X; k = 2.0)
    @test b isa PriorBundle
    @test size(b) == size(g)
    @test b.spread !== nothing
    # the bundle inherits the network's physical range by default
    @test b.log_rho_bounds == (0.5, 4.5)

    lo, hi = prior_bounds(b)
    @test all(0.5 .<= lo .<= hi .<= 4.5)

    # an air mask is honoured, and the bounds follow it
    air = falses(size(g))
    air[:, :, 1] .= true
    bair = prior_from_ensemble(g, ens, X; air = air)
    @test all(isnan, bair.mu[:, :, 1])
    lo2, _ = prior_bounds(bair)
    @test all(isnan, lo2[:, :, 1])

    # an explicit range overrides the network's
    bover = prior_from_ensemble(g, ens, X; log_rho_bounds = (1.0, 3.0))
    @test bover.log_rho_bounds == (1.0, 3.0)
    lo3, hi3 = prior_bounds(bover)
    @test all(1.0 .<= lo3 .<= hi3 .<= 3.0)
end
