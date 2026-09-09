using Test
using SmartPriorMT
using Random
using Statistics
using Zygote

_loss_grid() = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0, 225.0];
                         origin = [-1000.0, -1000.0, 0.0])

function _loss_sites(periods = 10 .^ range(-2, 2; length = 8))
    f = 1 ./ periods
    a, p = mt1d_apparent(f, [100.0], Float64[])
    return MTSites([-250.0, 250.0], [0.0, 0.0], periods,
                   hcat(a, a), hcat(p, p))
end

@testset "GravityCoupling and density_from_mu" begin
    # the slope is a dimensionless multiplier of DENSITY_SCALE, so -1.2 means
    # 120 kg/m3 per decade with dense rock conductive
    c = GravityCoupling(slope = -1.2, offset = 0.05)
    mu = [2.0, 3.0, 1.0]
    ref = [2.0, 2.0, 2.0]

    ρ = density_from_mu(c, mu, ref)
    @test ρ ≈ [5.0, -115.0, 125.0]
    @test physical_slope(c) ≈ -120.0

    # the scaling exists so a trained slope lands at order one rather than in the
    # hundreds, where a shared learning rate could never reach it
    @test DENSITY_SCALE == 100.0
    @test abs(c.slope) < 10

    # a zero slope decouples gravity entirely, which is the graceful failure mode
    zero_c = GravityCoupling(slope = 0.0, offset = 0.0)
    @test all(iszero, density_from_mu(zero_c, mu, ref))
    @test physical_slope(zero_c) == 0.0

    # the sign is free: flipping it flips which rocks are dense
    flipped = density_from_mu(GravityCoupling(slope = 1.2, offset = 0.05), mu, ref)
    @test flipped .- 5.0 ≈ -(ρ .- 5.0)

    @test_throws DimensionMismatch density_from_mu(c, mu, [1.0])
end

@testset "heteroscedastic_nll" begin
    target = [1.0, 2.0, 3.0]

    # a perfect fit still pays log(sigma), which is what stops sigma shrinking
    # to zero and what makes the uncertainty learnable at all
    exact = heteroscedastic_nll(target, fill(0.5, 3), target)
    @test exact ≈ log(0.5)

    # for a fixed residual the loss is minimised when sigma equals that residual
    res = 0.4
    mu = target .- res
    losses = [heteroscedastic_nll(mu, fill(s, 3), target) for s in 0.1:0.02:1.5]
    best = (0.1:0.02:1.5)[argmin(losses)]
    @test best ≈ res atol = 0.03

    # widening sigma reduces the quadratic part but costs the log part
    @test heteroscedastic_nll(mu, fill(0.05, 3), target) >
          heteroscedastic_nll(mu, fill(res, 3), target)
    @test heteroscedastic_nll(mu, fill(5.0, 3), target) >
          heteroscedastic_nll(mu, fill(res, 3), target)

    # weights select which anchors matter
    w = [1.0, 0.0, 0.0]
    mixed = [1.0, 99.0, 99.0]
    @test heteroscedastic_nll(mixed, fill(0.5, 3), target; weight = w) ≈ log(0.5)

    @test_throws DimensionMismatch heteroscedastic_nll([1.0], [1.0], [1.0, 2.0])
    @test_throws DimensionMismatch heteroscedastic_nll(mu, fill(0.5, 3), target; weight = [1.0])
    @test_throws ArgumentError heteroscedastic_nll(Float64[], Float64[], Float64[])
end

@testset "gravity_misfit" begin
    A = [1.0 0.0; 0.0 2.0]
    obs = [3.0, 4.0]
    err = [1.0, 1.0]

    # the exact solution gives zero
    @test gravity_misfit(A, [3.0, 2.0], obs, err) ≈ 0.0

    # with :none, a one-sigma error per datum gives chi-squared per datum of one
    @test gravity_misfit(A, [4.0, 2.5], obs, err; detrend = :none) ≈ 1.0

    # default :mean detrend removes a constant residual, so the same mismatch
    # scores zero
    @test gravity_misfit(A, [4.0, 2.5], obs, err) ≈ 0.0

    # larger error bars forgive the same residual
    @test gravity_misfit(A, [4.0, 2.5], obs, [2.0, 2.0]; detrend = :none) < 1.0

    @test_throws DimensionMismatch gravity_misfit(A, [1.0], obs, err)
    @test_throws DimensionMismatch gravity_misfit(A, [1.0, 2.0], [1.0], err)
    @test_throws ArgumentError gravity_misfit(A, [1.0, 2.0], obs, [0.0, 1.0])
    @test_throws ArgumentError gravity_misfit(A, [1.0, 2.0], obs, err; detrend = :offset)
end

@testset "gravity_misfit detrend" begin
    A = [1.0 0.5; 0.5 1.0]
    obs = [1.0, 2.0]
    err = [1.0, 2.0]
    density = [1.2, 1.8]

    r = A * density .- obs
    w = 1 ./ abs2.(err)
    r_dt = r .- (sum(w .* r) / sum(w))
    @test sum(w .* r_dt) / sum(w) ≈ 0.0 atol = 1e-12
    @test gravity_misfit(A, density, obs, err; detrend = :mean) ≈ mean(abs2.(r_dt ./ err))

    g = PriorGrid([1.0e5], fill(400.0, 20), fill(300.0, 8);
                  origin = [-5.0e4, -4000.0, 0.0])
    truth = add_block(truth_halfspace(g; log_rho = 2.6);
                      x = (-1.0e5, 1.0e5), y = (-2000.0, 0.0), z = (300.0, 1500.0),
                      log_rho = 0.6, density = 350.0)
    gx = collect(-4000.0:400.0:4000.0)
    obs_g = synth_gravity(truth, zeros(length(gx)), gx; noise = 0.02, rng = Xoshiro(3))
    A_full = gravity_matrix(g, obs_g.x, obs_g.y, obs_g.z)
    ρ = vec(truth.density)
    pred = A_full * ρ
    m_none = gravity_misfit(A_full, ρ, obs_g.value, obs_g.err; detrend = :none)
    m_mean = gravity_misfit(A_full, ρ, obs_g.value, obs_g.err; detrend = :mean)
    @test m_mean < m_none

    # a constant bias in the observations shifts the absolute level; :mean ignores it
    bias = 0.35
    m_none_bias = gravity_misfit(A_full, ρ, obs_g.value .+ bias, obs_g.err; detrend = :none)
    m_mean_bias = gravity_misfit(A_full, ρ, obs_g.value .+ bias, obs_g.err; detrend = :mean)
    @test m_mean_bias ≈ m_mean
    @test m_none_bias > m_none

    # structural match with wrong absolute level scores zero under :mean
    @test gravity_misfit(A_full, ρ, pred .+ bias, obs_g.err; detrend = :mean) ≈ 0.0 atol = 1e-12
    @test gravity_misfit(A_full, ρ, pred .+ bias, obs_g.err; detrend = :none) > 0.0
end

@testset "gravity slope sign on conductive dense slab" begin
    g = PriorGrid([1.0e5], fill(400.0, 20), fill(300.0, 8);
                  origin = [-5.0e4, -4000.0, 0.0])
    truth = add_block(truth_halfspace(g; log_rho = 2.6);
                      x = (-1.0e5, 1.0e5), y = (-2000.0, 0.0), z = (300.0, 1500.0),
                      log_rho = 0.6, density = 350.0)
    gx = collect(-4000.0:400.0:4000.0)
    obs_g = synth_gravity(truth, zeros(length(gx)), gx; noise = 0.02, rng = Xoshiro(4))
    A = gravity_matrix(g, obs_g.x, obs_g.y, obs_g.z)
    base = fill(2.4, ncells(g))
    μ = vec(truth.log_rho)

    neg = gravity_misfit(A, density_from_mu(GravityCoupling(-2.5, 0.0), μ, base),
                         obs_g.value, obs_g.err)
    pos = gravity_misfit(A, density_from_mu(GravityCoupling(2.5, 0.0), μ, base),
                         obs_g.value, obs_g.err)
    @test neg < pos
end

@testset "mt_column_misfit" begin
    g = _loss_grid()
    sites = _loss_sites()
    cells = [(2, 2), (3, 2)]

    # a model that is the half-space the data came from must fit essentially
    # perfectly, apart from the bottom cell being treated as the basement
    mu_true = fill(2.0, size(g))
    @test mt_column_misfit(mu_true, g, sites, cells) < 1e-6

    # a decade off costs far more
    mu_wrong = fill(3.0, size(g))
    @test mt_column_misfit(mu_wrong, g, sites, cells) >
          100 * mt_column_misfit(mu_true, g, sites, cells)

    # only the columns under the sites are looked at, so structure elsewhere is
    # invisible to this term
    mu_far = copy(mu_true)
    mu_far[1, 4, :] .= 4.0
    @test mt_column_misfit(mu_far, g, sites, cells) ≈
          mt_column_misfit(mu_true, g, sites, cells)

    @test_throws DimensionMismatch mt_column_misfit(mu_true, g, sites, [(1, 1)])
    @test_throws DimensionMismatch mt_column_misfit(zeros(2, 2, 2), g, sites, cells)
    @test_throws ArgumentError mt_column_misfit(mu_true, g, sites, [(99, 1), (1, 1)])
end

@testset "mt_column_misfit chi-squared errors" begin
    g = _loss_grid()
    periods = 10 .^ range(-2, 2; length = 8)
    f = 1 ./ periods
    a, p = mt1d_apparent(f, [100.0], Float64[])
    cells = [(2, 2), (3, 2)]
    mu_true = fill(2.0, size(g))
    mu_wrong = fill(3.0, size(g))
    sx, sy = [-250.0, 250.0], [0.0, 0.0]
    np, ns = length(periods), 2

    # a noise-free copy of the half-space the grid columns actually are: χ²/datum
    # must sit near zero, the same as gravity_misfit's exact-solution test
    err_a = 0.10 .* hcat(a, a)
    err_p = fill(2.0, np, ns)
    sites_exact = MTSites(sx, sy, periods, hcat(a, a), hcat(p, p);
                          err_rho_a = err_a, err_phase = err_p)
    @test mt_column_misfit(mu_true, g, sites_exact, cells) < 1e-4

    # shift ρ_a by exactly one sigma in log10 space so residual/err = 1 on every
    # resistivity datum (phase left exact). That is gravity_misfit with :none
    # and a one-sigma residual: χ²/datum = 1
    δlog10 = 0.05
    obs_a = hcat(a, a) ./ (10^δlog10)
    err_one = fill(δlog10, np, ns) .* obs_a .* log(10)
    sites_one = MTSites(sx, sy, periods, obs_a, hcat(p, p);
                        err_rho_a = err_one, err_phase = err_p)
    m_one = mt_column_misfit(mu_true, g, sites_one, cells)
    g_one = gravity_misfit([1.0 0.0; 0.0 2.0], [4.0, 2.5], [3.0, 4.0], [1.0, 1.0];
                           detrend = :none)
    @test g_one ≈ 1.0
    @test m_one ≈ 1.0 atol = 1e-4

    # a decade-off half-space is many sigma under the same errors
    @test mt_column_misfit(mu_wrong, g, sites_one, cells) > 100

    # phase_weight still scales the phase term once err_phase is in play
    obs_p = hcat(p, p) .+ 3.0
    sites_phase = MTSites(sx, sy, periods, hcat(a, a), obs_p;
                          err_rho_a = err_a, err_phase = fill(3.0, np, ns))
    m_phase = mt_column_misfit(mu_true, g, sites_phase, cells)
    @test m_phase ≈ 1.0 atol = 1e-4
    @test mt_column_misfit(mu_true, g, sites_phase, cells; phase_weight = 2.0) ≈
          2 * m_phase atol = 1e-4
end

@testset "smoothness" begin
    g = _loss_grid()

    # a constant field is perfectly smooth
    @test smoothness(fill(2.0, size(g)), g) ≈ 0.0

    # a checkerboard is rougher than a gentle ramp
    nx, ny, nz = size(g)
    rough = [iseven(i + j + k) ? 1.0 : -1.0 for i in 1:nx, j in 1:ny, k in 1:nz]
    ramp = [0.1 * i for i in 1:nx, j in 1:ny, k in 1:nz]
    @test smoothness(rough, g) > smoothness(ramp, g)

    # a smaller vertical weight makes layering cheaper
    layered = [0.0 + 1.0 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    @test smoothness(layered, g; vertical_weight = 0.1) <
          smoothness(layered, g; vertical_weight = 1.0)

    # a degenerate grid has no differences to take
    g1 = PriorGrid([100.0], [100.0], [100.0])
    @test smoothness(fill(3.0, 1, 1, 1), g1) ≈ 0.0

    @test_throws DimensionMismatch smoothness(zeros(2, 2, 2), g)
end

@testset "reference_penalty" begin
    @test reference_penalty(fill(2.0, 4), fill(2.0, 4)) ≈ 0.0
    @test reference_penalty(fill(2.5, 4), fill(2.0, 4)) ≈ 0.25
    # weights select which cells matter, the same way a depth taper would
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
end

@testset "prior_loss skips terms whose inputs are absent" begin
    g = _loss_grid()
    n = ncells(g)
    mu = fill(2.0, n)
    sigma = fill(0.5, n)
    c = GravityCoupling(slope = -1.0)

    bare = loss_report(mu, sigma, c, g, PriorTargets())
    @test isnan(bare.anchor)
    @test isnan(bare.gravity)
    @test isnan(bare.mt)
    @test isnan(bare.reference)
    @test bare.smooth ≈ 0.0
    @test bare.sigma ≈ 0.0
    @test bare.total ≈ 0.0

    # anchors alone
    t_anchor = PriorTargets(anchors = ([1, 5, 9], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]))
    r = loss_report(mu, sigma, c, g, t_anchor)
    @test r.anchor ≈ log(0.5)
    @test isnan(r.gravity)

    # gravity alone, and it must refuse to run without a reference level
    A = gravity_matrix(g, [0.0, 400.0], [0.0, 0.0], [0.0, 0.0])
    t_grav_bad = PriorTargets(gravity = (A, [0.1, 0.1], [0.05, 0.05]))
    @test_throws ArgumentError prior_loss(mu, sigma, c, g, t_grav_bad)

    t_grav = PriorTargets(gravity = (A, [0.1, 0.1], [0.05, 0.05]),
                          reference = fill(2.0, n))
    rg = loss_report(mu, sigma, c, g, t_grav)
    @test isfinite(rg.gravity)
    # mu equals the reference, so the predicted anomaly is zero; default
    # :mean detrend removes the constant observed level as well
    @test rg.gravity ≈ 0.0 atol = 1e-12

    t_grav_none = PriorTargets(gravity = (A, [0.1, 0.1], [0.05, 0.05]),
                               reference = fill(2.0, n),
                               gravity_detrend = :none)
    @test loss_report(mu, sigma, c, g, t_grav_none).gravity ≈
          mean(abs2.([0.1, 0.1] ./ [0.05, 0.05]))

    # mt alone
    sites = _loss_sites()
    t_mt = PriorTargets(mt = (sites, [(2, 2), (3, 2)]))
    rm = loss_report(mu, sigma, c, g, t_mt)
    @test rm.mt < 1e-6
    @test isnan(rm.anchor)
end

@testset "prior_loss weights and validation" begin
    g = _loss_grid()
    n = ncells(g)
    mu = fill(2.0, n)
    sigma = fill(0.8, n)
    c = GravityCoupling(slope = -1.0)
    t = PriorTargets(sigma_target = 0.4)

    w1 = LossWeights(sigma = 1.0)
    w2 = LossWeights(sigma = 2.0)
    @test prior_loss(mu, sigma, c, g, t, w2) ≈ 2 * prior_loss(mu, sigma, c, g, t, w1)

    @test_throws DimensionMismatch prior_loss(mu[1:end-1], sigma, c, g, t)
    @test_throws DimensionMismatch prior_loss(mu, sigma[1:end-1], c, g, t)

    bad_anchor = PriorTargets(anchors = ([n + 1], [2.0], [1.0]))
    @test_throws ArgumentError prior_loss(mu, sigma, c, g, bad_anchor)
end

@testset "prior_loss reference term" begin
    # isolate the damping term: constant mu ⇒ smooth == 0, sigma at target ⇒
    # sigma == 0, and no anchors / gravity / mt
    g = _loss_grid()
    n = ncells(g)
    mu = fill(2.5, n)
    sigma = fill(0.4, n)
    c = GravityCoupling(slope = 0.0)
    t = PriorTargets(reference = fill(2.0, n), sigma_target = 0.4)

    report = loss_report(mu, sigma, c, g, t)
    @test report.reference ≈ 0.25
    @test isnan(report.anchor)
    @test isnan(report.gravity)
    @test isnan(report.mt)
    @test report.smooth ≈ 0.0
    @test report.sigma ≈ 0.0

    off = loss_report(mu, sigma, c, g, t, LossWeights(reference = 0.0))
    @test off.total ≈ 0.0
    @test off.reference ≈ 0.25

    on = loss_report(mu, sigma, c, g, t, LossWeights(reference = 1.0))
    @test on.total ≈ 0.25
    doubled = loss_report(mu, sigma, c, g, t, LossWeights(reference = 2.0))
    @test doubled.total ≈ 2 * on.total

    tapered = PriorTargets(reference = fill(2.0, n), sigma_target = 0.4,
                           reference_weight = vcat(1.0, zeros(n - 1)))
    # only the first cell is charged, and it is 0.5 decades from the reference
    @test loss_report(mu, sigma, c, g, tapered).reference ≈ 0.25

    # reference weight zero must not change the total from a run with no damping
    mu_off = fill(2.5, n)
    t_ref = PriorTargets(reference = fill(2.0, n), sigma_target = 0.4)
    t_bare = PriorTargets(sigma_target = 0.4)
    w = LossWeights(smooth = 1.0, sigma = 1.0, reference = 0.0)
    @test prior_loss(mu_off, sigma, c, g, t_ref, w) ≈
          prior_loss(mu_off, sigma, c, g, t_bare, w)
    @test loss_report(mu_off, sigma, c, g, t_ref, w).reference ≈ 0.25
end

@testset "every loss term is differentiable on its own" begin
    # each term probed separately, because a single mutating call anywhere in the
    # assembled objective takes the whole gradient down and the combined test
    # cannot say which term did it
    g = _loss_grid()
    n = ncells(g)
    sites = _loss_sites()
    A = gravity_matrix(g, [-300.0, 300.0], [0.0, 0.0], [0.0, 0.0])
    ref = fill(2.0, n)

    # deliberately not the exact solution of any term, so a zero gradient means
    # a broken derivative rather than a converged one
    mu0 = [2.0 + 0.3 * sin(i / 3) for i in 1:n]

    terms = Dict(
        "anchor" => mu -> heteroscedastic_nll(mu[[1, 5, 9]], fill(0.5, 3), [2.0, 2.0, 2.0]),
        "gravity" => mu -> gravity_misfit(A, density_from_mu(GravityCoupling(-1.0, 0.0), mu, ref),
                                          [0.4, -0.2], [0.05, 0.05]),
        "smooth" => mu -> smoothness(reshape(mu, size(g)), g),
        "sigma" => mu -> sigma_penalty(mu; target = 0.4),
        "mt" => mu -> mt_column_misfit(reshape(mu, size(g)), g, sites, [(2, 2), (3, 2)]),
        "reference" => mu -> reference_penalty(mu, ref),
    )

    for (name, fn) in terms
        grad = Zygote.gradient(fn, mu0)[1]
        @test grad !== nothing
        @test all(isfinite, grad)
        @test any(!iszero, grad)

        # spot-check against a finite difference at the cell with the largest
        # sensitivity, which pins the value and not merely the existence
        k = argmax(abs.(grad))
        h = 1e-6
        up = copy(mu0); up[k] += h
        dn = copy(mu0); dn[k] -= h
        fd = (fn(up) - fn(dn)) / (2h)
        @test grad[k] ≈ fd rtol = 1e-4
    end
end

@testset "prior_loss is differentiable end to end through the network" begin
    g = _loss_grid()
    n = ncells(g)
    sites = _loss_sites()

    obs = GravityObs([-300.0, 300.0], [0.0, 0.0], [0.0, 0.0], [0.4, -0.2], [0.05, 0.05])
    s = build_features(g; gravity = obs, sites = sites)
    X = encode_features(s; n_bands = 3)

    net = PriorNet(size(X, 1); width = 16, depth = 2)
    ps, st = setup_prior(Xoshiro(21), net)

    A = gravity_matrix(g, obs.x, obs.y, obs.z)
    reference = vec(nb_baseline(g, sites))

    targets = PriorTargets(
        anchors = ([1, 10, 20], [2.0, 1.5, 3.0], [1.0, 1.0, 0.5]),
        gravity = (A, obs.value, obs.err),
        mt = (sites, [(2, 2), (3, 2)]),
        reference = reference,
        sigma_target = 0.4,
    )

    # the coupling slope is trained alongside the network weights, so the loss
    # closes over both and the gradient has to reach each of them
    function loss(p)
        (mu, sigma), _ = predict(net, X, p.net, st; offset = reference)
        coupling = GravityCoupling(p.slope, p.offset)
        return prior_loss(mu, sigma, coupling, g, targets)
    end

    params = (net = ps, slope = -0.8, offset = 0.0)
    l0 = loss(params)
    @test isfinite(l0)

    grad = Zygote.gradient(loss, params)[1]
    @test grad !== nothing
    @test all(isfinite, grad.net.layer_1.weight)
    @test any(!iszero, grad.net.layer_1.weight)

    # the learnable coupling only earns its place if gravity actually pushes on
    # the slope; a zero here would mean the term is decorative
    @test isfinite(grad.slope)
    @test grad.slope != 0

    report = loss_report(
        first(predict(net, X, ps, st; offset = reference))...,
        GravityCoupling(-0.8, 0.0), g, targets)
    @test isfinite(report.total)
    @test all(isfinite, (report.anchor, report.gravity, report.mt, report.smooth, report.sigma, report.reference))
end
