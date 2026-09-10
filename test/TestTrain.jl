using Test
using SmartPriorMT
using Random
using Statistics

# a small problem with a known answer: anchors pin every cell to a constant, so
# a converged run has to reproduce it and its sigma has to shrink towards zero
function _train_case(; nanchor = 40, truth = 2.4)
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0]; origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    s = build_features(g)
    X = encode_features(s; n_bands = 2)

    cells = collect(1:min(nanchor, n))
    targets = PriorTargets(
        anchors = (cells, fill(truth, length(cells)), fill(1.0, length(cells))),
        sigma_target = 0.5,
    )
    return g, X, targets, truth
end

@testset "init_params and coupling_of" begin
    net = PriorNet(6; width = 8, depth = 2)
    params, st = init_params(Xoshiro(1), net; slope = -75.0, offset = 3.0)

    @test haskey(params, :net)
    @test haskey(params, :coupling)
    # one-element vectors, because Optimisers.jl leaves bare scalars frozen
    @test params.coupling.slope isa Vector{Float64}
    @test params.coupling.offset isa Vector{Float64}

    c = coupling_of(params)
    @test c.slope ≈ -75.0
    @test c.offset ≈ 3.0

    # gravity starts decoupled by default, so the link has to be earned
    p0, _ = init_params(Xoshiro(1), net)
    @test coupling_of(p0).slope == 0.0
end

@testset "train_prior reduces the loss and recovers a known answer" begin
    g, X, targets, truth = _train_case()
    net = PriorNet(size(X, 1); width = 32, depth = 3, sigma_bounds = (0.02, 1.0))

    cfg = TrainConfig(epochs = 400, learning_rate = 5.0e-3,
                      log_every = 50, verbose = false, seed = 3)
    res = train_prior(net, X, g, targets; config = cfg)

    @test res isa TrainResult
    @test !isempty(res.history)
    @test isfinite(res.best_loss)
    @test res.best_epoch >= 1

    first_total = res.history[1].total
    @test res.best_loss < first_total

    # the anchors cover only part of the grid, so check where they actually are
    mu, sigma, _ = predict_grid(net, X, res.params.net, res.state, size(g))
    cells, values, _ = targets.anchors
    @test all(abs.(vec(mu)[cells] .- truth) .< 0.15)

    # a well-fitted anchor should have pulled its sigma below the default target,
    # which is the whole point of a learnable uncertainty
    @test mean(vec(sigma)[cells]) < 0.5
end

@testset "train_prior returns the best parameters, not the last" begin
    # full-batch Adam on a multi-term objective does not descend monotonically,
    # so the contract is that best_loss matches the best history entry
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 16, depth = 2)
    cfg = TrainConfig(epochs = 200, learning_rate = 2.0e-2,
                      log_every = 20, verbose = false, seed = 5)
    res = train_prior(net, X, g, targets; config = cfg)

    @test res.best_loss ≈ minimum(h.total for h in res.history)
    @test res.best_epoch == res.history[argmin([h.total for h in res.history])].epoch
end

@testset "train_prior history carries every term" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2)
    res = train_prior(net, X, g, targets;
                      config = TrainConfig(epochs = 20, log_every = 10, verbose = false))

    h = res.history[1]
    for k in (:epoch, :slope, :saturation, :total, :anchor, :gravity, :mt, :smooth, :sigma, :reference)
        @test haskey(h, k)
    end
    @test isfinite(h.anchor)
    @test isnan(h.gravity)      # no gravity supplied in this case
    @test isnan(h.mt)
    @test isnan(h.reference)
    @test 0.0 <= h.saturation <= 1.0
end

@testset "the history reports saturation when the band is too narrow" begin
    # a residual span far below the structure the anchors demand forces the
    # network against the edge of its band, where tanh has no gradient and the
    # affected cells stop learning. that is worth surfacing rather than hiding.
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0];
                  origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    s = build_features(g)
    X = encode_features(s; n_bands = 2)

    base = fill(2.0, n)
    cells = collect(1:n)
    # anchors two decades away, with a band only a tenth of a decade wide
    targets = PriorTargets(anchors = (cells, fill(4.0, n), fill(1.0, n)))

    tight = PriorNet(size(X, 1); width = 16, depth = 2, residual_span = 0.1,
                     log_rho_bounds = (0.0, 5.0))
    res_tight = train_prior(tight, X, g, targets;
                            config = TrainConfig(epochs = 400, learning_rate = 1.0e-2,
                                                 log_every = 100, verbose = false),
                            offset = base)
    @test res_tight.history[end].saturation > 0.5

    # given room, it reaches the anchors and stops saturating
    wide = PriorNet(size(X, 1); width = 16, depth = 2, residual_span = 2.5,
                    log_rho_bounds = (0.0, 5.0))
    res_wide = train_prior(wide, X, g, targets;
                           config = TrainConfig(epochs = 400, learning_rate = 1.0e-2,
                                                log_every = 100, verbose = false),
                           offset = base)
    @test res_wide.history[end].saturation < 0.1
    @test res_wide.history[end].anchor < res_tight.history[end].anchor
end

@testset "train_prior early stopping" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2)

    # an absurd improvement threshold means no evaluation ever counts as better,
    # so patience must trigger well before the epoch cap
    cfg = TrainConfig(epochs = 2000, log_every = 5, patience = 3,
                      min_delta = 1.0e6, verbose = false)
    res = train_prior(net, X, g, targets; config = cfg)
    @test last(res.history).epoch < 2000
end

@testset "train_prior validation" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2)
    @test_throws ArgumentError train_prior(net, X, g, targets;
                                           config = TrainConfig(epochs = 0))
    @test_throws ArgumentError train_prior(net, X, g, targets;
                                           config = TrainConfig(log_every = 0))
    @test_throws DimensionMismatch train_prior(net, X[:, 1:4], g, targets;
                                               config = TrainConfig(epochs = 2, verbose = false))
end

@testset "gravity coupling is trained when gravity data is present" begin
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0]; origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    obs = GravityObs([-400.0, 0.0, 400.0], [0.0, 0.0, 0.0], zeros(3),
                     [-0.6, 1.4, -0.5], fill(0.05, 3))
    s = build_features(g; gravity = obs)
    X = encode_features(s; n_bands = 2)

    A = gravity_matrix(g, obs.x, obs.y, obs.z)
    reference = fill(2.0, n)
    targets = PriorTargets(gravity = (A, obs.value, obs.err), reference = reference)

    net = PriorNet(size(X, 1); width = 32, depth = 3)
    cfg = TrainConfig(epochs = 300, learning_rate = 1.0e-2,
                      log_every = 50, verbose = false, seed = 8,
                      weights = LossWeights(gravity = 1.0, smooth = 1.0e-4,
                                            # damping off: this testset isolates the gravity term
                                            reference = 0.0))
    res = train_prior(net, X, g, targets;
                      config = cfg, offset = reference)

    # the slope starts at exactly zero, so any movement proves the gravity term
    # reaches it and that the link really is learned rather than assumed
    @test coupling_of(res.params).slope != 0.0
    @test isfinite(coupling_of(res.params).slope)
    @test res.history[end].gravity < res.history[1].gravity
end

@testset "gravity training actually fits the anomaly" begin
    # regression test for a scaling failure the unit tests could not see. When
    # the coupling slope was expressed directly in kg/m^3 per decade it had to
    # travel to the hundreds while the network weights sat at order one, and one
    # learning rate could not serve both: the misfit stayed at 1e4 and the
    # network flattened its own output against the residual bound to reduce it.
    # The signature was a perfectly uniform mu, so both are asserted here.
    g = PriorGrid([1.0e5], fill(400.0, 20), fill(300.0, 8);
                  origin = [-5.0e4, -4000.0, 0.0])
    n = ncells(g)

    truth = add_block(truth_halfspace(g; log_rho = 2.6);
                      x = (-1.0e5, 1.0e5), y = (-2000.0, 0.0), z = (300.0, 1500.0),
                      log_rho = 0.6, density = 350.0)

    gx = collect(-4000.0:400.0:4000.0)
    obs = synth_gravity(truth, zeros(length(gx)), gx; noise = 0.02, rng = Xoshiro(1))
    base = fill(2.4, n)

    s = build_features(g; gravity = obs, baseline = reshape(base, size(g)))
    X = encode_features(s; n_bands = 3)
    A = gravity_matrix(g, obs.x, obs.y, obs.z)
    targets = PriorTargets(gravity = (A, obs.value, obs.err), reference = base,
                           sigma_target = 0.35)

    net = PriorNet(size(X, 1); width = 48, depth = 3,
                   log_rho_bounds = (0.0, 4.0), residual_span = 2.5,
                   sigma_bounds = (0.05, 1.2))
    res = train_prior(net, X, g, targets;
                      config = TrainConfig(epochs = 1500, learning_rate = 3.0e-3,
                                           log_every = 250, verbose = false, seed = 7,
                                           weights = LossWeights(gravity = 1.0,
                                                                 smooth = 3.0e-3,
                                                                 sigma = 0.1,
                                                                 # damping off: this testset isolates the gravity term
                                                                 reference = 0.0)),
                      offset = base)

    # the misfit must fall by orders of magnitude, not percent
    @test res.history[end].gravity < res.history[1].gravity / 50

    (mu, sigma), _ = predict(net, X, res.params.net, res.state; offset = base)
    # a flat mu is the failure signature: it means the network gave up on the
    # spatial problem and saturated
    @test std(mu) > 0.05
    @test !all(≈(mu[1]), mu)
    # and it must not be pinned against the residual bound
    @test maximum(abs.(mu .- base)) < 2.5 - 1.0e-3

    # the learned coupling must be signed correctly: this body is conductive and
    # dense, so a lower resistivity has to imply a higher density
    @test physical_slope(coupling_of(res.params)) < 0

    # sigma is only touched by its own penalty here, so it must sit at the target
    @test isapprox(median(sigma), 0.35; atol = 0.05)
end

@testset "slope_bounds constrains coupling sign" begin
    bounds = (-4.0, -0.2)
    g = PriorGrid([1.0e5], fill(400.0, 20), fill(300.0, 8);
                  origin = [-5.0e4, -4000.0, 0.0])
    n = ncells(g)

    truth = add_block(truth_halfspace(g; log_rho = 2.6);
                      x = (-1.0e5, 1.0e5), y = (-2000.0, 0.0), z = (300.0, 1500.0),
                      log_rho = 0.6, density = 350.0)
    gx = collect(-4000.0:400.0:4000.0)
    obs = synth_gravity(truth, zeros(length(gx)), gx; noise = 0.02, rng = Xoshiro(5))
    base = fill(2.4, n)

    s = build_features(g; gravity = obs, baseline = reshape(base, size(g)))
    X = encode_features(s; n_bands = 3)
    A = gravity_matrix(g, obs.x, obs.y, obs.z)
    targets = PriorTargets(gravity = (A, obs.value, obs.err), reference = base,
                           sigma_target = 0.35)

    net = PriorNet(size(X, 1); width = 48, depth = 3,
                   log_rho_bounds = (0.0, 4.0), residual_span = 2.5,
                   sigma_bounds = (0.05, 1.2))
    params, _ = init_params(Xoshiro(1), net; slope_bounds = bounds)
    c0 = coupling_of(params)
    @test bounds[1] <= c0.slope <= bounds[2]

    res = train_prior(net, X, g, targets;
                      config = TrainConfig(epochs = 800, learning_rate = 3.0e-3,
                                           log_every = 800, verbose = false, seed = 9,
                                           slope_bounds = bounds,
                                           weights = LossWeights(gravity = 1.0,
                                                                 smooth = 3.0e-3,
                                                                 sigma = 0.1,
                                                                 reference = 0.0)),
                      offset = base)

    c = coupling_of(res.params)
    @test bounds[1] <= c.slope <= bounds[2]
    @test physical_slope(c) < 0
end

@testset "reference damping pulls residual-mode mu toward the offset" begin
    # direction test, not a convergence test: a large damping weight should keep
    # mu closer to the baseline than an unpenalised residual, when anchors try
    # to pull it away. offset sits well inside log_rho_bounds so the reachable
    # band is symmetric; an offset near a bound would clip the undamped run
    # against one edge and muddy the comparison.
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0];
                  origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    s = build_features(g)
    X = encode_features(s; n_bands = 2)

    offset = fill(2.5, n)
    cells = collect(1:n)
    targets = PriorTargets(
        anchors = (cells, fill(3.5, n), fill(1.0, n)),
        reference = offset,
        sigma_target = 0.5,
    )

    net = PriorNet(size(X, 1); width = 16, depth = 2, residual_span = 1.5,
                   log_rho_bounds = (0.0, 5.0))

    function run_damp(damp)
        res = train_prior(net, X, g, targets;
                          config = TrainConfig(epochs = 300, learning_rate = 1.0e-2,
                                               log_every = 300, verbose = false, seed = 11,
                                               weights = LossWeights(anchor = 1.0,
                                                                     smooth = 0.0,
                                                                     sigma = 0.0,
                                                                     reference = damp)),
                          offset = offset)
        (mu, _), _ = predict(net, X, res.params.net, res.state; offset = offset)
        return mu
    end

    mu_free = run_damp(0.0)
    mu_damp = run_damp(50.0)
    @test maximum(abs.(mu_free .- offset)) > 0.3
    @test maximum(abs.(mu_damp .- offset)) < 0.5 * maximum(abs.(mu_free .- offset))
end

@testset "train_ensemble and predict_ensemble" begin
    g, X, targets, truth = _train_case()
    net = PriorNet(size(X, 1); width = 16, depth = 2)
    cfg = TrainConfig(epochs = 150, learning_rate = 1.0e-2,
                      log_every = 50, verbose = false, seed = 100)

    ens, results = train_ensemble(net, X, g, targets; nmembers = 3, config = cfg)
    @test length(ens) == 3
    @test length(results) == 3

    # members must differ: identical seeds would make the epistemic term vanish
    # and the ensemble pointless
    @test results[1].params.net.layer_1.weight != results[2].params.net.layer_1.weight

    mu, sigma, spread = predict_ensemble(ens, X)
    @test length(mu) == ncells(g)
    @test all(isfinite, mu)
    @test all(sigma .> 0)
    @test all(spread .>= 0)

    # total uncertainty must exceed the disagreement alone, since it also carries
    # each member's own predicted spread
    @test all(sigma .>= spread)

    # and it must exceed the mean member sigma, which is the property that makes
    # the ensemble worth its cost
    members_sigma = map(ens.members) do m
        (_, s), _ = predict(ens.net, X, m.params.net, m.state)
        collect(Float64, s)
    end
    mean_member = reduce(+, members_sigma) ./ 3
    @test mean(sigma) >= mean(mean_member)

    mu3, sigma3, spread3 = predict_ensemble_grid(ens, X, size(g))
    @test size(mu3) == size(g)
    @test vec(mu3) ≈ mu
    @test vec(spread3) ≈ spread

    @test_throws DimensionMismatch predict_ensemble_grid(ens, X, (2, 2, 2))
    @test_throws ArgumentError train_ensemble(net, X, g, targets; nmembers = 0)
end

@testset "predict_ensemble on a single member has no disagreement" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2)
    ens, _ = train_ensemble(net, X, g, targets; nmembers = 1,
                            config = TrainConfig(epochs = 20, log_every = 20, verbose = false))

    mu, sigma, spread = predict_ensemble(ens, X)
    @test all(≈(0.0; atol = 1e-12), spread)

    (mu1, sigma1), _ = predict(net, X, ens.members[1].params.net, ens.members[1].state)
    @test mu ≈ mu1
    @test sigma ≈ sigma1
end

@testset "save_prior and load_prior round trip" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2,
                   log_rho_bounds = (0.5, 4.0), sigma_bounds = (0.03, 0.9),
                   residual_span = 1.25)
    res = train_prior(net, X, g, targets;
                      config = TrainConfig(epochs = 20, log_every = 10, verbose = false))

    path = joinpath(mktempdir(), "nested", "prior.jld2")
    save_prior(path, net, res.params, res.history; meta = Dict("width" => 8, "depth" => 2))
    @test isfile(path)

    loaded = load_prior(path)
    @test loaded.nin == net.nin
    @test loaded.log_rho_bounds == (0.5, 4.0)
    @test loaded.sigma_bounds == (0.03, 0.9)
    @test loaded.residual_span ≈ 1.25
    @test loaded.meta["width"] == 8
    @test length(loaded.history) == length(res.history)
    @test loaded.params.coupling.slope ≈ res.params.coupling.slope
    @test loaded.params.net.layer_1.weight ≈ res.params.net.layer_1.weight

    # the loaded parameters must reproduce the same prediction, which is the only
    # thing a checkpoint is actually for
    (mu_orig, _), _ = predict(net, X, res.params.net, res.state)
    (mu_load, _), _ = predict(net, X, loaded.params.net, res.state)
    @test mu_orig ≈ mu_load

    @test_throws ArgumentError load_prior(joinpath(mktempdir(), "missing.jld2"))
end

@testset "checkpointing during training" begin
    g, X, targets, _ = _train_case()
    net = PriorNet(size(X, 1); width = 8, depth = 2)
    path = joinpath(mktempdir(), "ckpt.jld2")

    train_prior(net, X, g, targets;
                config = TrainConfig(epochs = 40, log_every = 10, verbose = false,
                                     checkpoint_path = path, checkpoint_every = 20))
    @test isfile(path)
    @test load_prior(path).nin == net.nin
end

@testset "data-driven sigma targets via sigma_drive" begin
    # build a problem where MT sites constrain some columns but not others
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0, 225.0];
                  origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    periods = 10 .^ range(-2, 2; length = 8)
    f = 1 ./ periods
    a, p = mt1d_apparent(f, [100.0], Float64[])
    sites = MTSites([-250.0, 250.0], [0.0, 0.0], periods, hcat(a, a), hcat(p, p))
    cells = [(2, 2), (3, 2)]

    obs = GravityObs([-400.0, 0.0, 400.0], [0.0, 0.0, 0.0], zeros(3),
                     [-0.3, 0.8, -0.2], fill(0.05, 3))
    s = build_features(g; gravity = obs, sites = sites)
    X = encode_features(s; n_bands = 2)
    A = gravity_matrix(g, obs.x, obs.y, obs.z)
    reference = fill(2.0, n)

    targets = PriorTargets(
        gravity = (A, obs.value, obs.err),
        mt = (sites, cells),
        reference = reference,
        sigma_target = 0.35,
        sigma_drive = SigmaDriveConfig(alpha = 0.5, beta = 0.3, baseline = 0.15),
    )

    net = PriorNet(size(X, 1); width = 32, depth = 3,
                   sigma_bounds = (0.05, 0.9))

    cfg = TrainConfig(epochs = 100, learning_rate = 3.0e-3,
                      log_every = 50, verbose = false, seed = 42,
                      weights = LossWeights(gravity = 1.0, mt = 1.0,
                                            smooth = 0.01, sigma = 0.1,
                                            reference = 0.0))
    res = train_prior(net, X, g, targets; config = cfg, offset = reference)

    # 1. training should still converge
    @test res.best_loss < res.history[1].total

    # 2. history must contain sigma target statistics
    h = res.history[end]
    @test haskey(h, :sigma_target_mean)
    @test haskey(h, :sigma_target_std)
    @test isfinite(h.sigma_target_mean)
    @test isfinite(h.sigma_target_std)

    # 3. sigma targets should vary across cells (not all 0.35)
    @test h.sigma_target_std > 0.0
    @test h.sigma_target_mean != 0.35

    # 4. network σ should track the data-driven map, not sit at 0.35
    (mu, sigma), _ = predict(net, X, res.params.net, res.state; offset = reference)
    @test std(sigma) > 0.01
    @test !isapprox(median(sigma), 0.35; atol = 0.02)
    st_final = compute_sigma_targets(collect(Float64, mu), g, targets,
                                     coupling_of(res.params);
                                     config = targets.sigma_drive)
    @test cor(collect(Float64, sigma), st_final) > 0.2

    # 5. backward compat: without sigma_drive, same setup gives no sigma_target_mean
    #    and warns that σ will collapse to the scalar target
    targets_old = PriorTargets(
        gravity = (A, obs.value, obs.err),
        mt = (sites, cells),
        reference = reference,
        sigma_target = 0.35,
    )
    res_old = @test_logs (:warn, r"sigma_drive") match_mode = :any begin
        train_prior(net, X, g, targets_old; config = cfg, offset = reference)
    end
    @test !haskey(res_old.history[end], :sigma_target_mean)
end
