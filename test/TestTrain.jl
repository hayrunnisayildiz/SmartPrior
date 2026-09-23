using Test
using SmartPrior
using Random
using Statistics

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

@testset "init_params" begin
    net = PriorNet(6; width = 8, depth = 2)
    params, st = init_params(Xoshiro(1), net)
    @test haskey(params, :net)
    @test !haskey(params, :coupling)
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

    mu, sigma, _ = predict_grid(net, X, res.params.net, res.state, size(g))
    cells, values, _ = targets.anchors
    @test all(abs.(vec(mu)[cells] .- truth) .< 0.15)
    @test mean(vec(sigma)[cells]) < 0.5
end

@testset "train_prior returns the best parameters, not the last" begin
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
    for k in (:epoch, :saturation, :total, :anchor, :smooth, :sigma, :reference)
        @test haskey(h, k)
    end
    @test !haskey(h, :slope)
    @test !haskey(h, :gravity)
    @test !haskey(h, :mt)
    @test isfinite(h.anchor)
    @test isnan(h.reference)
    @test 0.0 <= h.saturation <= 1.0
end

@testset "the history reports saturation when the band is too narrow" begin
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0];
                  origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    s = build_features(g)
    X = encode_features(s; n_bands = 2)

    base = fill(2.0, n)
    cells = collect(1:n)
    targets = PriorTargets(anchors = (cells, fill(4.0, n), fill(1.0, n)))

    tight = PriorNet(size(X, 1); width = 16, depth = 2, residual_span = 0.1,
                     log_rho_bounds = (0.0, 5.0))
    res_tight = train_prior(tight, X, g, targets;
                            config = TrainConfig(epochs = 400, learning_rate = 1.0e-2,
                                                 log_every = 100, verbose = false),
                            offset = base)
    @test res_tight.history[end].saturation > 0.5

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

@testset "reference damping pulls residual-mode mu toward the offset" begin
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
    @test results[1].params.net.layer_1.weight != results[2].params.net.layer_1.weight

    mu, sigma, spread = predict_ensemble(ens, X)
    @test length(mu) == ncells(g)
    @test all(isfinite, mu)
    @test all(sigma .> 0)
    @test all(spread .>= 0)
    @test all(sigma .>= spread)

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
    @test loaded.sigma_bounds_per == net.sigma_bounds_per
    @test loaded.residual_span ≈ 1.25
    @test loaded.meta["width"] == 8
    @test length(loaded.history) == length(res.history)
    @test loaded.params.net.layer_1.weight ≈ res.params.net.layer_1.weight

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

@testset "train_prior fits four named anchor groups" begin
    g = PriorGrid(fill(500.0, 4), fill(500.0, 4), [100.0, 150.0];
                  origin = [-1000.0, -1000.0, 0.0])
    n = ncells(g)
    s = build_features(g)
    X = encode_features(s; n_bands = 2)
    names = ["grade", "density", "susceptibility", "resistivity"]
    cells = collect(1:min(20, n))
    targets = PriorTargets(
        anchors_grade = (cells, fill(1.0, length(cells)), ones(length(cells))),
        anchors_density = (cells, fill(2.5, length(cells)), ones(length(cells))),
        anchors_susceptibility = (cells, fill(1.5, length(cells)), ones(length(cells))),
        anchors_resistivity = (cells, fill(2.0, length(cells)), ones(length(cells))),
        property_names = names,
        sigma_target = 0.5,
    )
    net = PriorNet(size(X, 1); width = 32, depth = 3, nproperties = 4,
                   property_names = names,
                   mu_bounds = [(0.0, 3.0), (1.5, 3.5), (0.0, 3.0), (0.0, 4.0)],
                   sigma_bounds = (0.05, 1.2))
    res = train_prior(net, X, g, targets;
                      config = TrainConfig(epochs = 200, learning_rate = 5.0e-3,
                                           log_every = 50, verbose = false, seed = 8,
                                           weights = LossWeights(smooth = 1.0e-4, sigma = 1.0e-3)))
    @test res.best_loss < res.history[1].total
    @test isfinite(res.history[1].grade)
    (mus, sigmas), _ = predict(net, X, res.params.net, res.state)
    @test size(mus) == (4, n)
    @test mean(abs.(mus[1, cells] .- 1.0)) < 0.4
    @test mean(abs.(mus[2, cells] .- 2.5)) < 0.4
end
