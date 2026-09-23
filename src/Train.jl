# Training loop and deep ensemble.
#
# Training is full batch: every cell enters every step. The smoothness term
# couples neighbouring cells, so a minibatch of scattered cells would give a
# biased estimate. Prior grids run to at most a few hundred thousand cells, so
# a full-batch step is cheap.
#
# The ensemble exists because a single heteroscedastic network only reports the
# spread it can explain from the inputs. Training several networks from
# different initialisations and taking the spread of their means recovers the
# part that is "a different set of weights would have concluded something else".

"""
    init_params(rng, net::PriorNet; precision=Float64) -> (params, state)

Initial trainable parameters. Stored as `(net = ps,)` so Optimisers.jl updates
the network weights and checkpoints keep the same `.net` field that examples
already read.
"""
function init_params(rng::AbstractRNG, net::PriorNet;
                     precision::Type{<:AbstractFloat} = Float64)
    ps, st = setup_prior(rng, net; precision = precision)
    return (net = ps,), st
end

"""
    TrainConfig(; epochs=2000, learning_rate=1.0e-3, weights=LossWeights(),
                log_every=100, patience=0, min_delta=0.0, seed=1,
                checkpoint_path="", checkpoint_every=0)

Optimiser and bookkeeping settings.

`patience` above zero stops training after that many logged evaluations without
an improvement of at least `min_delta`. `checkpoint_every` above zero writes the
running best parameters to `checkpoint_path` at that epoch interval.
"""
Base.@kwdef struct TrainConfig
    epochs::Int = 2000
    learning_rate::Float64 = 1.0e-3
    weights::LossWeights = LossWeights()
    log_every::Int = 100
    patience::Int = 0
    min_delta::Float64 = 0.0
    seed::Int = 1
    checkpoint_path::String = ""
    checkpoint_every::Int = 0
    verbose::Bool = true
end

"""
    TrainResult

Outcome of a training run.

- `params`, `state`: the best parameters seen, not the last ones.
- `history`: one entry per logged epoch with the total, each unweighted term,
  and `saturation`.
- `best_loss`, `best_epoch`: where `params` came from.
"""
struct TrainResult{P,S}
    params::P
    state::S
    history::Vector{NamedTuple}
    best_loss::Float64
    best_epoch::Int
end

function _with(c::TrainConfig; kwargs...)
    base = NamedTuple{fieldnames(TrainConfig)}(
        ntuple(i -> getfield(c, i), fieldcount(TrainConfig)))
    return TrainConfig(; merge(base, NamedTuple(kwargs))...)
end

function _train_loss(net::PriorNet, X, st, grid, targets, weights, offset, p)
    pred, _ = predict(net, X, p.net, st; offset = offset)
    return prior_loss(pred[1], pred[2], grid, targets, weights)
end

function _net_params(p)
    return hasproperty(p, :net) ? p : (net = p,)
end

function _saturation(net::PriorNet, mu::AbstractVector,
                     offset::Union{Nothing,AbstractVector}; tol::Real = 1.0e-3)
    lo, hi = net.mu_bounds[1]
    n = length(mu)
    n == 0 && return 0.0
    hit = 0
    @inbounds for c in 1:n
        l, u = if offset === nothing
            lo, hi
        else
            o = clamp(offset[c], lo, hi)
            max(lo, o - net.residual_spans[1]), min(hi, o + net.residual_spans[1])
        end
        (mu[c] <= l + tol || mu[c] >= u - tol) && (hit += 1)
    end
    return hit / n
end

function _saturation(net::PriorNet, mus::AbstractMatrix,
                     offset::Union{Nothing,AbstractVector}; tol::Real = 1.0e-3)
    P, n = size(mus)
    n == 0 && return 0.0
    hit = 0
    @inbounds for p in 1:P
        lo, hi = net.mu_bounds[p]
        for c in 1:n
            (mus[p, c] <= lo + tol || mus[p, c] >= hi - tol) && (hit += 1)
        end
    end
    return hit / (P * n)
end

"""
    train_prior(net, X, grid, targets; config=TrainConfig(), offset=nothing, rng=nothing)
        -> TrainResult

Fit one prior field.

`X` is the `[nfeature, ncell]` input from [`encode_features`](@ref), `offset` the
optional baseline that puts the network in residual mode.

Without anchors, `sigma_penalty` is the only gradient on σ and it pulls every
cell to the scalar `sigma_target`.
"""
function train_prior(net::PriorNet,
                     X::AbstractMatrix,
                     grid::PriorGrid,
                     targets::PriorTargets;
                     config::TrainConfig = TrainConfig(),
                     offset::Union{Nothing,AbstractVector} = nothing,
                     rng::Union{Nothing,AbstractRNG} = nothing,
                     start_params = nothing,
                     epoch_offset::Integer = 0)
    config.epochs > 0 || throw(ArgumentError("train_prior: epochs must be positive"))
    config.log_every > 0 || throw(ArgumentError("train_prior: log_every must be positive"))
    size(X, 2) == ncells(grid) || throw(DimensionMismatch(
        "train_prior: X has $(size(X, 2)) cells but the grid has $(ncells(grid))"))

    generator = rng === nothing ? Xoshiro(config.seed) : rng
    params, st = init_params(generator, net)
    if start_params !== nothing
        params = _net_params(start_params)
    end
    epoch_offset >= 0 || throw(ArgumentError("train_prior: epoch_offset must be ≥ 0"))

    if offset !== nothing && targets.reference !== nothing &&
       config.weights.reference <= 0
        @warn """train_prior: residual-mode training with an unpenalised \
                 residual. The network may move residual_span from the \
                 baseline at no cost. Set LossWeights(reference = ...) so \
                 structure only survives if it buys more misfit reduction \
                 than it costs."""
    end

    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), params)

    history = NamedTuple[]
    best_loss = Inf
    best_epoch = 0
    best_params = params
    stale = 0

    if start_params !== nothing && epoch_offset > 0
        (mu0, sigma0), _ = predict(net, X, params.net, st; offset = offset)
        report0 = loss_report(mu0, sigma0, grid, targets, config.weights)
        if isfinite(report0.total)
            best_loss = report0.total
            best_epoch = Int(epoch_offset)
            best_params = params
            if config.verbose
                @printf("resume seed  epoch %d  total %.5e  (protects pre-resume best)\n",
                        best_epoch, best_loss)
                flush(stdout)
            end
        end
    end

    for epoch in 1:config.epochs
        logged_epoch = epoch + Int(epoch_offset)

        lossfn = p -> _train_loss(net, X, st, grid, targets, config.weights, offset, p)
        val, grads = Zygote.withgradient(lossfn, params)
        isfinite(val) || error("train_prior: loss became $(val) at epoch $(logged_epoch)")
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])

        if epoch % config.log_every == 0 || epoch == 1 || epoch == config.epochs
            (mu, sigma), _ = predict(net, X, params.net, st; offset = offset)
            report = loss_report(mu, sigma, grid, targets, config.weights)
            sat = _saturation(net, mu, offset)
            entry = merge((epoch = logged_epoch, saturation = sat), report)
            push!(history, entry)

            if config.verbose
                @printf("epoch %6d  total %.5e  sat %.3f",
                        logged_epoch, report.total, sat)
                for (label, term_field, w_field) in (
                        ("grade", :grade, :grade),
                        ("dens", :density, :density),
                        ("sus", :susceptibility, :susceptibility),
                        ("res", :resistivity, :resistivity),
                        ("cond", :conductivity_100kHz, :conductivity),
                        ("anc", :anchor, :anchor),
                        ("sm", :smooth, :smooth),
                        ("sig", :sigma, :sigma))
                    t = getfield(report, term_field)
                    isfinite(t) || continue
                    w = getfield(config.weights, w_field)
                    share = report.total != 0 ? 100 * w * t / report.total : NaN
                    @printf("  %s %.1f%%", label, share)
                end
                println()
                flush(stdout)
            end

            if report.total < best_loss - config.min_delta
                best_loss = report.total
                best_epoch = logged_epoch
                best_params = params
                stale = 0
            else
                stale += 1
                if config.patience > 0 && stale >= config.patience
                    config.verbose && @printf("early stop at epoch %d (best %.5e at %d)\n",
                                              logged_epoch, best_loss, best_epoch)
                    break
                end
            end
        end

        if config.checkpoint_every > 0 && !isempty(config.checkpoint_path) &&
           epoch % config.checkpoint_every == 0
            save_prior(config.checkpoint_path, net, best_params, history)
        end
    end

    return TrainResult(best_params, st, history, best_loss, best_epoch)
end

#---------- ensemble ----------

"""
    PriorEnsemble(net, members)

A trained deep ensemble: one shared architecture, several parameter sets.
"""
struct PriorEnsemble{N,M}
    net::N
    members::Vector{M}
end

Base.length(e::PriorEnsemble) = length(e.members)

"""
    train_ensemble(net, X, grid, targets; nmembers=5, config=TrainConfig(),
                   offset=nothing) -> (PriorEnsemble, Vector{TrainResult})

Train `nmembers` independent fits, seeding member `k` from `config.seed + k - 1`.
"""
function train_ensemble(net::PriorNet,
                        X::AbstractMatrix,
                        grid::PriorGrid,
                        targets::PriorTargets;
                        nmembers::Integer = 5,
                        config::TrainConfig = TrainConfig(),
                        offset::Union{Nothing,AbstractVector} = nothing)
    nmembers >= 1 || throw(ArgumentError("train_ensemble: nmembers must be at least 1"))

    results = TrainResult[]
    for k in 1:nmembers
        cfg = _with(config;
                    seed = config.seed + k - 1,
                    checkpoint_path = isempty(config.checkpoint_path) ? "" :
                                      _member_path(config.checkpoint_path, k))
        config.verbose && @printf("--- ensemble member %d/%d (seed %d) ---\n",
                                  k, nmembers, cfg.seed)
        push!(results, train_prior(net, X, grid, targets;
                                   config = cfg, offset = offset))
    end

    members = [(params = r.params, state = r.state) for r in results]
    return PriorEnsemble(net, members), results
end

function _member_path(path::AbstractString, k::Integer)
    base, ext = splitext(path)
    return string(base, "_member", k, isempty(ext) ? ".jld2" : ext)
end

"""
    predict_ensemble(e::PriorEnsemble, X; offset=nothing) -> (mu, sigma, spread)

Combined prediction of an ensemble.

`mu` is the member mean. `sigma` combines the two sources of uncertainty as

    sigma^2 = mean_k(sigma_k^2) + var_k(mu_k)
"""
function predict_ensemble(e::PriorEnsemble, X::AbstractMatrix;
                          offset::Union{Nothing,AbstractVector} = nothing)
    isempty(e.members) && throw(ArgumentError("predict_ensemble: empty ensemble"))

    K = length(e.members)
    mus = Vector{Vector{Float64}}(undef, K)
    vars = Vector{Vector{Float64}}(undef, K)

    for (k, m) in enumerate(e.members)
        (mu, sigma), _ = predict(e.net, X, m.params.net, m.state; offset = offset)
        mus[k] = collect(Float64, mu)
        vars[k] = collect(Float64, sigma) .^ 2
    end

    mu_bar = reduce(+, mus) ./ K
    aleatoric = reduce(+, vars) ./ K
    epistemic = reduce(+, (abs2.(m .- mu_bar) for m in mus)) ./ K

    return mu_bar, sqrt.(aleatoric .+ epistemic), sqrt.(epistemic)
end

"""
    predict_ensemble_grid(e, X, dims; offset=nothing) -> (mu, sigma, spread)

As [`predict_ensemble`](@ref) but reshaped onto a `(nx, ny, nz)` grid.
"""
function predict_ensemble_grid(e::PriorEnsemble, X::AbstractMatrix, dims::Tuple{Int,Int,Int};
                               offset::Union{Nothing,AbstractVector} = nothing)
    prod(dims) == size(X, 2) || throw(DimensionMismatch(
        "predict_ensemble_grid: dims $(dims) imply $(prod(dims)) cells but got $(size(X, 2))"))
    mu, sigma, spread = predict_ensemble(e, X; offset = offset)
    return reshape(mu, dims), reshape(sigma, dims), reshape(spread, dims)
end

#---------- persistence ----------

"""
    save_prior(path, net::PriorNet, params, history; meta=Dict())

Write parameters and training history to a JLD2 file.
"""
function save_prior(path::AbstractString, net::PriorNet, params, history;
                    meta::AbstractDict = Dict{String,Any}())
    mkpath(dirname(abspath(path)))
    JLD2.jldsave(String(path);
                 params = params,
                 history = history,
                 nin = net.nin,
                 log_rho_bounds = net.log_rho_bounds,
                 residual_span = net.residual_span,
                 sigma_bounds = net.sigma_bounds,
                 sigma_bounds_per = net.sigma_bounds_per,
                 meta = Dict{String,Any}(meta))
    return path
end

"""
    load_prior(path) -> NamedTuple

Read a file written by [`save_prior`](@ref), returning `params`, `history`, the
stored network configuration and `meta`.
"""
function load_prior(path::AbstractString)
    isfile(path) || throw(ArgumentError("load_prior: no such file: $path"))
    return JLD2.jldopen(String(path), "r") do file
        (params = file["params"],
         history = file["history"],
         nin = file["nin"],
         log_rho_bounds = file["log_rho_bounds"],
         residual_span = file["residual_span"],
         sigma_bounds = file["sigma_bounds"],
         sigma_bounds_per = haskey(file, "sigma_bounds_per") ? file["sigma_bounds_per"] : nothing,
         meta = file["meta"])
    end
end
