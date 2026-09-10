# Training loop and deep ensemble.
#
# Training is full batch: every cell enters every step. That is not a shortcut
# taken for simplicity but a consequence of the objective. The smoothness term
# couples neighbouring cells, the gravity term sums over the whole volume for
# each station, and the MT term reads entire columns; a minibatch of scattered
# cells would give a biased estimate of all three. Prior grids run to at most a
# few hundred thousand cells, so a full-batch step is cheap.
#
# The ensemble exists because a single heteroscedastic network only reports the
# spread it can explain from the inputs. It has no way to express "this is a
# region where a different set of weights would have concluded something else",
# which for a prior over an underdetermined inverse problem is most of the
# uncertainty that matters. Training several networks from different
# initialisations and taking the spread of their means recovers that part.

# Maps an unbounded raw parameter into `slope_bounds`, which is how the sign of
# the density-resistivity coupling can be fixed while its magnitude stays free.
# Infinite bounds mean no constraint and the raw value passes through.
function _map_slope(raw::Real, bounds::Tuple{Real,Real})
    lo, hi = bounds
    (isfinite(lo) && isfinite(hi)) || return raw
    return lo + (hi - lo) * sigmoid(raw)
end

function _unmap_slope(slope::Real, bounds::Tuple{Real,Real})
    lo, hi = bounds
    (isfinite(lo) && isfinite(hi)) || return slope
    lo < hi || throw(ArgumentError("slope_bounds must be increasing, got $(bounds)"))
    lo <= slope <= hi || throw(ArgumentError(
        "initial slope $(slope) is outside slope_bounds $(bounds)"))
    t = clamp((slope - lo) / (hi - lo), 1.0e-6, 1 - 1.0e-6)
    return log(t / (1 - t))
end

"""
    init_params(rng, net::PriorNet; slope=nothing, offset=0.0,
                slope_bounds=(-Inf, Inf), precision=Float64) -> (params, state)

Initial trainable parameters: network weights plus the gravity coupling.

The coupling scalars are stored as one-element vectors because Optimisers.jl only
updates array leaves and would silently leave bare scalars frozen.

`slope_bounds` restricts the coupling slope, in units of [`DENSITY_SCALE`](@ref)
per decade, and is worth setting whenever a gravity term is used. The sign of the
density-resistivity relationship is petrophysics, not something gravity data can
determine: a conductive body that is also dense, such as a sulphide or a mafic
intrusion, has a negative slope, while a resistive dense body has a positive one.
Left free, the fit will happily choose the wrong sign and place resistive
material where the mass is, because gravity constrains only the density field and
any sign can be absorbed by flipping the resistivity structure. The result fits
the data as well as the truth does while being anti-correlated with it, which is
worse than having no prior at all. Bounding the slope to the sign you know is what
turns the gravity term from ambiguous into informative.

`slope` defaults to the centre of `slope_bounds`, or zero when they are infinite,
so gravity begins decoupled and the network has to earn the connection.
"""
function init_params(rng::AbstractRNG, net::PriorNet;
                     slope::Union{Nothing,Real} = nothing,
                     offset::Real = 0.0,
                     slope_bounds::Tuple{Real,Real} = (-Inf, Inf),
                     precision::Type{<:AbstractFloat} = Float64)
    ps, st = setup_prior(rng, net; precision = precision)
    bounds = (Float64(slope_bounds[1]), Float64(slope_bounds[2]))
    s0 = if slope !== nothing
        Float64(slope)
    elseif all(isfinite, bounds)
        (bounds[1] + bounds[2]) / 2
    else
        0.0
    end
    params = (net = ps,
              coupling = (slope = [_unmap_slope(s0, bounds)],
                          offset = [Float64(offset)],
                          slope_bounds = bounds))
    return params, st
end

"""
    coupling_of(params) -> GravityCoupling

Read the gravity coupling out of a training parameter set, applying the slope
constraint recorded in it by [`init_params`](@ref).

`slope_bounds` is a plain tuple rather than an array, so Optimisers.jl treats it
as a frozen leaf and it rides along with the parameters without being trained.
"""
function coupling_of(params)
    bounds = hasproperty(params.coupling, :slope_bounds) ?
             params.coupling.slope_bounds : (-Inf, Inf)
    return GravityCoupling(_map_slope(params.coupling.slope[1], bounds),
                           params.coupling.offset[1])
end

"""
    TrainConfig(; epochs=2000, learning_rate=1.0e-3, weights=LossWeights(),
                log_every=100, patience=0, min_delta=0.0, seed=1,
                checkpoint_path="", checkpoint_every=0)

Optimiser and bookkeeping settings.

`patience` above zero stops training after that many logged evaluations without
an improvement of at least `min_delta`. `checkpoint_every` above zero writes the
running best parameters to `checkpoint_path` at that epoch interval.

`slope_bounds` is forwarded to [`init_params`](@ref); read the note there before
running a gravity term without it.
"""
Base.@kwdef struct TrainConfig
    epochs::Int = 2000
    learning_rate::Float64 = 1.0e-3
    weights::LossWeights = LossWeights()
    slope_bounds::Tuple{Float64,Float64} = (-Inf, Inf)
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

- `params`, `state`: the best parameters seen, not the last ones. Full-batch Adam
  on a multi-term objective does not descend monotonically, so the final step is
  frequently worse than one taken earlier.
- `history`: one entry per logged epoch with the total, each unweighted term, the
  coupling slope and `saturation`.
- `best_loss`, `best_epoch`: where `params` came from.

Watch `saturation`, the fraction of cells whose `mu` has been driven to the edge
of its reachable band. Beyond a few per cent it means the objective is asking for
something the band cannot express, and because `tanh` has no gradient there the
affected cells stop learning entirely. The usual causes are a `residual_span`
narrower than the structure actually present, or one loss term outweighing the
rest so heavily that saturating is the cheapest move available.
"""
struct TrainResult{P,S}
    params::P
    state::S
    history::Vector{NamedTuple}
    best_loss::Float64
    best_epoch::Int
end

# Copy of a config with some fields replaced. Enumerating the fields at each call
# site instead would silently drop any field added later, which is how an option
# ends up being honoured by `train_prior` but ignored by `train_ensemble`.
function _with(c::TrainConfig; kwargs...)
    base = NamedTuple{fieldnames(TrainConfig)}(
        ntuple(i -> getfield(c, i), fieldcount(TrainConfig)))
    return TrainConfig(; merge(base, NamedTuple(kwargs))...)
end

function _train_loss(net::PriorNet, X, st, grid, targets, weights, offset, p)
    (mu, sigma), _ = predict(net, X, p.net, st; offset = offset)
    return prior_loss(mu, sigma, coupling_of(p), grid, targets, weights)
end

# fraction of cells sitting within `tol` of the edge of their reachable band.
# tanh is flat there, so these cells have stopped contributing a gradient
function _saturation(net::PriorNet, mu::AbstractVector,
                     offset::Union{Nothing,AbstractVector}; tol::Real = 1.0e-3)
    lo, hi = net.log_rho_bounds
    n = length(mu)
    n == 0 && return 0.0
    hit = 0
    @inbounds for c in 1:n
        l, u = if offset === nothing
            lo, hi
        else
            o = clamp(offset[c], lo, hi)
            max(lo, o - net.residual_span), min(hi, o + net.residual_span)
        end
        (mu[c] <= l + tol || mu[c] >= u - tol) && (hit += 1)
    end
    return hit / n
end

"""
    train_prior(net, X, grid, targets; config=TrainConfig(), offset=nothing, rng=nothing)
        -> TrainResult

Fit one prior field.

`X` is the `[nfeature, ncell]` input from [`encode_features`](@ref), `offset` the
optional baseline that puts the network in residual mode.

When `targets.sigma_drive` is set, per-cell sigma targets are computed from the
current `mu` field at each epoch using [`compute_sigma_targets`](@ref). The targets
are computed **outside** the AD graph, so no gradient flows through them back to
`mu`. This is the EM-style update: the network's current best guess determines
what the uncertainty targets should be, and those targets then steer the next
gradient step.

Without `sigma_drive` and without anchors, `sigma_penalty` is the only gradient
on σ and it pulls every cell to the scalar `sigma_target`. A warning is emitted
when MT data are present in that configuration, because the search intervals
would then be a constant rather than a function of the data.
"""
function train_prior(net::PriorNet,
                     X::AbstractMatrix,
                     grid::PriorGrid,
                     targets::PriorTargets;
                     config::TrainConfig = TrainConfig(),
                     offset::Union{Nothing,AbstractVector} = nothing,
                     rng::Union{Nothing,AbstractRNG} = nothing)
    config.epochs > 0 || throw(ArgumentError("train_prior: epochs must be positive"))
    config.log_every > 0 || throw(ArgumentError("train_prior: log_every must be positive"))
    size(X, 2) == ncells(grid) || throw(DimensionMismatch(
        "train_prior: X has $(size(X, 2)) cells but the grid has $(ncells(grid))"))

    generator = rng === nothing ? Xoshiro(config.seed) : rng
    params, st = init_params(generator, net; slope_bounds = config.slope_bounds)

    if targets.gravity !== nothing && !all(isfinite, config.slope_bounds)
        @warn """train_prior: fitting gravity with an unbounded coupling slope. \
                 Gravity cannot determine the sign of the density-resistivity \
                 relationship, so the fit may place resistive material where the \
                 mass is and return a prior anti-correlated with the truth. Set \
                 TrainConfig(slope_bounds = ...) to the sign your target's \
                 petrophysics implies."""
    end

    if offset !== nothing && targets.reference !== nothing &&
       config.weights.reference <= 0
        @warn """train_prior: residual-mode training with an unpenalised \
                 residual. The network may move residual_span decades from the \
                 baseline at no cost; the symptom is data terms settling well \
                 below one while the prior scores worse than the baseline it \
                 started from. Set LossWeights(reference = ...) so structure \
                 only survives if it buys more misfit reduction than it costs."""
    end

    if targets.sigma_drive === nothing && targets.anchors === nothing &&
       targets.mt !== nothing
        @warn """train_prior: MT data are present but σ has no data term. \
                 Anchors are absent, so heteroscedastic_nll never fires, and \
                 sigma_penalty will pull every cell toward sigma_target=\
                 $(targets.sigma_target). Set PriorTargets(sigma_drive = \
                 SigmaDriveConfig()) so search intervals follow MT column \
                 residual and gravity sensitivity."""
    end

    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), params)

    history = NamedTuple[]
    best_loss = Inf
    best_epoch = 0
    best_params = params
    stale = 0

    use_sigma_drive = targets.sigma_drive !== nothing

    for epoch in 1:config.epochs
        # ---- data-driven sigma targets (outside AD) ----
        effective_targets = if use_sigma_drive
            (mu_now, _), _ = predict(net, X, params.net, st; offset = offset)
            sigma_vec = compute_sigma_targets(
                collect(Float64, mu_now), grid, targets, coupling_of(params);
                config = targets.sigma_drive)
            PriorTargets(
                anchors = targets.anchors,
                gravity = targets.gravity,
                mt = targets.mt,
                reference = targets.reference,
                gravity_detrend = targets.gravity_detrend,
                sigma_target = targets.sigma_target,
                sigma_drive = targets.sigma_drive,
                sigma_target_vec = sigma_vec,
                phase_weight = targets.phase_weight,
                vertical_weight = targets.vertical_weight,
                reference_weight = targets.reference_weight,
            )
        else
            targets
        end

        lossfn = p -> _train_loss(net, X, st, grid, effective_targets, config.weights, offset, p)
        val, grads = Zygote.withgradient(lossfn, params)
        isfinite(val) || error("train_prior: loss became $(val) at epoch $(epoch)")
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])

        if epoch % config.log_every == 0 || epoch == 1 || epoch == config.epochs
            (mu, sigma), _ = predict(net, X, params.net, st; offset = offset)
            report = loss_report(mu, sigma, coupling_of(params), grid,
                                 effective_targets, config.weights)
            sat = _saturation(net, mu, offset)
            slope = coupling_of(params).slope
            entry = merge((epoch = epoch, slope = slope, saturation = sat), report)

            # record sigma target statistics when data-driven
            if use_sigma_drive && effective_targets.sigma_target_vec !== nothing
                stv = effective_targets.sigma_target_vec
                entry = merge(entry, (sigma_target_mean = mean(stv),
                                      sigma_target_std = std(stv)))
            end

            push!(history, entry)

            config.verbose && @printf("epoch %6d  total %.5e  slope %+.3f  sat %.3f\n",
                                      epoch, report.total, slope, sat)

            if report.total < best_loss - config.min_delta
                best_loss = report.total
                best_epoch = epoch
                best_params = params
                stale = 0
            else
                stale += 1
                if config.patience > 0 && stale >= config.patience
                    config.verbose && @printf("early stop at epoch %d (best %.5e at %d)\n",
                                              epoch, best_loss, best_epoch)
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

Five members is the usual choice in the deep-ensemble literature and is enough
for a stable spread; the cost is linear, so raise it if the runs disagree a lot.
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

the first term being what each network believes it cannot resolve (aleatoric),
the second how much the members disagree (epistemic). `spread` returns the second
term's square root on its own, which is the useful diagnostic: where it dominates
the prior is poorly determined by the data rather than genuinely variable, and
more members or more constraints would help.
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

Only the scalar configuration of `net` is stored, not the layer objects: they
hold activation functions, and serialising functions across sessions is fragile.
The reconstructing script is expected to build the same architecture, so keep the
`width`, `depth` and activation choices in `meta` if they are not fixed in code.
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
                 meta = Dict{String,Any}(meta))
    return path
end

"""
    load_prior(path) -> NamedTuple

Read a file written by [`save_prior`](@ref), returning `params`, `history`, the
stored network configuration and `meta`.

Check `nin` against the network you rebuild: a feature stack assembled with
different options changes the input width, and a mismatch there is the most
likely way to load parameters into the wrong architecture.
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
         meta = file["meta"])
    end
end
