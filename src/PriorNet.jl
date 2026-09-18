# Lux neural field mapping features to a per-cell prior.
#
# The network outputs two numbers per cell: a nominal log10 resistivity `mu` and
# a confidence `sigma`. Together they are the prior: `mu` is the starting model
# and `mu +- k*sigma` the per-cell search interval.
#
# Both outputs are hard-bounded by construction rather than by penalty terms.
# That is the important design choice here. An unbounded heteroscedastic sigma
# gives the network a trivial escape from any cell it cannot explain: inflate
# sigma until the likelihood term stops caring. The resulting interval would be
# as wide as the global bounds, which is precisely the situation this package
# exists to improve on. Bounding sigma from above forces the network to commit;
# bounding it from below stops the likelihood from diverging and keeps intervals
# from collapsing onto a wrong value.

"""
    fourier_encode(X, rows, n_bands; base=2.0) -> Matrix{Float64}

Append sinusoidal positional encodings of selected rows of a `[nfeature, ncell]`
matrix, returning `[nfeature + 2*length(rows)*n_bands, ncell]`.

For each selected row and each band `b in 0:n_bands-1` a `sin` and `cos` of
`pi * base^b * x` is added, so successive bands double in frequency.

A plain MLP on raw coordinates is spectrally biased towards smooth functions and
cannot represent a sharp contact such as a dyke wall or a fault without an
impractical depth. Geometric-frequency encodings remove that bias, which is why
coordinate-network architectures use them almost universally.
"""
function fourier_encode(X::AbstractMatrix{<:Real},
                        rows,
                        n_bands::Integer;
                        base::Real = 2.0)
    n_bands >= 0 || throw(ArgumentError("fourier_encode: n_bands must be non-negative"))
    base > 1 || throw(ArgumentError("fourier_encode: base must exceed 1"))
    nrow, ncol = size(X)
    all(r -> 1 <= r <= nrow, rows) || throw(ArgumentError(
        "fourier_encode: row indices must lie in 1:$(nrow)"))

    (n_bands == 0 || isempty(rows)) && return Matrix{Float64}(X)

    extra = 2 * length(rows) * n_bands
    out = Matrix{Float64}(undef, nrow + extra, ncol)
    out[1:nrow, :] .= X

    r = nrow
    for row in rows, b in 0:(n_bands-1)
        ω = π * float(base)^b
        @views out[r+1, :] .= sin.(ω .* X[row, :])
        @views out[r+2, :] .= cos.(ω .* X[row, :])
        r += 2
    end
    return out
end

"""
    encode_features(s::FeatureStack; fourier_channels=("x_norm", "y_norm", "z_norm"),
                    n_bands=6, base=2.0) -> Matrix{Float64}

Turn a feature stack into the `[nfeature, ncell]` network input, adding Fourier
encodings for the named channels.

Names that are not present are skipped rather than raising, so the same call
works on a stack built with `coordinates = false` for a transferable model, where
the coordinate channels are deliberately absent.

Size the network from the result: `PriorNet(size(X, 1))`.
"""
function encode_features(s::FeatureStack;
                         fourier_channels = ("x_norm", "y_norm", "z_norm"),
                         n_bands::Integer = 6,
                         base::Real = 2.0)
    X = feature_matrix(s)
    rows = Int[]
    for name in fourier_channels
        k = findfirst(==(String(name)), s.names)
        k === nothing || push!(rows, k)
    end
    return fourier_encode(X, rows, n_bands; base = base)
end

"""
    PriorNet(nin; width=128, depth=4, nproperties=1, activation=gelu,
             log_rho_bounds=(0.0, 5.0), residual_span=1.5,
             sigma_bounds=(0.05, 1.2), mu_bounds=nothing,
             residual_spans=nothing, sigma_bounds_per=nothing,
             property_names=nothing)

Neural field mapping `nin` features per cell to `(mu, sigma)` per property.

`nproperties = 1` is the original resistivity prior: the last layer is
`Dense(width => 2)` and [`predict`](@ref) returns two length-`ncell` vectors.
`nproperties = 4` is the Keivitsa layout — grade, density, susceptibility,
resistivity — with `Dense(width => 8)` and a `(4, ncell)` pair of matrices.

- `log_rho_bounds`: physical limits on the first property (log10 resistivity in
  the single-property case), kept so export and VFSA code keep working.
- `mu_bounds`: optional length-`nproperties` vector of `(lo, hi)` squash limits.
  Defaults to `log_rho_bounds` repeated.
- `residual_span`: in residual mode, the largest deviation the network may add
  to the supplied baseline. Only defined for `nproperties = 1`.
- `sigma_bounds`: hard limits on the predicted standard deviation. Used for
  every property unless `sigma_bounds_per` is set.
- `sigma_bounds_per`: optional length-`nproperties` vector of `(lo, hi)` squash
  limits, one pair per property. The network cannot emit σ below `lo` — that is
  the floor that stops a heteroscedastic NLL from driving σ under the scale of
  the observations. [`sigma_bounds_from_anchors`](@ref) builds these from each
  group's weighted std.

Use with [`setup_prior`](@ref) and [`predict`](@ref).
"""
struct PriorNet{M} <: Lux.AbstractLuxWrapperLayer{:model}
    model::M
    nin::Int
    nproperties::Int
    log_rho_bounds::Tuple{Float64,Float64}
    mu_bounds::Vector{NTuple{2,Float64}}
    residual_span::Float64
    residual_spans::Vector{Float64}
    sigma_bounds::Tuple{Float64,Float64}
    sigma_bounds_per::Vector{NTuple{2,Float64}}
    property_names::Vector{String}
end

const DEFAULT_MULTI_PROPERTIES = ["grade", "density", "susceptibility", "resistivity"]

function _fill_bounds(value::Tuple{Float64,Float64}, n::Int)
    return fill(value, n)
end

function _fill_bounds(values::AbstractVector, n::Int)
    length(values) == n || throw(ArgumentError(
        "PriorNet: expected $(n) bound pairs, got $(length(values))"))
    out = Vector{NTuple{2,Float64}}(undef, n)
    for i in 1:n
        lo, hi = Float64(values[i][1]), Float64(values[i][2])
        lo < hi || throw(ArgumentError(
            "PriorNet: bounds must be increasing, got $(values[i]) at property $i"))
        out[i] = (lo, hi)
    end
    return out
end

function PriorNet(nin::Integer;
                  width::Integer = 128,
                  depth::Integer = 4,
                  nproperties::Integer = 1,
                  activation = gelu,
                  log_rho_bounds::Tuple{Real,Real} = (0.0, 5.0),
                  residual_span::Real = 1.5,
                  sigma_bounds::Tuple{Real,Real} = (0.05, 1.2),
                  mu_bounds = nothing,
                  residual_spans = nothing,
                  sigma_bounds_per = nothing,
                  property_names = nothing)
    nin > 0 || throw(ArgumentError("PriorNet: nin must be positive"))
    width > 0 || throw(ArgumentError("PriorNet: width must be positive"))
    depth >= 1 || throw(ArgumentError("PriorNet: depth must be at least 1"))
    nproperties >= 1 || throw(ArgumentError("PriorNet: nproperties must be at least 1"))
    log_rho_bounds[1] < log_rho_bounds[2] ||
        throw(ArgumentError("PriorNet: log_rho_bounds must be increasing"))
    0 < sigma_bounds[1] < sigma_bounds[2] ||
        throw(ArgumentError("PriorNet: sigma_bounds must satisfy 0 < lo < hi"))
    residual_span > 0 || throw(ArgumentError("PriorNet: residual_span must be positive"))

    rho = (Float64(log_rho_bounds[1]), Float64(log_rho_bounds[2]))
    sig = (Float64(sigma_bounds[1]), Float64(sigma_bounds[2]))
    P = Int(nproperties)

    mb = mu_bounds === nothing ? _fill_bounds(rho, P) : _fill_bounds(mu_bounds, P)
    sb = sigma_bounds_per === nothing ? _fill_bounds(sig, P) :
        _fill_bounds(sigma_bounds_per, P)
    for (i, (lo, hi)) in enumerate(sb)
        0 < lo < hi || throw(ArgumentError(
            "PriorNet: sigma_bounds_per[$i] must satisfy 0 < lo < hi, got $((lo, hi))"))
    end

    names = if property_names !== nothing
        nms = String.(collect(property_names))
        length(nms) == P || throw(ArgumentError(
            "PriorNet: property_names has $(length(nms)) entries, nproperties=$P"))
        nms
    elseif P == 1
        ["resistivity"]
    elseif P == 4
        copy(DEFAULT_MULTI_PROPERTIES)
    else
        ["prop$i" for i in 1:P]
    end

    rsp = if residual_spans === nothing
        fill(Float64(residual_span), P)
    else
        length(residual_spans) == P || throw(ArgumentError(
            "PriorNet: residual_spans has $(length(residual_spans)) entries, nproperties=$P"))
        rs = Float64.(residual_spans)
        all(>(0), rs) || throw(ArgumentError("PriorNet: residual_spans must be positive"))
        rs
    end

    hidden = ntuple(_ -> Dense(width => width, activation), depth - 1)
    model = Chain(Dense(nin => width, activation), hidden..., Dense(width => 2 * P))

    return PriorNet(model, Int(nin), P, mb[1], mb, Float64(residual_span), rsp,
                    sig, sb, names)
end

"""
    setup_prior(rng, net::PriorNet; precision=Float64) -> (ps, st)

Initialise parameters and state.

Defaults to `Float64` rather than Lux's `Float32`, because the physics terms in
the loss run in double precision and a mixed-precision graph would promote on
every step anyway. Prior grids are small enough that the cost is irrelevant.
"""
function setup_prior(rng::AbstractRNG, net::PriorNet;
                     precision::Type{<:AbstractFloat} = Float64)
    ps, st = Lux.setup(rng, net)
    precision === Float64 && return Lux.f64(ps), st
    precision === Float32 && return Lux.f32(ps), st
    throw(ArgumentError("setup_prior: precision must be Float32 or Float64"))
end

_squash_mu(raw, lo, hi) = ((lo + hi) / 2) .+ ((hi - lo) / 2) .* tanh.(raw)
_squash_sigma(raw, smin, smax) = smin .+ (smax - smin) .* sigmoid.(raw)

function _squash_mu_residual(raw, offset, lo, hi, span)
    o = clamp.(offset, lo, hi)
    l = max.(lo, o .- span)
    u = min.(hi, o .+ span)
    return (l .+ u) ./ 2 .+ (u .- l) ./ 2 .* tanh.(raw)
end

"""
    predict(net::PriorNet, X, ps, st; offset=nothing) -> ((mu, sigma), st)

Evaluate the field on a `[nfeature, ncell]` input.

For `nproperties = 1` this is the original contract: `mu` in log10 ohm-metres
and `sigma` in decades, both length `ncell`. For `nproperties > 1` both outputs
are `[nproperties, ncell]` matrices, one row per property in
`net.property_names` order, each squashed into that property's `mu_bounds` /
`sigma_bounds_per`.

With `offset === nothing` the network works in absolute mode. Passing `offset`
(normally a Niblett-Bostick baseline from [`nb_baseline`](@ref)) switches to
residual mode, where `mu` is squashed into the band `offset ± residual_span`
*intersected with* `log_rho_bounds`. Residual mode is only defined for a
single-property net: a four-property field has no shared baseline to residual
against.

The intersection matters. Without it a wide residual span lets `mu` leave the
physical range entirely, and since `mu` is written out as the inversion's starting
model, that is not something the export step can repair after the fact. The cost
is that the reachable band becomes asymmetric about the offset wherever the
offset sits within `residual_span` of a physical bound, which is the correct
behaviour: the field should not be able to propose resistivity it has been told
is impossible.
"""
function predict(net::PriorNet, X::AbstractMatrix, ps, st;
                 offset::Union{Nothing,AbstractVector} = nothing)
    size(X, 1) == net.nin || throw(DimensionMismatch(
        "predict: network expects $(net.nin) features, got $(size(X, 1))"))
    offset !== nothing && net.nproperties > 1 && throw(ArgumentError(
        "predict: residual mode (offset) is only defined for nproperties=1"))

    raw, st = Lux.apply(net.model, X, ps, st)
    ncell = size(X, 2)
    P = net.nproperties
    size(raw, 1) == 2 * P || throw(DimensionMismatch(
        "predict: last layer produced $(size(raw, 1)) rows, expected $(2 * P)"))

    if P == 1
        lo, hi = net.mu_bounds[1]
        smin, smax = net.sigma_bounds_per[1]
        r1 = raw[1, :]
        r2 = raw[2, :]
        mu = if offset === nothing
            _squash_mu(r1, lo, hi)
        else
            length(offset) == ncell || throw(DimensionMismatch(
                "predict: offset has $(length(offset)) entries but $(ncell) cells were given"))
            # interval endpoints are constants with respect to the parameters, so the
            # gradient still flows through tanh alone
            _squash_mu_residual(r1, offset, lo, hi, net.residual_spans[1])
        end
        sigma = _squash_sigma(r2, smin, smax)
        return (mu, sigma), st
    end

    mus = map(1:P) do p
        lo, hi = net.mu_bounds[p]
        _squash_mu(raw[2p - 1, :], lo, hi)
    end
    sigmas = map(1:P) do p
        smin, smax = net.sigma_bounds_per[p]
        _squash_sigma(raw[2p, :], smin, smax)
    end
    return (reduce(vcat, (reshape(m, 1, :) for m in mus)),
            reduce(vcat, (reshape(s, 1, :) for s in sigmas))), st
end

"""
    predict_grid(net::PriorNet, X, ps, st, dims; offset=nothing) -> (mu, sigma, st)

As [`predict`](@ref) but reshaped onto a `dims = (nx, ny, nz)` grid.

Safe because [`feature_matrix`](@ref) orders cells to match `vec` of a
`[nx, ny, nz]` array.
"""
function predict_grid(net::PriorNet, X::AbstractMatrix, ps, st, dims::Tuple{Int,Int,Int};
                      offset::Union{Nothing,AbstractVector} = nothing)
    net.nproperties == 1 || throw(ArgumentError(
        "predict_grid: single-property nets only; got nproperties=$(net.nproperties)"))
    prod(dims) == size(X, 2) || throw(DimensionMismatch(
        "predict_grid: dims $(dims) imply $(prod(dims)) cells but got $(size(X, 2))"))
    (mu, sigma), st = predict(net, X, ps, st; offset = offset)
    return reshape(mu, dims), reshape(sigma, dims), st
end
