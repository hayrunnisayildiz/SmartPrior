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
    PriorNet(nin; width=128, depth=4, activation=gelu, log_rho_bounds=(0.0, 5.0),
             residual_span=1.5, sigma_bounds=(0.05, 1.2))

Neural field mapping `nin` features per cell to `(mu, sigma)`.

- `log_rho_bounds`: physical limits on log10 resistivity, enforced exactly in
  absolute mode via a `tanh` squash.
- `residual_span`: in residual mode, the largest deviation in decades the network
  may add to the supplied baseline.
- `sigma_bounds`: hard limits on the predicted standard deviation, in decades.

Use with [`setup_prior`](@ref) and [`predict`](@ref).
"""
struct PriorNet{M} <: Lux.AbstractLuxWrapperLayer{:model}
    model::M
    nin::Int
    log_rho_bounds::Tuple{Float64,Float64}
    residual_span::Float64
    sigma_bounds::Tuple{Float64,Float64}
end

function PriorNet(nin::Integer;
                  width::Integer = 128,
                  depth::Integer = 4,
                  activation = gelu,
                  log_rho_bounds::Tuple{Real,Real} = (0.0, 5.0),
                  residual_span::Real = 1.5,
                  sigma_bounds::Tuple{Real,Real} = (0.05, 1.2))
    nin > 0 || throw(ArgumentError("PriorNet: nin must be positive"))
    width > 0 || throw(ArgumentError("PriorNet: width must be positive"))
    depth >= 1 || throw(ArgumentError("PriorNet: depth must be at least 1"))
    log_rho_bounds[1] < log_rho_bounds[2] ||
        throw(ArgumentError("PriorNet: log_rho_bounds must be increasing"))
    0 < sigma_bounds[1] < sigma_bounds[2] ||
        throw(ArgumentError("PriorNet: sigma_bounds must satisfy 0 < lo < hi"))
    residual_span > 0 || throw(ArgumentError("PriorNet: residual_span must be positive"))

    hidden = ntuple(_ -> Dense(width => width, activation), depth - 1)
    model = Chain(Dense(nin => width, activation), hidden..., Dense(width => 2))

    return PriorNet(model, Int(nin),
                    (Float64(log_rho_bounds[1]), Float64(log_rho_bounds[2])),
                    Float64(residual_span),
                    (Float64(sigma_bounds[1]), Float64(sigma_bounds[2])))
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

"""
    predict(net::PriorNet, X, ps, st; offset=nothing) -> ((mu, sigma), st)

Evaluate the field on a `[nfeature, ncell]` input.

Returns `mu` in log10 ohm-metres and `sigma` in decades, both length `ncell`.

With `offset === nothing` the network works in absolute mode and `mu` is squashed
into `log_rho_bounds`. Passing `offset` (normally a Niblett-Bostick baseline from
[`nb_baseline`](@ref)) switches to residual mode, where `mu` is squashed into the
band `offset ± residual_span` *intersected with* `log_rho_bounds`. Residual mode
is what makes a trained network reusable across surveys, since the absolute
resistivity level comes from the data rather than from the weights.

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

    raw, st = Lux.apply(net.model, X, ps, st)

    lo, hi = net.log_rho_bounds
    smin, smax = net.sigma_bounds

    r1 = raw[1, :]
    r2 = raw[2, :]

    mu = if offset === nothing
        c = (lo + hi) / 2
        h = (hi - lo) / 2
        c .+ h .* tanh.(r1)
    else
        length(offset) == size(X, 2) || throw(DimensionMismatch(
            "predict: offset has $(length(offset)) entries but $(size(X, 2)) cells were given"))
        # interval endpoints are constants with respect to the parameters, so the
        # gradient still flows through tanh alone
        o = clamp.(offset, lo, hi)
        l = max.(lo, o .- net.residual_span)
        u = min.(hi, o .+ net.residual_span)
        (l .+ u) ./ 2 .+ (u .- l) ./ 2 .* tanh.(r1)
    end

    sigma = smin .+ (smax - smin) .* sigmoid.(r2)

    return (mu, sigma), st
end

"""
    predict_grid(net::PriorNet, X, ps, st, dims; offset=nothing) -> (mu, sigma, st)

As [`predict`](@ref) but reshaped onto a `dims = (nx, ny, nz)` grid.

Safe because [`feature_matrix`](@ref) orders cells to match `vec` of a
`[nx, ny, nz]` array.
"""
function predict_grid(net::PriorNet, X::AbstractMatrix, ps, st, dims::Tuple{Int,Int,Int};
                      offset::Union{Nothing,AbstractVector} = nothing)
    prod(dims) == size(X, 2) || throw(DimensionMismatch(
        "predict_grid: dims $(dims) imply $(prod(dims)) cells but got $(size(X, 2))"))
    (mu, sigma), st = predict(net, X, ps, st; offset = offset)
    return reshape(mu, dims), reshape(sigma, dims), st
end
