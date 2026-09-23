# Continuous stationary Gaussian fields via random Fourier features.
# No grid: evaluate at arbitrary points. Covariance is exponential
# (Matérn ν = 1/2). Frequencies are multivariate Student-t with 1 d.f.

"""
Number of random Fourier features used by [`GaussianField`](@ref).
"""
const SYNTHETIC_RFF_M = 2048

"""
    GaussianField

Stationary Gaussian field with exponential covariance, optional nugget,
and vertical/horizontal anisotropy (`L_v = L_h / 2` by default).

The latent smooth field is
`Z_L(x) ≈ √(2/M) Σ cos(ωₘ · x̃ + bₘ)` with scaled coordinates
`x̃ = (x/L_h, y/L_h, z/L_v)` and `ωₘ ∼ Student-t₁(ℝ³)`. A sample is
`μ + σ · (√(1−η) · Z_L(x) + √η · ε)`, where `ε ∼ N(0,1)` is drawn
deterministically from `(seed, x, y, z)` so `evaluate` is pure and
subset-stable.
"""
struct GaussianField
    μ::Float64
    σ::Float64
    η::Float64
    L_h::Float64
    L_v::Float64
    seed::Int
    ω::Matrix{Float64}   # 3 × M
    b::Vector{Float64}   # M
end

"""
    gaussian_field(; μ=0, σ=1, η=0, L_h, L_v=L_h/2, seed, M=SYNTHETIC_RFF_M)

Build a reproducible [`GaussianField`](@ref). `η` is the nugget fraction
of total variance (`0 ≤ η < 1`). `L_h` and `L_v` are correlation lengths
in metres for the exponential covariance `exp(−r)` in scaled space.
"""
function gaussian_field(; μ::Real = 0.0,
                        σ::Real = 1.0,
                        η::Real = 0.0,
                        L_h::Real,
                        L_v::Real = L_h / 2,
                        seed::Integer,
                        M::Integer = SYNTHETIC_RFF_M)
    L_h > 0 || throw(ArgumentError("gaussian_field: L_h must be positive"))
    L_v > 0 || throw(ArgumentError("gaussian_field: L_v must be positive"))
    σ >= 0 || throw(ArgumentError("gaussian_field: σ must be non-negative"))
    (0 <= η < 1) || throw(ArgumentError(
        "gaussian_field: η must be in [0, 1), got $η"))
    M > 0 || throw(ArgumentError("gaussian_field: M must be positive"))
    rng = Xoshiro(Int(seed))
    ω = Matrix{Float64}(undef, 3, M)
    b = Vector{Float64}(undef, M)
    @inbounds for m in 1:M
        # Multivariate Student-t with 1 d.f.: g / √(χ²_1), χ²_1 = Z².
        g1 = randn(rng)
        g2 = randn(rng)
        g3 = randn(rng)
        z = randn(rng)
        s = abs(z)
        s = s < 1.0e-300 ? 1.0e-300 : s
        ω[1, m] = g1 / s
        ω[2, m] = g2 / s
        ω[3, m] = g3 / s
        b[m] = 2π * rand(rng)
    end
    return GaussianField(Float64(μ), Float64(σ), Float64(η),
                         Float64(L_h), Float64(L_v), Int(seed), ω, b)
end

@inline function _scale_xyz(field::GaussianField, x::Float64, y::Float64, z::Float64)
    return x / field.L_h, y / field.L_h, z / field.L_v
end

"""
    latent_field(field, xyz) -> Vector{Float64}

Nugget-free unit-variance field `Z_L` at columns of `xyz` (3 × n).
"""
function latent_field(field::GaussianField, xyz::AbstractMatrix{<:Real})
    size(xyz, 1) == 3 || throw(ArgumentError(
        "latent_field: xyz must be 3 × n, got size $(size(xyz))"))
    n = size(xyz, 2)
    M = size(field.ω, 2)
    out = zeros(Float64, n)
    scale = sqrt(2 / M)
    @inbounds for j in 1:n
        xs, ys, zs = _scale_xyz(field, Float64(xyz[1, j]), Float64(xyz[2, j]),
                                Float64(xyz[3, j]))
        s = 0.0
        for m in 1:M
            s += cos(field.ω[1, m] * xs + field.ω[2, m] * ys +
                     field.ω[3, m] * zs + field.b[m])
        end
        out[j] = scale * s
    end
    return out
end

"""
    smooth_field(field, xyz) -> Vector{Float64}

Block-scale target: `μ + σ · √(1−η) · Z_L(x)` (no nugget noise).
"""
function smooth_field(field::GaussianField, xyz::AbstractMatrix{<:Real})
    z = latent_field(field, xyz)
    a = field.σ * sqrt(1 - field.η)
    @inbounds for i in eachindex(z)
        z[i] = field.μ + a * z[i]
    end
    return z
end

@inline function _nugget_eps(seed::Int, x::Float64, y::Float64, z::Float64)
    # Deterministic N(0,1) from (seed, coordinates) so evaluate is pure and
    # subset-equivalent (does not depend on batch order or size).
    h = hash((seed, reinterpret(UInt64, x), reinterpret(UInt64, y),
              reinterpret(UInt64, z)), UInt(0x9e3779b97f4a7c15))
    rng = Xoshiro(h % typemax(UInt64))
    return randn(rng)
end

"""
    evaluate(field::GaussianField, xyz) -> Vector{Float64}

Pointwise samples including nugget noise. Pure: the same points always
yield the same values, and a column subset matches the corresponding
slice of a larger evaluation.
"""
function evaluate(field::GaussianField, xyz::AbstractMatrix{<:Real})
    size(xyz, 1) == 3 || throw(ArgumentError(
        "evaluate: xyz must be 3 × n, got size $(size(xyz))"))
    n = size(xyz, 2)
    z = latent_field(field, xyz)
    a = field.σ * sqrt(1 - field.η)
    b = field.σ * sqrt(field.η)
    out = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        eps = b == 0 ? 0.0 : _nugget_eps(field.seed, Float64(xyz[1, j]),
                                         Float64(xyz[2, j]), Float64(xyz[3, j]))
        out[j] = field.μ + a * z[j] + b * eps
    end
    return out
end
