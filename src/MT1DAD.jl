# AD-safe 1-D magnetotelluric forward physics.
#
# MTGeophysics.solve_mt1d_analytical implements the same recursion but casts its
# inputs with `Float64.(rho)` first. Reverse mode survives that, because Zygote
# tracks pullbacks over ordinary Float64 values and the cast is then an identity;
# forward mode does not, because the cast cannot accept a ForwardDiff.Dual and
# throws a MethodError. The functions here are element type generic so both modes
# work, which matters because per-column sensitivities have few inputs and are
# markedly cheaper in forward mode. Avoiding the per-call MT1DResponse allocation
# in an inner training loop is a secondary gain.
#
# test/TestMT1D.jl pins these against the reference so the physics cannot drift,
# and records the differentiability difference explicitly.
#
# Convention matches MTGeophysics: exp(+i*omega*t), Z = omega*mu0/k with
# k = sqrt(-i*omega*mu0*sigma).

const MU0 = 4π * 1e-7

"""
    mt1d_impedance(f, rho, t) -> Vector{Complex}

Complex MT impedance of a layered half-space at frequencies `f` (Hz).

`rho` holds layer resistivities in ohm-metres including the basement, `t` the
thicknesses in metres of everything above it, so `length(rho) == length(t) + 1`.

Differentiable with respect to `rho` and `t`. `tanh` saturates for thick layers
at high frequency, so no overflow guard is needed.
"""
function mt1d_impedance(f::AbstractVector{<:Real},
                        rho::AbstractVector{<:Real},
                        t::AbstractVector{<:Real})
    n = length(rho)
    n == length(t) + 1 || throw(ArgumentError(
        "mt1d_impedance: expected length(rho) == length(t) + 1, got $(n) and $(length(t))"))
    n >= 1 || throw(ArgumentError("mt1d_impedance: need at least a basement layer"))

    return map(f) do fi
        ω = 2π * fi
        k = sqrt(-1im * ω * MU0 / rho[n])
        Z = (ω * MU0) / k
        for i in (n-1):-1:1
            ki = sqrt(-1im * ω * MU0 / rho[i])
            Zi = (ω * MU0) / ki
            q = tanh(1im * ki * t[i])
            Z = Zi * (Z + Zi * q) / (Zi + Z * q)
        end
        Z
    end
end

"""
    mt1d_apparent(f, rho, t) -> (rho_a, phase_deg)

Apparent resistivity (ohm-metres) and impedance phase (degrees) of a layered
half-space. Arguments follow [`mt1d_impedance`](@ref).
"""
function mt1d_apparent(f::AbstractVector{<:Real},
                       rho::AbstractVector{<:Real},
                       t::AbstractVector{<:Real})
    Z = mt1d_impedance(f, rho, t)
    ω = 2π .* f
    rho_a = abs2.(Z) ./ (ω .* MU0)
    phase = rad2deg.(atan.(imag.(Z), real.(Z)))
    return rho_a, phase
end

"""
    mt1d_column_response(f, rho_cells, dz_cells) -> (rho_a, phase_deg)

1-D response of a single column of a 3-D model, treating the bottom cell as the
basement half-space.

`rho_cells` and `dz_cells` both have one entry per cell, which is how a WS3D
column is stored, so this is the natural entry point for the per-site physics
term in the training loss.
"""
function mt1d_column_response(f::AbstractVector{<:Real},
                              rho_cells::AbstractVector{<:Real},
                              dz_cells::AbstractVector{<:Real})
    nz = length(rho_cells)
    nz == length(dz_cells) || throw(ArgumentError(
        "mt1d_column_response: rho_cells and dz_cells must have equal length, got $(nz) and $(length(dz_cells))"))
    nz >= 1 || throw(ArgumentError("mt1d_column_response: empty column"))
    return mt1d_apparent(f, rho_cells, @view dz_cells[1:nz-1])
end

"""
    skin_depth(rho, T) -> Float64

Electromagnetic skin depth in metres for resistivity `rho` (ohm-metres) and
period `T` (seconds): `sqrt(2 * rho * T / (2 * pi * mu0))`, i.e. the familiar
`503 * sqrt(rho * T)`.

Used to turn absolute depth into the dimensionless `z / delta` that makes a
feature transferable between surveys of different size.
"""
skin_depth(rho::Real, T::Real) = sqrt(2 * rho * T / (2π * MU0))

"""
    bostick_depth(rho_a, T) -> Float64

Niblett-Bostick depth of investigation in metres, `sqrt(rho_a * T / (2*pi*mu0))`.
"""
bostick_depth(rho_a::Real, T::Real) = sqrt(rho_a * T / (2π * MU0))

"""
    bostick_resistivity(rho_a, phase_deg) -> Float64

Niblett-Bostick resistivity transform, `rho_a * (pi / (2 * phi) - 1)` with `phi`
in radians.

Phases at or below zero and at or above 90 degrees are non-physical for a 1-D
earth and would send the transform negative or to infinity, so the phase is
clamped just inside that band.
"""
function bostick_resistivity(rho_a::Real, phase_deg::Real)
    φ = deg2rad(clamp(float(phase_deg), 1.0e-3, 90.0 - 1.0e-3))
    return rho_a * (π / (2φ) - 1)
end

"""
    niblett_bostick(T, rho_a, phase_deg) -> (depth, rho)

Niblett-Bostick transform of a sounding curve into a depth-resistivity profile.

This is the cheap, data-driven background the neural field predicts a *residual*
against: it already carries the vertical resistivity level that 1-D MT can
resolve, leaving the network to supply only the 3-D structure that gravity and
topography imply.
"""
function niblett_bostick(T::AbstractVector{<:Real},
                         rho_a::AbstractVector{<:Real},
                         phase_deg::AbstractVector{<:Real})
    n = length(T)
    (length(rho_a) == n && length(phase_deg) == n) || throw(ArgumentError(
        "niblett_bostick: T, rho_a and phase_deg must have equal length"))

    depth = [bostick_depth(rho_a[i], T[i]) for i in 1:n]
    rho = [bostick_resistivity(rho_a[i], phase_deg[i]) for i in 1:n]
    perm = sortperm(depth)
    return depth[perm], rho[perm]
end
