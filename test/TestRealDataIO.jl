using Test
using SmartPriorMT
using SmartPriorMT: PRACTICAL_Z_TO_SI, MU0, mt_apparent_errors_from_impedance
using SmartPriorMT: MUSGRAVE_LON0, MUSGRAVE_LAT0, MUSGRAVE_EARTH_R

@testset "RealDataIO" begin

    @testset "PRACTICAL_Z_TO_SI matches 4π × 10⁻⁴" begin
        @test PRACTICAL_Z_TO_SI == 4π * 1e-4
    end

    @testset "SA227 ZXY at T=8 s gives ρ_a ≈ 13.66" begin
        # SA227 first period: FREQ 0.125 Hz, Z already in SI ohm. Same identity
        # RealDataIO uses after the practical-to-SI conversion: ρ_a = |Z|² / (ω μ₀).
        T = 8.0
        z_xy = 1.160000e-3 - 3.484400e-3im
        ω = 2π / T
        rho_a = abs2(z_xy) / (ω * MU0)
        @test rho_a ≈ 13.66 atol = 0.01
    end

    @testset "mt_apparent_errors_from_impedance magnitude" begin
        z = fill(1.0 + 0.0im, 1, 1)
        z_err = fill(0.1, 1, 1)
        frequencies = [1.0]
        rho_a = abs2.(z) ./ ((2π .* frequencies) .* MU0)

        err_rho, err_phase = mt_apparent_errors_from_impedance(rho_a, z, z_err, frequencies)

        @test isfinite(err_rho[1, 1]) && err_rho[1, 1] > 0
        @test isfinite(err_phase[1, 1]) && err_phase[1, 1] > 0
        # σ/|Z| = 0.1 rad ≈ 5.7°; relative ρ_a error is 2 σ/|Z| = 0.2
        @test 1.0 < err_phase[1, 1] < 15.0
        @test 0.1 < err_rho[1, 1] / rho_a[1, 1] < 0.4
    end

    @testset "phase-tensor skew β matches Caldwell et al. 2004 eq. (19)" begin
        # 1-D: Z = [0 z; -z 0] → Φ = tan(φ) I → β = 0 (Caldwell eq. 18–19).
        z1d = 0.01 * cis(deg2rad(40.0))
        @test abs(SmartPriorMT._phase_tensor_beta(0 + 0im, z1d, -z1d, 0 + 0im)) < 1e-12

        # 2-D, strike-aligned: antidiagonal Z with independent TE/TM → Φ diagonal → β = 0.
        zxy = 0.02 * cis(deg2rad(50.0))
        zyx = -0.008 * cis(deg2rad(25.0))
        @test abs(SmartPriorMT._phase_tensor_beta(0 + 0im, zxy, zyx, 0 + 0im)) < 1e-12

        # Coordinate invariance: rotating a 2-D tensor must leave β = 0.
        θ = deg2rad(33.0)
        c, s = cos(θ), sin(θ)
        R = [c s; -s c]
        Z = [0+0im zxy; zyx 0+0im]
        Zr = R * Z * transpose(R)
        @test abs(SmartPriorMT._phase_tensor_beta(Zr[1, 1], Zr[1, 2], Zr[2, 1], Zr[2, 2])) < 1e-12

        # Constructed Φ with X = I, Y = Φ so Z = I + iΦ. Trace > 0, so
        # atan2 and the ratio form of eq. (19) agree.
        Φ11, Φ12, Φ21, Φ22 = 2.0, 0.5, -0.3, 1.0
        expected = rad2deg(0.5 * atan((Φ12 - Φ21) / (Φ11 + Φ22)))
        β = SmartPriorMT._phase_tensor_beta(1 + Φ11 * im, 0 + Φ12 * im,
                                           0 + Φ21 * im, 1 + Φ22 * im)
        @test β ≈ expected rtol = 1e-14
        @test β ≈ rad2deg(0.5 * atan(Φ12 - Φ21, Φ11 + Φ22)) rtol = 1e-14
    end

    @testset "read_musgrave_edi parses ZXX/ZYY/ZYX on WA55" begin
        rec = read_musgrave_edi(SmartPriorMT._find_musgrave_edi("WA55"))
        @test length(rec.frequencies) == 23
        @test length(rec.z_xx) == 23
        @test length(rec.z_yy) == 23
        @test length(rec.z_yx) == 23
        # EDI FREQ decreasing → no reverse; first sample is T = 8 s.
        @test rec.frequencies[1] ≈ 0.125
        @test rec.z_xx[1] ≈ Complex(-14.593, -20.747) * PRACTICAL_Z_TO_SI rtol = 1e-5
        @test rec.z_yy[1] ≈ Complex(22.826, 27.668) * PRACTICAL_Z_TO_SI rtol = 1e-5
        @test rec.z_yx[1] ≈ Complex(-23.825, -41.966) * PRACTICAL_Z_TO_SI rtol = 1e-5
        # TE values must not have moved when the other components were added.
        @test rec.z_xy[1] ≈ Complex(41.671, 40.028) * PRACTICAL_Z_TO_SI rtol = 1e-5
    end

    @testset "build_musgrave_datafile2d is TE-only" begin
        rec = read_musgrave_edi(SmartPriorMT._find_musgrave_edi("WA55"))
        df = build_musgrave_datafile2d()
        @test size(df.z_yx) == (23, 9)
        @test all(isnan, real.(df.z_yx)) && all(isnan, imag.(df.z_yx))
        @test all(isnan, df.z_yx_error)
        @test all(isnan, df.rho_yx) && all(isnan, df.phase_yx)
        @test df.z_xy_error[:, 1] ≈
              sqrt.(rec.z_xy_error .^ 2 .+ (0.4 .* abs.(rec.z_xy)) .^ 2)
        @test all(iszero, df.z_xx) && all(iszero, df.z_yy)
        @test all(==(0.05), df.z_xx_error) && all(==(0.05), df.z_yy_error)
        @test occursin("TE-only", df.title)
    end

    @testset "local tangent plane: origin is (0, 0)" begin
        # inverse of the y → lon map in build_musgrave_surface_z, plus the
        # matching northing from the same spherical radius
        lon_per_m = (180 / π) / (MUSGRAVE_EARTH_R * cos(deg2rad(MUSGRAVE_LAT0)))
        y = (MUSGRAVE_LON0 - MUSGRAVE_LON0) / lon_per_m
        x = MUSGRAVE_EARTH_R * deg2rad(MUSGRAVE_LAT0 - MUSGRAVE_LAT0)
        @test x == 0.0
        @test y == 0.0

        y_east = (MUSGRAVE_LON0 + 0.01 - MUSGRAVE_LON0) / lon_per_m
        x_north = MUSGRAVE_EARTH_R * deg2rad((MUSGRAVE_LAT0 + 0.01) - MUSGRAVE_LAT0)
        @test y_east > 0
        @test x_north > 0
    end

end
