using Test
using SmartPrior

const _COLLAR = (1000.0, 2000.0, 300.0)

function _finite_path(path::DesurveyPath)
    return all(isfinite, path.east) && all(isfinite, path.north) && all(isfinite, path.z)
end

@testset "vertical hole ignores azimuth" begin
    path = desurvey(_COLLAR, [0.0, 10.0, 25.0], [15.0, 90.0, 240.0], [90.0, 90.0, 90.0])
    @test _finite_path(path)
    @test path.east ≈ [1000.0, 1000.0, 1000.0]
    @test path.north ≈ [2000.0, 2000.0, 2000.0]
    @test path.z ≈ [300.0, 290.0, 275.0]
    # Interval midpoint, and a depth that is not a station.
    e, n, z = positions(path, [7.5, 17.5])
    @test e ≈ [1000.0, 1000.0]
    @test n ≈ [2000.0, 2000.0]
    @test z ≈ [292.5, 282.5]
    down = desurvey(_COLLAR, [0.0, 25.0], [0.0, 180.0], [-90.0, -90.0];
                    dip_down_negative = true)
    @test down.z[2] ≈ 275.0
    @test down.east[2] ≈ 1000.0
    @test down.north[2] ≈ 2000.0
end

@testset "straight inclined hole at the cardinal azimuths" begin
    # 60° from horizontal: 10 m of horizontal advance and 10√3 m of drop
    # over 20 m of hole. Positive dip is downward.
    drop = -10 * √3
    expected = Dict(
        0.0 => (1000.0, 2010.0, 300.0 + drop),
        90.0 => (1010.0, 2000.0, 300.0 + drop),
        180.0 => (1000.0, 1990.0, 300.0 + drop),
        270.0 => (990.0, 2000.0, 300.0 + drop),
    )
    for az in (0.0, 90.0, 180.0, 270.0)
        path = desurvey(_COLLAR, [0.0, 20.0], [az, az], [60.0, 60.0])
        @test _finite_path(path)
        ee, nn, zz = expected[az]
        @test path.east[2] ≈ ee atol = 1e-12
        @test path.north[2] ≈ nn atol = 1e-12
        @test path.z[2] ≈ zz atol = 1e-12
        # Midpoint of a straight hole is the straight position at 10 m.
        e, n, z = positions(path, 10.0)
        @test e ≈ (1000.0 + ee) / 2 atol = 1e-12
        @test n ≈ (2000.0 + nn) / 2 atol = 1e-12
        @test z ≈ (300.0 + zz) / 2 atol = 1e-12
        flipped = desurvey(_COLLAR, [0.0, 20.0], [az, az], [-60.0, -60.0];
                           dip_down_negative = true)
        @test flipped.east[2] ≈ path.east[2] atol = 1e-12
        @test flipped.north[2] ≈ path.north[2] atol = 1e-12
        @test flipped.z[2] ≈ path.z[2] atol = 1e-12
    end
end

@testset "constant-curvature quarter circle" begin
    # Horizontal hole turning from north to east over an arc of length π/2.
    # Radius is 1, so the end point is (1, 1) east and north of the collar,
    # and the halfway point is the circle at π/4.
    path = desurvey(_COLLAR, [0.0, π / 2], [0.0, 90.0], [0.0, 0.0])
    @test _finite_path(path)
    @test path.east[2] ≈ 1000.0 + 1.0 atol = 1e-12
    @test path.north[2] ≈ 2000.0 + 1.0 atol = 1e-12
    @test path.z[2] ≈ 300.0 atol = 1e-12
    e, n, z = positions(path, π / 4)
    @test e ≈ 1000.0 + (1 - cos(π / 4)) atol = 1e-12
    @test n ≈ 2000.0 + sin(π / 4) atol = 1e-12
    @test z ≈ 300.0 atol = 1e-12
end

@testset "identical stations have zero dogleg and no NaN" begin
    # The repeated station uses azimuth 360°, the same direction as 0°.
    path = desurvey(_COLLAR, [0.0, 10.0, 10.0], [0.0, 0.0, 360.0], [45.0, 45.0, 45.0])
    @test _finite_path(path)
    @test path.east[2] ≈ path.east[3] atol = 1e-12
    @test path.north[2] ≈ path.north[3] atol = 1e-12
    @test path.z[2] ≈ path.z[3] atol = 1e-12
    advance = 10 * cos(deg2rad(45.0))
    drop = -10 * sin(deg2rad(45.0))
    @test path.east[2] ≈ 1000.0 atol = 1e-12
    @test path.north[2] ≈ 2000.0 + advance atol = 1e-12
    @test path.z[2] ≈ 300.0 + drop atol = 1e-12
    e, n, z = positions(path, 10.0)
    @test isfinite(e) && isfinite(n) && isfinite(z)
    @test e ≈ path.east[2] atol = 1e-12
    @test n ≈ path.north[2] atol = 1e-12
    @test z ≈ path.z[2] atol = 1e-12
end

@testset "gon input matches the same hole in degrees" begin
    # 100 gon = 90°, 50 gon = 45°. East, 45° from horizontal, 10 m of hole:
    # equal easting advance and elevation drop of 5√2.
    degrees = desurvey(_COLLAR, [0.0, 10.0], [90.0, 90.0], [45.0, 45.0];
                       angle_unit = :degree)
    gons = desurvey(_COLLAR, [0.0, 10.0], [100.0, 100.0], [50.0, 50.0];
                    angle_unit = :gon)
    @test gons.east ≈ degrees.east atol = 1e-12
    @test gons.north ≈ degrees.north atol = 1e-12
    @test gons.z ≈ degrees.z atol = 1e-12
    step = 5 * √2
    @test degrees.east[2] ≈ 1000.0 + step atol = 1e-12
    @test degrees.north[2] ≈ 2000.0 atol = 1e-12
    @test degrees.z[2] ≈ 300.0 - step atol = 1e-12
end

@testset "depth beyond the last station extends the last direction" begin
    path = desurvey(_COLLAR, [0.0, 10.0], [90.0, 90.0], [60.0, 60.0])
    # 10 m more of the same straight course: another 5 m east, no northing, drop 5√3.
    e, n, z = positions(path, 20.0)
    @test e ≈ 1000.0 + 10.0 atol = 1e-12
    @test n ≈ 2000.0 atol = 1e-12
    @test z ≈ 300.0 - 10 * √3 atol = 1e-12
    # One station is a straight line, including past that station and back to the collar.
    alone = desurvey(_COLLAR, [12.0], [0.0], [90.0])
    @test _finite_path(alone)
    for (md, zz) in ((0.0, 300.0), (12.0, 288.0), (20.0, 280.0))
        e, n, z = positions(alone, md)
        @test e ≈ 1000.0 atol = 1e-12
        @test n ≈ 2000.0 atol = 1e-12
        @test z ≈ zz atol = 1e-12
    end
end

@testset "desurvey rejects a bad survey" begin
    @test_throws ArgumentError desurvey(_COLLAR, Float64[], Float64[], Float64[])
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, 1.0], [0.0], [90.0, 90.0])
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, 1.0], [0.0, 0.0], [90.0, 90.0];
                                        angle_unit = :radian)
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, -1.0], [0.0, 0.0], [90.0, 90.0])
    @test_throws ArgumentError desurvey(_COLLAR, [5.0, 5.0], [0.0, 90.0], [45.0, 45.0])
    path = desurvey(_COLLAR, [0.0], [0.0], [90.0])
    @test_throws ArgumentError positions(path, -0.1)
end
