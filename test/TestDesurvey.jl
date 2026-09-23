using Test
using SmartPrior

const _COLLAR = (1000.0, 2000.0, 300.0)
const _DEG = (angle_unit = :degree, dip_down_negative = false)

function _finite_path(path::DesurveyPath)
    return all(isfinite, path.east) && all(isfinite, path.north) && all(isfinite, path.z)
end

@testset "vertical hole ignores azimuth" begin
    path = desurvey(_COLLAR, [0.0, 10.0, 25.0], [15.0, 90.0, 240.0], [90.0, 90.0, 90.0];
                    _DEG...)
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
                    angle_unit = :degree, dip_down_negative = true)
    @test down.z[2] ≈ 275.0
    @test down.east[2] ≈ 1000.0
    @test down.north[2] ≈ 2000.0
end

@testset "dip −90 with dip_down_negative matches dip +90" begin
    # Same azimuths, opposite dip sign, opposite flag: one path.
    pos = desurvey(_COLLAR, [0.0, 10.0, 25.0], [15.0, 90.0, 240.0], [90.0, 90.0, 90.0];
                   angle_unit = :degree, dip_down_negative = false)
    neg = desurvey(_COLLAR, [0.0, 10.0, 25.0], [15.0, 90.0, 240.0], [-90.0, -90.0, -90.0];
                   angle_unit = :degree, dip_down_negative = true)
    @test neg.east ≈ pos.east atol = 1e-12
    @test neg.north ≈ pos.north atol = 1e-12
    @test neg.z ≈ pos.z atol = 1e-12
    @test neg.de ≈ pos.de atol = 1e-12
    @test neg.dn ≈ pos.dn atol = 1e-12
    @test neg.dz ≈ pos.dz atol = 1e-12
    e1, n1, z1 = positions(pos, 12.5)
    e2, n2, z2 = positions(neg, 12.5)
    @test e2 ≈ e1 atol = 1e-12
    @test n2 ≈ n1 atol = 1e-12
    @test z2 ≈ z1 atol = 1e-12
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
        path = desurvey(_COLLAR, [0.0, 20.0], [az, az], [60.0, 60.0]; _DEG...)
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
                           angle_unit = :degree, dip_down_negative = true)
        @test flipped.east[2] ≈ path.east[2] atol = 1e-12
        @test flipped.north[2] ≈ path.north[2] atol = 1e-12
        @test flipped.z[2] ≈ path.z[2] atol = 1e-12
    end
end

@testset "constant-curvature quarter circle" begin
    # Horizontal hole turning from north to east over an arc of length π/2.
    # Radius is 1, so the end point is (1, 1) east and north of the collar,
    # and the halfway point is the circle at π/4.
    path = desurvey(_COLLAR, [0.0, π / 2], [0.0, 90.0], [0.0, 0.0]; _DEG...)
    @test _finite_path(path)
    @test path.east[2] ≈ 1000.0 + 1.0 atol = 1e-12
    @test path.north[2] ≈ 2000.0 + 1.0 atol = 1e-12
    @test path.z[2] ≈ 300.0 atol = 1e-12
    e, n, z = positions(path, π / 4)
    @test e ≈ 1000.0 + (1 - cos(π / 4)) atol = 1e-12
    @test n ≈ 2000.0 + sin(π / 4) atol = 1e-12
    @test z ≈ 300.0 atol = 1e-12
end

@testset "position between stations lies on the arc" begin
    # Horizontal quarter-circle of radius R, north to east. Arc length is
    # πR/2. The point at arc length πR/4 is the 45° point of that circle.
    # The chord from the collar to the end point is the line to (R, R);
    # its midpoint is (R/2, R/2), which is not the 45° point.
    R = 20.0
    path = desurvey(_COLLAR, [0.0, π * R / 2], [0.0, 90.0], [0.0, 0.0]; _DEG...)
    @test path.east[2] ≈ 1000.0 + R atol = 1e-12
    @test path.north[2] ≈ 2000.0 + R atol = 1e-12
    @test path.z[2] ≈ 300.0 atol = 1e-12
    e, n, z = positions(path, π * R / 4)
    @test e ≈ 1000.0 + R * (1 - cos(π / 4)) atol = 1e-12
    @test n ≈ 2000.0 + R * sin(π / 4) atol = 1e-12
    @test z ≈ 300.0 atol = 1e-12
    # Angle at the circle centre (one radius east of the collar), measured
    # from the westward radius at the collar.
    ang = atan(n - 2000.0, -(e - (1000.0 + R)))
    @test rad2deg(ang) ≈ 45.0 atol = 1e-9
    @test abs(e - (1000.0 + R / 2)) > 0.1 * R
    @test abs(n - (2000.0 + R / 2)) > 0.1 * R
end

@testset "azimuth crossing north takes the short dogleg" begin
    # 359° to 1° is a 2° clockwise turn, not a 358° turn the other way.
    # Both turns share the end tangent; the end point does not.
    L = 100.0
    path = desurvey(_COLLAR, [0.0, L], [359.0, 1.0], [0.0, 0.0]; _DEG...)
    u1 = (path.de[1], path.dn[1], path.dz[1])
    u2 = (path.de[2], path.dn[2], path.dz[2])
    cosβ = clamp(u1[1] * u2[1] + u1[2] * u2[2] + u1[3] * u2[3], -1.0, 1.0)
    βdeg = rad2deg(acos(cosβ))
    @test βdeg ≈ 2.0 atol = 1e-9
    β = deg2rad(2.0)
    # Chord of the 2° arc lies on the bisector, which is grid north.
    chord = L * sin(β / 2) / (β / 2)
    @test path.east[2] ≈ 1000.0 atol = 1e-9
    @test path.north[2] ≈ 2000.0 + chord atol = 1e-9
    @test path.z[2] ≈ 300.0 atol = 1e-9
    # The 358° arc with the same end tangents stays within a few metres of
    # the collar. Left of a heading (sin θ, cos θ) is (−cos θ, sin θ).
    βlong = deg2rad(358.0)
    Rlong = L / βlong
    left(θ) = (-cos(θ), sin(θ))
    le0, ln0 = left(deg2rad(359.0))
    le1, ln1 = left(deg2rad(1.0))
    long_e = 1000.0 + Rlong * (le0 - le1)
    long_n = 2000.0 + Rlong * (ln0 - ln1)
    @test hypot(path.east[2] - long_e, path.north[2] - long_n) > 50
end

@testset "identical stations have zero dogleg and no NaN" begin
    # The repeated station uses azimuth 360°, the same direction as 0°.
    path = desurvey(_COLLAR, [0.0, 10.0, 10.0], [0.0, 0.0, 360.0], [45.0, 45.0, 45.0];
                    _DEG...)
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
                       angle_unit = :degree, dip_down_negative = false)
    gons = desurvey(_COLLAR, [0.0, 10.0], [100.0, 100.0], [50.0, 50.0];
                    angle_unit = :gon, dip_down_negative = false)
    @test gons.east ≈ degrees.east atol = 1e-12
    @test gons.north ≈ degrees.north atol = 1e-12
    @test gons.z ≈ degrees.z atol = 1e-12
    step = 5 * √2
    @test degrees.east[2] ≈ 1000.0 + step atol = 1e-12
    @test degrees.north[2] ≈ 2000.0 atol = 1e-12
    @test degrees.z[2] ≈ 300.0 - step atol = 1e-12
end

@testset "depth beyond the last station extends the last direction" begin
    path = desurvey(_COLLAR, [0.0, 10.0], [90.0, 90.0], [60.0, 60.0]; _DEG...)
    # 10 m more of the same straight course: another 5 m east, no northing, drop 5√3.
    e, n, z = positions(path, 20.0)
    @test e ≈ 1000.0 + 10.0 atol = 1e-12
    @test n ≈ 2000.0 atol = 1e-12
    @test z ≈ 300.0 - 10 * √3 atol = 1e-12
    # One station is a straight line, including past that station and back to the collar.
    alone = desurvey(_COLLAR, [12.0], [0.0], [90.0]; _DEG...)
    @test _finite_path(alone)
    for (md, zz) in ((0.0, 300.0), (12.0, 288.0), (20.0, 280.0))
        e, n, z = positions(alone, md)
        @test e ≈ 1000.0 atol = 1e-12
        @test n ≈ 2000.0 atol = 1e-12
        @test z ≈ zz atol = 1e-12
    end
end

@testset "desurvey rejects a bad survey" begin
    @test_throws ArgumentError desurvey(_COLLAR, Float64[], Float64[], Float64[]; _DEG...)
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, 1.0], [0.0], [90.0, 90.0]; _DEG...)
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, 1.0], [0.0, 0.0], [90.0, 90.0];
                                        angle_unit = :radian, dip_down_negative = false)
    @test_throws ArgumentError desurvey(_COLLAR, [0.0, -1.0], [0.0, 0.0], [90.0, 90.0];
                                        _DEG...)
    @test_throws UndefKeywordError desurvey(_COLLAR, [0.0], [0.0], [90.0])
    @test_throws UndefKeywordError desurvey(_COLLAR, [0.0], [0.0], [90.0];
                                            angle_unit = :degree)
    @test_throws UndefKeywordError desurvey(_COLLAR, [0.0], [0.0], [90.0];
                                            dip_down_negative = false)
end

@testset "unsorted stations are rejected" begin
    # Depths are not sorted inside desurvey. Pairing them with the wrong
    # azimuth would be worse than refusing the row order.
    caught = nothing
    try
        desurvey(_COLLAR, [10.0, 0.0], [90.0, 0.0], [45.0, 45.0]; _DEG...)
    catch err
        caught = err
    end
    @test caught isa ArgumentError
    @test occursin("decreased", caught.msg)
end

@testset "duplicate depths with different directions name the depth" begin
    caught = nothing
    try
        desurvey(_COLLAR, [0.0, 17.25, 17.25], [0.0, 0.0, 90.0], [45.0, 45.0, 45.0];
                 _DEG...)
    catch err
        caught = err
    end
    @test caught isa ArgumentError
    @test occursin("17.25", caught.msg)
end

@testset "a negative query depth is rejected" begin
    path = desurvey(_COLLAR, [0.0], [0.0], [90.0]; _DEG...)
    caught = nothing
    try
        positions(path, -0.1)
    catch err
        caught = err
    end
    @test caught isa ArgumentError
    @test occursin("-0.1", caught.msg)
end
