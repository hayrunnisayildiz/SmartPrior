using Test
using Logging
using SmartPrior
using TOML

const KEI_FIXTURE = joinpath(@__DIR__, "fixtures", "keivitsa_tiny")
const KEI_SITE = joinpath(KEI_FIXTURE, "site.toml")

function _caret(path, header, body)
    open(path, "w") do io
        println(io, join(header, " ^ "))
        println(io, join(fill("char", length(header)), " ^ "))
        println(io, join(fill("8", length(header)), " ^ "))
        println(io, join(fill("0", length(header)), " ^ "))
        for row in body
            line = row isa AbstractString ? row : join(string.(row), " ^ ")
            println(io, line)
        end
    end
end

function _site_toml(dir)
    open(joinpath(dir, "site.toml"), "w") do io
        print(io, """
        name = "keivitsa"
        source = "keivitsa"
        crs = "EPSG:2393"
        root = "."
        collar = "collar.txt"
        survey = "kalte.txt"
        assays = "511P.txt"
        petrophysics = "petro.txt"

        [bounds]
        x = [0.0, 10000.0]
        y = [0.0, 10000.0]
        z = [0.0, 500.0]

        [properties.cu]
        kind = "continuous"
        transform = "log10"
        lod_policy = "fixed"
        lod = 1.0
        unit = "ppm"

        [properties.density]
        kind = "continuous"
        transform = "identity"
        unit = "kg/m3"

        [properties.susceptibility]
        kind = "continuous"
        transform = "log10"
        lod_policy = "min_positive"
        unit = "1e-6 SI"

        [covariates]
        use = ["coordinates", "depth"]

        [covariates.depth]
        d0 = 25.0
        """)
    end
    return joinpath(dir, "site.toml")
end

function _rows_of(table, hole, source)
    return [i for i in 1:nsamples(table)
            if table.hole[i] == hole && table.classes[:petro_source][i] == source]
end

function _logged_raw_susc(logger)
    raws = Float64[]
    for rec in logger.logs
        occursin("non-positive susceptibility", string(rec.message)) || continue
        for pair in rec.kwargs
            pair.first === :raw || continue
            append!(raws, pair.second)
        end
    end
    return raws
end

@testset "synthetic Keivitsa fixture: axes, censoring, exclusions" begin
    logger = TestLogger()
    table, report = with_logger(logger) do
        keivitsa_sample_table(KEI_FIXTURE, TOML.parsefile(KEI_SITE))
    end
    @test table isa SampleTable
    @test all(==("Keivitsa"), table.group)
    @test all(==("drillhole"), table.sample_type)
    # Collar id was "h1"; the assay id was "H1".
    @test "H1" in table.hole
    @test "H3" ∉ table.hole
    @test "H4" ∉ table.hole

    # Repeated assay coordinates are (easting 2, northing 1, z 3). The path
    # is the collar, swapped: KKJ_EAST → x, KKJ_NORTH → y.
    h1 = _rows_of(table, "H1", "")
    @test length(h1) == 4
    @test table.x[h1] ≈ fill(1000.0, 4) atol = 1e-9
    @test table.y[h1] ≈ fill(2000.0, 4) atol = 1e-9
    @test table.z[h1] ≈ [95.0, 89.0, 87.0, 85.0] atol = 1e-9
    @test table.values[:cu][h1] ≈ [10.0, 0.5, 1.0, 1.0] atol = 1e-12
    @test table.censored[:cu][h1] == BitVector([false, false, true, true])
    @test !any(table.x .== 2.0)
    @test !any(table.y .== 1.0)
    # 0.5 ppm unflagged is a measurement, not a percentile censor.
    @test table.values[:cu][h1[2]] == 0.5
    @test !table.censored[:cu][h1[2]]

    ptr = _rows_of(table, "H1", "PTR")
    @test length(ptr) == 1
    @test table.values[:density][ptr[1]] ≈ 3000.0
    @test table.values[:susceptibility][ptr[1]] ≈ 40000.0
    @test table.z[ptr[1]] ≈ 95.0 atol = 1e-9

    h2 = _rows_of(table, "H2", "")
    @test length(h2) == 1
    @test table.x[h2[1]] ≈ 1105.0 atol = 1e-9
    @test table.y[h2[1]] ≈ 2100.0 atol = 1e-9
    @test table.z[h2[1]] ≈ 100.0 atol = 1e-9
    @test table.values[:cu][h2[1]] ≈ 20.0

    dsr = _rows_of(table, "H2", "DSR")
    @test length(dsr) == 2
    @test table.x[dsr] ≈ [1105.0, 1108.0] atol = 1e-9
    pos = table.values[:susceptibility][.!table.censored[:susceptibility] .&
                                        isfinite.(table.values[:susceptibility])]
    @test minimum(pos) ≈ 40000.0
    @test table.censored[:susceptibility][dsr[2]]
    @test table.values[:susceptibility][dsr[2]] ≈ minimum(pos)
    @test table.values[:susceptibility][dsr[1]] ≈ 50000.0
    @test !table.censored[:susceptibility][dsr[1]]
    @test count(table.censored[:susceptibility]) == 1
    @test report.susc_censored == 1
    @test _logged_raw_susc(logger) ≈ [-20.0]
    @test !any(table.censored[:susceptibility][_rows_of(table, "H1", "")])
    @test !any(table.censored[:density])
    @test only(s.unit for s in table.specs if s.name === :density) == "kg/m3"
    @test only(s.unit for s in table.specs if s.name === :susceptibility) == "1e-6 SI"
    @test only(s.kind for s in table.specs if s.name === :petro_source) === :categorical

    @test report.cu_samples_read == 7
    @test report.cu_holes_read == 4
    @test report.cu_censored == 2
    @test report.cu_censored / report.cu_samples_read ≈ 2 / 7
    @test report.cu_samples_kept == 5
    @test report.cu_holes_kept == 2
    @test report.cu_outside_bounds == 1
    @test report.ptr_samples_kept == 1
    @test report.ptr_holes_kept == 1
    @test report.dsr_samples_kept == 2
    @test report.dsr_holes_kept == 1
    @test report.petro_rows_ignored == 1

    summary = Dict(row.reason => row for row in exclusion_summary(report))
    @test summary["other_method"].holes == 1
    @test summary["other_method"].cu_samples == 1
    @test summary["missing_elevation"].holes == 1
    @test summary["missing_elevation"].cu_samples == 1
    @test summary["outside_bounds"].holes == 1
    @test summary["outside_bounds"].cu_samples == 1
    elev = only(e for e in report.exclusions if e.reason == "missing_elevation")
    @test elev.hole == "H3"

    loaded, covs, cfg = load_site(KEI_SITE)
    @test nsamples(loaded) == nsamples(table)
    @test cfg["crs"] == "EPSG:2393"
    @test length(covs) == 2
    depth = only(c for c in covs if c isa DepthCovariate)
    @test depth.z_datum ≈ 50.0
    @test any(c -> c isa CoordinateCovariate, covs)
end

@testset "one geometry failure is dropped and logged" begin
    # 19 good 511P holes and 1 conflicting duplicate depth: 1/20 = 5%,
    # which is not more than 5%, so the load keeps the good holes.
    mktempdir() do dir
        collar = String[]
        survey = String[]
        assays = String[]
        for i in 1:19
            id = "G" * lpad(i, 2, "0")
            push!(collar, "$id ^ 2000 ^ $(1000 + i) ^ 100")
            push!(survey, "$id ^ 0 ^ 90 ^ 0")
            push!(assays, "$id ^ 0 ^ 2 ^ 10 ^  ^ 511P")
        end
        push!(collar, "BAD ^ 2000 ^ 1500 ^ 100")
        push!(survey, "BAD ^ 0 ^ 45 ^ 0")
        push!(survey, "BAD ^ 10 ^ 45 ^ 0")
        push!(survey, "BAD ^ 10 ^ 45 ^ 90")
        push!(assays, "BAD ^ 0 ^ 2 ^ 10 ^  ^ 511P")
        _caret(joinpath(dir, "collar.txt"),
               ["HOLE_ID", "KKJ_NORTH", "KKJ_EAST", "Z"], collar)
        _caret(joinpath(dir, "kalte.txt"),
               ["Tunnus", "Syvyys", "Kaltevuus", "Suunta"], survey)
        _caret(joinpath(dir, "511P.txt"),
               ["Tunnus", "Ylasyvyys", "Alasyvyys", "Cu", "Cu_L", "Method"], assays)
        _caret(joinpath(dir, "petro.txt"),
               ["Tunnus", "Syvyys", "PTR_D", "PTR_K", "DSR_D", "DSR_K"], String[])
        # desurvey itself still throws; the adapter is what catches it.
        @test_throws ArgumentError desurvey(
            (1500.0, 2000.0, 100.0), [0.0, 10.0, 10.0], [0.0, 0.0, 90.0],
            [45.0, 45.0, 45.0]; angle_unit = :degree, dip_down_negative = false)
        table, report = keivitsa_sample_table(dir, TOML.parsefile(_site_toml(dir)))
        @test "BAD" ∉ table.hole
        @test length(unique(table.hole)) == 19
        dropped = only(e for e in report.exclusions if e.hole == "BAD")
        @test dropped.reason == "duplicate_depth"
        @test occursin("same depth", dropped.message)
        @test occursin("10.0", dropped.message)
        @test dropped.n_cu == 1
        summary = only(row for row in exclusion_summary(report)
                       if row.reason == "duplicate_depth")
        @test summary.holes == 1
        @test summary.cu_samples == 1
    end
end

@testset "geometry failures above 5% of 511P holes abort the load" begin
    mktempdir() do dir
        collar = String[]
        survey = String[]
        assays = String[]
        for id in ("G1", "G2")
            push!(collar, "$id ^ 2000 ^ 1000 ^ 100")
            push!(survey, "$id ^ 0 ^ 90 ^ 0")
            push!(assays, "$id ^ 0 ^ 2 ^ 10 ^  ^ 511P")
        end
        for id in ("D1", "D2")
            push!(collar, "$id ^ 2000 ^ 1100 ^ 100")
            push!(survey, "$id ^ 0 ^ 45 ^ 0")
            push!(survey, "$id ^ 10 ^ 45 ^ 0")
            push!(survey, "$id ^ 10 ^ 45 ^ 90")
            push!(assays, "$id ^ 0 ^ 2 ^ 10 ^  ^ 511P")
        end
        push!(collar, "U1 ^ 2000 ^ 1200 ^ 100")
        push!(survey, "U1 ^ 10 ^ 90 ^ 0")
        push!(survey, "U1 ^ 0 ^ 90 ^ 0")
        push!(assays, "U1 ^ 0 ^ 2 ^ 10 ^  ^ 511P")
        _caret(joinpath(dir, "collar.txt"),
               ["HOLE_ID", "KKJ_NORTH", "KKJ_EAST", "Z"], collar)
        _caret(joinpath(dir, "kalte.txt"),
               ["Tunnus", "Syvyys", "Kaltevuus", "Suunta"], survey)
        _caret(joinpath(dir, "511P.txt"),
               ["Tunnus", "Ylasyvyys", "Alasyvyys", "Cu", "Cu_L", "Method"], assays)
        _caret(joinpath(dir, "petro.txt"),
               ["Tunnus", "Syvyys", "PTR_D", "PTR_K", "DSR_D", "DSR_K"], String[])
        caught = nothing
        try
            keivitsa_sample_table(dir, TOML.parsefile(_site_toml(dir)))
        catch err
            caught = err
        end
        @test caught isa ArgumentError
        @test occursin("5%", caught.msg)
        @test occursin("duplicate_depth", caught.msg)
        @test occursin("unsorted_stations", caught.msg)
    end
end

@testset "overlapping assay intervals are not averaged" begin
    mktempdir() do dir
        _caret(joinpath(dir, "collar.txt"),
               ["HOLE_ID", "KKJ_NORTH", "KKJ_EAST", "Z"],
               ["OK ^ 2000 ^ 1000 ^ 100", "OV ^ 2000 ^ 1100 ^ 100"])
        _caret(joinpath(dir, "kalte.txt"),
               ["Tunnus", "Syvyys", "Kaltevuus", "Suunta"],
               ["OK ^ 0 ^ 90 ^ 0", "OV ^ 0 ^ 90 ^ 0"])
        _caret(joinpath(dir, "511P.txt"),
               ["Tunnus", "Ylasyvyys", "Alasyvyys", "Cu", "Cu_L", "Method"],
               ["OK ^ 0 ^ 2 ^ 10 ^  ^ 511P",
                "OV ^ 0 ^ 10 ^ 5 ^  ^ 511P",
                "OV ^ 5 ^ 12 ^ 50 ^  ^ 511P"])
        _caret(joinpath(dir, "petro.txt"),
               ["Tunnus", "Syvyys", "PTR_D", "PTR_K", "DSR_D", "DSR_K"], String[])
        table, report = keivitsa_sample_table(dir, TOML.parsefile(_site_toml(dir)))
        @test unique(table.hole) == ["OK"]
        @test only(table.values[:cu]) ≈ 10.0
        dropped = only(e for e in report.exclusions if e.hole == "OV")
        @test dropped.reason == "overlapping_intervals"
        @test occursin("overlap", dropped.message)
        @test dropped.n_cu == 2
    end
end

@testset "Keivitsa site file needs a root or KEIVITSA_ROOT" begin
    path = joinpath(@__DIR__, "..", "sites", "keivitsa.toml")
    withenv("KEIVITSA_ROOT" => nothing) do
        @test_throws ArgumentError load_site(path)
    end
end

@testset "susceptibility lod_policy other than min_positive is rejected" begin
    cfg = TOML.parsefile(KEI_SITE)
    cfg["properties"]["susceptibility"]["lod_policy"] = "percentile"
    err = try
        keivitsa_sample_table(KEI_FIXTURE, cfg)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("min_positive", sprint(showerror, err))
end

@testset "depth_below_surface is depth under the collar IDW surface" begin
    flat = DepthBelowSurface([0.0, 40.0, 80.0], [5.0, -5.0, 15.0],
                             [200.0, 200.0, 200.0], 1.0;
                             log_mean = 0.0, log_std = 1.0)
    xyz = [10.0 25.0; 0.0 1.0; 100.0 190.0]
    out = evaluate(flat, xyz)
    @test channel_names(flat) == ["log_depth"]
    # d0 = 1 m, so both depths sit above the clamp and invert exactly.
    @test flat.d0 * 10 ^ out[1, 1] ≈ 200.0 - 100.0
    @test flat.d0 * 10 ^ out[1, 2] ≈ 200.0 - 190.0
    @test out[1, 1] > out[1, 2]
    @test evaluate(flat, xyz[:, 1:1]) ≈ out[:, 1:1]
    extra = [10.0 25.0 70.0; 0.0 1.0 0.0; 100.0 190.0 40.0]
    @test evaluate(flat, extra)[:, 1:2] ≈ out
    @test flat.log_mean == 0.0
    @test flat.log_std == 1.0

    # Power 2: at x = 25 the distances are 25 m and 75 m.
    # Weights 9 : 1, so the surface is (9*100 + 1*300) / 10 = 120.
    sloped = DepthBelowSurface([0.0, 100.0], [0.0, 0.0], [100.0, 300.0], 1.0;
                               log_mean = 0.0, log_std = 1.0)
    quarter = evaluate(sloped, reshape(Float64[25.0, 0.0, 0.0], :, 1))
    @test sloped.d0 * 10 ^ quarter[1, 1] ≈ 120.0

    mktempdir() do dir
        open(joinpath(dir, "site.toml"), "w") do io
            print(io, """
            name = "keivitsa"
            source = "keivitsa"
            crs = "EPSG:2393"
            root = "$(KEI_FIXTURE)"
            collar = "collar.txt"
            survey = "kalte.txt"
            assays = "511P.txt"
            petrophysics = "petro.txt"

            [bounds]
            x = [900.0, 1300.0]
            y = [1900.0, 2300.0]
            z = [50.0, 150.0]

            [properties.cu]
            kind = "continuous"
            transform = "log10"
            lod_policy = "fixed"
            lod = 1.0
            unit = "ppm"

            [properties.density]
            kind = "continuous"
            transform = "identity"
            unit = "kg/m3"

            [properties.susceptibility]
            kind = "continuous"
            transform = "log10"
            lod_policy = "min_positive"
            unit = "1e-6 SI"

            [properties.petro_source]
            kind = "categorical"
            transform = "identity"
            unit = ""

            [covariates]
            use = ["coordinates", "depth_below_surface"]

            [covariates.depth_below_surface]
            d0 = 25.0
            """)
        end
        table, covs, cfg = load_site(joinpath(dir, "site.toml"))
        @test cfg["properties"]["density"]["unit"] == "kg/m3"
        @test cfg["properties"]["susceptibility"]["unit"] == "1e-6 SI"
        @test cfg["properties"]["petro_source"]["kind"] == "categorical"
        @test "petro_source" ∉ cfg["covariates"]["use"]
        surf = only(c for c in covs if c isa DepthBelowSurface)
        # H1, H2, H4 have Z = 100. H3 has Z = 0 and is not a station.
        @test length(surf.elevation) == 3
        @test all(==(100.0), surf.elevation)
        @test surf.log_std > 0
        # Query at H3's collar. A kept Z = 0 station would make the surface 0.
        q = [1050.0 1050.0; 2000.0 2000.0; 0.0 90.0]
        ev = evaluate(surf, q)
        raw = ev .* surf.log_std .+ surf.log_mean
        d = surf.d0 .* 10 .^ raw
        @test d[1] ≈ 100.0 - 0.0
        @test ev[1] > ev[2]
        @test evaluate(surf, q[:, 1:1]) ≈ ev[:, 1:1]
        @test only(s.kind for s in table.specs if s.name === :petro_source) === :categorical
    end
end
