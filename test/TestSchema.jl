using Test
using SmartPrior
using Statistics

@testset "SampleTable rows, masks, and subset" begin
    specs = [
        PropertySpec(:cu, :continuous, :log10, "ppm"),
        PropertySpec(:lithology, :categorical, :identity, ""),
    ]
    table = SampleTable(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [10.0, 11.0, 12.0],
        ["H1", "H1", "H2"], ["D1", "D1", "D2"],
        ["drillhole", "drillhole", "drillhole"],
        Dict(:cu => [5.0, NaN, 1.0]),
        Dict(:cu => BitVector([true, false, false])),
        Dict(:lithology => ["MSD", "", "missing"]),
        specs)
    @test nsamples(table) == 3
    mask = observed_mask(table, :cu)
    @test mask == BitVector([true, false, true])
    @test observed_mask(table, :lithology) == BitVector([true, false, false])
    @test real_hole_mask(table) == BitVector([true, true, true])
    sub = subset(table, [1, 3])
    @test nsamples(sub) == 2
    @test sub.hole == ["H1", "H2"]
    @test sub.sample_type == ["drillhole", "drillhole"]
    @test sub.values[:cu] ≈ [5.0, 1.0]
    @test sub.censored[:cu] == BitVector([true, false])
    kept = subset(table, table.group .== "D1")
    @test nsamples(kept) == 2
    @test_throws ArgumentError SampleTable(
        [1.0], [0.0], [0.0], [""], ["D"], ["drillhole"],
        Dict(:cu => [1.0]), Dict(:cu => falses(1)),
        Dict{Symbol,Vector{String}}(),
        [PropertySpec(:cu, :continuous, :log10, "ppm")])
    @test_throws ArgumentError PropertySpec(:cu, :ordinal, :log10, "ppm")
    @test_throws ArgumentError PropertySpec(:cu, :continuous, :sqrt, "ppm")

    located = SampleTable(
        [1.0, NaN, 3.0], [0.0, 1.0, Inf], [4.0, 5.0, 6.0],
        ["A", "B", "C"], ["D", "D", "D"],
        ["drillhole", "outcrop", "drillhole"],
        Dict(:cu => [1.0, 2.0, 3.0]),
        Dict(:cu => falses(3)),
        Dict{Symbol,Vector{String}}(),
        [PropertySpec(:cu, :continuous, :log10, "ppm")])
    @test training_mask(located) == BitVector([true, false, false])
    @test real_hole_mask(located) == BitVector([true, false, true])
end

@testset "coordinate and depth covariates evaluate pointwise" begin
    coords = CoordinateCovariate(0.0, 100.0, -50.0, 50.0, -200.0, 0.0)
    xyz = [0.0 50.0 100.0 25.0; -50.0 0.0 50.0 10.0; -200.0 -100.0 0.0 -150.0]
    M = evaluate(coords, xyz)
    @test channel_names(coords) == ["x_norm", "y_norm", "z_norm"]
    @test M[:, 1] ≈ [-1.0, -1.0, -1.0]
    @test M[:, 2] ≈ [0.0, 0.0, 0.0]
    @test M[:, 3] ≈ [1.0, 1.0, 1.0]
    @test evaluate(coords, xyz[:, 1:2]) ≈ M[:, 1:2]

    # log_depth moments are frozen at construction; evaluate must not
    # recompute them from the query points.
    depth = DepthCovariate(-200.0, 0.0, 25.0, 100.0, 1000.0)
    D = evaluate(depth, xyz)
    @test channel_names(depth) == ["log_depth", "depth_over_skin"]
    @test size(D) == (2, 4)
    @test evaluate(depth, xyz[:, 1:2]) ≈ D[:, 1:2]
    raw = log10.(max.(xyz[3, :] .+ 200.0, 25.0) ./ 25.0)
    @test D[1, :] ≈ (raw .- depth.log_mean) ./ depth.log_std
    @test issorted(D[2, [1, 4, 2, 3]])

    M2, names = evaluate_all([coords, depth], xyz)
    @test names == vcat(channel_names(coords), channel_names(depth))
    @test M2 ≈ vcat(M, D)
    @test_throws ArgumentError evaluate_all([coords, coords], xyz)
end

@testset "pXRF, lithology, and sample distance are not covariates" begin
    mktempdir() do tmp
        open(joinpath(tmp, "bad.toml"), "w") do io
            println(io, """
            name = "bad"
            source = "keivitsa"
            crs = "EPSG:2393"
            root = "."
            [bounds]
            x = [0.0, 1.0]
            y = [0.0, 1.0]
            z = [0.0, 1.0]
            [properties.cu]
            kind = "continuous"
            transform = "log10"
            lod_policy = "fixed"
            lod = 1.0
            unit = "ppm"
            [covariates]
            use = ["sample_distance"]
            """)
        end
        err = try
            load_site(joinpath(tmp, "bad.toml"))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("sample_distance", sprint(showerror, err))
    end
end
