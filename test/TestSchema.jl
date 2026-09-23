using Test
using SmartPrior

const SCHEMA_FIXTURE = joinpath(@__DIR__, "fixtures", "site_tiny.toml")
const SCHEMA_PETRO = joinpath(@__DIR__, "fixtures", "cloncurry_petro_tiny.csv")
const SCHEMA_METAL = joinpath(@__DIR__, "fixtures", "cloncurry_metal_tiny.csv")
const SCHEMA_STRUCTURES = joinpath(@__DIR__, "fixtures", "cloncurry_structures_tiny.geojson")
const SCHEMA_SURFACE = joinpath(@__DIR__, "fixtures", "cloncurry_surface_geology_tiny.geojson")

function _csv_column(path, column)
    lines = readlines(path)
    header = String.(strip.(split(lines[1], ',')))
    j = findfirst(==(column), header)
    j === nothing && error("no column $column in $path")
    out = String[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        push!(out, length(fields) >= j ? strip(fields[j]) : "")
    end
    return out
end

function _cell_centre_xyz(g::PriorGrid)
    nx, ny, nz = size(g)
    xyz = Matrix{Float64}(undef, 3, nx * ny * nz)
    t = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        xyz[1, t] = g.cx[i]
        xyz[2, t] = g.cy[j]
        xyz[3, t] = g.cz[k]
        t += 1
    end
    return xyz
end

function _maxabs(a, b)
    return maximum(abs.(a .- b))
end

@testset "SampleTable rows, masks, and subset" begin
    specs = [
        PropertySpec(:cu, :continuous, :log10, "ppm"),
        PropertySpec(:lithology, :categorical, :identity, ""),
    ]
    table = SampleTable(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [10.0, 11.0, 12.0],
        ["H1", "H1", "H2"], ["Ernest Henry", "Ernest Henry", "Monakoff"],
        Dict(:cu => [5.0, NaN, 1.0]),
        Dict(:cu => BitVector([true, false, false])),
        Dict(:lithology => ["MSD", "", "missing"]),
        specs)
    @test nsamples(table) == 3
    mask = observed_mask(table, :cu)
    @test mask == BitVector([true, false, true])
    @test observed_mask(table, :lithology) == BitVector([true, false, false])
    sub = subset(table, [1, 3])
    @test nsamples(sub) == 2
    @test sub.hole == ["H1", "H2"]
    @test sub.values[:cu] ≈ [5.0, 1.0]
    @test sub.censored[:cu] == BitVector([true, false])
    kept = subset(table, table.group .== "Ernest Henry")
    @test nsamples(kept) == 2
    @test_throws ArgumentError SampleTable(
        [1.0], [0.0], [0.0], [""], ["D"],
        Dict(:cu => [1.0]), Dict(:cu => falses(1)),
        Dict{Symbol,Vector{String}}(),
        [PropertySpec(:cu, :continuous, :log10, "ppm")])
    @test_throws ArgumentError PropertySpec(:cu, :ordinal, :log10, "ppm")
    @test_throws ArgumentError PropertySpec(:cu, :continuous, :sqrt, "ppm")
end

@testset "load_site fixture returns a table and covariates" begin
    table, covs, cfg = load_site(SCHEMA_FIXTURE)
    @test table isa SampleTable
    @test nsamples(table) == 6
    @test covs isa Vector{<:Covariate}
    @test length(covs) == length(cfg["covariates"]["use"])
    @test cfg["source"] == "cloncurry"
    @test cfg["crs"] == "EPSG:28354"
    @test any(c -> c isa CoordinateCovariate, covs)
    @test any(c -> c isa DepthCovariate, covs)
    @test any(c -> c isa StructureDistance, covs)
    @test any(c -> c isa SurfaceGeology, covs)

    xyz = [1010.0 1020.0; 2010.0 2020.0; -15.0 -25.0]
    _, names = evaluate_all(covs, xyz)
    @test !any(n -> n == "sample_distance" || startswith(n, "geochem_") ||
                    startswith(n, "lith_"), names)

    @test all(s -> !isempty(strip(s)) && lowercase(strip(s)) != "missing", table.hole)
    @test all(s -> !isempty(strip(s)) && lowercase(strip(s)) != "missing", table.group)
end

@testset "censored Cu and conductivity match the fixture tokens" begin
    table, _, cfg = load_site(SCHEMA_FIXTURE)
    n_lod = count(s -> uppercase(s) == "<LOD", _csv_column(SCHEMA_METAL, "Cu_Concentration"))
    n_zero = count(_csv_column(SCHEMA_PETRO, "conductivity_mean_S_m_100kHz")) do s
        v = tryparse(Float64, s)
        v !== nothing && v == 0
    end
    @test n_lod == count(table.censored[:cu])
    @test n_zero == count(table.censored[:conductivity])

    # The stored Cu bound is the smallest positive assay, not the old 1 ppm fill.
    observed = observed_mask(table, :cu)
    positive = table.values[:cu][observed .& .!table.censored[:cu]]
    @test all(v -> v ≈ minimum(positive), table.values[:cu][table.censored[:cu]])
    @test minimum(positive) != CLONCURRY_CU_DL_PPM

    floor = Float64(cfg["properties"]["conductivity"]["floor"])
    @test all(v -> v ≈ floor, table.values[:conductivity][table.censored[:conductivity]])
    @test isnan(table.values[:cu][3])
    @test isnan(table.values[:conductivity][3])
    @test observed_mask(table, :cu)[2]
    @test !observed_mask(table, :lithology)[2]
end

@testset "blank drillhole and deposit ids are filled" begin
    mktempdir() do tmp
        lines = readlines(SCHEMA_PETRO)
        row = split(lines[2], ',')
        row[3] = ""
        lines[2] = join(row, ',')
        row3 = split(lines[4], ',')
        row3[2] = "missing"
        lines[4] = join(row3, ',')
        petro = joinpath(tmp, "petro.csv")
        open(petro, "w") do io
            for line in lines
                println(io, line)
            end
        end
        cp(SCHEMA_METAL, joinpath(tmp, "metal.csv"))
        cfg = Dict{String,Any}(
            "petrophysics" => "petro.csv",
            "metals" => "metal.csv",
            "properties" => Dict{String,Any}(
                "cu" => Dict{String,Any}(
                    "column" => "Cu_Concentration",
                    "kind" => "continuous",
                    "transform" => "log10",
                    "lod_policy" => "min_positive",
                    "unit" => "ppm"),
            ),
        )
        table = cloncurry_sample_table(tmp, cfg)
        @test table.hole[1] == "__ungrouped_1"
        @test table.group[3] == "__ungrouped_deposit_3"
        @test all(s -> !isempty(strip(s)) && lowercase(strip(s)) != "missing", table.hole)
        @test all(s -> !isempty(strip(s)) && lowercase(strip(s)) != "missing", table.group)
    end
end

@testset "covariates at cell centres match the grid channels" begin
    g = PriorGrid([400.0, 400.0, 400.0, 600.0], [400.0, 500.0, 700.0],
                  [50.0, 80.0, 120.0];
                  origin = [-1550.0, -1100.0, 0.0])
    xyz = _cell_centre_xyz(g)
    stack = build_features(g; coordinates = true)
    M, names = evaluate_all([CoordinateCovariate(g), DepthCovariate(g)], xyz)
    @test names == stack.names
    Fm = feature_matrix(stack)
    @test _maxabs(M, Fm) <= 1e-10

    lines = load_cloncurry_structures(SCHEMA_STRUCTURES)
    polys = load_cloncurry_surface_geology(SCHEMA_SURFACE)
    xs = Float64[]
    ys = Float64[]
    for p in polys
        for (rx, ry) in p.rings
            append!(xs, rx)
            append!(ys, ry)
        end
    end
    for line in lines
        append!(xs, line.x)
        append!(ys, line.y)
    end
    pad = 200.0
    span_x = maximum(xs) - minimum(xs) + 2pad
    span_y = maximum(ys) - minimum(ys) + 2pad
    geo = PriorGrid(fill(span_x / 4, 4), fill(span_y / 3, 3), [40.0, 90.0];
                    origin = [minimum(xs) - pad, minimum(ys) - pad, -30.0])
    xyz_g = _cell_centre_xyz(geo)

    sold, nold = structure_distance_channels(geo, lines)
    snew = StructureDistance(geo, lines)
    sm = evaluate(snew, xyz_g)
    @test channel_names(snew) == nold
    for k in eachindex(nold)
        @test _maxabs(sm[k, :], vec(sold[k])) <= 1e-10
    end

    uold, uold_n = surface_geology_channels(geo, polys)
    unew = SurfaceGeology(polys)
    um = evaluate(unew, xyz_g)
    @test channel_names(unew) == uold_n
    for k in eachindex(uold_n)
        @test _maxabs(um[k, :], vec(uold[k])) <= 1e-10
    end
end

@testset "pXRF, lithology, and sample distance are not covariates" begin
    mktempdir() do tmp
        open(joinpath(tmp, "bad.toml"), "w") do io
            println(io, """
            name = "bad"
            source = "cloncurry"
            crs = "EPSG:28354"
            root = "."
            [bounds]
            x = [0.0, 1.0]
            y = [0.0, 1.0]
            z = [0.0, 1.0]
            [properties.cu]
            column = "Cu_Concentration"
            kind = "continuous"
            transform = "log10"
            lod_policy = "min_positive"
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
