using Test
using SmartPriorMT
using CodecZlib
using Random

const PETRO_TINY = joinpath(@__DIR__, "fixtures", "cloncurry_petro_tiny.csv")
const METAL_TINY = joinpath(@__DIR__, "fixtures", "cloncurry_metal_tiny.csv")

function _tiny_grid()
    return PriorGrid(fill(10.0, 4), fill(10.0, 4), fill(10.0, 4);
                     origin = [1000.0, 2000.0, -50.0])
end

@testset "Cloncurry samples: Cu held out, units, missing-property anchors" begin
    s = load_cloncurry_samples(PETRO_TINY, METAL_TINY)
    @test length(s) == 6
    @test s.geochem_names == ["Mg", "Al", "Si", "Zn", "U"]
    @test !("Cu" in s.geochem_names)
    @test !("LE" in s.geochem_names)
    @test !("Co" in s.geochem_names)          # all <LOD, dropped
    @test s.cu_ppm[1] ≈ 100.0
    @test s.cu_ppm[2] ≈ CLONCURRY_CU_DL_PPM   # <LOD lifted for the target
    @test isnan(s.cu_ppm[3])                  # empty stays missing
    @test isnan(s.density_g_cm3[2])
    @test s.density_g_cm3[1] ≈ 2.80
    @test s.conductivity_S_m_100kHz[2] ≈ 0.0
    @test isnan(s.conductivity_S_m_100kHz[3])
    @test occursin("KT-20", CLONCURRY_CONDUCTIVITY_STATUS)
    @test occursin("100 kHz", CLONCURRY_CONDUCTIVITY_STATUS)
    @test occursin("not equivalent", lowercase(CLONCURRY_CONDUCTIVITY_STATUS))
    @test CLONCURRY_PROPERTY_NAMES[4] == "conductivity_100kHz"
    @test !("resistivity" in CLONCURRY_PROPERTY_NAMES)

    geochem = cloncurry_geochemistry(s)
    @test !("Cu" in geochem.names)
    @test geochem.names == s.geochem_names

    lith = cloncurry_lithology(s)
    @test "MSD" in lith.labels
    @test "OTHER" in lith.labels              # empty lithology_code

    g = _tiny_grid()
    anchors = load_cloncurry_anchors(g, s)
    # S4 (easting 9999) is outside the 40 m box
    @test containing_cell(g, s.east[4], s.north[4], s.elev[4]) == 0
    @test length(anchors.grade[1]) >= 1
    @test length(anchors.density[1]) >= 1
    @test length(anchors.susceptibility[1]) >= 1
    @test length(anchors.conductivity_100kHz[1]) >= 1
    @test anchors.stats.n_grade == 5
    @test anchors.stats.n_density == 5          # S2 density missing
    @test isnan(s.density_g_cm3[2])
    @test anchors.stats.n_conductivity_zero == 1
    @test anchors.stats.n_conductivity_100kHz == 5  # S3 missing; zero on S2 kept
    @test CLONCURRY_PROPERTY_NAMES[4] == "conductivity_100kHz"
end

@testset "Cloncurry gzip metals path and kg/m³ unit guard" begin
    mktempdir() do tmp
        gz = joinpath(tmp, "metal_all_fields.csv.gz")
        open(gz, "w") do io
            stream = GzipCompressorStream(io)
            write(stream, read(METAL_TINY))
            close(stream)
        end
        s = load_cloncurry_samples(PETRO_TINY, gz)
        @test length(s) == 6
        @test !("Cu" in s.geochem_names)

        bad = joinpath(tmp, "petro_bad.csv")
        open(bad, "w") do io
            for (i, line) in enumerate(eachline(PETRO_TINY))
                if i == 1
                    println(io, line)
                elseif startswith(line, "S1,")
                    println(io, replace(line, ",2800," => ",9999,"; count = 1))
                else
                    println(io, line)
                end
            end
        end
        @test_throws ArgumentError load_cloncurry_samples(bad, METAL_TINY)
    end
end

@testset "cloncurry_grid work box" begin
    bounds = cloncurry_work_bounds()
    g = cloncurry_grid(bounds; cell = 1000.0, cell_z = 100.0)
    @test g.x[1] ≈ bounds.x_min
    @test g.x[end] ≈ bounds.x_max
    @test g.y[end] ≈ bounds.y_max
    @test g.z[1] ≈ bounds.z_min
    @test g.z[end] ≈ bounds.z_max
    @test size(g, 1) == 42
    @test size(g, 2) == 76
    @test size(g, 3) == 18
    @test ncells(g) == 42 * 76 * 18
end

@testset "cloncurry_deposit_bounds Ernest Henry" begin
    s = load_cloncurry_samples(PETRO_TINY, METAL_TINY)
    b = cloncurry_deposit_bounds(s, "Ernest Henry"; pad_xy = 0.0, pad_z = 0.0)
    @test b.deposit == "Ernest Henry"
    @test b.n_samples == 4
    @test b.x_min ≈ 1005.0
    @test b.x_max ≈ 1036.0
    @test_throws ArgumentError cloncurry_deposit_bounds(s, "NoSuchDeposit")
end

@testset "spatial_holdout buffer" begin
    x = collect(0.0:100.0:900.0)
    y = zeros(10)
    z = zeros(10)
    eligible = trues(10)
    split = spatial_holdout(x, y, z, eligible; fraction = 0.2, buffer = 250.0,
                            rng = Xoshiro(1))
    @test length(split.test) == 2
    @test isempty(intersect(Set(split.train), Set(split.test)))
    @test isempty(intersect(Set(split.train), Set(split.buffer)))
    @test isempty(intersect(Set(split.test), Set(split.buffer)))
    @test length(split.train) + length(split.test) + length(split.buffer) == 10
    for i in split.test, j in split.train
        @test hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) >= 250.0 - 1e-12
    end
    for a in 1:length(split.test), b in (a + 1):length(split.test)
        i, j = split.test[a], split.test[b]
        @test hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) >= 250.0 - 1e-12
    end
    for i in split.buffer
        @test any(j -> hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) < 250.0,
                  split.test)
    end
    @test_throws ArgumentError spatial_holdout(x, y, z, eligible;
                                               fraction = 0.2, buffer = 1.0e6,
                                               rng = Xoshiro(1))
end

@testset "grade_keep hides Cu only" begin
    s = load_cloncurry_samples(PETRO_TINY, METAL_TINY)
    g = _tiny_grid()
    all_a = load_cloncurry_anchors(g, s)
    keep = cloncurry_grade_eligible(g, s)
    @test count(keep) == all_a.stats.n_grade_mapped
    keep[1] = false
    masked = load_cloncurry_anchors(g, s; grade_keep = keep)
    @test masked.stats.n_grade_mapped == all_a.stats.n_grade_mapped - 1
    @test masked.stats.n_density_mapped == all_a.stats.n_density_mapped
    @test masked.stats.n_susceptibility_mapped == all_a.stats.n_susceptibility_mapped
    @test masked.stats.n_conductivity_100kHz_mapped ==
          all_a.stats.n_conductivity_100kHz_mapped
end
