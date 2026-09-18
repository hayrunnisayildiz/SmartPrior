using Test
using SmartPriorMT

@testset "DBF Latin-1 decode" begin
    ascii = Vector{UInt8}(b"HOLE_ID\0\0\0")
    @test SmartPriorMT._decode_dbf_text(ascii) == "HOLE_ID"
    # Finnish Ä = 0xC4; treating as UTF-8 would be invalid
    a_umlaut = UInt8[0x6d, 0x61, 0x61, 0xc4]
    @test SmartPriorMT._decode_dbf_text(a_umlaut) == "maa" * Char(0xC4)
    @test isvalid(SmartPriorMT._decode_dbf_text(a_umlaut))
end

@testset "cu_log10 and merge_petro_pair" begin
    @test cu_log10(100.0) ≈ 2.0
    @test cu_log10(0.0) ≈ 0.0          # lifted to detection limit 1 ppm
    @test cu_log10(-5.0) ≈ 0.0         # BDL negative → detection limit
    @test cu_log10(0.1; detection_limit = 1.0) ≈ 0.0
    @test isnan(cu_log10(NaN))

    @test merge_petro_pair(3000.0, 3100.0) ≈ 3050.0
    @test merge_petro_pair(3000.0, NaN) ≈ 3000.0
    @test merge_petro_pair(NaN, 2800.0) ≈ 2800.0
    @test isnan(merge_petro_pair(NaN, NaN))
end

@testset "read_petro_txt skips four header rows and ignores K" begin
    path = joinpath(@__DIR__, "fixtures", "petro_tiny.txt")
    p = read_petro_txt(path)
    @test length(p.hole) == 3
    @test p.hole[1] == "HOLE1"
    @test p.depth[1] ≈ 10.0
    # petro.txt X is northing (~7.5e6), Y is easting (~3.5e6)
    @test p.kkj_x[1] ≈ 7512000.0
    @test p.kkj_y[1] ≈ 3498500.0

    @test merge_petro_pair(p.ptr_d[1], p.dsr_d[1]) ≈ 3050.0
    @test merge_petro_pair(p.ptr_j[1], p.dsr_j[1]) ≈ 90.0
    @test merge_petro_pair(p.ptr_d[2], p.dsr_d[2]) ≈ 2800.0
    @test merge_petro_pair(p.ptr_j[2], p.dsr_j[2]) ≈ 40.0
    @test merge_petro_pair(p.ptr_d[3], p.dsr_d[3]) ≈ 3200.0
    @test isnan(p.luo_r[3])
    @test p.luo_r[1] ≈ 50.0
    # K is parsed so a caller can inspect it, but it is not an anchor
    @test p.ptr_k[1] ≈ 999.0
    @test occursin("high-confidence", KEIVITSA_PETRO_STATUS.PTR_D)
    @test occursin("high-confidence", KEIVITSA_PETRO_STATUS.DSR_D)
    @test occursin("high-confidence", KEIVITSA_PETRO_STATUS.PTR_J)
    @test occursin("high-confidence", KEIVITSA_PETRO_STATUS.DSR_J)
    @test occursin("unused", KEIVITSA_PETRO_STATUS.PTR_K)
    @test occursin("remanent", KEIVITSA_PETRO_STATUS.PTR_K)
    @test occursin("unused", KEIVITSA_PETRO_STATUS.DSR_K)
    @test occursin("unverified", KEIVITSA_PETRO_STATUS.LUO_R)

    cov = petro_coverage(p)
    @test cov.n == 3
    @test cov.luo_r ≈ 2 / 3
    @test cov.density == 1.0
end

@testset "desurvey_depths vertical hole" begin
    e, n, z = desurvey_depths(100.0, 200.0, 50.0, [0.0, 10.0, 20.0];
                              azimuth = 0.0, dip = 90.0)
    @test e ≈ [100.0, 100.0, 100.0] atol = 1e-9
    @test n ≈ [200.0, 200.0, 200.0] atol = 1e-9
    @test z ≈ [50.0, 40.0, 30.0] atol = 1e-6
end

@testset "map_points_to_cells aggregates duplicates" begin
    g = PriorGrid(fill(10.0, 4), fill(10.0, 4), fill(10.0, 4);
                  origin = [0.0, 0.0, 0.0])
    # two samples in the same first cell, one outside
    x = [1.0, 2.0, 999.0]
    y = [1.0, 2.0, 1.0]
    z = [1.0, 2.0, 1.0]
    v = [1.0, 3.0, 9.0]
    cells, values, weights = map_points_to_cells(g, x, y, z, v; weights = [1.0, 1.0, 1.0])
    @test length(cells) == 1
    @test values[1] ≈ 2.0
    @test weights[1] ≈ 2.0
    @test containing_cell(g, 999.0, 1.0, 1.0) == 0
end

@testset "keivitsa_grid matches config bounds" begin
    bounds = (x_min = 0.0, x_max = 100.0, y_min = 0.0, y_max = 50.0,
              z_min = -10.0, z_max = 10.0)
    g = keivitsa_grid(bounds; cell = 25.0)
    @test g.x[1] ≈ 0.0
    @test g.x[end] ≈ 100.0
    @test g.y[end] ≈ 50.0
    @test g.z[1] ≈ -10.0
    @test g.z[end] ≈ 10.0
    g_z = keivitsa_grid(bounds; cell = 25.0, cell_z = 10.0)
    @test size(g_z, 1) == size(g, 1)
    @test size(g_z, 2) == size(g, 2)
    @test size(g_z, 3) == 2
    @test g_z.dz[1] ≈ 10.0
    @test g_z.z[end] ≈ 10.0
end

@testset "drill geochemistry and surface-priority combine" begin
    path = joinpath(@__DIR__, "fixtures", "intervals_tiny.csv")
    dh = load_keivitsa_drill_geochemistry(path)
    @test length(dh) == 3
    @test dh.names == ["S", "FE", "CR", "PT"]
    @test !("CU" in dh.names)
    @test dh.x[1] ≈ 3498500.0
    @test dh.y[1] ≈ 7512000.0
    @test dh.z[1] ≈ 100.0
    @test dh.values[1, findfirst(==("S"), dh.names)] ≈ 1.2
    @test isnan(dh.values[3, findfirst(==("S"), dh.names)])
    @test dh.values[3, findfirst(==("CR"), dh.names)] ≈ 500.0

    surface = PointSamples([1.0, 2.0], [3.0, 4.0],
                           [10.0 20.0; 11.0 21.0],
                           ["CU", "NI"]; z = [5.0, 6.0])
    # drill also carries CU: combine must keep the surface CU column
    drill_overlap = PointSamples([10.0], [20.0],
                                 [1.0 9.0 100.0],
                                 ["S", "CU", "FE"]; z = [1.0])
    comb, skipped = combine_geochemistry(surface, drill_overlap)
    @test skipped == ["CU"]
    @test comb.names == ["CU", "NI", "S", "FE"]
    @test length(comb) == 3
    @test comb.values[1, 1] ≈ 10.0          # surface CU
    @test isnan(comb.values[3, 1])          # drill row does not overwrite CU
    @test comb.values[3, 3] ≈ 1.0           # drill S
    @test isnan(comb.values[1, 3])          # surface row has no S
end
