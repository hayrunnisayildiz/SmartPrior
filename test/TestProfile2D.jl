using Test
using SmartPriorMT
import MTGeophysics
using MTGeophysics: BuildMesh2D, build_mt2d_halfspace_model, run_mt2d_forward,
    data_from_response2d

_p2d_mesh() = BuildMesh2D(frequencies = 10.0 .^ range(-2, 1; length = 5),
                          y_core_range = (-3000.0, 3000.0),
                          y_core_cell = 500.0,
                          y_padding = 4000.0,
                          air_cells = 4,
                          ground_layers = vcat(fill(200.0, 5), fill(400.0, 4)),
                          receiver_positions = collect(-2500.0:1000.0:2500.0))

@testset "grid_from_mt2dmesh matches the mesh's ground cells" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    na = mesh.n_air_cells

    nx, ny, nz = size(g)
    @test nx == 1
    @test ny == length(mesh.y_cell_sizes)
    @test nz == length(mesh.z_cell_sizes) - na

    # cell sizes and node positions must agree cell for cell, or the two grids
    # describe different geometry and every comparison downstream is meaningless
    @test g.dy ≈ mesh.y_cell_sizes
    @test g.dz ≈ mesh.z_cell_sizes[(na+1):end]
    @test g.y[1] ≈ mesh.y_nodes[1]
    @test g.y[end] ≈ mesh.y_nodes[end]
    @test g.z[1] ≈ mesh.z_nodes[na+1]
    @test g.z[end] ≈ mesh.z_nodes[end]

    # the ground starts at the surface, so the first ground node is the datum
    @test g.z[1] ≈ 0.0 atol = 1e-9

    # the strike cell is centred on the profile, so a 2D body is symmetric
    # about it and the prism gravity operator sees an elongated source
    @test g.x[1] ≈ -5.0e4
    @test g.x[end] ≈ 5.0e4
    @test g.dx == [1.0e5]

    @test grid_from_mt2dmesh(mesh; strike_extent = 2.0e5).dx == [2.0e5]
    @test_throws ArgumentError grid_from_mt2dmesh(mesh; strike_extent = 0.0)
end

@testset "to_mt2d and from_mt2d round trip" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    nx, ny, nz = size(g)
    na = mesh.n_air_cells

    # varying in both axes, so a transposed conversion cannot pass
    lr = [1.0 + 0.1 * j + 0.5 * k for i in 1:nx, j in 1:ny, k in 1:nz]

    rho = to_mt2d(lr, mesh)
    @test size(rho) == (length(mesh.z_cell_sizes), ny)
    @test all(rho[1:na, :] .== 1.0e9)                     # air re-added
    @test rho[na+1, 1] ≈ 10.0^lr[1, 1, 1]
    # depth first, profile second: check a cell that is asymmetric in the indices
    @test rho[na+3, 2] ≈ 10.0^lr[1, 2, 3]

    back = from_mt2d(rho, mesh)
    @test size(back) == size(lr)
    @test back ≈ lr

    @test_throws DimensionMismatch to_mt2d(zeros(1, 2, 2), mesh)
    @test_throws DimensionMismatch from_mt2d(zeros(2, 2), mesh)
end

@testset "to_mt2d turns non-finite cells into air" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    nx, ny, nz = size(g)

    lr = fill(2.0, nx, ny, nz)
    lr[1, 1, 1] = NaN
    rho = to_mt2d(lr, mesh)
    @test rho[mesh.n_air_cells+1, 1] == 1.0e9
    @test rho[mesh.n_air_cells+1, 2] ≈ 100.0

    # a custom air value is honoured, for meshes using a different convention
    @test to_mt2d(lr, mesh; air_resistivity = 1.0e12)[1, 1] == 1.0e12

    # a non-positive resistivity has no logarithm and must come back as air
    bad = fill(100.0, length(mesh.z_cell_sizes), ny)
    bad[mesh.n_air_cells+1, 1] = 0.0
    @test isnan(from_mt2d(bad, mesh)[1, 1, 1])
end

@testset "to_mt2d accepts a SyntheticModel" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    m = add_block(truth_halfspace(g; log_rho = 2.0);
                  x = (-1.0e5, 1.0e5), y = (-1000.0, 1000.0), z = (200.0, 800.0),
                  log_rho = 1.0, density = 300.0)

    rho = to_mt2d(m, mesh)
    @test size(rho) == (length(mesh.z_cell_sizes), length(mesh.y_cell_sizes))
    @test any(isapprox.(rho, 10.0))
    @test any(isapprox.(rho, 100.0))
end

@testset "a half-space survives the round trip into the 2D solver" begin
    # the strongest check available without hand-verifying the mesh: build a
    # half-space through this package's own grid, hand it to the 2D solver, and
    # require the analytic half-space response back. any axis, air-offset or
    # unit error would show up as a wrong apparent resistivity.
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    hs = truth_halfspace(g; log_rho = 2.0)

    rho_ours = to_mt2d(hs, mesh)
    rho_theirs = build_mt2d_halfspace_model(mesh; background_resistivity = 100.0)
    @test rho_ours ≈ rho_theirs

    resp = run_mt2d_forward(mesh, rho_ours)
    @test all(isapprox.(resp.rho_xy, 100.0; rtol = 0.05))
    @test all(isapprox.(resp.phase_xy, 45.0; atol = 2.0))
end

@testset "a conductive body is visible to the 2D solver at the right place" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    m = add_block(truth_halfspace(g; log_rho = 2.0);
                  x = (-1.0e5, 1.0e5), y = (-1500.0, -500.0), z = (200.0, 1000.0),
                  log_rho = 0.5)

    resp = run_mt2d_forward(mesh, to_mt2d(m, mesh))
    hs = run_mt2d_forward(mesh, to_mt2d(truth_halfspace(g; log_rho = 2.0), mesh))

    # long-period TE apparent resistivity must drop, and most over the body
    drop = 1 .- resp.rho_xy[end, :] ./ hs.rho_xy[end, :]
    @test maximum(drop) > 0.1
    # receivers run -2500:1000:2500, so the body at y in [-1500, -500] sits
    # under the second one. a transposed conversion would put it on the other side
    @test argmax(drop) in (1, 2, 3)
    @test mesh.receiver_positions[argmax(drop)] < 0
end

@testset "profile_sites from a response" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    resp = run_mt2d_forward(mesh, to_mt2d(truth_halfspace(g; log_rho = 2.0), mesh))

    sites = profile_sites(mesh, resp)
    @test sites isa MTSites
    @test nsites(sites) == length(mesh.receiver_positions)
    @test sites.y ≈ mesh.receiver_positions
    @test all(sites.x .== 0.0)          # the profile lies on the strike cell's centre line
    @test sites.periods ≈ resp.periods
    @test sites.rho_a ≈ resp.rho_xy
    @test sites.err_rho_a === nothing
    @test sites.err_phase === nothing

    # and it feeds the baseline, which is the point of carrying the data across
    base = nb_baseline(g, sites)
    @test size(base) == size(g)
    @test all(isapprox.(base, 2.0; atol = 0.2))
end

@testset "profile_sites from data with impedance errors" begin
    mesh = _p2d_mesh()
    g = grid_from_mt2dmesh(mesh)
    resp = run_mt2d_forward(mesh, to_mt2d(truth_halfspace(g; log_rho = 2.0), mesh))
    data = data_from_response2d(resp; impedance_error_fraction = 0.05)

    sites = profile_sites(mesh, data)
    @test sites.err_rho_a !== nothing
    @test sites.err_phase !== nothing
    @test all(>(0), sites.err_rho_a)
    @test all(>(0), sites.err_phase)
    @test sites.rho_a ≈ data.rho_xy
    @test sites.phase ≈ data.phase_xy
end

@testset "profile_sites from periods only" begin
    mesh = _p2d_mesh()
    sites = profile_sites(mesh, [1.0, 10.0, 100.0])
    @test nsites(sites) == length(mesh.receiver_positions)
    @test all(isnan, sites.rho_a)
    @test size(sites.rho_a) == (3, length(mesh.receiver_positions))

    # geometry-only features still work, which is what the placeholder is for
    g = grid_from_mt2dmesh(mesh)
    chans, names = coverage_channels(g, sites.x, sites.y)
    @test names == ["site_distance"]
    @test all(isfinite, chans[1])
end
