# Final Keivitsa Cu block model: nn_xyz (E2) + kriging (v1) on all Cu holes.
#
# E2 = Gaussian NLL, early stop on validation RMSE of μ, five seeds
# (same frozen settings as examples/feasibility_keivitsa.jl).
#
# Run:  KEIVITSA_ROOT=... julia --project=. examples/keivitsa_blockmodel.jl
# Out:  tmp_keivitsa_blockmodel/  (gitignored)

const ROOT = dirname(@__DIR__)
if !haskey(ENV, "SMARTPRIOR_WORK") || isempty(ENV["SMARTPRIOR_WORK"])
    ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, "tmp_keivitsa_blockmodel")
elseif !isabspath(ENV["SMARTPRIOR_WORK"])
    ENV["SMARTPRIOR_WORK"] = joinpath(ROOT, ENV["SMARTPRIOR_WORK"])
end
include(joinpath(@__DIR__, "feasibility_loho.jl"))

using WriteVTK

const SITE_PATH = joinpath(ROOT, "sites", "keivitsa.toml")
const DEPOSIT = "keivitsa"
const TARGET = :cu
const EXPECTED_CU_HOLES = 261

const NN_TRAIN_KW = (loss = :nll, stop_on = :val_rmse)
const LOSS_LOG_EVERY = 50

const CELL_XY = 20.0
const CELL_Z = 10.0
const MASK_RADIUS_M = 60.0

# Local metres for saved block-model coordinates (figure / export frame).
const LOCAL_X0 = 3_496_570.0
const LOCAL_Y0 = 7_509_950.0

const PRED_BATCH = 25_000

function claim_work_dir()
    lock_path = joinpath(WORK, ".run.lock")
    if isfile(lock_path)
        fail("refusing to start: lock file present at $lock_path")
    end
    if isdir(WORK)
        names = readdir(WORK)
        if !isempty(names)
            fail("refusing to start: work directory exists and is non-empty ($WORK)")
        end
    end
    mkpath(WORK)
    open(lock_path, "w") do io
        println(io, getpid())
    end
    return lock_path
end

function release_work_lock(lock_path)
    isfile(lock_path) && rm(lock_path)
    return nothing
end

function loss_at_step(history, step::Int)
    for h in history
        h.step == step || continue
        return h.train_loss, h.val_loss, h.val_rmse
    end
    return NaN, NaN, NaN
end

function append_loss_curves(io, method::AbstractString, seed::Int, history)
    for h in history
        println(io, join((
            method, string(seed), string(h.step),
            tsv_num(h.train_loss), tsv_num(h.val_loss), tsv_num(h.val_rmse),
        ), '\t'))
    end
    flush(io)
end

function train_ensemble_e2(model, Xtr, ytr, Xva, yva, ctx, method::AbstractString, loss_io)
    members = Tuple{Any,Any}[]
    epochs = Int[]
    for seed in NN_SEEDS
        history = Any[]
        ps, st, best_ep, last_ep, best_stop = train_member(
            model, seed, Xtr, ytr, Xva, yva, ctx;
            NN_TRAIN_KW..., log_every = LOSS_LOG_EVERY, history = history)
        append_loss_curves(loss_io, method, seed, history)
        tr_b, va_b, rm_b = loss_at_step(history, best_ep)
        tr_f, va_f, rm_f = loss_at_step(history, last_ep)
        logmsg(@sprintf("%s seed %d: best_step=%d train_loss=%.5e val_loss=%.5e val_rmse=%.5e | final_step=%d train_loss=%.5e val_loss=%.5e val_rmse=%.5e (stop_metric=%.5e)",
                        ctx, seed, best_ep, tr_b, va_b, rm_b, last_ep, tr_f, va_f, rm_f, best_stop))
        push!(members, (ps, st))
        push!(epochs, best_ep)
        last_ep == NN_EPOCHS &&
            logmsg("$ctx seed $seed ran all $(NN_EPOCHS) epochs (best $best_ep)")
    end
    return members, epochs
end

"nn_xyz Fourier features at arbitrary 3×N metre coordinates (same stack as finite_features)."
function features_xyz_at(covs, xyz::AbstractMatrix{<:Real})
    M, names = evaluate_all(covs, xyz)
    xyz_rows = Int[]
    for n in ("x_norm", "y_norm", "z_norm")
        k = findfirst(==(n), names)
        k === nothing && fail("covariate channels have no $n")
        push!(xyz_rows, k)
    end
    return fourier_append(M[xyz_rows, :], [1, 2, 3])
end

function build_sample_grid(xs, ys, zs; dx = CELL_XY, dy = CELL_XY, dz = CELL_Z)
    xlo, xhi = extrema(xs)
    ylo, yhi = extrema(ys)
    zlo, zhi = extrema(zs)
    # Cell-centred covering: edges snap outward so every sample sits inside a cell.
    x0 = floor(xlo / dx) * dx
    y0 = floor(ylo / dy) * dy
    z0 = floor(zlo / dz) * dz
    x1 = ceil(xhi / dx) * dx
    y1 = ceil(yhi / dy) * dy
    z1 = ceil(zhi / dz) * dz
    nx = max(1, round(Int, (x1 - x0) / dx))
    ny = max(1, round(Int, (y1 - y0) / dy))
    nz = max(1, round(Int, (z1 - z0) / dz))
    x_edges = collect(range(x0, x0 + nx * dx; length = nx + 1))
    y_edges = collect(range(y0, y0 + ny * dy; length = ny + 1))
    z_edges = collect(range(z0, z0 + nz * dz; length = nz + 1))
    cx = [(x_edges[i] + x_edges[i + 1]) / 2 for i in 1:nx]
    cy = [(y_edges[j] + y_edges[j + 1]) / 2 for j in 1:ny]
    cz = [(z_edges[k] + z_edges[k + 1]) / 2 for k in 1:nz]
    return (; x_edges, y_edges, z_edges, cx, cy, cz, nx, ny, nz, dx, dy, dz)
end

"""
Nearest-sample 3D distance at every cell centre via voxel bins (no extra package).
Returns a length-nx*ny*nz vector in Julia column-major cell order (i fastest).
"""
function nearest_sample_distance(grid, sx, sy, sz)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ncel = nx * ny * nz
    dist = fill(Inf, ncel)
    # Bin size ≈ mask radius so each query inspects a small stencil.
    bx = max(grid.dx, MASK_RADIUS_M)
    by = max(grid.dy, MASK_RADIUS_M)
    bz = max(grid.dz, MASK_RADIUS_M)
    bins = Dict{NTuple{3,Int},Vector{Int}}()
    @inbounds for p in eachindex(sx)
        key = (floor(Int, sx[p] / bx), floor(Int, sy[p] / by), floor(Int, sz[p] / bz))
        push!(get!(() -> Int[], bins, key), p)
    end
    rad = MASK_RADIUS_M + hypot(grid.dx, grid.dy, grid.dz)  # search a little past mask
    ib = ceil(Int, rad / bx)
    jb = ceil(Int, rad / by)
    kb = ceil(Int, rad / bz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        qx = grid.cx[i]
        qy = grid.cy[j]
        qz = grid.cz[k]
        iq = floor(Int, qx / bx)
        jq = floor(Int, qy / by)
        kq = floor(Int, qz / bz)
        best = Inf
        for dk in -kb:kb, dj in -jb:jb, di in -ib:ib
            pts = get(bins, (iq + di, jq + dj, kq + dk), nothing)
            pts === nothing && continue
            for p in pts
                d = hypot(qx - sx[p], qy - sy[p], qz - sz[p])
                d < best && (best = d)
            end
        end
        dist[i + (j - 1) * nx + (k - 1) * nx * ny] = best
    end
    return dist
end

function ensemble_predict_batched(model, members, X, shift, scale, context; batch = PRED_BATCH)
    n = size(X, 2)
    μ = Vector{Float64}(undef, n)
    σ = Vector{Float64}(undef, n)
    for a in 1:batch:n
        b = min(a + batch - 1, n)
        μb, σb = ensemble_predict(model, members, X[:, a:b], shift, scale, context)
        μ[a:b] .= μb
        σ[a:b] .= σb
    end
    return μ, σ
end

function kriging_predict_batched(tx, ty, tz, tv, qx, qy, qz, fit; batch = PRED_BATCH)
    n = length(qx)
    μ = Vector{Float64}(undef, n)
    σ = Vector{Float64}(undef, n)
    for a in 1:batch:n
        b = min(a + batch - 1, n)
        μb, σb = kriging_predict(tx, ty, tz, tv, qx[a:b], qy[a:b], qz[a:b], fit)
        μ[a:b] .= μb
        σ[a:b] .= σb
    end
    return μ, σ
end

function write_blockmodel_vts(path, x_edges, y_edges, z_edges, arrays)
    nx = length(x_edges) - 1
    ny = length(y_edges) - 1
    nz = length(z_edges) - 1
    (nx > 0 && ny > 0 && nz > 0) || fail("vts: need at least one cell on each axis")
    ncel = nx * ny * nz
    # StructuredGrid (.vts): full 3-D corner coordinate arrays (not 1-D → .vtr).
    Xp = Array{Float64}(undef, nx + 1, ny + 1, nz + 1)
    Yp = Array{Float64}(undef, nx + 1, ny + 1, nz + 1)
    Zp = Array{Float64}(undef, nx + 1, ny + 1, nz + 1)
    @inbounds for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        Xp[i, j, k] = x_edges[i]
        Yp[i, j, k] = y_edges[j]
        Zp[i, j, k] = z_edges[k]
    end
    endswith(path, ".vts") || fail("vts path must end with .vts, got $path")
    vtk_grid(path, Xp, Yp, Zp) do vtk
        for (name, a) in arrays
            length(a) == ncel || fail(
                "vts $(name): got $(length(a)) values, expected $ncel cells")
            vtk[String(name), VTKCellData()] = reshape(collect(Float64, a), nx, ny, nz)
        end
    end
    return path
end

function write_blockmodel_tsv(path, grid, xu, yu, zu, arrays, mask; kept_only::Bool)
    open(path, "w") do io
        names = ["x_local_m", "y_local_m", "z_m", "mask",
                 [String(n) for (n, _) in arrays]...]
        println(io, join(names, '\t'))
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            idx = i + (j - 1) * nx + (k - 1) * nx * ny
            kept_only && mask[idx] < 0.5 && continue
            row = String[tsv_num(xu[i]), tsv_num(yu[j]), tsv_num(zu[k]),
                         string(round(Int, mask[idx]))]
            for (_, a) in arrays
                push!(row, tsv_num(a[idx]))
            end
            println(io, join(row, '\t'))
        end
    end
    return path
end

function cell_ijk(idx::Int, nx::Int, ny::Int)
    t = idx - 1
    i = (t % nx) + 1
    j = ((t ÷ nx) % ny) + 1
    k = (t ÷ (nx * ny)) + 1
    return i, j, k
end

function main_blockmodel_locked()
    LOG[] = open(joinpath(WORK, "run.log"), "w")
    t0 = time()
    logmsg("Keivitsa Cu final block model (nn_xyz E2 + kriging v1)")
    logmsg("NN E2: loss=:nll stop_on=:val_rmse seeds=$(NN_SEEDS)")
    logmsg("grid $(CELL_XY)×$(CELL_XY)×$(CELL_Z) m; mask radius $(MASK_RADIUS_M) m")
    logmsg("julia " * string(VERSION))
    logmsg("work $WORK")

    table, covs, cfg = load_site(SITE_PATH)
    ids = [string(i) for i in 1:nsamples(table)]
    Xxyz, _, col_of, names = finite_features(covs, table)
    model_xyz = build_mlp(size(Xxyz, 1))
    logmsg(@sprintf("rows=%d xyz_nin=%d channels=%s",
                    nsamples(table), size(Xxyz, 1), join(names, ",")))

    spec = spec_of(table, TARGET)
    eligible = findall(
        training_mask(table) .& real_hole_mask(table) .& observed_mask(table, TARGET))
    all_holes = sort!(unique(table.hole[eligible]))
    n_holes = length(all_holes)
    if n_holes != EXPECTED_CU_HOLES
        logmsg("WARNING: expected $EXPECTED_CU_HOLES Cu holes, got $n_holes — using all")
    end
    logmsg("cu: $(length(eligible)) samples on $n_holes holes")

    y_all = fill(NaN, nsamples(table))
    y_all[eligible] = transformed(spec, table.values[TARGET][eligible])

    #----- final nn_xyz on ALL holes (val holes only for early stopping) -----
    train_rows = eligible
    y_tr = y_all[train_rows]
    m = mean(y_tr)
    s = std(y_tr)
    (isfinite(m) && isfinite(s) && s > 0) || fail("final train mean/std not usable")
    holes = sort!(unique(table.hole[train_rows]))
    # Same split scheme as CV: sort unique train holes + split_train_val(rng_for(tag)).
    tag = "val|$(DEPOSIT)|$(TARGET)|final"
    fit_holes, val_holes = split_train_val(holes, rng_for(tag))
    fit_set = Set(fit_holes)
    val_set = Set(val_holes)
    fit_rows = [i for i in train_rows if table.hole[i] in fit_set]
    val_rows = [i for i in train_rows if table.hole[i] in val_set]
    (isempty(fit_rows) || isempty(val_rows)) && fail("empty neural-net hole split")
    y_fit = (y_all[fit_rows] .- m) ./ s
    y_val = (y_all[val_rows] .- m) ./ s
    logmsg(@sprintf("nn_xyz val split: fit_holes=%d val_holes=%d tag=%s",
                    length(fit_holes), length(val_holes), tag))

    open(joinpath(WORK, "val_holes.tsv"), "w") do io
        println(io, "role\thole")
        for h in fit_holes
            println(io, "fit\t", h)
        end
        for h in val_holes
            println(io, "val\t", h)
        end
    end

    loss_path = joinpath(WORK, "loss_curves.tsv")
    loss_io = open(loss_path, "w")
    println(loss_io, "method\tseed\tstep\ttrain_loss\tval_loss\tval_rmse")
    flush(loss_io)

    t_nn = time()
    Xtr = take_cols(Xxyz, columns_of(col_of, fit_rows))
    Xva = take_cols(Xxyz, columns_of(col_of, val_rows))
    members, epochs = train_ensemble_e2(
        model_xyz, Xtr, y_fit, Xva, y_val, "keivitsa cu final nn_xyz", "nn_xyz", loss_io)
    close(loss_io)
    t_nn = time() - t_nn
    logmsg(@sprintf("nn_xyz 5-seed train done in %.1f s; best epochs=%s",
                    t_nn, join(string.(epochs), ",")))

    #----- kriging (v1) on all holes -----
    t_kr = time()
    dx, dy, dz, dv, dh, n_merged = dedupe_locations(
        table.x[train_rows], table.y[train_rows], table.z[train_rows],
        y_tr, table.hole[train_rows])
    n_merged > 0 && logmsg("kriging: averaged $n_merged coincident locations")
    fit = fit_variogram(dx, dy, dz, dv, dh; context = "keivitsa cu final kriging")
    ratio = fit.range_h / fit.range_v
    (isfinite(ratio) && ratio > 0) || fail("bad anisotropy ratio")
    open(joinpath(WORK, "kriging_params.tsv"), "w") do io
        println(io, "model\tnugget\tpartial_sill\ttotal_sill\trange_horizontal_m\t" *
                "range_vertical_m\tanisotropy_ratio\tfit_rmse\tn_pairs_horizontal\t" *
                "n_pairs_downhole\tn_bins_horizontal\tn_bins_downhole\tdegenerate\t" *
                "n_merged_locations")
        println(io, join((
            fit.model, tsv_num(fit.nugget), tsv_num(fit.partial_sill), tsv_num(fit.total_sill),
            tsv_num(fit.range_h), tsv_num(fit.range_v), tsv_num(ratio), tsv_num(fit.fit_rmse),
            string(fit.n_pairs_h), string(fit.n_pairs_v), string(fit.n_bins_h),
            string(fit.n_bins_v), fit.degenerate ? "1" : "0", string(n_merged),
        ), '\t'))
    end
    t_kr_fit = time() - t_kr
    logmsg(@sprintf("kriging (v1) fit in %.1f s (%s)", t_kr_fit, fit.model))

    #----- grid + mask -----
    sx = table.x[eligible]
    sy = table.y[eligible]
    sz = table.z[eligible]
    grid = build_sample_grid(sx, sy, sz)
    ncel = grid.nx * grid.ny * grid.nz
    logmsg(@sprintf("grid %d×%d×%d = %d cells", grid.nx, grid.ny, grid.nz, ncel))

    t_dist = time()
    nearest = nearest_sample_distance(grid, sx, sy, sz)
    t_dist = time() - t_dist
    mask = Float64[isfinite(d) && d <= MASK_RADIUS_M ? 1.0 : 0.0 for d in nearest]
    kept = findall(m -> m > 0.5, mask)
    n_kept = length(kept)
    logmsg(@sprintf("nearest-sample distances in %.1f s; kept=%d / %d (≤ %.0f m)",
                    t_dist, n_kept, ncel, MASK_RADIUS_M))
    n_kept >= 1 || fail("no blocks within mask radius")

    qx = Vector{Float64}(undef, n_kept)
    qy = Vector{Float64}(undef, n_kept)
    qz = Vector{Float64}(undef, n_kept)
    @inbounds for (t, idx) in enumerate(kept)
        i, j, k = cell_ijk(idx, grid.nx, grid.ny)
        qx[t] = grid.cx[i]
        qy[t] = grid.cy[j]
        qz[t] = grid.cz[k]
    end

    #----- predict on kept cells -----
    t_pred = time()
    xyz_q = Matrix{Float64}(undef, 3, n_kept)
    xyz_q[1, :] .= qx
    xyz_q[2, :] .= qy
    xyz_q[3, :] .= qz
    Xq = features_xyz_at(covs, xyz_q)
    size(Xq, 1) == size(Xxyz, 1) || fail("feature width mismatch at query points")
    μ_nn, σ_nn = ensemble_predict_batched(
        model_xyz, members, Xq, m, s, "keivitsa cu final nn_xyz grid")
    μ_kr, σ_kr = kriging_predict_batched(dx, dy, dz, dv, qx, qy, qz, fit)
    t_pred = time() - t_pred
    logmsg(@sprintf("grid predictions in %.1f s", t_pred))

    μ_nn_full = fill(NaN, ncel)
    σ_nn_full = fill(NaN, ncel)
    μ_kr_full = fill(NaN, ncel)
    σ_kr_full = fill(NaN, ncel)
    @inbounds for (t, idx) in enumerate(kept)
        μ_nn_full[idx] = μ_nn[t]
        σ_nn_full[idx] = σ_nn[t]
        μ_kr_full[idx] = μ_kr[t]
        σ_kr_full[idx] = σ_kr[t]
    end

    # Local-metre edges / centres for export (z stays elevation).
    x_edges_u = grid.x_edges .- LOCAL_X0
    y_edges_u = grid.y_edges .- LOCAL_Y0
    z_edges_u = copy(grid.z_edges)
    cx_u = grid.cx .- LOCAL_X0
    cy_u = grid.cy .- LOCAL_Y0
    cz_u = copy(grid.cz)

    arrays_vts = [
        "nearest_sample_m" => nearest,
        "mu_kriging_v1" => μ_kr_full,
        "sigma_kriging_v1" => σ_kr_full,
        "mu_neural_field" => μ_nn_full,
        "sigma_neural_field" => σ_nn_full,
        "mask" => mask,
    ]
    vts_path = joinpath(WORK, "keivitsa_blockmodel.vts")
    write_blockmodel_vts(vts_path, x_edges_u, y_edges_u, z_edges_u, arrays_vts)
    logmsg("wrote $vts_path")

    # Full grid lives in the .vts; TSV is kept cells (mask=1) to keep size tractable.
    arrays_tsv = [
        "nearest_sample_m" => nearest,
        "mu_kriging_v1" => μ_kr_full,
        "sigma_kriging_v1" => σ_kr_full,
        "mu_neural_field" => μ_nn_full,
        "sigma_neural_field" => σ_nn_full,
    ]
    tsv_path = joinpath(WORK, "keivitsa_blockmodel.tsv")
    write_blockmodel_tsv(tsv_path, grid, cx_u, cy_u, cz_u, arrays_tsv, mask; kept_only = true)
    logmsg("wrote $tsv_path (kept cells; mask column=1)")

    # Drill samples in local metres for figures (drillhole rows only).
    samples_path = joinpath(WORK, "samples_local.tsv")
    open(samples_path, "w") do io
        println(io, "hole\tx_local_m\ty_local_m\tz_m\tlog10_cu")
        for i in eligible
            println(io, join((
                tsv_field(table.hole[i]),
                tsv_num(table.x[i] - LOCAL_X0),
                tsv_num(table.y[i] - LOCAL_Y0),
                tsv_num(table.z[i]),
                tsv_num(y_all[i]),
            ), '\t'))
        end
    end
    logmsg("wrote $samples_path")

    elapsed = time() - t0
    report_path = joinpath(WORK, "report.txt")
    open(report_path, "w") do io
        println(io, "keivitsa_blockmodel")
        println(io, "cu_holes\t", n_holes)
        println(io, "cu_samples\t", length(eligible))
        println(io, "grid_nx\t", grid.nx)
        println(io, "grid_ny\t", grid.ny)
        println(io, "grid_nz\t", grid.nz)
        println(io, "n_cells\t", ncel)
        println(io, "mask_radius_m\t", MASK_RADIUS_M)
        println(io, "kept_blocks\t", n_kept)
        println(io, "nn_seeds\t", join(string.(NN_SEEDS), ","))
        println(io, "nn_best_epochs\t", join(string.(epochs), ","))
        println(io, "nn_train_s\t", tsv_num(t_nn))
        println(io, "kriging_fit_s\t", tsv_num(t_kr_fit))
        println(io, "nearest_dist_s\t", tsv_num(t_dist))
        println(io, "predict_s\t", tsv_num(t_pred))
        println(io, "wall_clock_s\t", tsv_num(elapsed))
        println(io, "wall_clock_h\t", tsv_num(elapsed / 3600))
        println(io, "full_5seed_e2_train\t1")
        println(io, "y_shift\t", tsv_num(m))
        println(io, "y_scale\t", tsv_num(s))
        println(io, "val_tag\t", tag)
        println(io, "n_fit_holes\t", length(fit_holes))
        println(io, "n_val_holes\t", length(val_holes))
        println(io, "local_x0\t", tsv_num(LOCAL_X0))
        println(io, "local_y0\t", tsv_num(LOCAL_Y0))
    end
    open(joinpath(WORK, "report.tsv"), "w") do io
        println(io, "key\tvalue")
        for line in eachline(report_path)
            occursin('\t', line) || continue
            println(io, line)
        end
    end
    logmsg("wrote $report_path")
    logmsg(@sprintf("kept_blocks=%d  wall=%.1f s (%.2f h)", n_kept, elapsed, elapsed / 3600))
    logmsg(@sprintf("finished in %.1f s", elapsed))
    return nothing
end

function main()
    lock_path = claim_work_dir()
    try
        main_blockmodel_locked()
    finally
        release_work_lock(lock_path)
        if isassigned(LOG) && isopen(LOG[])
            close(LOG[])
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
