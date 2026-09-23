# One survey: a sample table, the covariates the network is allowed to see,
# and the parsed site file. Adding a district is a new TOML file plus, if
# the rows are not already Cloncurry-shaped, an adapter. It is not a change
# to the network.

const _SITE_CRS = Dict(
    "cloncurry" => "EPSG:28354",
    "keivitsa" => "EPSG:2393",
)

"""
    load_site(path) -> (table::SampleTable, covariates, cfg)

Read a site TOML file. `source = "cloncurry"` loads rows through
[`cloncurry_sample_table`](@ref). `source = "keivitsa"` loads rows through
[`keivitsa_sample_table`](@ref). The dataset root is `root` in the file
(relative to the file's directory). Otherwise Cloncurry uses `CLONCURRY_ROOT`
or the usual Desktop dump, and Keivitsa uses `KEIVITSA_ROOT`.

Each source accepts one CRS (`EPSG:28354` or `EPSG:2393`). Covariates are
only those named in `covariates.use`. pXRF, drillhole lithology, and
`sample_distance` are rejected here.
"""
function load_site(path::AbstractString)
    isfile(path) || throw(ArgumentError("load_site: no such file: $path"))
    cfg = TOML.parsefile(String(path))
    source = String(get(cfg, "source", ""))
    haskey(_SITE_CRS, source) || throw(ArgumentError(
        "load_site: unsupported source $(repr(source))"))
    crs = String(get(cfg, "crs", ""))
    expected = _SITE_CRS[source]
    crs == expected || throw(ArgumentError(
        "load_site: source $(repr(source)) expects crs $(repr(expected)), got $(repr(crs))"))
    root = _site_root(cfg, path)
    # Reject a covariate list before touching the assay files.
    covs = _site_covariates(cfg, root)
    table = if source == "cloncurry"
        cloncurry_sample_table(root, cfg)
    else
        loaded, _ = keivitsa_sample_table(root, cfg)
        loaded
    end
    return table, covs, cfg
end

function _site_root(cfg, toml_path::AbstractString)
    if haskey(cfg, "root") && !isempty(strip(String(cfg["root"])))
        r = String(cfg["root"])
        return isabspath(r) ? r : normpath(joinpath(dirname(toml_path), r))
    end
    source = String(get(cfg, "source", ""))
    if source == "keivitsa"
        env = get(ENV, "KEIVITSA_ROOT", "")
        isempty(env) && throw(ArgumentError(
            "load_site: set root in the site file or KEIVITSA_ROOT"))
        return env
    end
    env = get(ENV, "CLONCURRY_ROOT", "")
    if !isempty(env)
        return env
    end
    for c in (
        joinpath(homedir(), "Desktop", "datasets4HY",
                 "Cloncurry_integrated_2026-09-17"),
        "/Users/hayrunnisayildiz/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17",
    )
        isdir(joinpath(c, "derived")) && return c
    end
    throw(ArgumentError(
        "load_site: set root in the site file or CLONCURRY_ROOT"))
end

function _bounds_pair(cfg, axis::AbstractString)
    haskey(cfg, "bounds") || throw(ArgumentError("load_site: missing [bounds]"))
    b = cfg["bounds"]
    haskey(b, axis) || throw(ArgumentError("load_site: bounds.$axis is missing"))
    v = b[axis]
    length(v) == 2 || throw(ArgumentError(
        "load_site: bounds.$axis must be [min, max], got $(v)"))
    lo = Float64(v[1])
    hi = Float64(v[2])
    hi > lo || throw(ArgumentError(
        "load_site: bounds.$axis max ($hi) must be greater than min ($lo)"))
    return lo, hi
end

function _cfg_table(cfg, key::AbstractString)
    haskey(cfg, key) || return Dict{String,Any}()
    t = cfg[key]
    t isa AbstractDict || throw(ArgumentError(
        "load_site: [$key] must be a table"))
    return t
end

function _resolve_file(root::AbstractString, rel::AbstractString)
    p = String(rel)
    return isabspath(p) ? p : normpath(joinpath(root, p))
end

function _required_number(table, key::AbstractString, what::AbstractString)
    haskey(table, key) || throw(ArgumentError("load_site: missing $what"))
    x = Float64(table[key])
    x > 0 || throw(ArgumentError("load_site: $what must be positive, got $x"))
    return x
end

function _site_covariates(cfg, root::AbstractString)
    block = _cfg_table(cfg, "covariates")
    haskey(block, "use") || throw(ArgumentError(
        "load_site: missing covariates.use"))
    use = block["use"]
    use isa AbstractVector || throw(ArgumentError(
        "load_site: covariates.use must be a list of names"))
    isempty(use) && throw(ArgumentError("load_site: covariates.use is empty"))

    xlo, xhi = _bounds_pair(cfg, "x")
    ylo, yhi = _bounds_pair(cfg, "y")
    zlo, zhi = _bounds_pair(cfg, "z")
    depth_cfg = _cfg_table(block, "depth")
    struct_cfg = _cfg_table(block, "structure_distance")
    surf_cfg = _cfg_table(block, "surface_geology")

    covs = Covariate[]
    for raw in use
        name = String(raw)
        _reject_non_covariate(name)
        if name == "coordinates"
            push!(covs, CoordinateCovariate(xlo, xhi, ylo, yhi, zlo, zhi))
        elseif name == "depth"
            rho = Float64(get(depth_cfg, "rho_ref", 100.0))
            period = Float64(get(depth_cfg, "period_max", 1000.0))
            d0 = _required_number(depth_cfg, "d0",
                                  "covariates.depth.d0 (half the reference cell thickness)")
            has_mean = haskey(depth_cfg, "log_mean")
            has_std = haskey(depth_cfg, "log_std")
            if has_mean || has_std
                (has_mean && has_std) || throw(ArgumentError(
                    "load_site: covariates.depth.log_mean and log_std must be set together"))
                push!(covs, DepthCovariate(zlo, d0, rho, period,
                                           Float64(depth_cfg["log_mean"]),
                                           Float64(depth_cfg["log_std"])))
            else
                push!(covs, DepthCovariate(zlo, zhi, d0, rho, period))
            end
        elseif name == "structure_distance"
            haskey(struct_cfg, "file") || throw(ArgumentError(
                "load_site: missing covariates.structure_distance.file"))
            h = _required_number(struct_cfg, "length_scale",
                                 "covariates.structure_distance.length_scale")
            path = _resolve_file(root, struct_cfg["file"])
            push!(covs, StructureDistance(path, h))
        elseif name == "surface_geology"
            haskey(surf_cfg, "file") || throw(ArgumentError(
                "load_site: missing covariates.surface_geology.file"))
            path = _resolve_file(root, surf_cfg["file"])
            push!(covs, SurfaceGeology(path))
        elseif name == "depth_below_surface"
            String(get(cfg, "source", "")) == "keivitsa" || throw(ArgumentError(
                "load_site: depth_below_surface uses Keivitsa collar elevations " *
                "(finite Z other than 0)"))
            below = _cfg_table(block, "depth_below_surface")
            d0 = _required_number(below, "d0",
                                  "covariates.depth_below_surface.d0 (half the reference cell thickness)")
            east, north, elev = _surface_collars(root, cfg)
            has_mean = haskey(below, "log_mean")
            has_std = haskey(below, "log_std")
            if has_mean || has_std
                (has_mean && has_std) || throw(ArgumentError(
                    "load_site: covariates.depth_below_surface.log_mean and log_std must be set together"))
                push!(covs, DepthBelowSurface(east, north, elev, d0;
                                              log_mean = Float64(below["log_mean"]),
                                              log_std = Float64(below["log_std"])))
            else
                push!(covs, DepthBelowSurface(east, north, elev, d0;
                                              z_min = zlo, z_max = zhi))
            end
        else
            throw(ArgumentError(
                "load_site: unknown covariate $(repr(name)). " *
                "Known: coordinates, depth, depth_below_surface, " *
                "structure_distance, surface_geology"))
        end
    end
    return covs
end
