# One survey: a sample table, the covariates the network is allowed to see,
# and the parsed site file. Adding a district is a new TOML file plus, if
# the rows are not already Cloncurry-shaped, an adapter. It is not a change
# to the network.

"""
    load_site(path) -> (table::SampleTable, covariates, cfg)

Read a site TOML file. `source = "cloncurry"` loads rows through
[`cloncurry_sample_table`](@ref). The dataset root is `root` in the file
(relative to the file's directory), otherwise `CLONCURRY_ROOT`, otherwise
the usual Desktop dump of `Cloncurry_integrated_2026-09-17`.

Covariates are only those named in `covariates.use`. pXRF, drillhole
lithology, and `sample_distance` are rejected here.
"""
function load_site(path::AbstractString)
    isfile(path) || throw(ArgumentError("load_site: no such file: $path"))
    cfg = TOML.parsefile(String(path))
    source = String(get(cfg, "source", ""))
    source == "cloncurry" || throw(ArgumentError(
        "load_site: unsupported source $(repr(source))"))
    crs = String(get(cfg, "crs", ""))
    crs == "EPSG:28354" || throw(ArgumentError(
        "load_site: expected crs \"EPSG:28354\", got $(repr(crs))"))
    root = _site_root(cfg, path)
    table = cloncurry_sample_table(root, cfg)
    covs = _site_covariates(cfg, root)
    return table, covs, cfg
end

function _site_root(cfg, toml_path::AbstractString)
    if haskey(cfg, "root") && !isempty(strip(String(cfg["root"])))
        r = String(cfg["root"])
        return isabspath(r) ? r : normpath(joinpath(dirname(toml_path), r))
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
            push!(covs, DepthCovariate(zlo, d0, rho, period))
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
        else
            throw(ArgumentError(
                "load_site: unknown covariate $(repr(name)). " *
                "Known: coordinates, depth, structure_distance, surface_geology"))
        end
    end
    return covs
end
