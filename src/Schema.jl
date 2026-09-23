# Site-agnostic sample table.
#
# One row is one physical specimen. Continuous properties live in `values`
# (NaN = not measured) with a parallel `censored` bit (true = the stored
# number is an upper bound, e.g. a detection limit). Categorical properties
# live in `classes` as one label per row ("" = not observed). `specs` records
# the transform a later training step will apply; this table stores numbers
# in the original unit and does not apply that transform.
#
# pXRF assays, drillhole lithology, and distance-to-sample are not covariates.
# Lithology and pXRF can sit here as properties (future outputs). Distance
# to the nearest sample is a confidence mask, not a column of this table.

"""
    PropertySpec(name, kind, transform, unit)

Description of one sample property.

`kind` is `:continuous` or `:categorical`. `transform` is `:log10` or
`:identity` and is metadata for a later training step; [`SampleTable`](@ref)
does not apply it. `unit` is a human-readable unit string (possibly empty
for a categorical property).
"""
struct PropertySpec
    name::Symbol
    kind::Symbol
    transform::Symbol
    unit::String

    function PropertySpec(name::Symbol, kind::Symbol, transform::Symbol,
                          unit::AbstractString)
        kind in (:continuous, :categorical) || throw(ArgumentError(
            "PropertySpec: kind must be :continuous or :categorical, got $(repr(kind))"))
        transform in (:log10, :identity) || throw(ArgumentError(
            "PropertySpec: transform must be :log10 or :identity, got $(repr(transform))"))
        return new(name, kind, transform, String(unit))
    end
end

"""
    SampleTable(x, y, z, hole, group, values, censored, classes, specs)

Row-aligned specimens in a projected Cartesian frame (metres).

`hole` is the drillhole id and `group` is the deposit (or other hold-out
group). Both are required on every row: an empty string or `"missing"` is
rejected. `values` and `censored` have one entry per continuous spec;
`classes` has one label vector per categorical spec. A censored row must
store a finite upper bound, never NaN.
"""
struct SampleTable
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    hole::Vector{String}
    group::Vector{String}
    values::Dict{Symbol,Vector{Float64}}
    censored::Dict{Symbol,BitVector}
    classes::Dict{Symbol,Vector{String}}
    specs::Vector{PropertySpec}

    function SampleTable(x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real},
                         z::AbstractVector{<:Real},
                         hole::AbstractVector{<:AbstractString},
                         group::AbstractVector{<:AbstractString},
                         values::AbstractDict,
                         censored::AbstractDict,
                         classes::AbstractDict,
                         specs::AbstractVector{PropertySpec})
        xv = collect(Float64, x)
        yv = collect(Float64, y)
        zv = collect(Float64, z)
        hv = String.(hole)
        gv = String.(group)
        n = length(xv)
        length(yv) == n || throw(ArgumentError(
            "SampleTable: y has $(length(yv)) rows but x has $n"))
        length(zv) == n || throw(ArgumentError(
            "SampleTable: z has $(length(zv)) rows but x has $n"))
        length(hv) == n || throw(ArgumentError(
            "SampleTable: hole has $(length(hv)) rows but x has $n"))
        length(gv) == n || throw(ArgumentError(
            "SampleTable: group has $(length(gv)) rows but x has $n"))
        for i in 1:n
            _require_filled_id(hv[i], "hole", i)
            _require_filled_id(gv[i], "group", i)
        end

        specv = collect(PropertySpec, specs)
        names = [s.name for s in specv]
        allunique(names) || throw(ArgumentError(
            "SampleTable: property names must be unique, got $(names)"))
        cont = Set(s.name for s in specv if s.kind === :continuous)
        cat = Set(s.name for s in specv if s.kind === :categorical)

        values_d = Dict{Symbol,Vector{Float64}}()
        for (k, v) in values
            key = k isa Symbol ? k : Symbol(k)
            values_d[key] = collect(Float64, v)
        end
        cens_d = Dict{Symbol,BitVector}()
        for (k, v) in censored
            key = k isa Symbol ? k : Symbol(k)
            bits = v isa BitVector ? copy(v) : BitVector(v)
            cens_d[key] = bits
        end
        class_d = Dict{Symbol,Vector{String}}()
        for (k, v) in classes
            key = k isa Symbol ? k : Symbol(k)
            class_d[key] = String.(v)
        end

        Set(keys(values_d)) == cont || throw(ArgumentError(
            "SampleTable: values keys $(collect(keys(values_d))) do not match " *
            "continuous specs $(collect(cont))"))
        Set(keys(cens_d)) == cont || throw(ArgumentError(
            "SampleTable: censored keys $(collect(keys(cens_d))) do not match " *
            "continuous specs $(collect(cont))"))
        Set(keys(class_d)) == cat || throw(ArgumentError(
            "SampleTable: classes keys $(collect(keys(class_d))) do not match " *
            "categorical specs $(collect(cat))"))

        for name in cont
            length(values_d[name]) == n || throw(ArgumentError(
                "SampleTable: values[:$(name)] has $(length(values_d[name])) rows, expected $n"))
            length(cens_d[name]) == n || throw(ArgumentError(
                "SampleTable: censored[:$(name)] has $(length(cens_d[name])) rows, expected $n"))
            @inbounds for i in 1:n
                if cens_d[name][i] && !isfinite(values_d[name][i])
                    throw(ArgumentError(
                        "SampleTable: censored[:$(name)][$i] is true but the value is not finite"))
                end
            end
        end
        for name in cat
            length(class_d[name]) == n || throw(ArgumentError(
                "SampleTable: classes[:$(name)] has $(length(class_d[name])) rows, expected $n"))
        end

        return new(xv, yv, zv, hv, gv, values_d, cens_d, class_d, specv)
    end
end

function _require_filled_id(s::AbstractString, what::AbstractString, i::Int)
    t = strip(s)
    if isempty(t) || lowercase(t) == "missing"
        throw(ArgumentError("SampleTable: $what[$i] is empty or missing"))
    end
    return nothing
end

"""
    nsamples(table::SampleTable) -> Int

Number of rows.
"""
nsamples(table::SampleTable) = length(table.x)

function _find_spec(table::SampleTable, prop::Symbol)
    for s in table.specs
        s.name === prop && return s
    end
    known = join((string(s.name) for s in table.specs), ", ")
    throw(ArgumentError("SampleTable: no property $(repr(prop)) (known: $known)"))
end

"""
    training_mask(table::SampleTable) -> BitVector

True on rows whose `(x, y, z)` are all finite. A row with a missing
coordinate is not a training location; it stays in the table.
"""
function training_mask(table::SampleTable)
    n = nsamples(table)
    mask = falses(n)
    @inbounds for i in 1:n
        mask[i] = isfinite(table.x[i]) && isfinite(table.y[i]) && isfinite(table.z[i])
    end
    return mask
end

"""
    observed_mask(table, prop) -> BitVector

True where `prop` was measured. Censored continuous values count as observed
(the upper bound is a measurement). NaN and, for categories, a blank or
`"missing"` label count as unobserved.
"""
function observed_mask(table::SampleTable, prop::Symbol)
    spec = _find_spec(table, prop)
    n = nsamples(table)
    mask = falses(n)
    if spec.kind === :continuous
        v = table.values[prop]
        @inbounds for i in 1:n
            mask[i] = isfinite(v[i])
        end
    else
        labs = table.classes[prop]
        @inbounds for i in 1:n
            t = strip(labs[i])
            mask[i] = !isempty(t) && lowercase(t) != "missing"
        end
    end
    return mask
end

"""
    subset(table, idx) -> SampleTable

Rows at integer positions, or rows where a `Bool` / `BitVector` mask is true.
`specs` are shared with the parent (they do not depend on the row set).
"""
function subset(table::SampleTable, idx::AbstractVector)
    n = nsamples(table)
    ii = if idx isa AbstractVector{Bool}
        length(idx) == n || throw(ArgumentError(
            "subset: mask length $(length(idx)) ≠ $n"))
        findall(idx)
    elseif idx isa AbstractVector{<:Integer}
        for i in idx
            1 <= i <= n || throw(ArgumentError(
                "subset: index $i is outside 1:$n"))
        end
        idx
    else
        throw(ArgumentError(
            "subset: index must be integer positions or a Bool mask, got $(typeof(idx))"))
    end
    values = Dict{Symbol,Vector{Float64}}(
        k => v[ii] for (k, v) in table.values)
    censored = Dict{Symbol,BitVector}(
        k => v[ii] for (k, v) in table.censored)
    classes = Dict{Symbol,Vector{String}}(
        k => v[ii] for (k, v) in table.classes)
    return SampleTable(table.x[ii], table.y[ii], table.z[ii],
                       table.hole[ii], table.group[ii],
                       values, censored, classes, table.specs)
end

function Base.show(io::IO, table::SampleTable)
    names = join((string(s.name) for s in table.specs), ", ")
    print(io, "SampleTable($(nsamples(table)) samples, ",
          length(table.specs), " properties: ", names, ")")
end
