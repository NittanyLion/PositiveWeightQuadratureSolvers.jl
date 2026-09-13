"""
    PositiveWeightQuadratureSolvers

Solvers that produce interior cubature rules with strictly positive weights for
the Gaussian weight N(0, I_d) on R^d ("GH") and the uniform weight on [0,1]^d
("Le"), together with the extended-precision polish and the verification gate
that decide whether a rule is banked.

The modules are the ones the research campaign ran, copied unchanged except for
path handling (see `CHANGES_FROM_RESEARCH_TREE.md`).  Each ansatz lives in its
own submodule, and several of them export the *same* names on purpose —
`OType`, `build_type`, `orbit_types`, `conditions`, `random_state`, `solve!`,
`eliminate!`, `orbit_search` — because they implement one interface for
different symmetry groups.  They are therefore NOT re-exported wholesale;
reach them as `PositiveWeightQuadratureSolvers.SymmetricDQ.orbit_search` and so
on, or through the disambiguated aliases below.

Re-exported directly (no clashes):

  * the moment system and the verification gate — `total_degree_indices`,
    `verify_exactness`, `basis_target`;
  * rule input/output — `load_rule`, `save_rule`, `load_pairs`, `save_pairs`;
  * the solvers — `find_rule`, `designed_quadrature`, `designed_quadrature_v2`,
    `node_elimination`, `degree_continuation`, `pairs_from_rule`;
  * the spectral start — `spectral_nodes`;
  * weight-to-weight node maps — `WeightTransforms`' exports.

Orbit-search entry points, disambiguated:

  * `b_orbit_search`         — B_d / D_d orbits (`SymmetricDQ`)
  * `sc_orbit_search`        — S_d × Z_2 orbits (`PermCentralDQ`)
  * `simplex_orbit_search`   — S_{d+1} × Z_2 orbits (`SimplexDQ`)
  * `laguerre_orbit_search`  — orthant orbits for the Gamma weight (`LaguerreDQ`)

Reading a deposited rule file (the Zenodo deposits carry a comment block and a
column header that `load_rule` does not skip): `load_deposit_rule` and
`verify_deposit`.
"""
module PositiveWeightQuadratureSolvers

# DesignedQuadratureV2.jl includes DesignedQuadrature.jl itself, at its own top
# level, so including V2 defines BOTH submodules here.  Including
# DesignedQuadrature.jl separately as well would only replace the module and
# warn, so it is deliberately not done.
include(joinpath(@__DIR__, "WeightTransforms.jl"))
include(joinpath(@__DIR__, "DesignedQuadratureV2.jl"))
include(joinpath(@__DIR__, "SpectralInit.jl"))

# Orbit ansätze.  SymmetricDQ, PermCentralDQ and LaguerreDQ each include their
# own nested copy of WeightTransforms — left exactly as in the research tree.
include(joinpath(@__DIR__, "SymmetricDQ.jl"))
include(joinpath(@__DIR__, "LegendreProductDQ.jl"))     # needs ..SymmetricDQ
include(joinpath(@__DIR__, "PermCentralDQ.jl"))
include(joinpath(@__DIR__, "SimplexDQ.jl"))
include(joinpath(@__DIR__, "IcosahedralDQ.jl"))
include(joinpath(@__DIR__, "OPSpaceDQ.jl"))
include(joinpath(@__DIR__, "LaguerreDQ.jl"))
include(joinpath(@__DIR__, "LaguerreProductDQ.jl"))     # needs ..LaguerreDQ
include(joinpath(@__DIR__, "LaguerrePolish.jl"))

using .WeightTransforms
using .DesignedQuadrature
using .DesignedQuadratureV2
using .SpectralInit

export WeightTransforms, DesignedQuadrature, DesignedQuadratureV2, SpectralInit,
       SymmetricDQ, LegendreProductDQ, PermCentralDQ, SimplexDQ, IcosahedralDQ,
       OPSpaceDQ, LaguerreDQ, LaguerreProductDQ, LaguerrePolish

# moment system, gate, io, solvers
export total_degree_indices, verify_exactness, basis_target, lhsdesign,
       elaplace_smoment,
       load_rule, save_rule, load_pairs, save_pairs,
       find_rule, designed_quadrature, designed_quadrature_v2,
       node_elimination, degree_continuation, pairs_from_rule,
       spectral_nodes

# node maps between weights (WeightTransforms)
export erf64, erfc64, normcdf, normccdf, norminv,
       gauss_to_unit, unit_to_gauss, gauss_to_exp, gauss_to_gamma,
       gauss_hermite_nodes, gauss_legendre_nodes, gauss_laguerre_nodes,
       node_map, monotone_interp

# ---------------------------------------------------------------------------
# Orbit-search entry points, one alias per symmetry group.  The submodules all
# call theirs `orbit_search`; these names let a caller pick one without
# qualifying every argument type.
# ---------------------------------------------------------------------------
const b_orbit_search        = SymmetricDQ.orbit_search
const sc_orbit_search       = PermCentralDQ.orbit_search
const simplex_orbit_search  = SimplexDQ.orbit_search
const laguerre_orbit_search = LaguerreDQ.orbit_search
export b_orbit_search, sc_orbit_search, simplex_orbit_search, laguerre_orbit_search

# ---------------------------------------------------------------------------
# Deposited rule files
# ---------------------------------------------------------------------------

"""
    load_deposit_rule(path) -> (nodes, weights)

Read a rule file as deposited on Zenodo (`publish/gh/rules/…`,
`publish/le/rules/…`): a block of `#` comment lines, one `x1,…,xd,w` column
header, then one node per line.  Returns an `n × d` node matrix and the length-`n`
weight vector, exactly as `load_rule` does for a bare bank file.

`load_rule` itself is left as it is in the research tree — it parses every line
as numbers, which is what the bank files are.
"""
function load_deposit_rule(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    for line ∈ eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        v = tryparse.(Float64, split(s, ','))
        any(x -> x ≡ nothing, v) && continue          # the x1,…,xd,w header row
        push!(rows, Vector{Float64}(v))
    end
    isempty(rows) && error("no numeric rows in $path")
    M = permutedims(reduce(hcat, rows))
    return (nodes = M[:, 1:end-1], weights = M[:, end])
end

"""
    verify_deposit(path; basis = :auto, p = nothing) -> (d, p, n, err)

Load a deposited rule file and report its worst RELATIVE monomial error over all
monomials of total degree ≤ p — the project's gate, whose threshold is 1.0e-11.

`basis` defaults to `:hermite` for a file whose name starts with `hermite`
(Gaussian weight, N(0, I_d), weights summing to 1) and `:legendre` for one
starting with `legendre` (uniform weight on [0,1]^d).  `p` defaults to the `_pP_`
field of the file name.
"""
function verify_deposit(path::AbstractString; basis::Symbol = :auto, p = nothing)
    name = basename(path)
    if basis ≡ :auto
        basis = startswith(name, "hermite")  ? :hermite  :
                startswith(name, "legendre") ? :legendre :
                startswith(name, "laguerre") ? :laguerre :
                error("cannot infer the basis from $name; pass basis = …")
    end
    if p ≡ nothing
        m = match(r"_p(\d+)_", name)
        m ≡ nothing && error("cannot infer p from $name; pass p = …")
        p = parse(Int, m.captures[1])
    end
    rule = load_deposit_rule(path)
    err = verify_exactness(rule.nodes, rule.weights, p; basis = basis, relative = true)
    return (d = size(rule.nodes, 2), p = p, n = length(rule.weights), err = err)
end

export load_deposit_rule, verify_deposit

end # module
