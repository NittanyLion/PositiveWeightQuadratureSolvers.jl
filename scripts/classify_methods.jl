# Tag each banked-best rule with the method that found it, for the status
# window.  B  = full B_d closure (single-coord sign flips + transpositions
#               generate the group) with orbit-constant weights;
#          SC = symmetric but smaller group (e.g. permutations x central
#               inversion — the Stroud-type d5 p5 structure);
#          F  = free nodes.  Literature transcriptions overridden below.
using DelimitedFiles

# Bank and sidecar locations (packaged copy, 2026-09-13): the research tree read
# them from the script's own directory; here they come from the environment, with
# the working directory as the default.
const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
function classify(path, d)
    A = readdlm(path, ','); X = A[:, 1:d]; w = A[:, d+1]
    key(v) = ntuple(j -> round(v[j], digits=8) + 0.0, d)
    S = Set(key(X[i, :]) for i in 1:size(X, 1))
    inS(v) = key(v) in S
    perm_closed = true; flip_closed = true
    for i in 1:size(X, 1)
        x = X[i, :]
        for j in 1:d
            y = copy(x); y[j] = -y[j]
            flip_closed &= inS(y)
        end
        for j in 1:d-1
            y = copy(x); y[j], y[j+1] = y[j+1], y[j]
            perm_closed &= inS(y)
        end
        # no early F return: icosahedral rules fail BOTH closures, and their
        # check lives after this loop
        flip_closed || perm_closed || break
    end
    # d=4 exceptional groups, tested BEFORE the B branch: B_4 IS a
    # subgroup of F_4, so an F_4 rule passes the B test and would otherwise be
    # filed as an ordinary B_4 find, hiding exactly the result we are hunting.
    # Each test is invariance under one quaternion left-multiplication that
    # lies in the group but not in B_4 (added 2026-08-13 with PolytopeDQ4.jl).
    if d == 4
        φ4 = (1 + sqrt(5)) / 2
        qinv(l, X, w) = begin
            n = size(X, 1)
            key(v) = ntuple(j -> round(v[j], digits = 7) + 0.0, 4)
            SS = Set(key(X[i, :]) for i in 1:n)
            for i in 1:n
                a = l; b = X[i, :]
                q = [a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4],
                     a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3],
                     a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2],
                     a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1]]
                key(q) in SS || return false
            end
            true
        end
        qinv([0.0, 1/φ4, 1.0, φ4] ./ 2, X, w) && return "H4"   # icosian: not in B_4
        qinv([0.5, 0.5, 0.5, 0.5], X, w)      && return "F4"   # 24-cell: not in B_4
    end
    if flip_closed && perm_closed
        orb = Dict{NTuple{d,Float64},Vector{Float64}}()
        for i in 1:size(X, 1)
            push!(get!(orb, key(sort(abs.(X[i, :]))), Float64[]), w[i])
        end
        all(maximum(v) - minimum(v) <= 1e-9 * maximum(v) for v in values(orb)) &&
            return "B"
    end
    # icosahedral (d=3 only): an order-5 rotation axis is impossible in B_3,
    # so finding one is definitive.  Tag I so ico-ansatz finds read correctly
    # in the window (added 2026-08-13 with IcosahedralDQ.jl).
    d == 3 && has5fold(X, w) && return "I"
    # some symmetry survives (permutations and/or global inversion)?
    (perm_closed && all(inS(-X[i, :]) for i in 1:size(X, 1))) ? "SC" : "F"
end
using LinearAlgebra
function has5fold(X, w; tol = 1e-7)
    n = size(X, 1)
    for c in 1:n
        norm(X[c, :]) < tol && continue
        a = X[c, :] / norm(X[c, :])
        K = [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
        R = I(3) + sin(2π/5) * K + (1 - cos(2π/5)) * K * K
        ok = true
        for i in 1:n
            q = R * X[i, :]
            any(j -> norm(X[j, :] - q) < tol && abs(w[j] - w[i]) < tol, 1:n) ||
                (ok = false; break)
        end
        ok && return true
    end
    return false
end
LIT = Dict("hermite_d2_p9_n18.csv" => "L",   # Haegemans–Piessens 1977 (PROVENANCE.md)
           "hermite_d3_p5_n13.csv" => "L",   # Stroud 1971 closed form (PROVENANCE.md)
           # Haegemans–Piessens 1976 Table 6 (PROVENANCE.md): hexagonal C_6
           # orbits, so the classifier above finds no B_2 / S_2×Z_2 structure
           # and would tag it F.
           "hermite_d2_p11_n25.csv" => "L",
           # Cools–Haegemans 1988 Table II (PROVENANCE.md)
           "hermite_d2_p13_n34.csv" => "L",
           # Stroud–Secrest 1963 E_5^{r^2}:5-1 option 2.  Structurally this IS
           # an S_5 × Z_2 rule, so the classifier above calls it SC; it is
           # literature, and our SC solver's own 32-node solution turned out to
           # be the option-1 branch of the same published formula (matched to
           # 4.9e-14 — PROVENANCE.md), so neither branch is ours.
           "hermite_d5_p5_n32.csv" => "L",
           # Festa & Sommariva 2012 (rules/literature.tsv, 2026-09-10): p25 transcribed;
           # p13/p15 are what our spectral search banked, but they match SMR13/SMR15
           # node for node, so the rules are theirs.
           "legendre_d2_p13_n33.csv" => "L",
           "legendre_d2_p15_n43.csv" => "L",
           "legendre_d2_p25_n113.csv" => "L")
# Tensor-product fills (tensor_fill.jl, 2026-09-05) are listed in
# symq/tensor_rules.tsv; structurally they pass the flip test but not the
# permutation test and would read F, which is not what they are.
TENSOR = Set{String}()
let f = joinpath(SYMQ, "tensor_rules.tsv")
    isfile(f) && for ln in eachline(f)
        push!(TENSOR, first(split(ln, '\t')))
    end
end
function main()
    best = Dict{Tuple{Int,Int},Tuple{Int,String}}()
    for f in readdir(RULES)
        m = match(r"^hermite_d(\d)_p(\d+)_n(\d+)\.csv$", f)
        m === nothing && continue
        d, p, n = parse.(Int, m.captures)
        (!haskey(best, (d, p)) || n < best[(d, p)][1]) && (best[(d, p)] = (n, f))
    end
    open(joinpath(SYMQ, "method_cache.tsv"), "w") do io
        for ((d, p), (n, f)) in sort(collect(best))
            println(io, "$f\t$(f ∈ TENSOR ? "T" : get(LIT, f, classify(joinpath(RULES, f), d)))")
        end
    end
    nothing
end
main()
