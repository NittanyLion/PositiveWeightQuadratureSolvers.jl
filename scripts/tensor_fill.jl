#=
tensor_fill.jl -- fill blank (d, p) cells, and cells that tensoring beats,
with the best PROPER tensor product of banked lower-dimensional rules
(2026-09-05, IDEAS_future.md §11).

The status window's ⊗ column already treats "the cheapest product of banked
lower-dim rules of the same degree" as the baseline every cell must beat; a
blank cell is one where the bank holds NOTHING while that baseline is a
perfectly valid positive-weight exact rule.  This banks the baseline itself:

  * a blank cell gets its product rule (⊗ reads 1.00 — honest: no better
    than tensoring), so it stops being blank, and the elimination workers
    (pairleg / free-elim / ladder) get a valid warm start instead of a
    cold one;
  * a cell whose banked count exceeds the product (⊗ ≥ 1, e.g. Le d5 p19
    20320 vs 1612 × 10 = 16120) is replaced by the product.

The product of centrally symmetric rules is centrally symmetric, so the
GH/Le fills are valid ±pair starts for update_legendre.jl and the V2 tools;
the La fills are plain Gauss–Laguerre grids (an S_d orbit start is what
symq_lag_tensor.jl builds itself).  1-D factors are the m = ⌈(p+1)/2⌉-point
Gauss rule of the family (Golub–Welsch).  Every product is re-verified at
the campaign gate (relative monomial exactness ≤ 1e-11, positive weights,
unit mass) before it is written.  Files above --cap nodes are reported, not
written (a 161,051-node La d5 p21 grid is a 19 MB CSV on every synced
machine; leave those to the orbit-space descent).

Banked fills are listed in symq/tensor_rules.tsv so classify_methods.jl can
tag them T in the status window.

usage: julia tensor_fill.jl [hermite|legendre|laguerre ...] [--cap=N] [--dry]
=#
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
using .DesignedQuadrature
using LinearAlgebra, Printf

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const GRID  = Dict(2 => 1:2:35, 3 => 1:2:25, 4 => 1:2:23, 5 => 1:2:21)
# 2026-09-09 (user): GH and Le run out to the Diallo–Worku ceilings at d = 2, 3
# (square degree 77, cube degree 45) — see symq_policy.jl and
# litcheck/ARXIV_SWEEP_2026-09-09.md.  The new cells are blank, so they need
# their tensor warm start here before any eliminator can descend on them.
const GRID_FAM = Dict(("hermite", 2)  => 1:2:77, ("hermite", 3)  => 1:2:45,
                      ("legendre", 2) => 1:2:77, ("legendre", 3) => 1:2:45)
gridof(fam, d) = get(GRID_FAM, (fam, d), GRID[d])
const EXTOL = 1e-11

fams = [a for a in ARGS if a ∈ ("hermite", "legendre", "laguerre")]
isempty(fams) && (fams = ["hermite", "legendre", "laguerre"])
const CAP = something(findfirst(a -> startswith(a, "--cap="), ARGS) |>
                      (i -> i ≡ nothing ? nothing : parse(Int, ARGS[i][7:end])), 40_000)
const DRY = "--dry" ∈ ARGS
const BASIS = Dict("hermite" => :hermite, "legendre" => :legendre, "laguerre" => :laguerre)

# 1-D Gauss rule of the family, weights summing to one, in the bank's frame
# (Hermite on ℝ, Legendre on [0,1], Laguerre on [0,∞))
function gauss1(fam::String, m::Int)
    m == 1 && return (fam == "hermite" ? [0.0] : fam == "legendre" ? [0.5] : [1.0]), [1.0]
    J1 = fam == "hermite"  ? SymTridiagonal(zeros(m), [sqrt(k) for k in 1.0:m-1]) :
         fam == "legendre" ? SymTridiagonal(zeros(m), [k / sqrt(4k^2 - 1) for k in 1.0:m-1]) :
                             SymTridiagonal([2k + 1.0 for k in 0:m-1], [Float64(k) for k in 1:m-1])
    E = eigen(J1)
    x = E.values; w = vec(E.vectors[1, :]) .^ 2
    fam == "legendre" && (x = (x .+ 1) ./ 2)
    return x, w
end

function bank(fam)
    b = Dict{Tuple{Int,Int},Int}()
    for f in readdir(RULES)
        m = match(Regex("^$(fam)_d(\\d)_p(\\d+)_n(\\d+)\\.csv\$"), f)
        m ≡ nothing && continue
        k = (parse(Int, m[1]), parse(Int, m[2]))
        b[k] = min(get(b, k, typemax(Int)), parse(Int, m[3]))
    end
    return b
end

function load_rule(fam, d, p, n)
    A = zeros(n, d + 1); i = 0
    for ln in eachline(joinpath(RULES, "$(fam)_d$(d)_p$(p)_n$(n).csv"))
        s = strip(ln); (isempty(s) || startswith(s, "#")) && continue
        i += 1; A[i, :] = parse.(Float64, split(s, ','))
    end
    i == n || error("$(fam)_d$(d)_p$(p)_n$(n).csv: expected $n rows, read $i")
    return A[:, 1:d], A[:, d+1]
end

# factor(d, p): the cheapest rule for (d, p) counting bank AND products
# (recursive, same T(d) as status_symq.sh's ⊗ column); returns (count, how)
function factors(fam, b, p, dmax)
    T = Dict{Int,Tuple{Int,Any}}()
    T[1] = (cld(p + 1, 2), :gauss)
    for dd in 2:dmax
        best = (typemax(Int), nothing)
        for d1 in 1:dd-1
            c = T[d1][1] * T[dd-d1][1]
            c < best[1] && (best = (c, (d1, dd - d1)))
        end
        if haskey(b, (dd, p)) && b[(dd, p)] < best[1]
            T[dd] = (b[(dd, p)], :bank)
        else
            T[dd] = best
        end
    end
    return T
end

# materialize the rule T describes for dimension dd (nodes, weights)
function materialize(fam, b, p, T, dd)
    how = T[dd][2]
    if how ≡ :gauss
        x, w = gauss1(fam, cld(p + 1, 2))
        return reshape(x, :, 1), w
    end
    how ≡ :bank && return load_rule(fam, dd, p, b[(dd, p)])
    d1, d2 = how
    X1, w1 = materialize(fam, b, p, T, d1)
    X2, w2 = materialize(fam, b, p, T, d2)
    n1, n2 = length(w1), length(w2)
    X = zeros(n1 * n2, d1 + d2); w = zeros(n1 * n2)
    k = 0
    for i in 1:n1, j in 1:n2
        k += 1
        X[k, 1:d1] = X1[i, :]; X[k, d1+1:end] = X2[j, :]
        w[k] = w1[i] * w2[j]
    end
    return X, w
end

describe(T, dd) = (h = T[dd][2]; h ≡ :gauss ? "G$(T[1][1])" : h ≡ :bank ? "bank$(T[dd][1])" :
                   "(" * describe(T, h[1]) * "×" * describe(T, h[2]) * ")")

ledger = joinpath(SYMQ, "tensor_rules.tsv")
written = String[]
for fam in fams
    b = bank(fam)
    for d in 2:5, p in gridof(fam, d)
        T = factors(fam, b, p, d)
        have = get(b, (d, p), typemax(Int))
        # best PROPER product (the ⊗ denominator): exclude the bank at level d
        prodc = minimum(T[d1][1] * T[d-d1][1] for d1 in 1:d-1)
        prodc < have || continue                 # bank already beats tensoring
        Tp = copy(T); Tp[d] = (prodc, argmin(d1 -> T[d1][1] * T[d-d1][1], 1:d-1) |> (d1 -> (d1, d - d1)))
        tag = have == typemax(Int) ? "blank" : "⊗=$(round(have / prodc, digits = 2))"
        if prodc > CAP
            @printf("%-8s d%d p%-3d %-10s product %7d = %s  -- above cap %d, not written\n",
                    fam, d, p, tag, prodc, describe(Tp, d), CAP)
            continue
        end
        X, w = materialize(fam, b, p, Tp, d)
        n = length(w)
        n == prodc || error("size mismatch $n vs $prodc")
        ex = verify_exactness(X, w, p; basis = BASIS[fam], relative = true)
        ok = ex ≤ EXTOL && minimum(w) > 0 && abs(sum(w) - 1) ≤ 1e-10
        @printf("%-8s d%d p%-3d %-10s product %7d = %-28s exact %.1e %s\n",
                fam, d, p, tag, n, describe(Tp, d), ex, ok ? (DRY ? "(dry)" : "BANKED") : "REJECTED")
        (ok && !DRY) || continue
        f = "$(fam)_d$(d)_p$(p)_n$(n).csv"
        open(joinpath(RULES, f), "w") do io
            for i in 1:n
                println(io, join(string.([X[i, :]; w[i]]), ","))
            end
        end
        # no parent RULE file: this is a product of lower-dimensional factors and
        # 1-D Gauss rules, and `how` records exactly which product it is
        symq_lineage!(f, "none",
                      "tensor fill: $(describe(Tp, d)) (tensor_fill.jl)"; rules = RULES)
        push!(written, "$f\t$(describe(Tp, d))")
    end
end
if !isempty(written)
    open(ledger, "a") do io
        for l in written; println(io, l); end
    end
end
println("$(length(written)) rule(s) written")
