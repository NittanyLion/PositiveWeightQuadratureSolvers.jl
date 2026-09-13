# Icosahedral-ansatz worker for one d=3 case: enumerate shell-type
# combinations cheaper than the banked best, in order of increasing node
# count, and multistart-fit the radii for each (weights come free by variable
# projection -- see IcosahedralDQ.jl).  Banks any verified improvement.
#
# usage: julia symq_ico.jl <p> <minutes> [seed]
#
# Same campaign contract as symq_run.jl: progress in symq/prog_ico_d3p{p}.txt,
# log in symq/ico_d3p{p}.log, rules banked to rules/hermite_d3_p{p}_n{n}.csv.
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "IcosahedralDQ.jl"))
using .DesignedQuadrature, .IcosahedralDQ
using LinearAlgebra, Random, Printf

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

BLAS.set_num_threads(1)

const p       = parse(Int, ARGS[1])
const minutes = parse(Float64, ARGS[2])
const seed    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const EXTOL = 1e-9                       # same relative gate as the campaign

logio = open(joinpath(SYMQ, "ico_d3p$(p).log"), "a")
prog  = joinpath(SYMQ, "prog_ico_d3p$(p).txt")
rng   = MersenneTwister(41 * seed + p)

function best_banked()
    best = typemax(Int)
    for f in readdir(RULES)
        m = match(Regex("^hermite_d3_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing || (best = min(best, parse(Int, m[1])))
    end
    return best
end

# all multisets of shell types {1, 12, 20, 30, 60} (origin at most once) whose
# node total lands in [moller, nmax], ordered by node count then shell count --
# fewer shells = fewer radii = easier fit, so try those first
function shell_plans(nmax::Int)
    plans = Tuple{Int,Vector{Int}}[]
    maxof(t) = t == 1 ? 1 : nmax ÷ t
    for n1 in 0:1, n12 in 0:maxof(12), n20 in 0:maxof(20), n30 in 0:maxof(30), n60 in 0:maxof(60)
        n = n1 + 12n12 + 20n20 + 30n30 + 60n60
        (0 < n <= nmax) || continue
        types = vcat(fill(1, n1), fill(12, n12), fill(20, n20), fill(30, n30), fill(60, n60))
        # enough shells to plausibly match the condition count: each shell
        # carries 2 dof (radius + weight), origin 1
        push!(plans, (n, types))
    end
    sort!(plans, by = q -> (q[1], length(q[2])))
    return plans
end

function bank(nodes, w, ex, n)
    path = joinpath(RULES, "hermite_d3_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    # no parent rule: an icosahedral shell ansatz solved from scratch
    symq_lineage!(basename(path), "none",
                  "icosahedral shell ansatz, solved from scratch (symq_ico.jl seed $seed)";
                  rules = RULES)
    println(logio, "BANKED $path (exactness $ex)"); flush(logio)
end

function main()
    deadline = time() + minutes * 60
    b0 = best_banked()
    b0 == typemax(Int) && (b0 = 10^6)
    println(logio, "=== ico d=3 p=$p below $b0 nodes, $(round(minutes))min, seed $seed ===")
    flush(logio)
    write(prog, "d=3 p=$p ICO  start $b0, run-best none, none banked (ico r0)\n")

    run_best = b0
    tried = 0
    for (n, types) in shell_plans(b0 - 1)
        time() > deadline && break
        tried += 1
        r = IcosahedralDQ.fit(p, types; tries = 40, rng = rng, deadline = deadline)
        tried % 10 == 0 && write(prog,
            "d=3 p=$p ICO  start $b0, run-best $(run_best < b0 ? run_best : "none"), none banked (ico r$tried)\n")
        r === nothing && continue
        nodes, _ = IcosahedralDQ.build_rule(types, r.radii)
        w = Float64[r.weights[k] for (k, t) in enumerate(types)
                    for _ in 1:(t == 1 ? 1 : IcosahedralDQ.orbit_size(
                        t isa Integer ? IcosahedralDQ.ICO_DIRS[t] : t))]
        ex = verify_exactness(nodes, w, p; basis = :hermite, relative = true)
        println(logio, "candidate n=$n types=$types resid=$(r.residual) verify=$ex")
        flush(logio)
        if ex <= EXTOL && minimum(w) > 0 && n < run_best
            bank(nodes, w, ex, n)
            run_best = n
            write(prog, "d=3 p=$p ICO  start $b0, run-best $n, banked $n (ico r$tried)\n")
        end
    end
    println(logio, "=== ico done: best $run_best (was $b0), $tried plans tried ===")
    write(prog, "d=3 p=$p ICO  start $b0, run-best $(run_best < b0 ? run_best : "none"), ico  (done)\n")
    close(logio)
end

main()
