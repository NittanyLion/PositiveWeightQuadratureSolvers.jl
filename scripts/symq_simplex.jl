# Simplex-symmetry (S_{d+1} × Z_2) orbit search for one (d, p): random
# restarts + greedy reduction, banking any verified improvement over the
# current best rule file.  The simplex counterpart of symq_sc.jl -- same
# campaign contract (segments, donors via Dropbox, shared rule bank), separate
# state namespace (abest_*.txt) because the states live in the simplex frame.
#
# usage: julia symq_simplex.jl <d> <p> <minutes> [seed] [explore]
#
# Progress:  julia/symq/simplex_d{d}p{p}_s{seed}.log
#            julia/symq/abest_d{d}_p{p}.txt
#            julia/symq/prog_simplex_d{d}p{p}_s{seed}.txt
#
# Test hooks (all default to the campaign locations): SYMQ_RULES_DIR (rule
# bank), SYMQ_SIDECAR_DIR (logs, prog and abest files), SYMQ_DONORS_DIR
# (Dropbox donor exchange).  Point all three at a scratch directory to run
# without touching the live bank.
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "SimplexDQ.jl"))
using .DesignedQuadrature, .SimplexDQ
using Random, Printf, LinearAlgebra

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

BLAS.set_num_threads(1)

const d       = parse(Int, ARGS[1])
const p       = parse(Int, ARGS[2])
const minutes = parse(Float64, ARGS[3])
const seed    = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 20260827
const explore = any(a -> a == "explore", ARGS[5:end])

const J = get(ENV, "SYMQ_ROOT", pwd())
const RULES  = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ   = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const DONORS = get(ENV, "SYMQ_DONORS_DIR",
                   joinpath(get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms")), "donors"))
mkpath(RULES); mkpath(SYMQ); mkpath(DONORS)

function freshest(name)
    # "freshest" = the BEST copy: fewest nodes per the header, mtime only as a
    # tie-break (see symq_sc.jl, 2026-08-26).
    cands = [f for f in (joinpath(SYMQ, name), joinpath(DONORS, name)) if isfile(f)]
    isempty(cands) && return nothing
    function nodes_of(f)
        m = match(r"nodes=(\d+)", readline(f))
        m === nothing ? typemax(Int) : parse(Int, m[1])
    end
    return cands[argmin([(nodes_of(f), -mtime(f)) for f in cands])]
end

function current_best()
    best = typemax(Int)
    for f in readdir(RULES)
        m = match(Regex("^hermite_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        best = min(best, parse(Int, m[1]))
    end
    return best
end

logio = open(joinpath(SYMQ, "simplex_d$(d)p$(p)_s$(seed).log"), "a")
prog  = joinpath(SYMQ, "prog_simplex_d$(d)p$(p)_s$(seed).txt")

# collapse coincident nodes (degenerate configurations: a value at 0, two
# values equal, an orbit at the origin); weights add
function dedupe(nodes, w)
    key(i) = ntuple(j -> round(nodes[i, j]; digits = 12) + 0.0, size(nodes, 2))
    seen = Dict{NTuple{size(nodes,2),Float64},Int}()
    keep = Int[]; wo = Float64[]
    for i in axes(nodes, 1)
        k = key(i)
        if haskey(seen, k)
            wo[seen[k]] += w[i]
        else
            push!(keep, i); push!(wo, w[i])
            seen[k] = length(keep)
        end
    end
    return nodes[keep, :], wo
end

# relative gate -- see symq_run.jl / verify_exactness for the rationale
verify(nodes, w) = verify_exactness(nodes, w, p; basis = :hermite, relative = true)

best0 = current_best()
beststr() = best0 == typemax(Int) ? "none" : string(best0)

function bank(st, δ, nodes, w, ex)
    nodes, w = dedupe(nodes, w)
    n = length(w)
    # re-verify after any collapse; the collapsed rule is what gets banked
    ex2 = verify(nodes, w)
    if ex2 > 1e-11 || minimum(w) ≤ 0
        println(logio, "collapsed rule failed re-verify ($ex2, min w $(minimum(w))) — not banked")
        return
    end
    if n ≥ best0
        println(logio, "collapsed rule has $n nodes ≥ best $(beststr()) — not banked")
        return
    end
    path = joinpath(RULES, "hermite_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    # no parent rule: simplex-orbit search from restarts — do not invent a lineage
    symq_lineage!(basename(path), "none",
                  "simplex-orbit search from restarts (symq_simplex.jl seed $seed)";
                  rules = RULES)
    save_state(joinpath(SYMQ, "abest_d$(d)_p$(p).txt"), st, δ)
    try   # publish to the other machines via Dropbox
        cp(joinpath(SYMQ, "abest_d$(d)_p$(p).txt"),
           joinpath(DONORS, "abest_d$(d)_p$(p).txt"); force = true)
    catch e
        println(logio, "donor publish failed: ", sprint(showerror, e))
    end
    write(prog, "d=$d p=$p SX  best $n (was $(beststr()))  exact $(round(ex2, sigdigits=2))\n")
    println(logio, "BANKED $path  (exactness $ex2)  $(describe(st))")
    flush(logio)
end

function report(restart, best_run, banked)
    b = banked < best0 ? "banked $banked" : "none banked"
    write(prog, "d=$d p=$p SX  start $(beststr()), run-best $best_run, $b  (r$restart)\n")
end

rng = MersenneTwister(seed)
deadline = time() + minutes * 60
seg = 0
while time() < deadline - 30
    global seg += 1
    global best0 = current_best()
    seg_min = min(45.0, (deadline - time()) / 60)
    println(logio, "=== SX $(d)/$(p) seg $seg ($(round(seg_min, digits=1))min) seed=$seed  current best $(beststr()) ===")
    flush(logio)
    warmfile = freshest("abest_d$(d)_p$(p).txt")
    warm = warmfile === nothing ? nothing : load_state(warmfile, d, p)
    templates = SimplexDQ.MixState[]
    for (dd, pp) in ((d, p - 2), (d - 1, p), (d, p + 2))
        dd ≥ 2 || continue
        f = freshest("abest_d$(dd)_p$(pp).txt")
        f === nothing || push!(templates, load_state(f, dd, pp))
    end
    best = orbit_search(d, p; seconds = seg_min * 60, rng,
                        best_nodes = best0, verify, on_improve = bank,
                        warm, templates, log_io = logio, progress = report,
                        template_prob = explore ? 0.25 : 0.45,
                        hops_budget = explore ? 5 : 3,
                        factor_range = explore ? (1.3, 3.0) : (1.6, 2.4),
                        extol = 1e-11)  # relative gate (verify above)
    println(logio, "=== seg $seg done: best $best (segment start $best0) ===")
    flush(logio)
end
write(prog, "d=$d p=$p SX  best $(current_best())  (done)\n")
close(logio)
