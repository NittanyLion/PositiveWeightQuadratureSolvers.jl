# Pair / node elimination for a GENERIC weight from the banked best rule
# (2026-09-08, PROPOSAL_scale_mixtures.md §3.4).  Written for :elaplace — the
# elliptical Laplace, whose exact mixture rules (mixture_rule.jl) carry a
# factor m ≈ p/4 of slack over the GH count — but basis-agnostic: anything
# DesignedQuadrature.jl's poly_tables!/basis_target/verify_exactness know.
#
# usage: julia symq_pairelim.jl <d> <p> <minutes> [seed] [basis]   (basis default elaplace)
#
# Start = the smallest banked <basis>_d{d}_p{p}_n*.csv.  If it is centrally
# symmetric (±x pairs) the V2 solver runs in pair mode (half the unknowns,
# even moments only); otherwise node mode.  Drops the lightest pairs in
# batches, halving the batch on failure, re-converging from the warm start;
# EVERY verified improvement is banked at once (relative exactness ≤ 1e-11,
# w > 0), so a killed run keeps its progress and the next run resumes from
# the bank — stateless apart from the bank, like symq_freeelim.jl.
# Logs symq/pairelim_<basis>_d{d}p{p}.log; progress prog_pairelim_<basis>_…L.txt.
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "DesignedQuadratureV2.jl"))
using .DesignedQuadrature, .DesignedQuadratureV2
using LinearAlgebra, Random, Printf

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
const seed    = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 1
const BASIS   = length(ARGS) ≥ 5 ? Symbol(ARGS[5]) : :elaplace
const PFX     = String(BASIS)
const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const EXTOL = 1e-11
mkpath(RULES); mkpath(SYMQ)

function best_rule()
    best = typemax(Int); path = ""
    for f ∈ readdir(RULES)
        m = match(Regex("^$(PFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        n = parse(Int, m[1]); n < best && (best = n; path = joinpath(RULES, f))
    end
    best, path
end
verify(nodes, w) = verify_exactness(nodes, w, p; basis = BASIS, relative = true)
logio = open(joinpath(SYMQ, "pairelim_$(PFX)_d$(d)p$(p).log"), "a")
prog  = joinpath(SYMQ, "prog_pairelim_$(PFX)_d$(d)p$(p)L.txt")
say(x...) = (println(logio, x...); flush(logio))

n0, path = best_rule()
n0 == typemax(Int) && (say("no banked $PFX rule for d=$d p=$p — nothing to start from"); exit(0))
r0 = load_rule(path)
nodes0, w0 = r0.nodes, r0.weights
function start_pairs(nodes0, w0)
    try
        b, w = pairs_from_rule(nodes0, w0)
        return true, b, w
    catch e
        say("start is not ±x symmetric ($(sprint(showerror, e))) — node mode")
        return false, Matrix{Float64}(nodes0), Vector{Float64}(w0)
    end
end
const symmetric, b0, wp0 = start_pairs(nodes0, w0)
b = b0; w = wp0
m = length(w)
expand(b, w) = symmetric ? (vcat(b, .-b), vcat(w ./ 2, w ./ 2)) : (b, w)
say("=== PAIRELIM $PFX d=$d p=$p seed=$seed: start $n0 nodes = $m $(symmetric ? "pairs" : "nodes"), $(round(minutes, digits = 1)) min ===")

# Lineage chain (2026-09-11): elimination is a chain, so each banked rule's parent
# is the rule banked just before it, rooted at the start rule this run read.
const LIN_PARENT = Ref(basename(path))

function bank(b, w)
    nodes, wx = expand(b, w)
    n = length(wx)
    n < best_rule()[1] || return false
    ex = verify(nodes, wx)
    ok = ex ≤ EXTOL && minimum(wx) > 0 && abs(sum(wx) - 1) ≤ 1e-10
    ok || (say("  n=$n failed the gate (ex $ex, min w $(minimum(wx)), Σw−1 $(sum(wx)-1)) — not banked"); return false)
    out = joinpath(RULES, "$(PFX)_d$(d)_p$(p)_n$(n).csv")
    open(out, "w") do io
        for i ∈ 1:n; println(io, join(string.([nodes[i, :]; wx[i]]), ",")); end
    end
    symq_lineage!(basename(out), LIN_PARENT[],
                  "pair elimination: dropped pair(s), reconverged (symq_pairelim.jl seed $seed)";
                  rules = RULES)
    LIN_PARENT[] = basename(out)          # the next drop descends from this rule
    say("BANKED $out  (exactness $ex)")
    write(prog, "d=$d p=$p $(uppercase(PFX))  best $n (was $n0)  exact $(round(ex, sigdigits = 2))\n")
    true
end

deadline = time() + 60minutes
batch = max(1, m ÷ 20)          # the mixture start has a factor ≈ p/4 of slack: drop boldly
tries = 8
rng = MersenneTwister(97seed + 13d + p)
# a solve that is going nowhere is cut early (symq_freeelim.jl's numbers)
solve(bb, ww, mm) = designed_quadrature_v2(d, p, mm; basis = BASIS, symmetric,
                                           init_pairs = (bb, ww), tol = 1e-12,
                                           maxiter = 3000, stall_window = 150,
                                           abort_iters = 400, abort_resid = 1e-3)
while time() < deadline && m > 1
    order = sortperm(w)
    success = false
    while batch > 1 && m - batch ≥ 1
        keep = order[batch+1:end]
        ts = time()
        rt = solve(b[keep, :], w[keep], m - batch)
        if rt.converged
            global b, w = rt.pairs; global m -= batch
            say("m=$m converged (batch $batch, $(round(Int, time() - ts)) s)")
            bank(b, w); success = true; break
        else
            say("batch $batch → m=$(m - batch) failed ($(rt.status), ‖R‖=$(round(rt.residual, sigdigits = 3)), $(round(Int, time() - ts)) s) — halving")
            global batch ÷= 2
        end
        time() < deadline || break
    end
    success && continue
    time() < deadline || break
    # single drops: the `tries` lightest, then the same with a jittered start
    # (seeded, so a re-launch from the bank does not repeat the identical
    # failures), then give up
    cands = order[1:min(tries, m)]
    for (jit, c) ∈ Iterators.flatten((((0.0, c) for c ∈ cands), ((1e-3, c) for c ∈ cands)))
        keep = setdiff(1:m, c); ts = time()
        bb = b[keep, :]
        jit > 0 && (bb = bb .+ jit .* randn(rng, size(bb)...) .* max.(abs.(bb), 1.0))
        rt = solve(bb, w[keep], m - 1)
        if rt.converged
            global b, w = rt.pairs; global m -= 1
            say("m=$m converged (dropped #$c$(jit > 0 ? ", jittered" : ""), $(round(Int, time() - ts)) s)")
            bank(b, w); success = true; break
        else
            say("m=$(m - 1) attempt (drop #$c$(jit > 0 ? ", jittered" : "")) failed ($(rt.status), ‖R‖=$(round(rt.residual, sigdigits = 3)), $(round(Int, time() - ts)) s)")
        end
        time() < deadline || break
    end
    success || (say("no single drop converged from m=$m — done"); break)
end
say("=== end: $(best_rule()[1]) nodes banked for $PFX d=$d p=$p ===")
close(logio)
