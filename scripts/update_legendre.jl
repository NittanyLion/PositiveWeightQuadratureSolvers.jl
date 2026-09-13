# Keep the Legendre (uniform-weight) rule bank in step with the Hermite bank.
#
# The campaign searches only the Gaussian weight; the legendre_* rules froze on
# 2026-08-04 when the symq campaign took over the machines.  This brings one
# case at a time up to the Hermite bank's node count, reusing the V2 solver's
# node_elimination with basis = :legendre.
#
#   julia update_legendre.jl --list          print stale cases as "d p m_start cost",
#                                            cheapest first (cost-capped)
#   julia update_legendre.jl <d> <p>         work one case; banks every verified
#                                            improvement into julia/rules/
#
# Designed to run under `timeout`: every converged size is staged to disk the
# moment it exists, so a killed run keeps its progress.  Staged rules are
# verified with the campaign's banking gate (relative monomial exactness
# ≤ EXTOL = 1e-11 — the comment said 1e-8 until 2026-09-02, but the constant
# has matched the Gaussian bank's gate since 2026-08-24 — positive weights,
# unit mass) and only then promoted into rules/ —
# and only at node counts the bank does not already have (add-only, a better
# rule is a new filename).
#
# Cases whose estimated LM cost n_terms × n exceeds COST_CAP are skipped by
# --list: at today's bank that keeps d=2..4 low/mid-p and drops the d=5
# monsters, which no nightly slot could finish anyway.

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

# One core per worker, like symq_run.jl and spectral_start.jl.  Without this
# each worker spins up a full OpenBLAS pool: measured 2026-09-09 at home, five
# unpinned workers ran at 650-770% CPU each (~35 of 64 cores) and starved
# every single-threaded worker on the box to ~40% of a core, Tctl 83°C.
BLAS.set_num_threads(1)

const J     = get(ENV, "SYMQ_ROOT", pwd())
# Test hooks, matching symq_lag.jl / symq_ladder.jl / symq_freeelim.jl
# (added 2026-09-05 so the retargeted cells can be exercised off the bank).
const RULES = get(ENV, "SYMQ_RULES_DIR", joinpath(J, "rules"))
const STAGE = get(ENV, "SYMQ_STAGE_DIR",
                  joinpath(get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq")), "legendre_stage"))
const COST_CAP = 5.0e7
const EXTOL    = 1e-11         # the Gaussian bank's gate (was 1e-8; every
                               # banked uniform rule measures ≤ 6e-13, 2026-08-29)

# Cases the symmetric pair ansatz provably or measurably cannot bring to the
# Hermite count, skipped so they stop eating the nightly budget (2026-08-29:
# the cheapest-first list began with these every night — p=1 wants a single
# center node, which ±pairs cannot express; d3 p5 (13) and d3 p9 (45) are
# icosahedral records — and the 45-min per-case cap on each meant the run
# never reached d3 p11, the first genuine gap).
const SKIP = Set([(3, 1), (4, 1), (5, 1), (3, 5), (3, 9)])

# Pair-ansatz floor in PAIRS (moller.jl:36-43, METHODS §3): a symmetric ±pair
# rule carries d+1 free parameters per pair against the even-degree conditions,
# so below n_even/(d+1) pairs that system is overdetermined and LM provably
# cannot converge.  A cell already within FLOOR_SLACK pairs of the floor has
# essentially nothing left for THIS solver to win — 2026-09-05 measured 33
# pair-elimination slots (office + Roar + OSPool) pointed at 11 such cells,
# holding 13 pairs of headroom between them, while d4 p21 and d5 p15/17/19
# (10,820 pairs of headroom) had no slot on any 24/7 machine.
pair_floor(d::Int, p::Int) =
    cld(sum(iseven(q) ? binomial(q + d - 1, d - 1) : 0 for q in 0:p), d + 1)
const FLOOR_SLACK = 4

best_of(prefix) = begin
    b = Dict{Tuple{Int,Int},Int}()
    for f in readdir(RULES)
        m = match(Regex("^$(prefix)_d(\\d+)_p(\\d+)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        d, p, n = parse.(Int, m.captures)
        b[(d, p)] = min(get(b, (d, p), typemax(Int)), n)
    end
    b
end

function targets()
    h, l = best_of("hermite"), best_of("legendre")
    t = Tuple{Int,Int,Int,Float64,Bool}[]   # (d, p, target_n, cost, gap?)
    for ((d, p), nh) in h
        d == 1 && continue
        (d, p) in SKIP && continue
        nl = get(l, (d, p), typemax(Int))
        nl <= nh && continue           # up to date (or legendre already better)
        # the pair solver is spent here — don't offer it as a nightly target
        nl < typemax(Int) && cld(nl, 2) - pair_floor(d, p) <= FLOOR_SLACK && continue
        cost = Float64(binomial(p + d, d)) * nh
        cost > COST_CAP && continue
        push!(t, (d, p, nh, cost, nl == typemax(Int)))
    end
    # genuine gaps (no uniform rule at all) first, then catch-ups; cheapest
    # first within each group
    sort!(t; by = x -> (!x[5], x[4]))
    [(d, p, nh, cost) for (d, p, nh, cost, _) in t]
end

if !isempty(ARGS) && ARGS[1] == "--list"
    for (d, p, nh, cost) in targets()
        @printf("%d %d %d %.3g\n", d, p, nh, cost)
    end
    exit(0)
end

length(ARGS) >= 2 || (println("usage: update_legendre.jl --list | <d> <p>"); exit(1))
const d = parse(Int, ARGS[1])
const p = parse(Int, ARGS[2])
# --promote-only (2026-09-08 22:40): promote this cell's staged leftovers and
# exit (Roar: the 47 h `timeout` kills every big-cell run before the end-of-run
# promotion and the harvest ships rules/ but not the stage;
# cluster/roar_activate_promote_stage.sh runs this at a harvest).
const PROMOTE_ONLY = "--promote-only" ∈ ARGS

# Promote verified staged improvements (strictly better than n_cut, what the
# bank holds).  A FUNCTION since 2026-09-08 22:05, called at the START of every
# run as well as at the end: this block used to run only at the end, and on the
# big cells the run never gets there — symq_stop.sh's 08:00 SIGTERM (and any
# relaunch) kills it mid-rung, so the rules node_elimination had staged sat in
# symq/legendre_stage unpromoted.  Found holding d5 p15 n=2660 (bank 3802),
# d4 p21 n=2754 (bank 2774) and d4 p23 n=3882 (bank 7344, the tensor fill), all
# three passing the gate — i.e. the window's "discontinuities" at d4 q=12 and
# d5 q=8 were partly this bug (METHODS §23.5).
# Lineage of what this run banks (2026-09-11).  The cold start ladder has no
# parent rule; the fallback warm start further down descends from the banked
# incumbent and sets this to it.
const LIN_PARENT = Ref("none")

function promote_staged(n_cut; parent = "none",
                       how = "pair elimination (update_legendre.jl)")
    promoted = 0
    isdir(STAGE) || return 0
    for f in sort(readdir(STAGE))
        m = match(Regex("^legendre_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        n = parse(Int, m.captures[1])
        src = joinpath(STAGE, f)
        if n >= n_cut                        # not better than what the bank holds
            rm(src; force = true); continue
        end
        nodes, w = load_rule(src)
        ex = verify_exactness(nodes, w, p; basis = :legendre, relative = true)
        ok = ex <= EXTOL && minimum(w) > 0 && abs(sum(w) - 1) <= 1e-10
        if ok && !isfile(joinpath(RULES, f))
            mv(src, joinpath(RULES, f); force = false)
            symq_lineage!(f, parent, how; rules = RULES)
            @printf("banked %s  (relative exactness %.2e)\n", f, ex)
            promoted += 1
        else
            ok || @printf("REJECTED %s  (exact=%.2e, min w=%.2e, |Σw-1|=%.1e)\n",
                          f, ex, minimum(w), abs(sum(w) - 1))
            rm(src; force = true)
        end
    end
    return promoted
end

# leftovers of a killed run go into the bank BEFORE the incumbent is read
let k = promote_staged(get(best_of("legendre"), (d, p), typemax(Int));
                       how = "promoted from symq/legendre_stage, staged by an earlier run of this tool (update_legendre.jl)")
    k > 0 && println("promoted $k staged rule(s) left by an earlier run"); flush(stdout)
    PROMOTE_ONLY && (println("promote-only: done, $k rule(s) banked for legendre d=$d p=$p"); exit(0))
end

h, l = best_of("hermite"), best_of("legendre")
haskey(h, (d, p)) || (println("no hermite rule for d=$d p=$p — nothing to match"); exit(0))
const n_target = h[(d, p)]
const n_have   = get(l, (d, p), typemax(Int))
# 2026-09-09: the hermite count is a target only when it is a DESIGNED count.
# In the extension cells (d2 p37–77, d3 p27–45) both banks hold nothing but
# their tensor warm starts, of identical size, and "already at n ≤ hermite"
# made every pairleg seat there exit in 3 s (probe, update_legendre.jl 3 45).
# When hermite is itself at the grid, anything smaller than what we have is
# progress: fall through, and the ladder below anchors on the pair floor and
# caps at m_have − 1 as it always did.
const n_grid = ((p + 2) ÷ 2)^d
n_have <= n_target && n_target < n_grid &&
    (println("legendre d=$d p=$p already at n=$n_have ≤ hermite $n_target"); exit(0))

# Optional seed, so several slots on one cell are not bit-identical duplicates
# (the ladder used a fixed Xoshiro(10 + k) until 2026-09-05).
# --promote-only (2026-09-08 22:40): promote this cell's staged leftovers and
# exit — for Roar, where the 47 h `timeout` kills every big-cell run before
# the end-of-run promotion and the harvest ships rules/ but not the stage
# (cluster/roar_activate_promote_stage.sh runs this at a harvest).
const SEED = length(ARGS) >= 3 && ARGS[3] != "--promote-only" ? parse(Int, ARGS[3]) : 10

const m_floor = pair_floor(d, p)
const m_have  = n_have == typemax(Int) ? typemax(Int) : cld(n_have, 2)

# The pair ansatz is spent once the bank sits at the floor: stop burning the
# slot re-deriving what is already banked.  (Measured 2026-09-05: d4 p17 banked
# nothing twice a night for 5,144 s + 5,245 s; d5 p13 spent 18,340 s to
# re-derive its own bank entry.)
if m_have != typemax(Int) && m_have - m_floor <= FLOOR_SLACK
    println("legendre d=$d p=$p: n=$n_have is $(2 * (m_have - m_floor)) nodes above the " *
            "pair floor $(2 * m_floor) — pair ansatz exhausted, nothing for this solver")
    exit(0)
end

mkpath(STAGE)
prefix = joinpath(STAGE, "legendre_d$(d)_p$(p)")

# Start-size ladder, ANCHORED ON THE PAIR FLOOR (2026-09-05).  It used to be
# anchored on the bank — `m_start = cld(n_have,2) + 1`, i.e. one pair ABOVE the
# incumbent — so the night was arithmetically capped at a single drop and
# measured zero: d4 p21 spent 10,809 s and 11,569 s on consecutive nights for
# no net progress, and d4 p19's ladder climbed to 1938 nodes against a bank of
# 1612 and spent the night walking back down.  Anchoring on the floor puts the
# first solve BELOW the incumbent, where elimination has somewhere to go; every
# rung is capped at m_have - 1 so no rung can start at or above the bank.
const LADDER = (1.20, 1.45, 1.75, 2.10)
rungs = Int[]
for fac in LADDER
    m = ceil(Int, m_floor * fac)
    m_have == typemax(Int) || (m = min(m, m_have - 1))
    m > m_floor && (isempty(rungs) || m != rungs[end]) && push!(rungs, m)
end
isempty(rungs) && push!(rungs, max(m_floor + 1, m_have == typemax(Int) ? m_floor + 1 : m_have - 1))

println("legendre d=$d p=$p: have $(n_have == typemax(Int) ? "nothing" : "n=$n_have"), " *
        "hermite best n=$n_target, pair floor $(2 * m_floor) nodes ($m_floor pairs), " *
        "start ladder $(rungs) pairs, seed $SEED")
flush(stdout)   # this worker is designed to run under `timeout`; an unflushed
                # header means a killed run loses its own configuration line

solved = false
for (k, m) in enumerate(rungs)
    r = node_elimination(d, p; basis = :legendre, symmetric = true,
                         m_start = m, save_prefix = prefix,
                         tag = "legendre d$d p$p", rng = Xoshiro(SEED + k))
    if r !== nothing
        global solved = true
        break
    end
    println("start m=$m did not converge — climbing the ladder"); flush(stdout)
end

# Fallback: every cold rung stalled, so warm start the first solve from the
# banked rule minus its lightest pair.  `node_elimination` has accepted
# `init_pairs` since 2026-08-29 (DesignedQuadratureV2.jl:489) and its own inner
# loop warm starts every subsequent step this way — only the FIRST solve was
# throwing the incumbent away.
if !solved && m_have != typemax(Int)
    bankfile = joinpath(RULES, "legendre_d$(d)_p$(p)_n$(n_have).csv")
    try
        X, wv = load_rule(bankfile)
        b0, w0 = pairs_from_rule(2 .* X .- 1, wv)      # bank is [0,1], solver [-1,1]
        keep = sortperm(w0)[2:end]                     # drop the lightest pair
        println("cold ladder exhausted — warm starting from the bank at m=$(length(keep)) pairs")
        LIN_PARENT[] = basename(bankfile)     # what this run's rules descend from
        r = node_elimination(d, p; basis = :legendre, symmetric = true,
                             m_start = length(keep), init_pairs = (b0[keep, :], w0[keep]),
                             save_prefix = prefix, tag = "legendre d$d p$p warm",
                             rng = Xoshiro(SEED + 100))
        r === nothing && println("warm start did not converge either")
    catch e
        println("warm start unavailable: ", sprint(showerror, e))
    end
end

how2 = LIN_PARENT[] == "none" ?
    "pair elimination from the cold start ladder (update_legendre.jl seed $SEED)" :
    "pair elimination warm started from the banked rule minus its lightest pair (update_legendre.jl seed $SEED)"
println("done: $(promote_staged(n_have; parent = LIN_PARENT[], how = how2)) rule(s) banked for legendre d=$d p=$p")
