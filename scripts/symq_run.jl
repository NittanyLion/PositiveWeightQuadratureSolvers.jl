# Symmetric-orbit search for one (d, p): random restarts + greedy reduction,
# banking any verified improvement over the current best rule file.
#
# usage: julia symq_run.jl <d> <p> <minutes> [seed] [mode...]
#
# mode "explore" widens the search (wider parameter budgets, fewer template
# restarts, deeper basin hopping) — used by the office/explorer machine.
# mode "D" (2026-08-26) searches the D_d half-orbit ansatz instead of B_d
# (see SymmetricDQ.jl): its sidecar is dbest_d{d}_p{p}.txt, it warm-starts
# from the better of the B_d and D_d sidecars (a B_d state is a valid D_d
# state), and its logs carry a D suffix.  The bank is shared — a D_d rule is
# banked under the same hermite_d{d}_p{p}_n{n}.csv name.
# mode "L" (2026-08-29, PROPOSAL_nonnormal_native_search.md) searches the
# UNIFORM weight natively (SymmetricDQ.jl basis = :legendre; combines with
# "D" and "explore").  Its bank is legendre_d{d}_p{p}_n{n}.csv (nodes on
# [0,1]^d), its sidecars are best_uniform_d{d}_p{p}.txt / dbest_uniform_…
# (names chosen to match every existing best_*/dbest_* transfer glob), and
# its logs/prog files carry an L suffix.  Templates additionally include the
# Gaussian sidecars of the same and neighboring cases mapped into the uniform
# frame by SymmetricDQ.from_hermite (mode "nohermite" switches that off) —
# measured 2026-08-29, those transformed structures are NOT solution basins
# by themselves (see legendre_from_hermite.jl), so they serve only as donors
# of orbit-type mixes, never as the warm incumbent.
#
# Cross-machine communication: banked best structures are copied to the
# Dropbox comms directory, and donor templates are read from whichever copy
# (local or Dropbox) is newest — so improvements made on either machine
# seed the other within a segment, not overnight.
#
# The total budget is split into ≤45-minute segments; the shared warm-start
# file and the rule bank are re-read between segments, so sibling workers on
# the same case propagate improvements to each other as they happen.
#
# Progress:  julia/symq/d{d}p{p}_s{seed}.log   (restart lines, improvements)
#            julia/symq/best_d{d}_p{p}.txt     (orbit structure of current best)
#            julia/symq/prog_d{d}p{p}_s{seed}.txt  (one line, for the window)
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "SymmetricDQ.jl"))
using .DesignedQuadrature, .SymmetricDQ
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
const seed    = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 20260808
const explore = any(a -> a == "explore", ARGS[5:end])
const dhalf   = any(a -> a == "D", ARGS[5:end])
const uniform = any(a -> a in ("L", "legendre", "uniform"), ARGS[5:end])
const fromh   = uniform && !any(a -> a == "nohermite", ARGS[5:end])
const BASIS   = uniform ? :legendre : :hermite
# uniform → Gaussian orbit-type donors (2026-09-10): OPT-IN, mode
# "legendredonor".  It shipped on-by-default and was measured the same hour on
# GH d2 p37/p39, where no Gaussian sidecar exists so every template was a
# uniform donor: over ~1000 restarts the donor starts landed at δ = 3e-5…4e-4
# while plain random starts reached δ = 8e-7…4e-6, 10-100× better, and
# template_prob = 0.80 at high p meant 77% of restarts (98 of 128 on p37) went
# to the worse option.  The orbit TYPES transfer no better than the values
# did.  Left in the tree because d3 is untested and its Le structures are 2×
# better than the Gaussian ones, but nothing gets it without asking.
const froml   = !uniform && any(a -> a == "legendredonor", ARGS[5:end])
const BANKPFX = uniform ? "legendre" : "hermite"   # rule-file prefix
const WSUF    = uniform ? "_uniform" : ""          # sidecar name infix
const SIDE    = (dhalf ? "dbest" : "best") * WSUF  # this ansatz's sidecar prefix
const SUF     = (dhalf ? "D" : "") * (uniform ? "L" : "")

# Refine-first mode at high p (2026-08-16, PROPOSAL_high_p.md P0).  Measured
# over the full seg logs: at p ≥ 17 no cold restart has ever come within 10%
# of the incumbent (d=4 p=21: 0/256, p=23: 0/101), so cold restarts get cut
# to ~1 in 10 and the budget moves onto the incumbent: most restarts re-seed
# warm on a jitter ladder, hops get 20-30 attempts instead of 3-5, jitter
# also hits the log-weights, and some hops mutate the orbit structure itself
# (reshape — the monotone eliminate! cannot reroute a frozen incumbent).
const highp = p ≥ 17

const J = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const COMMS = get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms"))
const DONORS = joinpath(COMMS, "donors")
mkpath(SYMQ); mkpath(DONORS)

# newest existing candidate among local and Dropbox copies of a best-file
function freshest(names...)
    # "freshest" = the BEST copy: fewest nodes per the header, mtime only as a
    # tie-break.  Newest-mtime-wins let stale sidecars (OSPool jobs ship back
    # every one they were given) regress the warm start (2026-08-26).
    cands = [f for name in names for f in (joinpath(SYMQ, name), joinpath(DONORS, name)) if isfile(f)]
    isempty(cands) && return nothing
    function nodes_of(f)
        m = match(r"nodes=(\d+)", readline(f))
        m === nothing ? typemax(Int) : parse(Int, m[1])
    end
    return cands[argmin([(nodes_of(f), -mtime(f)) for f in cands])]
end

# current best banked rule for this (d, p)
function current_best()
    best = typemax(Int)
    for f in readdir(RULES)
        m = match(Regex("^$(BANKPFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        best = min(best, parse(Int, m[1]))
    end
    return best
end

logio = open(joinpath(SYMQ, "d$(d)p$(p)_s$(seed)$(SUF).log"), "a")
prog  = joinpath(SYMQ, "prog_d$(d)p$(p)_s$(seed)$(SUF).txt")

# relative (backward-error) gate: the absolute monomial check rejects
# provably exact rules for p ≥ 17 (see verify_exactness).  Gate 1e-9,
# validated 2026-08-12 over all 46 banked bests: good rules measure
# ≤ 1.1e-10, rules with one weight off by 1e-6 measure ≥ 1.2e-8.
verify(nodes, w) = verify_exactness(to_bank_nodes(nodes, BASIS), w, p; basis = BASIS, relative = true)

best0 = current_best()
beststr() = best0 == typemax(Int) ? "none" : string(best0)

function bank(st, δ, nodes, w, ex)
    n = length(w)
    nodes = to_bank_nodes(nodes, BASIS)      # uniform rules are banked on [0,1]^d
    path = joinpath(RULES, "$(BANKPFX)_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    # no parent rule: this is a cold/warm restart search, and the warm start is a
    # sidecar orbit STATE, not a banked rule — do not invent a lineage
    symq_lineage!(basename(path), "none",
                  "symmetric-orbit search from restarts (symq_run.jl seed $seed, basis $BASIS, suffix '$SUF')";
                  rules = RULES)
    save_state(joinpath(SYMQ, "$(SIDE)_d$(d)_p$(p).txt"), st, δ)
    try   # publish to the other machine via Dropbox
        cp(joinpath(SYMQ, "$(SIDE)_d$(d)_p$(p).txt"),
           joinpath(DONORS, "$(SIDE)_d$(d)_p$(p).txt"); force = true)
    catch e
        println(logio, "donor publish failed: ", sprint(showerror, e))
    end
    write(prog, "d=$d p=$p  best $n (was $(beststr()))  exact $(round(ex, sigdigits=2))\n")
    println(logio, "BANKED $path  (exactness $ex)  $(describe(st))")
    flush(logio)
end

function report(restart, best_run, banked)
    b = banked < best0 ? "banked $banked" : "none banked"
    write(prog, "d=$d p=$p  start $(beststr()), run-best $best_run, $b  (r$restart)\n")
end

rng = MersenneTwister(seed)
deadline = time() + minutes * 60
seg = 0
while time() < deadline - 30
    global seg += 1
    global best0 = current_best()
    seg_min = min(45.0, (deadline - time()) / 60)
    println(logio, "=== $(d)/$(p) seg $seg ($(round(seg_min, digits=1))min) seed=$seed$(dhalf ? " D_$d half-orbits" : "")$(uniform ? " UNIFORM weight" : "")  current best $(beststr()) ===")
    flush(logio)
    # sidecar lookup for this ansatz/weight: D workers may warm-start from
    # the B_d sidecar of the same weight (a B_d state is a valid D_d state)
    sidecar(dd, pp) = dhalf ? freshest("dbest$(WSUF)_d$(dd)_p$(pp).txt", "best$(WSUF)_d$(dd)_p$(pp).txt") :
                              freshest("best$(WSUF)_d$(dd)_p$(pp).txt")
    warmfile = sidecar(d, p)
    warm = warmfile === nothing ? nothing : load_state(warmfile, d, p)
    # structure transfer: donor structures from neighboring cases (either
    # machine, whichever copy is newest), re-read every segment so wins
    # cascade between workers and between machines
    templates = SymmetricDQ.MixState[]
    for (dd, pp) in ((d, p - 2), (d - 1, p), (d, p + 2))
        f = sidecar(dd, pp)
        f === nothing || push!(templates, load_state(f, dd, pp))
    end
    if fromh
        # uniform search: Gaussian sidecars of this and neighboring cases,
        # mapped into the uniform frame, as extra donors of orbit mixes
        for (dd, pp) in ((d, p), (d, p - 2), (d - 1, p), (d, p + 2))
            f = dhalf ? freshest("dbest_d$(dd)_p$(pp).txt", "best_d$(dd)_p$(pp).txt") :
                        freshest("best_d$(dd)_p$(pp).txt")
            f === nothing && continue
            sth = load_state(f, dd, pp)
            sth.basis === :hermite && push!(templates, from_hermite(sth))
        end
    end
    if froml
        # Gaussian search: uniform sidecars of this and neighboring cases as
        # extra donors of orbit mixes — the mirror of the block above, added
        # 2026-09-10.  This is the ONLY use the 2026-08-29 measurement leaves
        # open for a cross-weight transfer ("donors of orbit-type mixes,
        # never the warm incumbent"), and at d2 p≥37 it is the only structural
        # information about these cells that exists: the uniform bank has
        # designed rules there (p37 242 nodes, p39 268, p41 294) while the
        # Gaussian bank is still the tensor fill (361, 400, 441).  Mode
        # "nolegendre" switches it off.
        for (dd, pp) in ((d, p), (d, p - 2), (d - 1, p), (d, p + 2))
            f = freshest("best_uniform_d$(dd)_p$(pp).txt")
            f === nothing && continue
            stl = load_state(f, dd, pp)
            stl.basis === :legendre && push!(templates, from_legendre(stl))
        end
    end
    # high p: warm 50% of restarts; of the rest, 80% template → cold-random
    # lands at ~1 in 10.  Low p keeps the historical balance.
    best = orbit_search(d, p; seconds = seg_min * 60, rng,
                        best_nodes = best0, verify, on_improve = bank,
                        warm, templates, log_io = logio, progress = report,
                        template_prob = highp ? 0.80 : (explore ? 0.25 : 0.45),
                        warm_prob     = highp ? 0.50 : 0.0,
                        hops_budget   = highp ? (explore ? 30 : 20) :
                                                (explore ? 5 : 3),
                        hop_jitter    = highp ? (0.02, 0.35) : (0.08, 0.23),
                        u_jitter      = highp ? 0.5 : 0.0,
                        reshape_prob  = highp ? 0.4 : 0.0,
                        factor_range = explore ? (1.3, 3.0) : (1.6, 2.4),
                        halves = dhalf, basis = BASIS,
                        extol = 1e-11)  # relative gate (verify above; tightened 2026-08-24)
    println(logio, "=== seg $seg done: best $best (segment start $best0) ===")
    flush(logio)
end
write(prog, "d=$d p=$p  best $(current_best())  (done)\n")
close(logio)
