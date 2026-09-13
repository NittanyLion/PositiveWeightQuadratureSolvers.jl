#=
symq_ladder.jl -- degree-ladder worker (Xiao-Gimbutas-style sequential
degree climbing; litcheck/STRATEGIES_NOT_YET_EMPLOYED.md #2, wired
2026-08-30).  Climbs p in steps of 2; every rung warm-starts from the best
rule two degrees below (banked/sidecar, or the rung just climbed), so the
expensive high-p cases inherit structure instead of starting cold.

Routes per basis:
  laguerre  -- LaguerreDQ's native machinery: the (d, p-2) sidecar is fed
               to orbit_search as a near-mandatory template
               (template_prob 0.95; seed_from_template scales radially by
               (p+1)/(p_src+1)).  V2 has no :laguerre basis, so the V2
               degree_continuation path is not available here.
  hermite / legendre -- V2 degree_continuation from the banked (d, p-2)
               rule's +/- pairs (center node becomes a zero pair), with a
               start-size ladder (1.0/1.15/1.35 x the DOF count -- a start
               AT the count stalls at d >= 4, METHODS §17), free-mode
               fallback for non-centrally-symmetric sources, then
               node_elimination downward; staged sizes promoted through
               the standard gates.

Rung ledger (2026-09-05).  Rungs used to run unconditionally, so a rung on a
solved, static cell spent its whole `minutes` budget re-solving the incumbent
(warm restart 1 re-solves it by construction) and a laguerre rung with no
donor burned the same budget cold.  A rung now runs only when an input has
moved -- the donor at p-2 improved, the cell's own best improved, or it has
never been tried -- and otherwise backs off, retrying every SYMQ_LADDER_RETRY
passes (default 4) so nothing is skipped forever.  State lives in
`symq/ladder_ledger_<basis>_d<d>.tsv`; SYMQ_LADDER_FORCE=1 restores the old
unconditional full pass.  No tolerance, gate or precision rule is affected --
this only decides where the wall clock goes.

usage: julia symq_ladder.jl <d> <p_first_target> <p_end> <basis> <minutes_per_rung> [seed]
       (rung budget is enforced by orbit_search for laguerre; the V2 route's
        elimination has no internal deadline -- cap the process with
        `timeout` when the budget is strict)

Stop-cron: julia/symq_stop.sh's pkill regex includes `ladder` (extended
2026-08-30) so the 08:00 safety net covers this worker.
=#
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "DesignedQuadratureV2.jl"))
include(joinpath(@__DIR__, "..", "src", "LaguerreDQ.jl"))
using .DesignedQuadrature, .DesignedQuadratureV2, .LaguerreDQ
using Random, Printf, LinearAlgebra, Dates

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

BLAS.set_num_threads(1)

const d       = parse(Int, ARGS[1])
const pfirst  = parse(Int, ARGS[2])
const pend    = parse(Int, ARGS[3])
const basis   = Symbol(ARGS[4])
const minutes = parse(Float64, ARGS[5])
const seed    = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20260830
basis in (:hermite, :legendre, :laguerre) || error("basis must be hermite|legendre|laguerre")

const J      = get(ENV, "SYMQ_ROOT", pwd())
const RULES  = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ   = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const DONORS = get(ENV, "SYMQ_DONORS_DIR",
                   joinpath(get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms")), "donors"))
const EXTOL  = 1e-11
const PFX    = basis === :laguerre ? "laguerre" : basis === :legendre ? "legendre" : "hermite"
const SIDE   = "best_laguerre"          # alpha = 0 only; extend with ATAG if needed

logio = open(joinpath(SYMQ, "ladder_$(PFX)_d$(d).log"), "a")
prog  = joinpath(SYMQ, "prog_ladder_$(PFX)_d$(d)_s$(seed)L.txt")
rng   = MersenneTwister(seed)

bank_best(p) = minimum([parse(Int, m[1]) for f in readdir(RULES)
                        for m in [match(Regex("^$(PFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)]
                        if m !== nothing]; init = typemax(Int))
bank_file(p) = (n = bank_best(p); n == typemax(Int) ? nothing :
                joinpath(RULES, "$(PFX)_d$(d)_p$(p)_n$(n).csv"))

function write_rule(p, nodes, w, ex)
    n = length(w)
    path = joinpath(RULES, "$(PFX)_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    # a rung continues this cell's (d, p-2) rule.  That parent is in ANOTHER cell,
    # so make_appendices.jl ends the credit chain there — which is correct.
    par = (f2 = bank_file(p - 2); f2 === nothing ? "none" : basename(f2))
    symq_lineage!(basename(path), par,
                  "degree ladder rung: continued from the d=$d p=$(p - 2) rule (symq_ladder.jl basis $basis, seed $seed)";
                  rules = RULES)
    println(logio, "BANKED $path (exactness $ex)")
end

# --- laguerre rung: native orbit search, donor template forced ------------
function freshest(name)
    cands = [f for f in (joinpath(SYMQ, name), joinpath(DONORS, name)) if isfile(f)]
    isempty(cands) && return nothing
    nodes_of(f) = (m = match(r"nodes=(\d+)", readline(f)); m === nothing ? typemax(Int) : parse(Int, m[1]))
    return cands[argmin([(nodes_of(f), -mtime(f)) for f in cands])]
end

function dedupe(nodes, w)
    key(i) = ntuple(j -> round(nodes[i, j]; digits = 12) + 0.0, size(nodes, 2))
    seen = Dict{NTuple{size(nodes,2),Float64},Int}()
    keep = Int[]; wo = Float64[]
    for i in axes(nodes, 1)
        k = key(i)
        if haskey(seen, k); wo[seen[k]] += w[i]
        else push!(keep, i); push!(wo, w[i]); seen[k] = length(keep) end
    end
    return nodes[keep, :], wo
end

function rung_laguerre(p)
    best0 = bank_best(p)
    verify(nodes, w) = verify_exactness(nodes, w, p; basis = :laguerre, relative = true, alpha = 0.0)
    function on_improve(st, δ, nodes, w, ex)
        nodes, w = dedupe(nodes, w)
        ex2 = verify(nodes, w)
        (ex2 > EXTOL || minimum(w) <= 0 || minimum(nodes) < 0) && return
        write_rule(p, nodes, w, ex2)
        LaguerreDQ.save_state(joinpath(SYMQ, "$(SIDE)_d$(d)_p$(p).txt"), st, δ)
        try cp(joinpath(SYMQ, "$(SIDE)_d$(d)_p$(p).txt"),
               joinpath(DONORS, "$(SIDE)_d$(d)_p$(p).txt"); force = true) catch; end
    end
    src = freshest("$(SIDE)_d$(d)_p$(p - 2).txt")
    # A ladder rung IS the continuation from p-2; with no donor it degenerates
    # into the cold search symq_lag.jl already runs, and orbit_search would
    # still burn the whole `seconds` budget.  Skip, as rung_v2 does.
    if src === nothing
        println(logio, "=== ladder rung d=$d p=$p (laguerre): no donor at p=$(p - 2) — rung skipped ===")
        flush(logio)
        return best0
    end
    templates = LaguerreDQ.MixState[LaguerreDQ.load_state(src, d, p - 2)]
    wf = freshest("$(SIDE)_d$(d)_p$(p).txt")
    warm = wf === nothing ? nothing : LaguerreDQ.load_state(wf, d, p)
    println(logio, "=== ladder rung d=$d p=$p (laguerre): donor $(src === nothing ? "NONE (cold)" : basename(src)), best $best0 ===")
    flush(logio)
    best = orbit_search(d, p; seconds = minutes * 60, rng, best_nodes = best0,
                        verify, on_improve, warm, templates, log_io = logio,
                        template_prob = isempty(templates) ? 0.0 : 0.95,
                        warm_prob = warm === nothing ? 0.0 : 0.3,
                        alpha = 0.0, extol = EXTOL)
    write(prog, "d=$d p=$p LADDER  best $(bank_best(p))  (rung done)\n")
    return best
end

# --- hermite / legendre rung: V2 degree continuation + elimination --------
n_even(p) = sum(binomial(s + d - 1, d - 1) for s in 0:2:p)

function rung_v2(p)
    best0 = bank_best(p)
    srcf = bank_file(p - 2)
    srcf === nothing && (println(logio, "no source rule at p=$(p-2) — rung skipped"); return)
    ru = load_rule(srcf)
    X = Matrix{Float64}(ru.nodes); w = Vector{Float64}(ru.weights)
    Tm = basis === :legendre ? 2 .* X .- 1 : X
    n0 = length(w)
    symmetric = true
    local b0, w0
    origin = [i for i in 1:n0 if norm(Tm[i, :]) < 1e-9]
    rest = setdiff(1:n0, origin)
    try
        b0, w0 = pairs_from_rule(Tm[rest, :], w[rest])
        isempty(origin) || (b0 = vcat(b0, zeros(1, d)); w0 = vcat(w0, [sum(w[origin])]))
    catch e
        e isa ArgumentError || rethrow()
        symmetric = false
        b0, w0 = copy(Tm), copy(w)
    end
    m0 = length(w0)
    mbase = symmetric ? max(m0, cld(n_even(p), d + 1)) : max(n0, cld(binomial(d + p, d), d + 1))
    println(logio, "=== ladder rung d=$d p=$p ($basis): source n=$n0 ($(symmetric ? "$m0 pairs" : "free")), best $best0 ===")
    flush(logio)
    r = nothing
    for fac in (1.0, 1.15, 1.35)
        mt = ceil(Int, mbase * fac)
        if symmetric
            r = degree_continuation(d, p, (b0, w0), mt; basis, rng, tol = 1e-11,
                                    maxiter = 4000, stall_window = 300)
        else
            nadd = mt - n0
            lhs = nadd > 0 ? rand(rng, nadd, d) : zeros(0, d)
            badd = basis === :hermite ? DesignedQuadratureV2.norminvcdf.(lhs) : lhs .* 2 .- 1
            r = designed_quadrature_v2(d, p, mt; basis, symmetric = false, tol = 1e-11,
                                       init_pairs = (vcat(b0, badd), vcat(w0, fill(1e-4, nadd))),
                                       maxiter = 4000, stall_window = 300)
        end
        r.converged && minimum(r.weights) > 0 && break
        println(logio, "  start size $mt did not converge (res $(r.residual))")
        r = nothing
    end
    r === nothing && (println(logio, "rung failed to converge"); return)
    stage = joinpath(SYMQ, "ladder_stage")
    mkpath(stage)
    node_elimination(d, p; basis, symmetric, m_start = length(r.pairs[2]),
                     init_pairs = r.pairs, rng,
                     save_prefix = joinpath(stage, "$(PFX)_d$(d)_p$(p)"),
                     tag = "ladder d$d p$p")
    for f in sort(readdir(stage))
        m = match(Regex("^$(PFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        nd, wd = load_rule(joinpath(stage, f))
        nd, wd = dedupe(nd, wd)
        nn = length(wd)
        ex = verify_exactness(nd, wd, p; basis, relative = true)
        ok = ex <= EXTOL && minimum(wd) > 0 && abs(sum(wd) - 1) <= 1e-10 &&
             (basis !== :legendre || all(0 <= x <= 1 for x in nd))
        ok && nn < bank_best(p) && write_rule(p, nd, wd, ex)
        rm(joinpath(stage, f); force = true)
    end
    write(prog, "d=$d p=$p LADDER  best $(bank_best(p))  (rung done)\n")
end

# --- rung ledger: spend the budget only where an input has changed --------
# A rung re-run is worth its wall clock when something upstream moved: the
# donor at p-2 improved, this cell's own best improved, or the rung has never
# been tried.  Otherwise the rung can only reconfirm what is already banked
# (warm restart 1 re-solves the incumbent by construction), so back off and
# retry every RETRY_EVERY passes rather than skipping forever.  This changes
# only WHERE the budget goes -- no tolerance, gate or precision rule is
# touched.  SYMQ_LADDER_FORCE=1 restores the old unconditional full pass.
const LEDGER      = joinpath(SYMQ, "ladder_ledger_$(PFX)_d$(d).tsv")
const RETRY_EVERY = parse(Int, get(ENV, "SYMQ_LADDER_RETRY", "4"))
const FORCE       = get(ENV, "SYMQ_LADDER_FORCE", "0") != "0"

# The laguerre rung continues from the SIDECAR, not the bank, so measure the
# donor there when one exists; hermite/legendre continue from the banked rule.
function donor_best(p)
    if basis === :laguerre
        f = freshest("$(SIDE)_d$(d)_p$(p - 2).txt")
        if f !== nothing
            m = match(r"nodes=(\d+)", readline(f))
            m === nothing || return parse(Int, m[1])
        end
    end
    return bank_best(p - 2)
end

function read_ledger()
    led = Dict{Int,NTuple{3,Int}}()        # p => (own_best, donor_best, skips)
    isfile(LEDGER) || return led
    for ln in eachline(LEDGER)
        f = split(strip(ln), '\t')
        (length(f) >= 4 && all(isdigit, f[1])) || continue
        try
            led[parse(Int, f[1])] = (parse(Int, f[2]), parse(Int, f[3]), parse(Int, f[4]))
        catch
        end
    end
    return led
end

function write_ledger(led)
    tmp = LEDGER * ".tmp"
    open(tmp, "w") do io
        println(io, "p\town_best\tdonor_best\tskips\tupdated")
        for p in sort(collect(keys(led)))
            o, dn, sk = led[p]
            println(io, join((p, o, dn, sk,
                              Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")), '\t'))
        end
    end
    mv(tmp, LEDGER; force = true)
end

function should_run(p, led)
    FORCE && return (true, "forced")
    haskey(led, p) || return (true, "first pass")
    own, dn = bank_best(p), donor_best(p)
    o0, d0, skips = led[p]
    own < o0 && return (true, "own best improved $o0 -> $own")
    dn  < d0 && return (true, "donor improved $d0 -> $dn")
    dn == typemax(Int) && return (false, "no donor at p=$(p - 2)")
    skips + 1 >= RETRY_EVERY && return (true, "periodic retry after $skips skips")
    return (false, "inputs unchanged (own $own, donor $dn), skip $(skips + 1)/$RETRY_EVERY")
end

led = read_ledger()
for p in pfirst:2:pend
    go, why = should_run(p, led)
    if !go
        _, _, sk = get(led, p, (bank_best(p), donor_best(p), 0))
        led[p] = (bank_best(p), donor_best(p), sk + 1)
        write_ledger(led)                    # symq_stop.sh may pkill us at 08:00
        @printf("rung d=%d p=%d skipped (%s)\n", d, p, why)
        println(logio, "rung p=$p skipped: $why")
        flush(logio); flush(stdout)
        continue
    end
    t0 = time()
    basis === :laguerre ? rung_laguerre(p) : rung_v2(p)
    led[p] = (bank_best(p), donor_best(p), 0)
    write_ledger(led)
    @printf("rung d=%d p=%d done in %.0fs, bank best now %s (ran: %s)\n", d, p, time() - t0,
            (b = bank_best(p)) == typemax(Int) ? "none" : string(b), why)
    println(logio, "rung p=$p done in $(round(time() - t0))s, bank best $(bank_best(p)) (ran: $why)")
    flush(logio); flush(stdout)
end
close(logio)
