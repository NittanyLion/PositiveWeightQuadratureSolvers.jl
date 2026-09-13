# S_d × Z_2 (permutation + central) orbit search for one (d, p): random
# restarts + greedy reduction, banking any verified improvement over the
# current best rule file.  The SC counterpart of symq_run.jl -- same campaign
# contract (segments, donors via Dropbox, shared rule bank), separate state
# namespace (sc_best_*.txt) because SC states carry SIGNED values.
#
# usage: julia symq_sc.jl <d> <p> <minutes> [seed] [mode...]
#
# mode "explore" widens the search; mode "L" (2026-08-29) searches the UNIFORM
# weight natively (PermCentralDQ.jl basis = :legendre): bank
# legendre_d{d}_p{p}_n{n}.csv, sidecar sc_best_uniform_d{d}_p{p}.txt (matches
# the existing sc_best_* transfer globs), logs/prog with an L suffix; the
# Gaussian SC sidecars of the same and neighboring cases are added as donor
# templates through PermCentralDQ.from_hermite ("nohermite" switches that off).
#
# Progress:  julia/symq/sc_d{d}p{p}_s{seed}.log
#            julia/symq/sc_best_d{d}_p{p}.txt
#            julia/symq/prog_sc_d{d}p{p}_s{seed}.txt
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "PermCentralDQ.jl"))
using .DesignedQuadrature, .PermCentralDQ
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
const seed    = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 20260811
const explore = any(a -> a == "explore", ARGS[5:end])
const uniform = any(a -> a in ("L", "legendre", "uniform"), ARGS[5:end])
const fromh   = uniform && !any(a -> a == "nohermite", ARGS[5:end])
const BASIS   = uniform ? :legendre : :hermite
const BANKPFX = uniform ? "legendre" : "hermite"
const WSUF    = uniform ? "_uniform" : ""
const SUF     = uniform ? "L" : ""

const J = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const COMMS = get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms"))
const DONORS = joinpath(COMMS, "donors")
mkpath(SYMQ); mkpath(DONORS)

function freshest(name)
    # "freshest" = the BEST copy: fewest nodes per the header, mtime only as a
    # tie-break.  Newest-mtime-wins let stale sidecars (OSPool jobs ship back
    # every one they were given) regress the warm start (2026-08-26).
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
        m = match(Regex("^$(BANKPFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        best = min(best, parse(Int, m[1]))
    end
    return best
end

logio = open(joinpath(SYMQ, "sc_d$(d)p$(p)_s$(seed)$(SUF).log"), "a")
prog  = joinpath(SYMQ, "prog_sc_d$(d)p$(p)_s$(seed)$(SUF).txt")

# collapse coincident nodes (degenerate SC configurations can list a point
# twice, e.g. a centrally self-symmetric orbit); weights add
function dedupe(nodes, w)
    key(i) = ntuple(j -> round(nodes[i, j]; digits = 12), size(nodes, 2))
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

# relative gate — see symq_run.jl / verify_exactness for the rationale
verify(nodes, w) = verify_exactness(to_bank_nodes(nodes, BASIS), w, p; basis = BASIS, relative = true)

best0 = current_best()
beststr() = best0 == typemax(Int) ? "none" : string(best0)

function bank(st, δ, nodes, w, ex)
    nodes, w = dedupe(nodes, w)
    n = length(w)
    # re-verify after any collapse; the collapsed rule is what gets banked
    ex2 = verify(nodes, w)
    if ex2 > (p ≥ 21 ? 1e-7 : 1e-8)
        println(logio, "collapsed rule failed re-verify ($ex2) — not banked")
        return
    end
    nodes = to_bank_nodes(nodes, BASIS)      # uniform rules are banked on [0,1]^d
    path = joinpath(RULES, "$(BANKPFX)_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    # no parent rule: S_d × Z_2 orbit search from restarts (the warm start is a
    # sidecar orbit STATE, not a banked rule) — do not invent a lineage
    symq_lineage!(basename(path), "none",
                  "S_d x Z_2 orbit search from restarts (symq_sc.jl seed $seed, basis $BASIS)";
                  rules = RULES)
    save_state(joinpath(SYMQ, "sc_best$(WSUF)_d$(d)_p$(p).txt"), st, δ)
    try   # publish to the other machine via Dropbox
        cp(joinpath(SYMQ, "sc_best$(WSUF)_d$(d)_p$(p).txt"),
           joinpath(DONORS, "sc_best$(WSUF)_d$(d)_p$(p).txt"); force = true)
    catch e
        println(logio, "donor publish failed: ", sprint(showerror, e))
    end
    write(prog, "d=$d p=$p SC  best $n (was $(beststr()))  exact $(round(ex2, sigdigits=2))\n")
    println(logio, "BANKED $path  (exactness $ex2)  $(describe(st))")
    flush(logio)
end

function report(restart, best_run, banked)
    b = banked < best0 ? "banked $banked" : "none banked"
    write(prog, "d=$d p=$p SC  start $(beststr()), run-best $best_run, $b  (r$restart)\n")
end

rng = MersenneTwister(seed)
deadline = time() + minutes * 60
seg = 0
while time() < deadline - 30
    global seg += 1
    global best0 = current_best()
    seg_min = min(45.0, (deadline - time()) / 60)
    println(logio, "=== SC $(d)/$(p) seg $seg ($(round(seg_min, digits=1))min) seed=$seed$(uniform ? " UNIFORM weight" : "")  current best $(beststr()) ===")
    flush(logio)
    warmfile = freshest("sc_best$(WSUF)_d$(d)_p$(p).txt")
    warm = warmfile === nothing ? nothing : load_state(warmfile, d, p)
    templates = PermCentralDQ.MixState[]
    for (dd, pp) in ((d, p - 2), (d - 1, p), (d, p + 2))
        f = freshest("sc_best$(WSUF)_d$(dd)_p$(pp).txt")
        f === nothing || push!(templates, load_state(f, dd, pp))
    end
    if fromh   # Gaussian SC structures mapped into the uniform frame, as donors
        for (dd, pp) in ((d, p), (d, p - 2), (d - 1, p), (d, p + 2))
            f = freshest("sc_best_d$(dd)_p$(pp).txt")
            f === nothing && continue
            sth = load_state(f, dd, pp)
            sth.basis === :hermite && push!(templates, from_hermite(sth))
        end
    end
    best = orbit_search(d, p; seconds = seg_min * 60, rng,
                        best_nodes = best0, verify, on_improve = bank,
                        warm, templates, log_io = logio, progress = report,
                        template_prob = explore ? 0.25 : 0.45,
                        hops_budget = explore ? 5 : 3,
                        factor_range = explore ? (1.3, 3.0) : (1.6, 2.4),
                        basis = BASIS,
                        extol = 1e-11)  # relative gate (verify above; tightened 2026-08-24)
    println(logio, "=== seg $seg done: best $best (segment start $best0) ===")
    flush(logio)
end
write(prog, "d=$d p=$p SC  best $(current_best())  (done)\n")
close(logio)
