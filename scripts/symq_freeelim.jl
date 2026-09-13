# Free-node elimination worker: attack the current best banked rule for
# (d, p) with NO symmetry assumption — repeatedly drop a light node and
# reconverge (jittered retries), plus basin-hopping perturbations when stuck.
# Symmetric-ansatz floors (retired.tsv) do NOT bind here: this is the tool
# for gaps like d=3 p=9 (48 banked vs 45 published, non-B_3 structure).
#
# usage: julia symq_freeelim.jl <d> <p> <minutes> [seed] [hermite|legendre|laguerre]
#
# The basis argument arrived 2026-09-05; the worker was Hermite-only, so the
# uniform bank had no free-node elimination at all.  That matters because free
# elimination is exactly the tool that breaks a symmetric-ansatz floor, and
# Legendre's floor-bound cells are the same situation Hermite's were: Le d3 p9
# = 48 = the pair floor while GH d3 p9 = 45 (found by this worker, non-B_3
# structure); Le d4 p11 = 234 = floor vs GH 193; Le d5 p7 = 100 = floor vs
# GH 83; Le d3 p23 = 612 = floor vs GH 597.
#
# laguerre arrived 2026-09-06 (user): the exponential bank's undesigned cells
# are RAW tensor products — a designed lower-dim rule times a 1-D Gauss–
# Laguerre rule (d3 p17 = 62×9, d4 p17 = 62², d5 p15 = 51×237) or the plain
# grid — which are only S_{d-1}×S_1 (or less) symmetric and so out of reach of
# the S_d orbit tools; a symmetry-free eliminator is exactly what takes them.
# Nodes live on the orthant x ≥ 0 (half-space penalty in the V2 solver); the
# "origin" merge target is the corner, a legitimate node of the weight.
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
const BASIS   = length(ARGS) ≥ 5 ? Symbol(ARGS[5]) : :hermite
BASIS in (:hermite, :legendre, :laguerre) || error("basis must be hermite, legendre or laguerre")
const BANKPFX = String(BASIS)
const BTAG    = BASIS === :hermite ? "" : "_$(BANKPFX)"
# The solver's uniform frame is [-1,1] and it returns nodes in [0,1]
# (DesignedQuadratureV2.jl:464), which is also the bank's frame; only the
# init_pairs handed BACK to it need converting.  The centre of the domain —
# where the merge-into-origin move aims — is 0 for Hermite, ½ for uniform.
to_solver(X) = BASIS === :legendre ? 2 .* X .- 1 : X
const CENTER = BASIS === :legendre ? 0.5 : 0.0
in_box(X) = BASIS === :legendre ? all(0 .<= X .<= 1) :
            BASIS === :laguerre ? all(X .>= 0) : true

const J = get(ENV, "SYMQ_ROOT", pwd())
# Test hooks, matching symq_lag.jl / symq_ladder.jl (added 2026-09-05 so the
# legendre route can be exercised without writing to the live bank).
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
# relative (backward-error) gate — the absolute monomial check rejects
# provably exact rules for p ≥ 17; see verify_exactness in DesignedQuadrature.jl
# Was 1e-9 until 2026-08-24.  Too loose: a rule that never really converged
# (~3e-10) was banked and then eliminated down ten generations, every one
# inheriting the defect (d=4 p=19 n=1495..1504, see rules/suspect/README.md).
# Genuine convergence lands at ~1e-14, so 1e-11 keeps ~40x headroom over the
# worst honest rule in the bank while shutting out that whole failure mode.
const EXTOL = 1e-11

# Lineage chain (2026-09-11).  Elimination is a chain: the run reconverges the
# banked start, then drops one node at a time, so each banked rule's parent is
# the rule banked just before it — main() seeds this with the start rule and
# bank() advances it, exactly as dw_warm.jl records its pair chain.
const LIN_PARENT = Ref("none")

function best_rule()
    best = typemax(Int); path = ""
    for f in readdir(RULES)
        m = match(Regex("^$(BANKPFX)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        n = parse(Int, m[1])
        n < best && (best = n; path = joinpath(RULES, f))
    end
    return best, path
end

logio = open(joinpath(SYMQ, "free$(BTAG)_d$(d)p$(p).log"), "a")
# the trailing L keeps every non-Gaussian worker out of the Gaussian rows
# (publish_status.sh / status_symq.sh grep -v L.txt); laguerre rides on it too
prog  = joinpath(SYMQ, "prog_free$(BTAG)_d$(d)p$(p)$(BASIS === :hermite ? "" : "L").txt")
rng = MersenneTwister(97 * seed + 13 * d + p)

function converge(nds, ws; jitter = 0.0)
    b0 = copy(nds)
    jitter > 0 && (b0 .+= jitter .* randn(rng, size(b0)...))
    # abort_*: measured cutoff — attempts still above 1e-4 at iteration 400
    # never converged in trajectory stats, while genuine near-basin runs are
    # below ~1e-5 by iteration 200 (25–40× margin)
    # tol 1e-11, not 1e-13: some banked rules (e.g. d=3 p=9 n=48) have a
    # conditioning floor just above 1e-13, which made the warm start "fail"
    # and killed the worker at startup.  Correctness is gated by the
    # verify_exactness ≤ EXTOL check at banking, and the solver's polish
    # stage pushes to the machine floor regardless of tol.
    # Weight floor 1e-300, NOT 1e-14 (2026-09-10, user: "are you sure there is
    # not something stopping valid solutions in the way we're doing things?").
    # A d=2 Gauss–Hermite product rule has corner weights far below 1e-14 —
    # 5.6e-25 at p=37, 1.6e-26 at p=39 — because they are products of two tiny
    # 1-D weights, and 16-21% of the grid's weights sit under the old clamp.
    # Raising them to 1e-14 puts mass where the rule has essentially none, at
    # |x| ~ 7.4 where the degree-37 basis values reach 5.5e5, so the warm start
    # arrived at the solver already broken.  Measured on the banked tensor
    # fills, reconverging the SAME rule:
    #     p=35  as banked ‖R‖ 5.3e-16 converged | clamped 9.98e-12 converged
    #     p=37  as banked ‖R‖ 7.4e-16 converged | clamped 3.60e-10 FAILED
    #     p=39  as banked ‖R‖ 6.2e-16 converged | clamped 3.36e-09 FAILED
    # p=35 squeaked under the 1e-11 tol and p=37 did not, which is exactly why
    # free elimination — the tool that made every banked GH d2 rule from p25 to
    # p35 — died at startup ("warm start failed to reconverge") on every GH d2
    # cell from p37 to p77, on every machine, and those cells stayed at their
    # tensor fill.  The solver already floors the log parameterization itself
    # (`u = log.(max.(w0, 1e-300))`), so this clamp was never load-bearing.
    return designed_quadrature_v2(d, p, length(ws); basis = BASIS,
                                  symmetric = false,
                                  init_pairs = (to_solver(b0), max.(ws, 1e-300)),
                                  tol = 1e-11, maxiter = 2000,
                                  stall_window = 250,
                                  abort_iters = 400, abort_resid = 1e-4)
end

function bank(nodes, w, ex)
    n = length(w)
    if !in_box(nodes)
        println(logio, "rule left the box (min $(minimum(nodes)), max $(maximum(nodes))) — not banked")
        return
    end
    path = joinpath(RULES, "$(BANKPFX)_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes[i, :]; w[i]]), ","))
        end
    end
    symq_lineage!(basename(path), LIN_PARENT[],
                  "free node elimination: dropped a node/merged a pair, reconverged (symq_freeelim.jl seed $seed)";
                  rules = RULES)
    LIN_PARENT[] = basename(path)          # the next drop descends from this rule
    println(logio, "BANKED $path (exactness $ex)")
    flush(logio)
end

# Merge move (HP76 lesson, added 2026-08-13): drop-and-reconverge can only
# shed one node per step, but Haegemans-Piessens got 28->25 by driving TWO
# orbits into COINCIDENCE at the origin -- several nodes gone in one move,
# from a basin no sequence of single drops reaches.  Here: pick the closest
# node pair, replace both with their weighted centroid carrying the summed
# weight, and reconverge.  Also tried against the origin (node -> 0) when a
# node is light and central.
function try_merge(cur_nodes, cur_w, deadline)
    n = length(cur_w)
    n < 2 && return nothing
    # candidate pairs: the 6 closest, plus the 2 most central light nodes vs origin
    dists = [(norm(cur_nodes[i, :] - cur_nodes[j, :]), i, j)
             for i in 1:n for j in (i+1):n]
    sort!(dists, by = first)
    cands = Tuple{Int,Int}[(i, j) for (_, i, j) in dists[1:min(6, length(dists))]]
    ord = sortperm([norm(cur_nodes[i, :] .- CENTER) * cur_w[i] for i in 1:n])
    for k in ord[1:min(2, n)]
        push!(cands, (k, 0))                       # 0 = merge into the origin
    end
    for (i, j) in cands
        time() > deadline && return nothing
        if j == 0
            # merge into the origin: if an origin node exists, fold i into it
            # (true n-1 reduction); otherwise move i there and let the drop
            # pass try again from that basin
            o = findfirst(s -> norm(cur_nodes[s, :] .- CENTER) < 1e-9, 1:n)
            if o !== nothing && o != i
                keep = setdiff(1:n, i)
                nds = cur_nodes[keep, :]
                ws = copy(cur_w[keep])
                ws[o > i ? o - 1 : o] += cur_w[i]
            else
                nds = copy(cur_nodes); nds[i, :] .= CENTER
                ws = copy(cur_w)
            end
        else
            keep = setdiff(1:n, j)
            nds = cur_nodes[keep, :]
            ws = cur_w[keep]
            i2 = i > j ? i - 1 : i
            wsum = cur_w[i] + cur_w[j]
            nds[i2, :] .= (cur_w[i] .* cur_nodes[i, :] .+ cur_w[j] .* cur_nodes[j, :]) ./ wsum
            ws[i2] = wsum
        end
        ws ./= sum(ws)
        for t in 0:3
            r = converge(nds, ws; jitter = t == 0 ? 0.0 : 0.01 * t)
            if r.converged && minimum(r.weights) > 0
                ex = verify_exactness(r.nodes, r.weights, p; basis = BASIS,
                                      relative = true)
                ex <= EXTOL && return (nodes = r.nodes, weights = r.weights, ex = ex)
            end
            time() > deadline && return nothing
        end
    end
    return nothing
end

function main()
    deadline = time() + minutes * 60
    best0, path0 = best_rule()
    if best0 == typemax(Int)
        println(logio, "no banked $BANKPFX rule for d=$d p=$p")
        flush(logio)          # an early return skips close(logio) below
        return
    end
    LIN_PARENT[] = basename(path0)        # chain root: what this run started from
    println(logio, "=== free-elim $BANKPFX d=$d p=$p from $best0 nodes, $(round(minutes))min, seed $seed ===")
    flush(logio)
    write(prog, "d=$d p=$p  start $best0, run-best $best0, none banked (free-elim r0)\n")

    ru = load_rule(path0)
    r0 = converge(Matrix{Float64}(ru.nodes), Vector{Float64}(ru.weights))
    if !(r0.converged && minimum(r0.weights) > 0)
        println(logio, "warm start failed to reconverge (δ=$(r0.residual))")
        flush(logio)          # an early return skips close(logio) below
        return
    end
    cur_nodes, cur_w = r0.nodes, r0.weights
    attempt = 0

    while time() < deadline
        attempt += 1
        n = length(cur_w)
        ord = sortperm(cur_w)
        success = false
        for k in ord[1:min(4, n)]
            time() > deadline && break
            keep = setdiff(1:n, k)
            nds = cur_nodes[keep, :]
            ws = cur_w[keep] ./ sum(cur_w[keep])
            for t in 0:5
                r = converge(nds, ws; jitter = t == 0 ? 0.0 : 0.015 * t)
                if r.converged && minimum(r.weights) > 0
                    ex = verify_exactness(r.nodes, r.weights, p; basis = BASIS,
                                          relative = true)
                    if ex ≤ EXTOL
                        cur_nodes, cur_w = r.nodes, r.weights
                        bank(cur_nodes, cur_w, ex)
                        write(prog, "d=$d p=$p  start $best0, run-best $(n - 1), banked $(n - 1) (free-elim r$attempt)\n")
                        success = true
                        break
                    end
                end
                time() > deadline && break
            end
            success && break
        end
        if !success
            # merge move before hopping: coincidence reductions live in basins
            # single drops cannot reach (see try_merge above)
            m = try_merge(cur_nodes, cur_w, deadline)
            if m !== nothing
                if length(m.weights) < length(cur_w)
                    cur_nodes, cur_w = m.nodes, m.weights
                    bank(cur_nodes, cur_w, m.ex)
                    write(prog, "d=$d p=$p  start $best0, run-best $(length(cur_w)), banked $(length(cur_w)) (free-elim merge r$attempt)\n")
                    continue
                else
                    cur_nodes, cur_w = m.nodes, m.weights   # basin move only
                end
            end
            # basin hop: perturb the whole configuration and reconverge; if
            # it lands elsewhere (still a valid rule), retry the drops there
            for _ in 1:4
                time() > deadline && break
                r = converge(cur_nodes, cur_w; jitter = 0.03 + 0.05 * rand(rng))
                if r.converged && minimum(r.weights) > 0 &&
                   verify_exactness(r.nodes, r.weights, p; basis = BASIS,
                                    relative = true) ≤ EXTOL
                    cur_nodes, cur_w = r.nodes, r.weights
                    break
                end
            end
            attempt % 5 == 0 && write(prog,
                "d=$d p=$p  start $best0, run-best $(length(cur_w)), none banked (free-elim r$attempt)\n")
        end
    end
    bfin, _ = best_rule()
    println(logio, "=== free-elim done: best $bfin (was $best0), $attempt attempts ===")
    write(prog, "d=$d p=$p  start $best0, run-best $bfin, free-elim  (done)\n")
    close(logio)
end

main()