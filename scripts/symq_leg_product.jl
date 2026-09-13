# Product-ansatz descent for one (d, p) under the UNIFORM weight (2026-09-08,
# user: Le d5 q9/q10/q11): start from (banked D_{d-1}/B_{d-1} orbit factor) ×
# (1-D Gauss–Legendre rule) and eliminate downward in D_{d-1} × B_1 orbit
# space.  Sibling of symq_lag_product.jl; module LegendreProductDQ.jl.
#
# The (d-1) factor is the freshest of symq/dbest_uniform_d{d-1}_p{p}.txt and
# symq/best_uniform_d{d-1}_p{p}.txt (also looked up in comms/donors), verified
# exact under SymmetricDQ.  The 1-D factor is the m1-point Gauss–Legendre
# rule, m1 = ⌈(p+1)/2⌉; a segment that ends where it began re-seeds with m1+1
# (headroom), up to m1+3, then wraps.
#
# usage: julia symq_leg_product.jl <d> <p> <minutes> [seed] [m=<m1>] [grid]
#
# Bank legendre_d{d}_p{p}_n{n}.csv on [0,1]^d when n beats the bank and
# n ≤ SYMQ_BANK_MAXN (40000), relative exactness ≤ 1e-11;
# prog_legprod_d{d}p{p}_s{seed}L.txt; state sidecar best_uniform_prod_d{d}_p{p}.txt
# (its own format — never a donor for the B_d tools).
#
# Test hooks: SYMQ_RULES_DIR, SYMQ_SIDECAR_DIR, SYMQ_DONORS_DIR.
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "SymmetricDQ.jl"))
include(joinpath(@__DIR__, "..", "src", "LegendreProductDQ.jl"))
using .DesignedQuadrature, .SymmetricDQ, .LegendreProductDQ
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
const seed    = length(ARGS) ≥ 4 && !occursin('=', ARGS[4]) ? parse(Int, ARGS[4]) : 20260908
d ≥ 2 || error("the product ansatz needs d ≥ 2")
kwarg(k, dflt) = something(findfirst(a -> startswith(a, "$k="), ARGS) |>
                           (i -> i === nothing ? nothing : parse(Float64, ARGS[i][length(k)+2:end])),
                           dflt)
const M1 = Int(kwarg("m", Float64(cld(p + 1, 2))))
const GRID = any(a -> a == "grid", ARGS)     # 2026-09-08 11:15: force the GL grid factor even when a sidecar exists
                                              # (q8's grid start shed 78 % in 10 min while the D_4-sidecar starts crawled)
2 * M1 - 1 ≥ p || error("m = $M1 is exact only to degree $(2M1-1) < p = $p")

const J      = get(ENV, "SYMQ_ROOT", pwd())
const RULES  = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ   = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const DONORS = get(ENV, "SYMQ_DONORS_DIR",
                   joinpath(get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms")), "donors"))
mkpath(RULES); mkpath(SYMQ)
logio = open(joinpath(SYMQ, "legprod_d$(d)p$(p)_s$(seed).log"), "a")
prog  = joinpath(SYMQ, "prog_legprod_d$(d)p$(p)_s$(seed)L.txt")

function current_best()
    best = typemax(Int)
    for f in readdir(RULES)
        m = match(Regex("^legendre_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing || (best = min(best, parse(Int, m[1])))
    end
    return best
end
best0 = current_best()
beststr() = best0 == typemax(Int) ? "none" : string(best0)

function dedupe(nodes, w)     # coincident nodes — weights add
    key(i) = ntuple(j -> round(nodes[i, j]; digits = 12) + 0.0, size(nodes, 2))
    seen = Dict{NTuple{size(nodes,2),Float64},Int}()
    keep = Int[]; wo = Float64[]
    for i in axes(nodes, 1)
        k = key(i)
        if haskey(seen, k)
            wo[seen[k]] += w[i]
        else
            push!(keep, i); push!(wo, w[i]); seen[k] = length(keep)
        end
    end
    return nodes[keep, :], wo
end

to_bank(nodes) = (nodes .+ 1.0) ./ 2.0          # [-1,1]^d → [0,1]^d, as symq_run.jl
verify(nodes01, w) = verify_exactness(nodes01, w, p; basis = :legendre, relative = true)

const BANK_MAXN = parse(Int, get(ENV, "SYMQ_BANK_MAXN", "40000"))
const STATE = joinpath(SYMQ, "best_uniform_prod$(GRID ? "grid" : "")_d$(d)_p$(p).txt")
run_best = typemax(Int)
function persist(st, δ)
    n = prod_nodes(st)
    n < run_best || return
    global run_best = n
    LegendreProductDQ.save_state(STATE, st, δ)
    write(prog, "d=$d p=$p LEGPROD  descent $n (bank $(beststr()))\n")
end

function bank(st, δ)
    nodes, w = LegendreProductDQ.expand_rule(st)
    nodes, w = dedupe(nodes, w)
    n = length(w)
    persist(st, δ)
    n < current_best() || return
    n ≤ BANK_MAXN || (println(logio, "  n=$n above SYMQ_BANK_MAXN=$BANK_MAXN — not written"); return)
    nodes01 = to_bank(nodes)
    ex = verify(nodes01, w)
    if ex > 1e-11 || minimum(w) ≤ 0 || minimum(nodes01) < 0 || maximum(nodes01) > 1
        println(logio, "  n=$n failed re-verify (ex $ex, min w $(minimum(w)), range $(extrema(nodes01))) — not banked")
        return
    end
    path = joinpath(RULES, "legendre_d$(d)_p$(p)_n$(n).csv")
    open(path, "w") do io
        for i in 1:n
            println(io, join(string.([nodes01[i, :]; w[i]]), ","))
        end
    end
    # no parent RULE: the start is a (d-1) factor STATE (or the Gauss–Legendre
    # grid) times a 1-D rule, built here — not a banked rule file
    symq_lineage!(basename(path), "none",
                  "product-ansatz descent from a (d-1) factor x 1-D Gauss-Legendre rule (symq_leg_product.jl seed $seed, m=$M1)";
                  rules = RULES)
    write(prog, "d=$d p=$p LEGPROD  best $n (was $(beststr()))  exact $(round(ex, sigdigits=2))\n")
    println(logio, "BANKED $path  (exactness $ex)  $(LegendreProductDQ.describe(st))")
    flush(logio)
end

# fallback factor (2026-09-08, user: q=8 has no d4 orbit sidecar): the m-point
# Gauss–Legendre grid on [-1,1]^(d-1) in B_{d-1} orbit form — one orbit per
# multiset of |node| values, weight per node = product of the 1-D weights.
function grid_factor(m)
    dm = d - 1
    x, w = gauss_legendre_rule(m)
    pos = [(x[j], w[j]) for j in eachindex(x) if x[j] > 1e-14]
    w0  = [w[j] for j in eachindex(x) if abs(x[j]) ≤ 1e-14]
    mix = SymmetricDQ.OType[]; vals = Vector{Float64}[]; u = Float64[]
    # counts c_j ≥ 0 over the positive values plus z zeros, Σ c_j + z = dm
    function rec(j, left, counts)
        if j > length(pos)
            z = left
            (z > 0 && isempty(w0)) && return
            mults = Int[]; vs = Float64[]; lu = z * (isempty(w0) ? 0.0 : log(w0[1]))
            for (jj, c) in enumerate(counts)
                c == 0 && continue
                push!(mults, c); push!(vs, pos[jj][1]); lu += c * log(pos[jj][2])
            end
            push!(mix, SymmetricDQ.build_type(mults, z, dm)); push!(vals, vs); push!(u, lu)
            return
        end
        for c in 0:left
            counts[j] = c; rec(j + 1, left - c, counts)
        end
        counts[j] = 0
    end
    rec(1, dm, zeros(Int, length(pos)))
    return SymmetricDQ.MixState(dm, p, mix, vals, u, :legendre)
end

# the (d-1) factor: freshest exact D/B orbit sidecar for (d-1, p)
function factor_state()
    cands = String[]
    for dir in (SYMQ, DONORS), pfx in ("dbest_uniform", "best_uniform")
        f = joinpath(dir, "$(pfx)_d$(d-1)_p$(p).txt")
        isfile(f) && push!(cands, f)
    end
    sort!(cands; by = mtime, rev = true)
    Wf = SymmetricDQ.Work(d - 1, p; halves = true, basis = :legendre)
    GRID && (println(logio, "grid factor requested"); return (grid_factor(M1), "GL$(M1)^$(d-1) grid ($(M1^(d-1)) nodes)"))
    isempty(cands) && (println(logio, "no (d-1) orbit sidecar for d=$(d-1) p=$p — using the GL$(M1)^$(d-1) grid factor");
                       return (grid_factor(M1), "GL$(M1)^$(d-1) grid ($(M1^(d-1)) nodes)"))
    for f in cands
        try
            stf = SymmetricDQ.load_state(f, d - 1, p)
            stf.basis === :legendre || continue
            δf = SymmetricDQ.eval_rj!(zeros(Wf.C), nothing, stf, Wf)
            δf ≤ 1e-11 && return (stf, "$(basename(f)) ($(mix_nodes(stf)) nodes, resid $δf)")
            println(logio, "factor $(basename(f)) is not exact (resid $δf) — skipped")
        catch e
            println(logio, "factor $(basename(f)) unreadable: ", sprint(showerror, e))
        end
    end
    error("no exact (d-1) factor found among $(join(basename.(cands), ", "))")
end

rng = MersenneTwister(seed)
W   = ProdWork(d, p)
deadline = time() + minutes * 60
seg = 0
mcur = M1
while time() < deadline - 30
    global seg += 1
    global best0 = current_best()
    stf, how = factor_state()
    x1, w1 = gauss_legendre_rule(mcur)
    st = product_state(stf, x1, w1)
    if seg == 1 && isfile(STATE)
        try
            sr = LegendreProductDQ.load_state(STATE, d, p)
            δr = LegendreProductDQ.eval_rj!(zeros(W.C), nothing, sr, W)
            if δr ≤ 1e-11 && prod_nodes(sr) < prod_nodes(st)
                println(logio, "resuming from $(basename(STATE)): $(prod_nodes(sr)) nodes (fresh product would be $(prod_nodes(st))), resid $δr")
                st = sr; how = "resumed state"
                global run_best = prod_nodes(sr)
            else
                println(logio, "state file $(basename(STATE)) not used (resid $δr, $(prod_nodes(sr)) nodes vs fresh $(prod_nodes(st)))")
            end
        catch e
            println(logio, "state file unreadable: ", sprint(showerror, e))
        end
    end
    n_start = prod_nodes(st)
    δ0 = LegendreProductDQ.eval_rj!(zeros(W.C), nothing, st, W)
    println(logio, "=== LEGPROD $(d)/$(p) seg $seg seed=$seed m1=$mcur: factor = $how × GL$(mcur) → " *
                   "$(prod_nodes(st)) nodes in $(length(st.mix)) product orbits " *
                   "($(W.C) conditions, $(prod_params(st)) unknowns), resid $δ0, current best $(beststr()) ===")
    flush(logio)
    if δ0 > 1e-11
        println(logio, "product seed is not exact (resid $δ0) — aborting segment")
        break
    end
    if seg > 1 || how == "resumed state"
        for v in st.vals
            v .= clamp.(v .+ 0.02 .* randn(rng, length(v)), 0.0, 1.0)
        end
        for o in eachindex(st.t)
            st.tz[o] || (st.t[o] = clamp(st.t[o] + 0.02 * randn(rng), 0.0, 1.0))
        end
        δj, ok = LegendreProductDQ.solve!(st, W; tol = 1e-12)
        ok || (println(logio, "jittered seed did not re-solve (resid $δj) — using the exact seed");
               st = how == "resumed state" ? LegendreProductDQ.load_state(STATE, d, p) : product_state(stf, x1, w1))
    end
    bank(st, δ0)
    LegendreProductDQ.eliminate!(st, W, rng; tol = 1e-12, jitter_tries = 3, swap_rounds = 20,
                                 log_io = logio, on_improve = (s, δ) -> (persist(s, δ); bank(s, δ)),
                                 deadline = deadline)
    println(logio, "=== seg $seg done: $(LegendreProductDQ.describe(st)), bank best $(current_best()) ===")
    if prod_nodes(st) ≥ n_start
        global mcur = mcur < M1 + 3 ? mcur + 1 : M1
        println(logio, "no reduction from the m1=$(mcur == M1 ? M1 + 3 : mcur - 1) product — next segment seeds with m1=$mcur")
    end
    flush(logio)
end
write(prog, "d=$d p=$p LEGPROD  best $(current_best())  (done)\n")
close(logio)
