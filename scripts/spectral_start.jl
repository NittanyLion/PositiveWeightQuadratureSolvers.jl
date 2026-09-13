# Spectral-start worker: attack one (d, p) at a FIXED node count n from
# Vioreanu-Rokhlin spectral candidates (SpectralInit.jl) instead of random
# draws -- litcheck/STRATEGIES_NOT_YET_EMPLOYED.md item 1, adopted
# 2026-08-31.  Each outer attempt draws a fresh random operator combination
# (a new, equally principled candidate set) and tries it raw plus a few
# jittered retries through the free V2 solver.  A converged rule is
# verified at the campaign gate (relative monomial exactness <= 1e-11);
# it is banked ONLY when strictly below the banked best -- ties are
# printed, never banked.
#
# The solve supports basis hermite | legendre (the free solver's bases);
# SpectralInit also produces :laguerre candidates, but those go to the
# S_d orbit tools, not here.
#
# usage: julia spectral_start.jl <d> <p> <n> <hermite|legendre> [minutes] [seed]
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "DesignedQuadratureV2.jl"))
include(joinpath(@__DIR__, "..", "src", "SpectralInit.jl"))
using .DesignedQuadrature, .DesignedQuadratureV2, .SpectralInit
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
const n       = parse(Int, ARGS[3])
const basis   = Symbol(ARGS[4])
const minutes = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 10.0
const seed    = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
basis in (:hermite, :legendre) ||
    (println("basis must be hermite or legendre (V2's free bases)"); exit(1))

const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const EXTOL = 1e-11
const prefix = String(basis)

bank_best() = minimum([parse(Int, m[1]) for f in readdir(RULES)
                       for m in [match(Regex("^$(prefix)_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)]
                       if m !== nothing]; init = typemax(Int))

rng = Xoshiro(1009 * seed + 31 * d + p)
best0 = bank_best()
@printf("spectral-start %s d=%d p=%d at n=%d (bank best %s), %.0f min, seed %d\n",
        prefix, d, p, n, best0 == typemax(Int) ? "none" : string(best0),
        minutes, seed)

deadline = time() + minutes * 60
attempt = 0
while time() < deadline
    global attempt += 1
    sp = spectral_nodes(d, p, n; basis, rng)
    scale = max(maximum(abs, sp.nodes), 0.1)
    for jit in (0.0, 0.005, 0.015, 0.04)
        time() > deadline && break
        b0 = jit == 0 ? copy(sp.nodes) :
             sp.nodes .+ (jit * scale) .* randn(rng, size(sp.nodes)...)
        r = designed_quadrature_v2(d, p, n; basis, symmetric = false,
                                   init_pairs = (b0, copy(sp.weights)),
                                   tol = 1e-11, maxiter = 3000,
                                   stall_window = 300,
                                   abort_iters = 500, abort_resid = 1e-3)
        if r.converged && minimum(r.weights) > 0
            ex = verify_exactness(r.nodes, r.weights, p; basis, relative = true)
            inbox = basis === :legendre ?
                all(0.0 .<= r.nodes .<= 1.0) : true
            if ex <= EXTOL && inbox
                @printf("attempt %d (jit %.3f): CONVERGED n=%d, exactness %.2e\n",
                        attempt, jit, n, ex)
                if n < bank_best()
                    out = joinpath(RULES, "$(prefix)_d$(d)_p$(p)_n$(n).csv")
                    open(out, "w") do io
                        for i in 1:n
                            println(io, join(string.([r.nodes[i, :]; r.weights[i]]), ","))
                        end
                    end
                    # no parent rule: a spectral initialization solved from scratch
                    symq_lineage!(basename(out), "none",
                                  "spectral initialization then free solve at n=$n (spectral_start.jl seed $seed, basis $basis)";
                                  rules = RULES)
                    println("BANKED $out")
                else
                    println("ties the bank -- not banked")
                end
                exit(0)
            end
        end
    end
    attempt % 5 == 0 &&
        @printf("  ... %d spectral draws tried, best so far not converged\n", attempt)
end
println("done: no verified rule at n=$n in $(round(minutes)) min ($attempt spectral draws)")
