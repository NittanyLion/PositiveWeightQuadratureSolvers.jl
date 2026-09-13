#=
LaguerreProductDQ.jl — S_{d-1} × S_1 product-ansatz orbit search for the
product exponential weight exp(-x_1-…-x_d) on the orthant (2026-09-06, user).

WHY: a designed (d-1)-dimensional Laguerre rule times the 1-D Gauss–Laguerre
rule is an exact, positive, much-smaller-than-the-grid start for the blank
d-dimensional cells (La d5 q=11: designed d4 × G11 instead of 11^5).  But it
is only S_{d-1} × S_1 symmetric — the last coordinate is not exchangeable
with the first d-1 — so LaguerreDQ's S_d orbit mix cannot represent it and
symq_lag_tensor.jl cannot descend from it.  This module is the same machinery
(orbit types, LM with box projection, greedy reduction moves) for the
subgroup S_{d-1} × S_1:

  * an orbit is an S_{d-1} orbit type T over the first d-1 coordinates
    (LaguerreDQ.OType built with d-1) with its value vector, PLUS one free
    last coordinate t ≥ 0, PLUS one log-weight u shared by all T.size nodes;
  * the invariant polynomials are (symmetrized monomial in x_1..x_{d-1}) ×
    (power of x_d): conditions are the pairs (λ, m) with λ a partition into
    ≤ d-1 parts and |λ| + m ≤ p — more conditions than S_d's, fewer unknowns
    per node than a free search;
  * the residual factorizes: R_(λ,m) = Σ_o w_o · S_o(λ) · l_m(t_o), so the
    S_{d-1} inner sum is LaguerreDQ's eval_rj! kernel and the 1-D factor is a
    single orthonormal Laguerre value.

`product_state(st, x, w)` builds the start from an S_{d-1} MixState (bank
sidecar best_laguerre_d{d-1}_p{p}.txt, or LaguerreDQ.tensor_state) and a 1-D
rule (x, w): one product orbit per (orbit, 1-D node).  `eliminate!` is the
same greedy descent as LaguerreDQ.eliminate! (drop / zero / merge / swap),
`expand_rule` returns the full node set for the bank gate.
=#
module LaguerreProductDQ

using LinearAlgebra, Random, Printf
using ..LaguerreDQ: OType, build_type, orbit_types, partitions_le, poly_cols!,
                    MixState, gauss_laguerre_rule, tensor_state, jscale,
                    basis_vmax, random_values

# describe / eval_rj! / solve! / eliminate! / expand_rule / save_state / load_state
# are deliberately NOT exported: LaguerreDQ exports the same names, and a
# worker that uses both modules must qualify them (LaguerreProductDQ.solve!)
export ProdState, ProdWork, prod_nodes, prod_params, product_state

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
mutable struct ProdState
    d::Int
    p::Int
    mix::Vector{OType}              # S_{d-1} orbit types (built with d-1)
    vals::Vector{Vector{Float64}}   # per orbit, length k — nonnegative values
    t::Vector{Float64}              # per orbit, the last coordinate ≥ 0
    u::Vector{Float64}              # log weight per orbit (weight per node)
    alpha::Float64
end

prod_nodes(st::ProdState)  = sum(T.size for T in st.mix; init = 0)
prod_params(st::ProdState) = sum(T.k + 2 for T in st.mix; init = 0)

Base.copy(st::ProdState) = ProdState(st.d, st.p, copy(st.mix),
                                     [copy(v) for v in st.vals], copy(st.t),
                                     copy(st.u), st.alpha)

function describe(st::ProdState)
    # group the orbits by S_{d-1} signature: "1+1|1z×3(6)" = three product
    # orbits of that type, 6 nodes each
    cnt = Dict{String,Tuple{Int,Int}}()
    for T in st.mix
        sig = isempty(T.mults) ? "origin" :
              join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "")
        c, s = get(cnt, sig, (0, T.size))
        cnt[sig] = (c + 1, s)
    end
    parts = ["$(sig)×$(c)($(s))" for (sig, (c, s)) in sort(collect(cnt); by = x -> -x[2][2])]
    return "$(prod_nodes(st)) nodes in $(length(st.mix)) product orbits = " * join(parts, " ")
end

# ---------------------------------------------------------------------------
# conditions: (λ padded to d-1, m), all |λ| + m ≤ p; (0, 0) first
# ---------------------------------------------------------------------------
struct ProdWork
    lam::Vector{Vector{Int}}    # length d-1 each
    m::Vector{Int}
    C::Int
    alpha::Float64
end
function ProdWork(d::Int, p::Int; alpha::Real = 0.0)
    d ≥ 2 || throw(ArgumentError("the product ansatz needs d ≥ 2"))
    alpha > -1 || throw(ArgumentError("Laguerre parameter must exceed -1"))
    lam = Vector{Int}[]; ms = Int[]
    for n in 0:p, m in 0:n, λ in partitions_le(n - m, d - 1)
        push!(lam, vcat(λ, zeros(Int, d - 1 - length(λ)))); push!(ms, m)
    end
    ProdWork(lam, ms, length(ms), Float64(alpha))
end

# ---------------------------------------------------------------------------
# residual and Jacobian.  Parameter layout per orbit: vals (k), t, u.
# ---------------------------------------------------------------------------
function eval_rj!(R::Vector{Float64}, J::Union{Nothing,Matrix{Float64}},
                  st::ProdState, W::ProdWork)
    d, p = st.d, st.p
    dm = d - 1
    st.alpha == W.alpha ||
        throw(ArgumentError("state alpha $(st.alpha) ≠ work alpha $(W.alpha)"))
    fill!(R, 0.0)
    J === nothing || fill!(J, 0.0)
    kmax = maximum(T.k for T in st.mix; init = 0)
    H  = zeros(p + 1, kmax + 2)      # col 1: x = 0; cols 2..k+1: vals; col k+2: t
    Hd = zeros(p + 1, kmax + 2)
    f   = zeros(dm)
    pre = zeros(dm + 1)
    suf = zeros(dm + 1)
    dS  = zeros(max(kmax, 1))
    off = 0
    for (o, T) in enumerate(st.mix)
        w = exp(st.u[o])
        poly_cols!(H, Hd, 1, 0.0, p, W.alpha)
        for j in 1:T.k
            poly_cols!(H, Hd, j + 1, st.vals[o][j], p, W.alpha)
        end
        tc = T.k + 2
        poly_cols!(H, Hd, tc, st.t[o], p, W.alpha)
        for c in 1:W.C
            λ = W.lam[c]; m = W.m[c]
            S = 0.0
            T.k > 0 && fill!(dS, 0.0)
            for r in axes(T.asg, 1)
                @inbounds for i in 1:dm
                    f[i] = H[λ[i] + 1, T.asg[r, i] + 1]
                end
                pre[1] = 1.0
                @inbounds for i in 1:dm
                    pre[i+1] = pre[i] * f[i]
                end
                S += pre[dm+1]
                if J !== nothing && T.k > 0
                    suf[dm+1] = 1.0
                    @inbounds for i in dm:-1:1
                        suf[i] = suf[i+1] * f[i]
                    end
                    @inbounds for i in 1:dm
                        j = T.asg[r, i]
                        (j == 0 || λ[i] == 0) && continue
                        dS[j] += pre[i] * suf[i+1] * Hd[λ[i] + 1, j + 1]
                    end
                end
            end
            lt = H[m + 1, tc]
            R[c] += w * S * lt
            if J !== nothing
                for j in 1:T.k
                    J[c, off + j] += w * dS[j] * lt
                end
                J[c, off + T.k + 1] += w * S * Hd[m + 1, tc]   # d/dt
                J[c, off + T.k + 2] += w * S * lt              # d/du of exp(u)·(...)
            end
        end
        off += T.k + 2
    end
    R[1] -= 1.0
    return norm(R)
end

# ---------------------------------------------------------------------------
# LM (box-projected on vals and t) + minimum-norm Newton polish — the
# LaguerreDQ.solve! loop verbatim over this state's pack/unpack/mask
# ---------------------------------------------------------------------------
function pack!(θ, st::ProdState)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            θ[off + j] = st.vals[o][j]
        end
        θ[off + T.k + 1] = st.t[o]
        θ[off + T.k + 2] = st.u[o]
        off += T.k + 2
    end
    return θ
end

function unpack!(st::ProdState, θ)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            st.vals[o][j] = θ[off + j]
        end
        st.t[o] = θ[off + T.k + 1]
        st.u[o] = clamp(θ[off + T.k + 2], -690.0, 5.0)   # 2026-09-07: was -46, see LaguerreDQ.unpack!
        off += T.k + 2
    end
    return st
end

function value_mask(st::ProdState)   # coordinates that live in [0, vmax]
    mask = falses(prod_params(st))
    off = 0
    for T in st.mix
        for j in 1:T.k + 1
            mask[off + j] = true
        end
        off += T.k + 2
    end
    return mask
end

function solve!(st::ProdState, W::ProdWork; tol::Float64 = 1e-12,
                maxiter::Int = 600, λ0::Float64 = 1e-3,
                vmax::Float64 = basis_vmax(st.p, W.alpha))
    C = W.C
    N = prod_params(st)
    N == 0 && return (Inf, false)
    θ  = pack!(zeros(N), st)
    θt = similar(θ)
    R  = zeros(C); Rt = zeros(C)
    J  = zeros(C, N)
    st2 = copy(st)
    isval = value_mask(st)

    δ = eval_rj!(R, J, st, W)
    colmax = maximum(sum(abs2, J, dims = 1))
    λ = λ0 * max(colmax, 1e-12)
    ν = 2.0
    for _ in 1:maxiter
        δ ≤ tol && break
        g = J' * R
        accepted = false
        for _ in 1:40
            dθ = try
                if C < N
                    A = J * J'
                    @inbounds for i in 1:C
                        A[i, i] += λ
                    end
                    -(J' * (cholesky!(Symmetric(A)) \ R))
                else
                    A = J' * J
                    @inbounds for i in 1:N
                        A[i, i] += λ
                    end
                    -(cholesky!(Symmetric(A)) \ g)
                end
            catch e
                e isa LinearAlgebra.PosDefException || rethrow()
                λ *= ν; ν *= 2.0
                λ > 1e14 && break
                continue
            end
            θt .= θ .+ dθ
            @inbounds for i in eachindex(θt)
                isval[i] && (θt[i] = clamp(θt[i], 0.0, vmax))
            end
            dθ = θt .- θ
            unpack!(st2, θt)
            δn = eval_rj!(Rt, nothing, st2, W)
            pred = δ^2 - sum(abs2, R .+ J * dθ)
            ρ = pred > 0 ? (δ^2 - δn^2) / pred : -1.0
            if δn < δ && ρ > 1e-4
                θ .= θt
                unpack!(st, θ); pack!(θ, st)
                λ *= max(1/3, 1 - (2ρ - 1)^3)
                ν = 2.0
                δ = eval_rj!(R, J, st, W)
                accepted = true
                break
            else
                λ *= ν; ν *= 2.0
                (λ > 1e14 || ν > 1e8) && break
            end
        end
        accepted || break
    end

    if δ < 1e-6
        θbest = copy(θ); δbest = δ
        cutfacs = (1e-10, 1e-12, 1e-14)
        ci = 1
        for _ in 1:40
            eval_rj!(R, J, st, W)
            F = try
                svd(J)
            catch e
                e isa LinearAlgebra.LAPACKException || rethrow()
                break
            end
            thresh = cutfacs[ci] * F.S[1]
            dθp = F.V * [σ > thresh ? y / σ : 0.0
                         for (y, σ) in zip(F.U' * R, F.S)]
            θ .-= dθp
            @inbounds for i in eachindex(θ)
                isval[i] && (θ[i] = clamp(θ[i], 0.0, vmax))
            end
            unpack!(st, θ)
            δn = eval_rj!(R, nothing, st, W)
            if δn < δbest
                δbest = δn
                θbest .= θ
                δn ≤ 1e-15 && break
            else
                θ .= θbest
                unpack!(st, θ)
                ci += 1
                ci > length(cutfacs) && break
            end
        end
        θ .= θbest
        unpack!(st, θ)
        δ = eval_rj!(R, nothing, st, W)
    end
    return (δ, δ ≤ tol)
end

# ---------------------------------------------------------------------------
# the product start
# ---------------------------------------------------------------------------
"""
    product_state(st::MixState, x, w) -> ProdState

The product of an S_{d-1} orbit mix `st` (dimension d-1, degree p) with the
1-D rule (x, w), as an S_{d-1} × S_1 state of dimension d: one product orbit
per (orbit of st, node of x), value vector copied, t = x_j, log-weight
u_o + log w_j.  If st is exact to degree p and (x, w) is exact to degree p,
the product is exact to total degree p with positive weights — a feasible
start, verified by the caller.
"""
function product_state(st::MixState, x::AbstractVector, w::AbstractVector)
    length(x) == length(w) || throw(ArgumentError("1-D rule: nodes and weights differ in length"))
    d = st.d + 1
    mix = OType[]; vals = Vector{Float64}[]; t = Float64[]; u = Float64[]
    for (o, T) in enumerate(st.mix), j in eachindex(x)
        push!(mix, T); push!(vals, copy(st.vals[o])); push!(t, x[j]); push!(u, st.u[o] + log(w[j]))
    end
    return ProdState(d, st.p, mix, vals, t, u, st.alpha)
end

# ---------------------------------------------------------------------------
# reduction moves: drop an orbit, zero a value, merge two values
# (the S_{d-1} moves of LaguerreDQ.apply_move; t rides along unchanged)
# ---------------------------------------------------------------------------
function apply_move(st::ProdState, move)
    kind, o, arg = move
    st2 = copy(st)
    dm = st.d - 1
    if kind === :drop
        deleteat!(st2.mix, o); deleteat!(st2.vals, o); deleteat!(st2.t, o); deleteat!(st2.u, o)
    elseif kind === :zero
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])
        deleteat!(m2, arg); deleteat!(v2, arg)
        st2.mix[o] = build_type(m2, T.z + T.mults[arg], dm)
        st2.vals[o] = v2
    elseif kind === :merge
        a, b = arg
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])
        vnew = (T.mults[a] * v2[a] + T.mults[b] * v2[b]) / (T.mults[a] + T.mults[b])
        m2[a] += m2[b]; v2[a] = vnew
        deleteat!(m2, b); deleteat!(v2, b)
        st2.mix[o] = build_type(m2, T.z, dm)
        st2.vals[o] = v2
    end
    return st2
end

function reduction_moves(st::ProdState)
    moves = Tuple{Symbol,Int,Any}[]
    pr = Float64[]
    for o in eachindex(st.mix)
        push!(moves, (:drop, o, nothing))
        push!(pr, exp(st.u[o]) * st.mix[o].size)
    end
    n0 = length(moves)
    for o in eachindex(st.mix)
        T = st.mix[o]; v = st.vals[o]
        T.k == 0 && continue
        jm = argmin(v)
        push!(moves, (:zero, o, jm)); push!(pr, v[jm])
        if T.k ≥ 2
            best = (Inf, 0, 0)
            for a in 1:T.k, b in (a+1):T.k
                gp = abs(v[a] - v[b])
                gp < best[1] && (best = (gp, a, b))
            end
            push!(moves, (:merge, o, (best[2], best[3]))); push!(pr, best[1])
        end
    end
    o1 = sortperm(pr[1:n0])
    o2 = n0 .+ sortperm(pr[(n0+1):end])
    return moves[vcat(o1, o2)]
end

"""
    eliminate!(st, W, rng; tol, jitter_tries, swap_rounds, log_io,
               on_improve, deadline, max_tries) -> st

Greedy reduction, as LaguerreDQ.eliminate!: try the ranked moves, reconverge
warm (plus jittered retries), accept the first success; when blocked, swap
an orbit's S_{d-1} type for a strictly smaller one with fresh values.
`max_tries` caps the moves attempted per round (the ranked list is one drop
per orbit — thousands on a product start — and each try is a full LM solve).
"""
function eliminate!(st::ProdState, W::ProdWork, rng::AbstractRNG;
                    tol::Float64 = 1e-12, jitter_tries::Int = 1,
                    swap_rounds::Int = 12, log_io = nothing,
                    on_improve = nothing, deadline::Float64 = Inf,
                    max_tries::Int = 200)
    dm = st.d - 1
    types = orbit_types(dm)
    js = jscale(st.p, W.alpha)
    while true
        time() < deadline || break
        improved = false
        tries = 0
        for move in reduction_moves(st)
            prod_nodes(st) ≤ 1 && break
            time() < deadline || break
            tries += 1
            tries > max_tries && break
            cand = apply_move(st, move)
            prod_params(cand) == 0 && continue
            δ, ok = solve!(cand, W; tol)
            if !ok
                for _ in 1:jitter_tries
                    c2 = copy(cand)
                    for v in c2.vals
                        v .= abs.(v .+ (0.02 * js) .* randn(rng, length(v)))
                    end
                    c2.t .= abs.(c2.t .+ (0.02 * js) .* randn(rng, length(c2.t)))
                    δ, ok = solve!(c2, W; tol)
                    ok && (cand = c2; break)
                end
            end
            ok || continue
            st.mix = cand.mix; st.vals = cand.vals; st.t = cand.t; st.u = cand.u
            improved = true
            log_io === nothing ||
                println(log_io, "    -$(move[1]) → $(describe(st))")
            on_improve === nothing || on_improve(st, δ)
            break
        end
        if !improved
            wt = [exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix)]
            ord = sortperm(wt)
            for _ in 1:swap_rounds
                time() < deadline || break
                o = ord[min(1 + floor(Int, abs(randn(rng)) * 2), length(ord))]
                smaller = [T for T in types if T.size < st.mix[o].size]
                isempty(smaller) && continue
                T2 = smaller[rand(rng, 1:length(smaller))]
                cand = copy(st)
                cand.mix[o] = T2
                cand.vals[o] = random_values(rng, T2.k, st.p, W.alpha)
                cand.u[o] = st.u[o] + log(st.mix[o].size / max(T2.size, 1))
                δ, ok = solve!(cand, W; tol)
                ok || continue
                st.mix = cand.mix; st.vals = cand.vals; st.t = cand.t; st.u = cand.u
                improved = true
                log_io === nothing ||
                    println(log_io, "    -swap → $(describe(st))")
                on_improve === nothing || on_improve(st, δ)
                break
            end
        end
        improved || break
    end
    return st
end

# ---------------------------------------------------------------------------
# expansion, persistence
# ---------------------------------------------------------------------------
"""Expand to the full node set (all distinct permutations of the first d-1
coordinates, last coordinate t), weights per node."""
function expand_rule(st::ProdState)
    d = st.d; dm = d - 1
    N = prod_nodes(st)
    nodes = zeros(N, d)
    w = zeros(N)
    row = 0
    for (o, T) in enumerate(st.mix)
        wo = exp(st.u[o])
        for r in axes(T.asg, 1)
            row += 1
            for i in 1:dm
                j = T.asg[r, i]
                nodes[row, i] = j == 0 ? 0.0 : st.vals[o][j]
            end
            nodes[row, d] = st.t[o]
            w[row] = wo
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::ProdState, δ::Float64)
    open(path, "w") do io
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(prod_nodes(st)) resid=$δ group=S(d-1)xS1 alpha=$(st.alpha)")
        for (o, T) in enumerate(st.mix)
            println(io, join(T.mults, ","), "|", T.z, "|",
                    join(st.vals[o], ","), "|", st.t[o], "|", st.u[o])
        end
    end
end

function load_state(path::AbstractString, d::Int, p::Int)
    mix = OType[]; vals = Vector{Float64}[]; t = Float64[]; u = Float64[]
    a = 0.0
    for line in eachline(path)
        if startswith(line, "#")
            m = match(r"alpha=([-0-9.eE+]+)", line)
            m === nothing || (a = parse(Float64, m[1]))
            continue
        end
        isempty(strip(line)) && continue
        parts = split(line, "|")
        m = isempty(parts[1]) ? Int[] : parse.(Int, split(parts[1], ","))
        z = parse(Int, parts[2])
        v = isempty(parts[3]) ? Float64[] : parse.(Float64, split(parts[3], ","))
        push!(mix, build_type(m, z, d - 1))
        push!(vals, v)
        push!(t, parse(Float64, parts[4]))
        push!(u, parse(Float64, parts[5]))
    end
    return ProdState(d, p, mix, vals, t, u, a)
end

end # module
