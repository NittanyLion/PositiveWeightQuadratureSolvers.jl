#=
LegendreProductDQ.jl — D_{d-1} × B_1 product-ansatz orbit descent for the
UNIFORM weight on [-1,1]^d (2026-09-08, user: "brilliant ideas re Le d=5
q=9,10,11 — go for it").

WHY: Le d5 q9/q10/q11 hold a 7200-node ladder output and two tensor
placeholders (1612 × 10, 2774 × 11) against GH's 3986 / 7174 / 13199, and the
pair elimination that would move them is a 50 GB system.  The La column was
in the same place a week ago and the product-orbit descent
(LaguerreProductDQ.jl) is what moved it: a designed (d-1) factor times the
1-D Gauss rule is exact, positive, and lives in a product-orbit space where
one LM solve is milliseconds instead of minutes.  This is the same module for
the cube: the (d-1) factor is a B_{d-1} / D_{d-1} orbit state
(SymmetricDQ.jl, basis :legendre — the dbest_uniform_d{d-1}_p{p}.txt sidecar),
the 1-D factor is the m-point Gauss–Legendre rule, and every product orbit
carries one free last coordinate t ≥ 0 (the B_1 orbit {t, -t}, or the single
point 0 when `tz`).

  * conditions: (λ, m) with λ a D_{d-1} invariant of SymmetricDQ.conditions
    (even partitions, plus the all-odd vectors for half-orbits) and m EVEN,
    |λ| + m ≤ p — odd m vanish on {t, -t} by symmetry;
  * residual: R_(λ,m) = Σ_o w_o · signfac_o · [par_o] · S_o(λ) · (2 ℓ_m(t_o)
    or ℓ_m(0)), with S_o the SymmetricDQ kernel and ℓ_m orthonormal Legendre;
  * moves: drop, zero a value, merge two values, t → 0 (halves the orbit),
    B-orbit → D half-orbit (halves the orbit), and type swaps.

Frame: everything here is on [-1,1]^d (SymmetricDQ's :legendre frame); the
worker maps to [0,1]^d for the bank exactly as symq_run.jl does.
=#
module LegendreProductDQ

using LinearAlgebra, Random, Printf
using ..SymmetricDQ: OType, build_type, orbit_types, partitions_le, poly_cols!,
                     MixState, conditions, random_values

export ProdState, ProdWork, prod_nodes, prod_params, product_state, gauss_legendre_rule

# ---------------------------------------------------------------------------
# 1-D Gauss–Legendre on [-1,1] for the uniform PROBABILITY measure (Σw = 1)
# ---------------------------------------------------------------------------
function gauss_legendre_rule(m::Int)
    m ≥ 1 || throw(ArgumentError("need m ≥ 1"))
    β = [n / sqrt(4n^2 - 1) for n in 1:m-1]
    Jm = SymTridiagonal(zeros(m), β)
    F = eigen(Jm)
    x = F.values
    w = F.vectors[1, :] .^ 2
    return x, w ./ sum(w)
end

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
mutable struct ProdState
    d::Int
    p::Int
    mix::Vector{OType}              # D_{d-1}/B_{d-1} orbit types (built with d-1)
    vals::Vector{Vector{Float64}}   # per orbit, length k — values in [0,1]
    t::Vector{Float64}              # per orbit, the last coordinate ≥ 0 (unused when tz)
    tz::Vector{Bool}                # last coordinate structurally 0
    u::Vector{Float64}              # log weight per orbit (weight per node)
end

prod_nodes(st::ProdState)  = sum(T.size * (st.tz[o] ? 1 : 2) for (o, T) in enumerate(st.mix); init = 0)
prod_params(st::ProdState) = sum(T.k + (st.tz[o] ? 1 : 2) for (o, T) in enumerate(st.mix); init = 0)

Base.copy(st::ProdState) = ProdState(st.d, st.p, copy(st.mix),
                                     [copy(v) for v in st.vals], copy(st.t),
                                     copy(st.tz), copy(st.u))

function describe(st::ProdState)
    cnt = Dict{String,Tuple{Int,Int}}()
    for (o, T) in enumerate(st.mix)
        sig = (isempty(T.mults) ? "origin" :
               join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "") *
               (T.par == 1 ? "+" : T.par == -1 ? "-" : "")) * (st.tz[o] ? "×0" : "×±")
        c, s = get(cnt, sig, (0, T.size * (st.tz[o] ? 1 : 2)))
        cnt[sig] = (c + 1, s)
    end
    parts = ["$(sig)×$(c)($(s))" for (sig, (c, s)) in sort(collect(cnt); by = x -> -x[2][2])]
    return "$(prod_nodes(st)) nodes in $(length(st.mix)) product orbits = " * join(parts, " ")
end

# ---------------------------------------------------------------------------
# conditions: (λ padded to d-1, m even), |λ| + m ≤ p; (0, 0) first
# ---------------------------------------------------------------------------
struct ProdWork
    lam::Vector{Vector{Int}}    # length d-1 each
    m::Vector{Int}
    odd::Vector{Bool}           # all-odd λ (D_{d-1} condition)
    C::Int
end
function ProdWork(d::Int, p::Int)
    d ≥ 2 || throw(ArgumentError("the product ansatz needs d ≥ 2"))
    lam = Vector{Int}[]; ms = Int[]; odd = Bool[]
    for m in 0:2:p, λ in conditions(d - 1, p - m; halves = true)
        push!(lam, λ); push!(ms, m); push!(odd, isodd(λ[1]))
    end
    ProdWork(lam, ms, odd, length(ms))
end

# ---------------------------------------------------------------------------
# residual and Jacobian.  Parameter layout per orbit: vals (k), [t], u.
# ---------------------------------------------------------------------------
function eval_rj!(R::Vector{Float64}, J::Union{Nothing,Matrix{Float64}},
                  st::ProdState, W::ProdWork)
    d, p = st.d, st.p
    dm = d - 1
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
        poly_cols!(H, Hd, 1, 0.0, p, :legendre)
        for j in 1:T.k
            poly_cols!(H, Hd, j + 1, st.vals[o][j], p, :legendre)
        end
        tc = T.k + 2
        st.tz[o] || poly_cols!(H, Hd, tc, st.t[o], p, :legendre)
        for c in 1:W.C
            λ = W.lam[c]; m = W.m[c]
            fac = w * T.signfac
            if W.odd[c]
                T.par == 0 && continue
                fac *= T.par
            end
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
            if st.tz[o]
                lt = H[m + 1, 1]; ltd = 0.0
            else
                lt = 2 * H[m + 1, tc]; ltd = 2 * Hd[m + 1, tc]
            end
            R[c] += fac * S * lt
            if J !== nothing
                for j in 1:T.k
                    J[c, off + j] += fac * dS[j] * lt
                end
                if !st.tz[o]
                    J[c, off + T.k + 1] += fac * S * ltd          # d/dt
                    J[c, off + T.k + 2] += fac * S * lt           # d/du
                else
                    J[c, off + T.k + 1] += fac * S * lt           # d/du
                end
            end
        end
        off += T.k + (st.tz[o] ? 1 : 2)
    end
    R[1] -= 1.0
    return norm(R)
end

# ---------------------------------------------------------------------------
# LM (box-projected on vals and t to [0,1]) + minimum-norm Newton polish —
# the LaguerreProductDQ.solve! loop over this state's pack/unpack/mask
# ---------------------------------------------------------------------------
function pack!(θ, st::ProdState)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            θ[off + j] = st.vals[o][j]
        end
        if st.tz[o]
            θ[off + T.k + 1] = st.u[o]; off += T.k + 1
        else
            θ[off + T.k + 1] = st.t[o]; θ[off + T.k + 2] = st.u[o]; off += T.k + 2
        end
    end
    return θ
end

function unpack!(st::ProdState, θ)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            st.vals[o][j] = θ[off + j]
        end
        if st.tz[o]
            st.u[o] = clamp(θ[off + T.k + 1], -690.0, 5.0); off += T.k + 1
        else
            st.t[o] = θ[off + T.k + 1]
            st.u[o] = clamp(θ[off + T.k + 2], -690.0, 5.0); off += T.k + 2
        end
    end
    return st
end

function value_mask(st::ProdState)   # coordinates that live in [0, 1]
    mask = falses(prod_params(st))
    off = 0
    for (o, T) in enumerate(st.mix)
        nv = T.k + (st.tz[o] ? 0 : 1)
        for j in 1:nv
            mask[off + j] = true
        end
        off += T.k + (st.tz[o] ? 1 : 2)
    end
    return mask
end

function solve!(st::ProdState, W::ProdWork; tol::Float64 = 1e-12,
                maxiter::Int = 600, λ0::Float64 = 1e-3, vmax::Float64 = 1.0)
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

The product of a B_{d-1}/D_{d-1} orbit mix `st` (dimension d-1, degree p,
:legendre frame) with the symmetric 1-D rule (x, w) on [-1,1]: one product
orbit per (orbit of st, node x_j ≥ 0) — the pair {x_j, -x_j} as t = x_j, or
the point 0 as tz — log-weight u_o + log w_j.  Exact to total degree p with
positive weights when both factors are.
"""
function product_state(st::MixState, x::AbstractVector, w::AbstractVector)
    length(x) == length(w) || throw(ArgumentError("1-D rule: nodes and weights differ in length"))
    st.basis === :legendre || throw(ArgumentError("the factor must be a :legendre state"))
    d = st.d + 1
    mix = OType[]; vals = Vector{Float64}[]; t = Float64[]; tz = Bool[]; u = Float64[]
    for (o, T) in enumerate(st.mix), j in eachindex(x)
        x[j] < -1e-14 && continue                    # the negative twin is implied
        push!(mix, T); push!(vals, copy(st.vals[o]))
        if x[j] ≤ 1e-14
            push!(t, 0.0); push!(tz, true)
        else
            push!(t, x[j]); push!(tz, false)
        end
        push!(u, st.u[o] + log(w[j]))
    end
    return ProdState(d, st.p, mix, vals, t, tz, u)
end

# ---------------------------------------------------------------------------
# reduction moves
# ---------------------------------------------------------------------------
function apply_move(st::ProdState, move)
    kind, o, arg = move
    st2 = copy(st)
    dm = st.d - 1
    if kind === :drop
        deleteat!(st2.mix, o); deleteat!(st2.vals, o); deleteat!(st2.t, o)
        deleteat!(st2.tz, o); deleteat!(st2.u, o)
    elseif kind === :tzero
        st2.tz[o] = true; st2.t[o] = 0.0
        st2.u[o] = st.u[o] + log(2.0)      # two nodes become one: keep the orbit mass
    elseif kind === :half
        T = st.mix[o]
        st2.mix[o] = build_type(copy(T.mults), 0, dm; par = arg)
        st2.u[o] = st.u[o] + log(2.0)
    elseif kind === :zero
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])
        deleteat!(m2, arg); deleteat!(v2, arg)
        st2.mix[o] = build_type(m2, T.z + T.mults[arg], dm)      # a zero coordinate ends a half-orbit
        st2.vals[o] = v2
    elseif kind === :merge
        a, b = arg
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])
        vnew = (T.mults[a] * v2[a] + T.mults[b] * v2[b]) / (T.mults[a] + T.mults[b])
        m2[a] += m2[b]; v2[a] = vnew
        deleteat!(m2, b); deleteat!(v2, b)
        st2.mix[o] = build_type(m2, T.z, dm; par = T.par)
        st2.vals[o] = v2
    end
    return st2
end

function reduction_moves(st::ProdState)
    moves = Tuple{Symbol,Int,Any}[]
    pr = Float64[]
    for (o, T) in enumerate(st.mix)
        push!(moves, (:drop, o, nothing))
        push!(pr, exp(st.u[o]) * T.size * (st.tz[o] ? 1 : 2))
    end
    n0 = length(moves)
    for (o, T) in enumerate(st.mix)
        st.tz[o] || (push!(moves, (:tzero, o, nothing)); push!(pr, st.t[o]))
        if T.par == 0 && T.z == 0 && T.k > 0
            for par in (1, -1)
                push!(moves, (:half, o, par)); push!(pr, 0.5 + exp(st.u[o]) * T.size)
            end
        end
        T.k == 0 && continue
        v = st.vals[o]
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

Greedy reduction (LaguerreProductDQ.eliminate! with the two extra halving
moves): try the ranked moves, reconverge warm (plus jittered retries), accept
the first success; when blocked, swap an orbit's type for a strictly smaller
one with fresh values.
"""
function eliminate!(st::ProdState, W::ProdWork, rng::AbstractRNG;
                    tol::Float64 = 1e-12, jitter_tries::Int = 1,
                    swap_rounds::Int = 12, log_io = nothing,
                    on_improve = nothing, deadline::Float64 = Inf,
                    max_tries::Int = 200)
    dm = st.d - 1
    types = orbit_types(dm; halves = true)
    js = 0.02
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
                        v .= clamp.(v .+ js .* randn(rng, length(v)), 0.0, 1.0)
                    end
                    for o in eachindex(c2.t)
                        c2.tz[o] || (c2.t[o] = clamp(c2.t[o] + js * randn(rng), 0.0, 1.0))
                    end
                    δ, ok = solve!(c2, W; tol)
                    ok && (cand = c2; break)
                end
            end
            ok || continue
            st.mix = cand.mix; st.vals = cand.vals; st.t = cand.t; st.tz = cand.tz; st.u = cand.u
            improved = true
            log_io === nothing ||
                (println(log_io, "    -$(move[1]) → $(describe(st))"); flush(log_io))
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
                cand.vals[o] = random_values(rng, T2.k, st.p; basis = :legendre)
                cand.u[o] = st.u[o] + log(st.mix[o].size / max(T2.size, 1))
                δ, ok = solve!(cand, W; tol)
                ok || continue
                st.mix = cand.mix; st.vals = cand.vals; st.t = cand.t; st.tz = cand.tz; st.u = cand.u
                improved = true
                log_io === nothing ||
                    (println(log_io, "    -swap → $(describe(st))"); flush(log_io))
                on_improve === nothing || on_improve(st, δ)
                break
            end
        end
        improved || break
    end
    return st
end

# ---------------------------------------------------------------------------
# expansion (on [-1,1]^d), persistence
# ---------------------------------------------------------------------------
function expand_rule(st::ProdState)
    d = st.d; dm = d - 1
    N = prod_nodes(st)
    nodes = zeros(N, d)
    w = zeros(N)
    row = 0
    for (o, T) in enumerate(st.mix)
        wo = exp(st.u[o])
        nz = sum(T.mults)
        tvals = st.tz[o] ? (0.0,) : (st.t[o], -st.t[o])
        for r in axes(T.asg, 1)
            pos = [i for i in 1:dm if T.asg[r, i] > 0]
            for smask in 0:(2^nz - 1)
                T.par != 0 && (iseven(count_ones(smask)) ? 1 : -1) != T.par && continue
                for tv in tvals
                    row += 1
                    for i in 1:dm
                        j = T.asg[r, i]
                        nodes[row, i] = j == 0 ? 0.0 : st.vals[o][j]
                    end
                    for (b, i) in enumerate(pos)
                        ((smask >> (b - 1)) & 1) == 1 && (nodes[row, i] *= -1)
                    end
                    nodes[row, d] = tv
                    w[row] = wo
                end
            end
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::ProdState, δ::Float64)
    open(path, "w") do io
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(prod_nodes(st)) resid=$δ group=D(d-1)xB1 basis=legendre")
        for (o, T) in enumerate(st.mix)
            println(io, join(T.mults, ","), "|", T.z, "|",
                    join(st.vals[o], ","), "|", st.t[o], "|", st.tz[o] ? 1 : 0, "|", st.u[o], "|", T.par)
        end
    end
end

function load_state(path::AbstractString, d::Int, p::Int)
    mix = OType[]; vals = Vector{Float64}[]; t = Float64[]; tz = Bool[]; u = Float64[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        parts = split(line, "|")
        m = isempty(parts[1]) ? Int[] : parse.(Int, split(parts[1], ","))
        z = parse(Int, parts[2])
        v = isempty(parts[3]) ? Float64[] : parse.(Float64, split(parts[3], ","))
        par = parse(Int, parts[7])
        push!(mix, build_type(m, z, d - 1; par))
        push!(vals, v)
        push!(t, parse(Float64, parts[4]))
        push!(tz, parse(Int, parts[5]) == 1)
        push!(u, parse(Float64, parts[6]))
    end
    return ProdState(d, p, mix, vals, t, tz, u)
end

end # module
