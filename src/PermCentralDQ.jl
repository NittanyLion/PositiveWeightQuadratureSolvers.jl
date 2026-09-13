#=
PermCentralDQ.jl -- designed quadrature under S_d × Z_2 symmetry (coordinate
permutations + global negation x -> -x), for the Gaussian weight N(0, I_d).

Why.  The B_d ansatz (SymmetricDQ.jl) imposes independent sign flips, which
excludes rules whose orbits mix coordinate signs.  Stroud-Secrest's
E_5^{r^2}:5-1 (32 nodes, d=5 p=5, all positive -- banked 2026-08-11, see
rules/PROVENANCE.md) is exactly such a rule: orbits are permutations of
(eta,eta,eta,eta,eta), (lambda,xi,xi,xi,xi), (mu,mu,gamma,gamma,gamma) with
signed values, closed only under permutations and central inversion.  Our B_5
search sat 10 nodes above it at p=5, so this symmetry class demonstrably
holds sub-B_d rules in d=5; this module searches it directly.

Ansatz.  A rule is a union of S_d × Z_2 orbits.  An orbit type is (mults, z):
z coordinates are 0, the rest split into groups of equal SIGNED value, group
sizes mults[1..k].  Free parameters: the k distinct signed values and one
weight.  Orbit = distinct permutations of the value pattern, each paired with
its global negation: size = 2 · d!/(z!·prod(mults!)).

Conditions.  Only |alpha| EVEN survives (global negation kills odd totals on
both sides), but individual alpha_i may be odd -- unlike B_d, where every
alpha_i must be even.  One condition per sorted alpha (permutation
invariance): sum over nodes of w · prod_i h_{alpha_i}(y_i) = delta_{alpha=0},
h_n the orthonormal probabilists' Hermite polynomials.  The negated copy of
each permutation contributes identically (parity), giving a flat factor 2.

Everything else -- LM solver with Nielsen damping and dual form, minimum-norm
Newton polish, greedy elimination (drop/zero/merge/swap), basin hopping,
structure transfer -- mirrors SymmetricDQ.jl.  Differences are marked SIGNED.
Kept as a separate module so the live B_d campaign is untouched.

Bases (2026-08-29).  As in SymmetricDQ.jl: `basis = :legendre` searches the
uniform weight on [-1,1]^d natively (S_d × Z_2 is a symmetry of the cube),
with orthonormal Legendre polynomials, value box |v| ≤ 1 (box-projected LM
steps), flat initial weights and range-scaled jitters; `:hermite` is
byte-identical to the historical search.  `from_hermite` maps a Gaussian
state into the uniform frame (odd map, signs kept) — a donor of orbit-type
mixes only: measured 2026-08-29, transformed Gaussian structures are not
solution basins of the uniform equations.
=#
module PermCentralDQ

using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "WeightTransforms.jl"))
using .WeightTransforms: gauss_to_unit

export OType, build_type, orbit_types, conditions, MixState, mix_nodes,
       mix_params, random_state, solve!, eliminate!, expand_rule,
       orbit_search, save_state, load_state, describe, Work, from_hermite,
       to_bank_nodes

const BASES = (:hermite, :legendre)
check_basis(b::Symbol) = b in BASES || throw(ArgumentError("basis must be one of $BASES, got $b"))
basis_vmax(b::Symbol)  = b === :legendre ? 1.0 : 15.0
jscale(b::Symbol, p::Int) = b === :legendre ? 1.0 / sqrt(2p + 2) : 1.0
init_u(T, v, b::Symbol) = b === :legendre ? 0.0 :
    (T.k == 0 ? 0.0 : -0.25 * sum(T.mults[j] * v[j]^2 for j in 1:T.k))
to_bank_nodes(nodes, b::Symbol) = b === :legendre ? nodes ./ 2 .+ 0.5 : nodes

# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------
function distinct_perms(v::Vector{Int})
    out = Vector{Vector{Int}}()
    n = length(v)
    vs = sort(v)
    used = falses(n)
    cur = zeros(Int, n)
    function rec(pos)
        if pos > n
            push!(out, copy(cur)); return
        end
        prev = typemin(Int)
        for i in 1:n
            (used[i] || vs[i] == prev) && continue
            prev = vs[i]
            used[i] = true; cur[pos] = vs[i]
            rec(pos + 1)
            used[i] = false
        end
    end
    rec(1)
    return out
end

struct OType
    mults::Vector{Int}   # sizes of the equal-value groups (k = length)
    z::Int               # number of zero coordinates
    k::Int
    asg::Matrix{Int}     # distinct assignments: rows × d, entries 0..k
    signfac::Float64     # 2.0 -- the central pair (SIGNED: not 2^(d-z))
    size::Int            # full orbit cardinality = 2 · #rows
end

function build_type(mults::Vector{Int}, z::Int, d::Int)
    sum(mults) + z == d || throw(ArgumentError("mults + zeros must fill d"))
    pattern = Int[]
    for (j, m) in enumerate(mults), _ in 1:m
        push!(pattern, j)
    end
    append!(pattern, zeros(Int, z))
    perms = distinct_perms(pattern)
    asg = Matrix{Int}(undef, length(perms), d)
    for (r, pm) in enumerate(perms)
        asg[r, :] .= pm
    end
    OType(copy(mults), z, length(mults), asg, 2.0, 2 * length(perms))
end

# partitions of n into at most d parts, parts ≤ maxp, descending
function partitions_le(n::Int, d::Int, maxp::Int = n)
    n == 0 && return [Int[]]
    d == 0 && return Vector{Int}[]
    out = Vector{Int}[]
    for f in min(n, maxp):-1:1, rest in partitions_le(n - f, d - 1, f)
        push!(out, [f; rest])
    end
    return out
end

# all orbit types in dimension d; z = d yields the origin.  NOTE the origin
# orbit under S_d × Z_2 is the single point 0 listed twice (its own negation)
# -- expand_rule/dedupe below collapses it, but node counting treats it as 2,
# so the origin type is EXCLUDED here and the point 0 arises only as the
# z -> d degeneration limit, which the zero-move never completes.  Rules that
# want a center point get it from a (mults=[1], z=d-1) orbit collapsing --
# cheaper to simply not offer the origin and let dedupe fix coincidences.
function orbit_types(d::Int)
    types = OType[]
    for z in 0:(d - 1), m in partitions_le(d - z, d)
        push!(types, build_type(m, z, d))
    end
    return types
end

# invariant conditions: sorted alpha with |alpha| even (any parts), padded.
# alpha = 0 (the weight-sum condition) comes first.
conditions(d::Int, p::Int) =
    [vcat(λ, zeros(Int, d - length(λ)))
     for n in 0:2:p for λ in partitions_le(n, d)]

# ---------------------------------------------------------------------------
# mix state
# ---------------------------------------------------------------------------
mutable struct MixState
    d::Int
    p::Int
    mix::Vector{OType}
    vals::Vector{Vector{Float64}}   # per orbit, length k -- SIGNED values
    u::Vector{Float64}              # log weight per orbit (weight per node)
    basis::Symbol                   # :hermite (Gaussian) or :legendre (uniform)
end
MixState(d, p, mix, vals, u) = MixState(d, p, mix, vals, u, :hermite)

mix_nodes(st::MixState)  = sum(T.size for T in st.mix; init = 0)
mix_params(st::MixState) = sum(T.k + 1 for T in st.mix; init = 0)

Base.copy(st::MixState) = MixState(st.d, st.p, copy(st.mix),
                                   [copy(v) for v in st.vals], copy(st.u), st.basis)

function describe(st::MixState)
    parts = String[]
    for (o, T) in enumerate(st.mix)
        sig = join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "")
        push!(parts, "$(sig)($(T.size))")
    end
    return "$(mix_nodes(st)) nodes = " * join(parts, " ")
end

# ---------------------------------------------------------------------------
# residual and Jacobian
# ---------------------------------------------------------------------------
struct Work
    conds::Vector{Vector{Int}}
    C::Int
    basis::Symbol
end
function Work(d::Int, p::Int; basis::Symbol = :hermite)
    check_basis(basis)
    c = conditions(d, p)
    Work(c, length(c), basis)
end

# value/derivative tables of the orthonormal polynomials of `basis` at x
# (see SymmetricDQ.poly_cols!)
function poly_cols!(H::Matrix{Float64}, Hd::Matrix{Float64}, col::Int,
                    x::Float64, p::Int, basis::Symbol)
    H[1, col] = 1.0; Hd[1, col] = 0.0
    p ≥ 1 || return
    H[2, col] = x; Hd[2, col] = 1.0
    if basis === :hermite
        for n in 1:(p - 1)
            H[n+2, col] = (x * H[n+1, col] - sqrt(n) * H[n, col]) / sqrt(n + 1)
        end
        for n in 1:(p - 1)
            Hd[n+2, col] = sqrt(n + 1) * H[n+1, col]
        end
    else
        for n in 1:(p - 1)
            H[n+2, col]  = ((2n + 1) * x * H[n+1, col] - n * H[n, col]) / (n + 1)
            Hd[n+2, col] = ((2n + 1) * (H[n+1, col] + x * Hd[n+1, col]) - n * Hd[n, col]) / (n + 1)
        end
        for n in 1:p
            s = sqrt(2n + 1)
            H[n+1, col] *= s; Hd[n+1, col] *= s
        end
    end
end

# R (length C) and optionally J (C × N); returns ‖R‖
function eval_rj!(R::Vector{Float64}, J::Union{Nothing,Matrix{Float64}},
                  st::MixState, W::Work)
    d, p = st.d, st.p
    st.basis === W.basis ||
        throw(ArgumentError("state basis $(st.basis) ≠ work basis $(W.basis)"))
    fill!(R, 0.0)
    J === nothing || fill!(J, 0.0)
    kmax = maximum(T.k for T in st.mix; init = 0)
    H  = zeros(p + 1, kmax + 1)
    Hd = zeros(p + 1, kmax + 1)
    f   = zeros(d)
    pre = zeros(d + 1)
    suf = zeros(d + 1)
    dS  = zeros(max(kmax, 1))
    off = 0
    for (o, T) in enumerate(st.mix)
        w = exp(st.u[o])
        poly_cols!(H, Hd, 1, 0.0, p, W.basis)
        for j in 1:T.k
            poly_cols!(H, Hd, j + 1, st.vals[o][j], p, W.basis)   # SIGNED evaluation
        end
        fac = w * T.signfac
        for (c, α) in enumerate(W.conds)
            S = 0.0
            T.k > 0 && fill!(dS, 0.0)
            for r in axes(T.asg, 1)
                @inbounds for i in 1:d
                    f[i] = H[α[i] + 1, T.asg[r, i] + 1]
                end
                pre[1] = 1.0
                @inbounds for i in 1:d
                    pre[i+1] = pre[i] * f[i]
                end
                S += pre[d+1]
                if J !== nothing && T.k > 0
                    suf[d+1] = 1.0
                    @inbounds for i in d:-1:1
                        suf[i] = suf[i+1] * f[i]
                    end
                    @inbounds for i in 1:d
                        j = T.asg[r, i]
                        (j == 0 || α[i] == 0) && continue
                        dS[j] += W.basis === :hermite ?
                                 pre[i] * suf[i+1] * sqrt(α[i]) * H[α[i], j + 1] :
                                 pre[i] * suf[i+1] * Hd[α[i] + 1, j + 1]
                    end
                end
            end
            R[c] += fac * S
            if J !== nothing
                for j in 1:T.k
                    J[c, off + j] += fac * dS[j]
                end
                J[c, off + T.k + 1] += fac * S    # d/du of exp(u)·(...)
            end
        end
        off += T.k + 1
    end
    R[1] -= 1.0
    return norm(R)
end

# ---------------------------------------------------------------------------
# Levenberg-Marquardt + minimum-norm Newton polish
# ---------------------------------------------------------------------------
function pack!(θ, st::MixState)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            θ[off + j] = st.vals[o][j]
        end
        θ[off + T.k + 1] = st.u[o]
        off += T.k + 1
    end
    return θ
end

function unpack!(st::MixState, θ)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            st.vals[o][j] = θ[off + j]
        end
        st.u[o] = clamp(θ[off + T.k + 1], -46.0, 5.0)
        off += T.k + 1
    end
    return st
end

# marks which entries of θ are node values (as opposed to log weights)
function value_mask(st::MixState)
    mask = falses(mix_params(st))
    off = 0
    for T in st.mix
        for j in 1:T.k
            mask[off + j] = true
        end
        off += T.k + 1
    end
    return mask
end

"""
    solve!(st, W; tol, maxiter) -> (residual, converged)

Converge the mix state in place.  `tol` is on the invariant-basis residual;
after the main loop an undamped minimum-norm Newton polish drives it toward
machine precision.  SIGNED: values keep their signs -- no |v| at the end.
"""
function solve!(st::MixState, W::Work; tol::Float64 = 1e-12,
                maxiter::Int = 600, λ0::Float64 = 1e-3,
                vmax::Float64 = basis_vmax(W.basis))
    C = W.C
    N = mix_params(st)
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
            if W.basis === :hermite
                if any(i -> isval[i] && abs(θt[i]) > vmax, eachindex(θt))
                    λ *= ν; ν *= 2.0; continue
                end
            else   # bounded support: project the step onto the box
                @inbounds for i in eachindex(θt)
                    isval[i] && (θt[i] = clamp(θt[i], -vmax, vmax))
                end
                dθ = θt .- θ
            end
            unpack!(st2, θt)
            δn = eval_rj!(Rt, nothing, st2, W)
            pred = δ^2 - sum(abs2, R .+ J * dθ)
            ρ = pred > 0 ? (δ^2 - δn^2) / pred : -1.0
            if δn < δ && ρ > 1e-4
                θ .= θt
                unpack!(st, θ); pack!(θ, st)    # apply u clamp consistently
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

    # polish: undamped minimum-norm Newton with rank cutoffs
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
    # SIGNED: values represent themselves; do NOT take absolute values here
    return (δ, δ ≤ tol)
end

# ---------------------------------------------------------------------------
# random initial states
# ---------------------------------------------------------------------------
# SIGNED: sample magnitudes as before, then a random sign per value; keep
# min separation in SIGNED distance so distinct groups stay distinct
function random_values(rng::AbstractRNG, k::Int, p::Int; basis::Symbol = :hermite)
    if basis === :legendre
        v = Float64[]
        while length(v) < k
            c = (0.02 + rand(rng) * 0.96) * (rand(rng) < 0.5 ? -1 : 1)
            all(abs(c - x) > 0.02 for x in v) && push!(v, c)
        end
        return v
    end
    vmax = sqrt(2p + 2)
    v = Float64[]
    while length(v) < k
        c = (0.15 + rand(rng) * vmax) * (rand(rng) < 0.5 ? -1 : 1)
        all(abs(c - x) > 0.12 for x in v) && push!(v, c)
    end
    return v
end

function random_state(d::Int, p::Int, rng::AbstractRNG;
                      factor::Float64 = 1.3, types = orbit_types(d),
                      bias::Float64 = 1.0, basis::Symbol = :hermite)
    check_basis(basis)
    C = length(conditions(d, p))
    target = ceil(Int, factor * C)
    nonorigin = [T for T in types if T.k > 0]
    wts = [((T.k + 1) / T.size)^bias for T in nonorigin]
    cw = cumsum(wts) ./ sum(wts)
    mix = OType[]
    while sum(T.k + 1 for T in mix; init = 0) < target
        r = rand(rng)
        push!(mix, nonorigin[findfirst(≥(r), cw)])
    end
    vals = [random_values(rng, T.k, p; basis) for T in mix]
    u = [init_u(T, vals[o], basis) for (o, T) in enumerate(mix)]
    st = MixState(d, p, mix, vals, u, basis)
    tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
    st.u .-= log(tot)
    return st
end

# ---------------------------------------------------------------------------
# structure transfer: seed a target case from a donor case's best structure
# ---------------------------------------------------------------------------
"""
    seed_from_template(donor, d, p, rng; factor) -> MixState

Same-d donors: radii scaled by √((2p+1)/(2p_src+1)), signs preserved.
(d-1)-donors: embedded via one more zero coordinate per orbit type.
Filler orbits are added to `factor` × #conditions.
"""
function seed_from_template(donor::MixState, d::Int, p::Int, rng::AbstractRNG;
                            factor::Float64 = 1.35, types = orbit_types(d))
    donor.d == d || donor.d == d - 1 ||
        throw(ArgumentError("donor must have dimension d or d-1"))
    basis = donor.basis
    scale = basis === :legendre ? 1.0 : sqrt((2p + 1) / (2 * donor.p + 1))
    lo, hi = basis === :legendre ? (0.02, 0.98) : (0.05, 14.0)
    js = jscale(basis, p)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    for (o, T) in enumerate(donor.mix)
        z = T.z + (d - donor.d)
        push!(mix, build_type(copy(T.mults), z, d))
        # SIGNED: scale magnitudes, keep signs, jitter
        v = sign.(donor.vals[o]) .* clamp.(abs.(donor.vals[o]) .* scale, lo, hi)
        v .+= (0.03 * js) .* randn(rng, length(v))
        push!(vals, v)
        push!(u, donor.u[o])                 # relative weights carry over
    end
    C = length(conditions(d, p))
    target = ceil(Int, factor * C)
    nonorigin = [T for T in types if T.k > 0]
    while sum(T.k + 1 for T in mix; init = 0) < target
        T = nonorigin[rand(rng, 1:length(nonorigin))]
        push!(mix, T)
        v = random_values(rng, T.k, p; basis)
        push!(vals, v)
        push!(u, init_u(T, v, basis))
    end
    st = MixState(d, p, mix, vals, u, basis)
    tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
    st.u .-= log(tot)
    return st
end

# ---------------------------------------------------------------------------
# reduction moves: drop an orbit, zero a value, merge two values
# ---------------------------------------------------------------------------
function apply_move(st::MixState, move)
    kind, o, arg = move
    st2 = copy(st)
    if kind === :drop
        deleteat!(st2.mix, o); deleteat!(st2.vals, o); deleteat!(st2.u, o)
    elseif kind === :zero
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])
        deleteat!(m2, arg); deleteat!(v2, arg)
        if isempty(m2)
            # SIGNED: zeroing the last group would leave the origin orbit,
            # which this ansatz does not carry -- drop the orbit instead
            deleteat!(st2.mix, o); deleteat!(st2.vals, o); deleteat!(st2.u, o)
        else
            st2.mix[o] = build_type(m2, T.z + T.mults[arg], st.d)
            st2.vals[o] = v2
        end
    elseif kind === :merge
        a, b = arg
        T = st.mix[o]
        m2 = copy(T.mults); v2 = copy(st.vals[o])   # SIGNED: no abs
        vnew = (T.mults[a] * v2[a] + T.mults[b] * v2[b]) / (T.mults[a] + T.mults[b])
        m2[a] += m2[b]; v2[a] = vnew
        deleteat!(m2, b); deleteat!(v2, b)
        st2.mix[o] = build_type(m2, T.z, st.d)
        st2.vals[o] = v2
    end
    return st2
end

function reduction_moves(st::MixState)
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
        jm = argmin(abs.(v))
        push!(moves, (:zero, o, jm)); push!(pr, abs(v[jm]))
        if T.k ≥ 2
            best = (Inf, 0, 0)
            for a in 1:T.k, b in (a+1):T.k
                gp = abs(v[a] - v[b])       # SIGNED distance
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
    eliminate!(st, W; tol, jitter_tries, log) -> st

Greedy reduction: repeatedly attempt the ranked moves, reconverging from the
warm start (plus jittered retries) after each; accept the first success.
"""
function eliminate!(st::MixState, W::Work, rng::AbstractRNG;
                    tol::Float64 = 1e-12, jitter_tries::Int = 1,
                    swap_rounds::Int = 12, log_io = nothing)
    types = orbit_types(st.d)
    js = jscale(W.basis, st.p)
    while true
        improved = false
        for move in reduction_moves(st)
            mix_nodes(st) ≤ 2 && break
            cand = apply_move(st, move)
            mix_params(cand) == 0 && continue
            δ, ok = solve!(cand, W; tol)
            if !ok
                for _ in 1:jitter_tries
                    c2 = copy(cand)
                    for v in c2.vals
                        v .+= (0.02 * js) .* randn(rng, length(v))
                    end
                    δ, ok = solve!(c2, W; tol)
                    ok && (cand = c2; break)
                end
            end
            ok || continue
            st.mix = cand.mix; st.vals = cand.vals; st.u = cand.u
            improved = true
            log_io === nothing ||
                println(log_io, "    -$(move[1]) → $(describe(st))")
            break
        end
        if !improved
            wt = [exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix)]
            ord = sortperm(wt)
            for _ in 1:swap_rounds
                o = ord[min(1 + floor(Int, abs(randn(rng)) * 2), length(ord))]
                smaller = [T for T in types if T.size < st.mix[o].size]
                isempty(smaller) && continue
                T2 = smaller[rand(rng, 1:length(smaller))]
                cand = copy(st)
                cand.mix[o] = T2
                cand.vals[o] = random_values(rng, T2.k, st.p; basis = W.basis)
                cand.u[o] = st.u[o] + log(st.mix[o].size / max(T2.size, 1))
                δ, ok = solve!(cand, W; tol)
                ok || continue
                st.mix = cand.mix; st.vals = cand.vals; st.u = cand.u
                improved = true
                log_io === nothing ||
                    println(log_io, "    -swap → $(describe(st))")
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
"""Expand to the full node set: every distinct permutation plus its global
negation, weights per node.  Coincident nodes (degenerate configurations,
e.g. a value hitting 0 or an orbit that is centrally self-symmetric) are NOT
merged here -- the runner dedupes before banking."""
function expand_rule(st::MixState)
    d = st.d
    N = mix_nodes(st)
    nodes = zeros(N, d)
    w = zeros(N)
    row = 0
    for (o, T) in enumerate(st.mix)
        wo = exp(st.u[o])
        for r in axes(T.asg, 1)
            row += 1
            for i in 1:d
                j = T.asg[r, i]
                nodes[row, i] = j == 0 ? 0.0 : st.vals[o][j]
            end
            w[row] = wo
            row += 1
            nodes[row, :] .= .-nodes[row-1, :]
            w[row] = wo
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::MixState, δ::Float64)
    open(path, "w") do io
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(mix_nodes(st)) resid=$δ group=SC",
                st.basis === :hermite ? "" : " basis=$(st.basis)")
        for (o, T) in enumerate(st.mix)
            println(io, join(T.mults, ","), "|", T.z, "|",
                    join(st.vals[o], ","), "|", st.u[o])
        end
    end
end

function load_state(path::AbstractString, d::Int, p::Int)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    basis = :hermite
    for line in eachline(path)
        if startswith(line, "#")
            m = match(r"basis=(\w+)", line)
            m === nothing || (basis = Symbol(m[1]))
            continue
        end
        isempty(strip(line)) && continue
        parts = split(line, "|")
        m = isempty(parts[1]) ? Int[] : parse.(Int, split(parts[1], ","))
        z = parse(Int, parts[2])
        v = isempty(parts[3]) ? Float64[] : parse.(Float64, split(parts[3], ","))
        push!(mix, build_type(m, z, d))
        push!(vals, v)
        push!(u, parse(Float64, parts[4]))
    end
    check_basis(basis)
    return MixState(d, p, mix, vals, u, basis)
end

"""
    from_hermite(st) -> MixState in the :legendre frame

Signed values mapped by the odd node-matching map (WeightTransforms.node_map),
orbit types and weights kept.  Donor of orbit mixes for the uniform search —
not a solution basin by itself (measured 2026-08-29).
"""
function from_hermite(st::MixState)
    st.basis === :hermite || throw(ArgumentError("from_hermite needs a :hermite state"))
    f = WeightTransforms.node_map(st.p, :legendre)
    vals = [Float64[f(x) for x in v] for v in st.vals]
    return MixState(st.d, st.p, copy(st.mix), vals, copy(st.u), :legendre)
end

# ---------------------------------------------------------------------------
# search driver
# ---------------------------------------------------------------------------
"""
    orbit_search(d, p; seconds, rng, best_nodes, verify, on_improve, log_io)

Random-restart + greedy-reduction search over S_d × Z_2 orbit mixes.  Same
contract as SymmetricDQ.orbit_search: `verify(nodes, weights)` returns the
raw-monomial exactness error; improvements are reported through `on_improve`
only if that error is ≤ `extol`.
"""
function orbit_search(d::Int, p::Int; seconds::Real = 300,
                      rng::AbstractRNG = Random.default_rng(),
                      best_nodes::Int = typemax(Int),
                      verify = nothing, on_improve = nothing,
                      warm::Union{Nothing,MixState} = nothing,
                      templates::Vector{MixState} = MixState[],
                      template_prob::Float64 = 0.45,
                      hops_budget::Int = 3,
                      factor_range = (1.6, 2.4),
                      basis::Symbol = :hermite,
                      tol::Float64 = 1e-12, extol::Float64 = 1e-8,
                      log_io = stdout, progress = nothing)
    check_basis(basis)
    W = Work(d, p; basis)
    js = jscale(basis, p)          # jitters are in Hermite units
    warm === nothing || warm.basis === basis ||
        throw(ArgumentError("warm state is $(warm.basis), search is $basis"))
    all(t.basis === basis for t in templates) ||
        throw(ArgumentError("a template's basis differs from the search basis $basis"))
    deadline = time() + seconds
    restart = 0
    best_run = typemax(Int)
    while time() < deadline
        restart += 1
        local st
        if warm !== nothing && restart == 1
            st = copy(warm)
            for v in st.vals
                v .+= (0.01 * js) .* randn(rng, length(v))
            end
        elseif !isempty(templates) && rand(rng) < template_prob
            donor = templates[rand(rng, 1:length(templates))]
            st = seed_from_template(donor, d, p, rng;
                                    factor = 1.2 + 0.4 * rand(rng))
        else
            fac = factor_range[1] + rand(rng) * (factor_range[2] - factor_range[1])
            st = random_state(d, p, rng; factor = fac, bias = rand(rng)^2 * 0.8, basis)
        end
        δ, ok = solve!(st, W; tol)
        if !ok
            restart % 20 == 0 &&
                println(log_io, "  restart $restart: no initial convergence (δ=$(round(δ, sigdigits=3)))")
            continue
        end
        eliminate!(st, W, rng; tol, log_io = nothing)
        hops = hops_budget
        while hops > 0
            cand = copy(st)
            for v in cand.vals
                v .+= ((0.08 + 0.15 * rand(rng)) * js) .* randn(rng, length(v))
            end
            _, hok = solve!(cand, W; tol)
            if hok
                eliminate!(cand, W, rng; tol, log_io = nothing)
                if mix_nodes(cand) < mix_nodes(st)
                    st = cand
                    continue
                end
            end
            hops -= 1
        end
        δ, ok = solve!(st, W; tol)
        n = mix_nodes(st)
        if n < best_run || restart ≤ 5 || restart % 25 == 0
            println(log_io, "  restart $restart: $(n) nodes  δ=$(round(δ, sigdigits=3))  [$(describe(st))]")
            flush(log_io)
        end
        best_run = min(best_run, n)
        progress === nothing || progress(restart, best_run, best_nodes)
        if ok && n < best_nodes
            nodes, w = expand_rule(st)
            ex = verify === nothing ? 0.0 : verify(nodes, w)
            if ex ≤ extol
                best_nodes = n
                on_improve === nothing || on_improve(st, δ, nodes, w, ex)
                println(log_io, "  *** improvement: $n nodes (exactness $(round(ex, sigdigits=3)))")
                flush(log_io)
            else
                println(log_io, "  (candidate $n nodes failed monomial verify: $(round(ex, sigdigits=3)))")
            end
        end
    end
    return best_nodes
end

end # module
