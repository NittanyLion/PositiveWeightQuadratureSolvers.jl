#=
SymmetricDQ.jl -- designed quadrature under FULL B_d symmetry (signed
coordinate permutations), for the Gaussian weight N(0, I_d).

Why.  Our centrally symmetric pair ansatz cannot go below its parameter-count
floor (e.g. d=3 p=21: 474 nodes), yet Möller's proven bound sits far lower
(321).  Rules below the pair-ansatz floor are non-generic there.  Imposing the
larger group B_d changes the game twice over (Sobolev's theorem):

  * only one moment condition per B_d-orbit of exponent vectors survives --
    67 conditions at d=3 p=21 instead of 946;
  * orbits through special positions (axis, diagonal, coordinate planes)
    carry many nodes with very few parameters, so the ansatz reaches
    configurations that are non-generic -- exactly the structure published
    sub-counting rules use.

Ansatz.  A rule is a union of B_d-orbits.  An orbit type is (mults, z):
z coordinates are 0, and the remaining split into groups of equal |value|,
group sizes mults[1..k].  Free parameters: the k distinct positive values and
one weight (equal across the orbit).  Orbit size = d!/(z!·prod(mults!)) · 2^(d-z).

Conditions.  For each partition alpha (all parts even, |alpha| <= p, padded
to length d): sum over nodes of w · prod_i h_{alpha_i}(y_i) = delta_{alpha=0},
with h_n the orthonormal probabilists' Hermite polynomials.  Because every
alpha_i is even, each product is invariant under sign flips, so the sum over a
full orbit is 2^(d-z) times the sum over the distinct unsigned permutations.

D_d half-orbits (2026-08-26, PROPOSAL_high_p.md P3).  With `halves = true`
the ansatz is enlarged to the demihypercube group D_d (permutations + EVEN
numbers of sign changes, index 2 in B_d).  A full-support B_d orbit splits
into two D_d orbits — the points with an even / odd number of negative
coordinates — that get independent values and weights; orbits with a zero
coordinate do not split (a sign flip on the zero is free).  Sobolev now keeps
two families of conditions: all exponents even (as before) and all exponents
ODD (the sign product is D_d-invariant), the latter with zero right-hand
side.  Half-orbits enter as types with `par = ±1`; everything else — LM,
elimination, templates, hops — is unchanged, and `halves = false` is
byte-identical to the historical B_d search.

Bases (2026-08-29, PROPOSAL_nonnormal_native_search.md).  `basis = :hermite`
is the Gaussian search above.  `basis = :legendre` searches the UNIFORM weight
on [-1,1]^d natively: the same orbit machinery (the cube has the full B_d
symmetry, and D_d half-orbits work unchanged because the uniform weight is
centrally symmetric, so the all-odd conditions still have zero right-hand
side), with the orthonormal Legendre polynomials √(2n+1)·P_n in place of the
Hermite ones, the value box |v| ≤ 1 instead of 15, flat initial weights, no
radius scaling in templates, and jitters shrunk by the ratio of the node
ranges.  `MixState` and `Work` carry the basis; sidecars record it in their
header; `:hermite` remains byte-identical.  `from_hermite(st)` maps a Hermite
state to the uniform frame by the quantile map v ↦ 2Φ(v)−1 (odd, so every
orbit keeps its type and weight) — the warm start that makes the uniform
search a re-solve rather than a campaign.  Banked uniform rules live on
[0,1]^d (x/2 + 1/2 of the expanded nodes), as `legendre_d{d}_p{p}_n{n}.csv`.

Solver.  Levenberg-Marquardt with Nielsen damping (dual form when conditions
< unknowns), then undamped minimum-norm Newton polish with rank cutoffs --
the same recipe that drives DesignedQuadratureV2.  Search: random orbit-type
mixes, then greedy reduction: drop the lightest orbit, zero a value, or merge
the two closest values (type degenerations), reconverging warm after each.
=#
module SymmetricDQ

using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "WeightTransforms.jl"))
using .WeightTransforms: gauss_to_unit

export OType, build_type, orbit_types, conditions, MixState, mix_nodes,
       mix_params, random_state, solve!, eliminate!, expand_rule,
       orbit_search, save_state, load_state, describe, Work, from_hermite,
       continue_from_hermite, continue_to_hermite, from_legendre, to_bank_nodes

# ---------------------------------------------------------------------------
# bases: everything basis-specific lives here
# ---------------------------------------------------------------------------
const BASES = (:hermite, :legendre)
check_basis(b::Symbol) = b in BASES || throw(ArgumentError("basis must be one of $BASES, got $b"))
# value box for the LM step (nodes must stay inside the support for :legendre)
basis_vmax(b::Symbol)  = b === :legendre ? 1.0 : 15.0
# typical node range, for scaling random values and jitters: Hermite nodes
# reach √(2p+2); uniform ones sit in (0,1)
basis_range(b::Symbol, p::Int) = b === :legendre ? 1.0 : sqrt(2p + 2)
jscale(b::Symbol, p::Int) = basis_range(b, p) / sqrt(2p + 2)
# initial log-weight of an orbit with values v: Gaussian density for Hermite
# (weights fall off like exp(-|x|²/4) empirically), flat for uniform
init_u(T, v, b::Symbol) = b === :legendre ? 0.0 :
    (T.k == 0 ? 0.0 : -0.25 * sum(T.mults[j] * v[j]^2 for j in 1:T.k))
"""Expanded nodes in the frame the rule bank uses: ℝ^d for :hermite,
[0,1]^d for :legendre (x/2 + 1/2)."""
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
    mults::Vector{Int}   # sizes of the equal-|value| groups (k = length)
    z::Int               # number of zero coordinates
    k::Int
    asg::Matrix{Int}     # distinct unsigned assignments: rows × d, entries 0..k
    signfac::Float64     # 2^(d - z), or 2^(d - 1) for a half-orbit
    size::Int            # orbit cardinality
    par::Int             # 0 = B_d orbit; ±1 = D_d half-orbit (even/odd sign count)
end

function build_type(mults::Vector{Int}, z::Int, d::Int; par::Int = 0)
    sum(mults) + z == d || throw(ArgumentError("mults + zeros must fill d"))
    par in (-1, 0, 1) || throw(ArgumentError("par must be -1, 0 or 1"))
    par == 0 || z == 0 || throw(ArgumentError("half-orbits need full support (z = 0)"))
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
    sf = par == 0 ? 2.0^sum(mults) : 2.0^(sum(mults) - 1)
    OType(copy(mults), z, length(mults), asg, sf, length(perms) * Int(sf), par)
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

# all orbit types in dimension d; the z = d iteration yields the origin
# (mults = [], z = d), so nothing needs appending
function orbit_types(d::Int; halves::Bool = false)
    types = OType[]
    for z in 0:d, m in partitions_le(d - z, d)
        push!(types, build_type(m, z, d))
        if halves && z == 0
            push!(types, build_type(m, 0, d; par = 1))
            push!(types, build_type(m, 0, d; par = -1))
        end
    end
    return types
end

# invariant conditions: even partitions padded to length d; weight sum first.
# With `halves`, also the all-odd exponent vectors 2λ+1 (|α| ≤ p), which are
# D_d-invariant and have zero Gaussian moment.
function conditions(d::Int, p::Int; halves::Bool = false)
    conds = [vcat(2 .* λ, zeros(Int, d - length(λ)))
             for n in 0:(p ÷ 2) for λ in partitions_le(n, d)]
    if halves
        for n in 0:((p - d) ÷ 2), λ in partitions_le(n, d)
            push!(conds, vcat(2 .* λ, zeros(Int, d - length(λ))) .+ 1)
        end
    end
    return conds
end

# ---------------------------------------------------------------------------
# mix state
# ---------------------------------------------------------------------------
mutable struct MixState
    d::Int
    p::Int
    mix::Vector{OType}
    vals::Vector{Vector{Float64}}   # per orbit, length k
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
        sig = isempty(T.mults) ? "origin" :
              join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "") *
              (T.par == 1 ? "+" : T.par == -1 ? "-" : "")
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
    odd::Vector{Bool}      # all-odd (D_d) condition?
    halves::Bool
    basis::Symbol
    alpha::Float64         # :legendre frame only — Jacobi(α,α) weight (1-x²)^α; 0 = uniform
end
function Work(d::Int, p::Int; halves::Bool = false, basis::Symbol = :hermite,
              alpha::Real = 0.0)
    check_basis(basis)
    basis === :hermite && alpha != 0 && throw(ArgumentError("alpha applies to the :legendre frame only"))
    alpha > -1 || throw(ArgumentError("Jacobi parameter must exceed -1"))
    c = conditions(d, p; halves)
    Work(c, length(c), [isodd(α[1]) for α in c], halves, basis, Float64(alpha))
end

# column `col` of the value table H and derivative table Hd: the orthonormal
# degree-n polynomial of `basis` and its derivative at x, n = 0..p.
#   :hermite  — probabilists' h_n = He_n/√(n!), h_n' = √n·h_{n-1}
#   :legendre — √(2n+1)·P_n, orthonormal for the probability weight dx/2 on
#               [-1,1]; P_n' by the differentiated three-term recurrence.
#               With alpha ≠ 0: the symmetric Jacobi (Gegenbauer) polynomials
#               orthonormal for the probability weight ∝ (1-x²)^α, by the
#               orthonormal recurrence p_{n+1} = (x p_n − β_n p_{n-1}) / β_{n+1},
#               β_n² = n(n+2α) / ((2n+2α+1)(2n+2α−1))  (α = 0: n²/(4n²−1))
function poly_cols!(H::Matrix{Float64}, Hd::Matrix{Float64}, col::Int,
                    x::Float64, p::Int, basis::Symbol, alpha::Float64 = 0.0)
    H[1, col] = 1.0; Hd[1, col] = 0.0
    p ≥ 1 || return
    if basis === :hermite
        H[2, col] = x; Hd[2, col] = 1.0
        for n in 1:(p - 1)
            H[n+2, col] = (x * H[n+1, col] - sqrt(n) * H[n, col]) / sqrt(n + 1)
        end
        for n in 1:(p - 1)
            Hd[n+2, col] = sqrt(n + 1) * H[n+1, col]
        end
    elseif alpha == 0  # :legendre — build raw P_n, P_n', then scale
        H[2, col] = x; Hd[2, col] = 1.0
        for n in 1:(p - 1)
            H[n+2, col]  = ((2n + 1) * x * H[n+1, col] - n * H[n, col]) / (n + 1)
            Hd[n+2, col] = ((2n + 1) * (H[n+1, col] + x * Hd[n+1, col]) - n * Hd[n, col]) / (n + 1)
        end
        for n in 1:p
            s = sqrt(2n + 1)
            H[n+1, col] *= s; Hd[n+1, col] *= s
        end
    else               # symmetric Jacobi, orthonormal recurrence
        β(n) = sqrt(n * (n + 2alpha) / ((2n + 2alpha + 1) * (2n + 2alpha - 1)))
        β1 = β(1)
        H[2, col] = x / β1; Hd[2, col] = 1 / β1
        for n in 1:(p - 1)
            bn, bn1 = β(n), β(n + 1)
            H[n+2, col]  = (x * H[n+1, col] - bn * H[n, col]) / bn1
            Hd[n+2, col] = (H[n+1, col] + x * Hd[n+1, col] - bn * Hd[n, col]) / bn1
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
        poly_cols!(H, Hd, 1, 0.0, p, W.basis, W.alpha)
        for j in 1:T.k
            poly_cols!(H, Hd, j + 1, st.vals[o][j], p, W.basis, W.alpha)
        end
        for (c, α) in enumerate(W.conds)
            # all-odd conditions: a B_d orbit (or any orbit with a zero
            # coordinate) contributes nothing; a half-orbit contributes with
            # the sign of its parity
            fac = w * T.signfac
            if W.odd[c]
                T.par == 0 && continue
                fac *= T.par
            end
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
                        # Hermite keeps the historical product order so the
                        # Gaussian search stays bit-for-bit reproducible
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
machine precision (needed for the raw-monomial verification at high p).
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
                # unbounded support: a runaway value only signals a bad
                # step, so reject it and damp harder (historical behavior)
                if any(i -> isval[i] && abs(θt[i]) > vmax, eachindex(θt))
                    λ *= ν; ν *= 2.0; continue
                end
            else
                # bounded support: PROJECT the step onto the box.  Rejecting
                # would freeze a warm start whose outer orbits sit near a
                # face of the cube (the Φ-map puts them at 0.9999…), since
                # every step then crosses the face and λ just runs away.
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
    # |v| represents the same rule: every even condition is even in each
    # value, and a half-orbit whose negated value-groups total an odd number
    # of coordinates is the SAME point set as the opposite-parity half of |v|
    for (o, v) in enumerate(st.vals)
        T = st.mix[o]
        if T.par != 0
            flips = sum(T.mults[j] for j in 1:T.k if v[j] < 0; init = 0)
            isodd(flips) && (st.mix[o] = build_type(T.mults, T.z, st.d; par = -T.par))
        end
        v .= abs.(v)
    end
    return (δ, δ ≤ tol)
end

# ---------------------------------------------------------------------------
# random initial states
# ---------------------------------------------------------------------------
function random_values(rng::AbstractRNG, k::Int, p::Int; basis::Symbol = :hermite)
    if basis === :legendre
        # uniform frame: values in (0,1), separation scaled like the range
        v = Float64[]
        while length(v) < k
            c = 0.02 + rand(rng) * 0.96
            all(abs(c - x) > 0.02 for x in v) && push!(v, c)
        end
        return v
    end
    vmax = sqrt(2p + 2)
    v = Float64[]
    while length(v) < k
        c = 0.15 + rand(rng) * vmax
        all(abs(c - x) > 0.12 for x in v) && push!(v, c)
    end
    return v
end

function random_state(d::Int, p::Int, rng::AbstractRNG;
                      factor::Float64 = 1.3, types = orbit_types(d),
                      bias::Float64 = 1.0, basis::Symbol = :hermite)
    check_basis(basis)
    halves = any(T.par != 0 for T in types)
    C = length(conditions(d, p; halves))
    target = ceil(Int, factor * C)
    nonorigin = [T for T in types if T.k > 0]
    # sample types weighted toward parameter-efficient (special-position)
    # orbits: the good configurations are special-orbit-heavy, and generic
    # orbits carry 48/384 nodes for only d+1 parameters
    wts = [((T.k + 1) / T.size)^bias for T in nonorigin]
    cw = cumsum(wts) ./ sum(wts)
    mix = OType[]
    if rand(rng) < 0.7
        push!(mix, types[findfirst(T -> T.k == 0, types)])
    end
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

Build an initial state for (d, p) from a donor MixState of a neighboring
case.  Same-d donors have their radii scaled by √((2p+1)/(2p_src+1)) (the
Hermite node range grows like √p); (d-1)-donors are embedded in the
hyperplane x_d = 0 by giving every orbit type one more zero coordinate.
Filler orbits (uniformly sampled types) are added until the parameter count
reaches `factor` × #conditions; the solver and elimination trim any excess.
"""
function seed_from_template(donor::MixState, d::Int, p::Int, rng::AbstractRNG;
                            factor::Float64 = 1.35, types = orbit_types(d))
    donor.d == d || donor.d == d - 1 ||
        throw(ArgumentError("donor must have dimension d or d-1"))
    basis = donor.basis
    # Hermite node ranges grow like √p; the uniform frame is fixed
    scale = basis === :legendre ? 1.0 : sqrt((2p + 1) / (2 * donor.p + 1))
    lo, hi = basis === :legendre ? (0.02, 0.98) : (0.05, 14.0)
    js = jscale(basis, p)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    for (o, T) in enumerate(donor.mix)
        z = T.z + (d - donor.d)
        # embedding adds a zero coordinate, which turns a half-orbit into a
        # full one; same-d donors keep their parity
        push!(mix, build_type(copy(T.mults), z, d; par = z == T.z ? T.par : 0))
        v = clamp.(abs.(donor.vals[o]) .* scale, lo, hi)
        v .+= (0.03 * js) .* randn(rng, length(v))
        push!(vals, abs.(v))
        push!(u, donor.u[o])                 # relative weights carry over
    end
    halves = any(T.par != 0 for T in types)
    C = length(conditions(d, p; halves))
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
        st2.mix[o] = build_type(m2, T.z + T.mults[arg], st.d)
        st2.vals[o] = v2
    elseif kind === :merge
        a, b = arg
        T = st.mix[o]
        m2 = copy(T.mults); v2 = abs.(st.vals[o])
        vnew = (T.mults[a] * v2[a] + T.mults[b] * v2[b]) / (T.mults[a] + T.mults[b])
        m2[a] += m2[b]; v2[a] = vnew
        deleteat!(m2, b); deleteat!(v2, b)
        st2.mix[o] = build_type(m2, T.z, st.d; par = T.par)
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
                gp = abs(abs(v[a]) - abs(v[b]))
                gp < best[1] && (best = (gp, a, b))
            end
            push!(moves, (:merge, o, (best[2], best[3]))); push!(pr, best[1])
        end
    end
    # drops lightest-first, then degenerations by smallness
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
    types = orbit_types(st.d; halves = W.halves)
    js = jscale(W.basis, st.p)
    while true
        improved = false
        for move in reduction_moves(st)
            mix_nodes(st) ≤ 1 && break
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
            # reduction blocked: try replacing an orbit by a strictly smaller
            # type with fresh values (keeps the descent monotone in nodes)
            wt = [exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix)]
            ord = sortperm(wt)
            for _ in 1:swap_rounds
                o = ord[min(1 + floor(Int, abs(randn(rng)) * 2), length(ord))]
                smaller = [T for T in types if T.size < st.mix[o].size &&
                           !(T.k == 0 && any(S.k == 0 for S in st.mix))]
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
"""Expand to the full node set (all signs), weights per node."""
function expand_rule(st::MixState)
    d = st.d
    N = mix_nodes(st)
    nodes = zeros(N, d)
    w = zeros(N)
    row = 0
    for (o, T) in enumerate(st.mix)
        wo = exp(st.u[o])
        nz = sum(T.mults)
        for r in axes(T.asg, 1)
            pos = [i for i in 1:d if T.asg[r, i] > 0]
            for smask in 0:(2^nz - 1)
                # half-orbit: keep only the sign patterns of its parity
                T.par != 0 && (iseven(count_ones(smask)) ? 1 : -1) != T.par && continue
                row += 1
                for i in 1:d
                    j = T.asg[r, i]
                    nodes[row, i] = j == 0 ? 0.0 : st.vals[o][j]
                end
                for (b, i) in enumerate(pos)
                    ((smask >> (b - 1)) & 1) == 1 && (nodes[row, i] *= -1)
                end
                w[row] = wo
            end
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::MixState, δ::Float64)
    open(path, "w") do io
        # basis= only for non-Gaussian states, so Hermite sidecars keep the
        # header every reader (merge_best.sh, status window) already parses
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(mix_nodes(st)) resid=$δ",
                st.basis === :hermite ? "" : " basis=$(st.basis)")
        for (o, T) in enumerate(st.mix)
            println(io, join(T.mults, ","), "|", T.z, "|",
                    join(st.vals[o], ","), "|", st.u[o],
                    T.par == 0 ? "" : "|$(T.par)")
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
        par = length(parts) ≥ 5 ? parse(Int, parts[5]) : 0
        push!(mix, build_type(m, z, d; par))
        push!(vals, v)
        push!(u, parse(Float64, parts[4]))
    end
    check_basis(basis)
    return MixState(d, p, mix, vals, u, basis)
end

# ---------------------------------------------------------------------------
# quantile-map warm start (2026-08-29)
# ---------------------------------------------------------------------------
"""
    from_hermite(st; map = :nodes) -> MixState in the :legendre frame

Map a Gaussian (Hermite) state to the uniform weight coordinate-wise.  Both
maps are odd and monotone, so every orbit keeps its type, its parity and its
weight; the result is a positive-weight rule with the right node count and
orbit structure.  MEASURED 2026-08-29 (METHODS.md §17): it is NOT a solution
basin of the uniform equations — LM stalls at ‖R‖ ≈ 0.1–0.5 on every case
tried, boxed or not, and the Jacobi-family continuation
(`continue_from_hermite`) shows why: the Hermite structures are non-generic
solutions that only the Gaussian equations admit.  Use the result as a donor
of orbit-type mixes for the native search (symq_run.jl does), nothing more.

  * `:nodes` (default) — WeightTransforms.node_map: the 1-D Gauss–Hermite
    nodes of degree p go to the Gauss–Legendre nodes, interpolated
    monotonically; outer orbits land at the outer GL node (≈0.97).
  * `:cdf` — the quantile map 2Φ(v)−1.  Structurally the same, but it puts
    the outer orbits within 1e-4 of the cube's face, where the box-projected
    LM has no room to move them (measured 2026-08-29: every case stalled).
"""
function from_hermite(st::MixState; map::Symbol = :nodes)
    st.basis === :hermite || throw(ArgumentError("from_hermite needs a :hermite state"))
    f = map === :cdf ? gauss_to_unit :
        map === :nodes ? WeightTransforms.node_map(st.p, :legendre) :
        throw(ArgumentError("map must be :nodes or :cdf"))
    vals = [Float64[f(x) for x in v] for v in st.vals]
    return MixState(st.d, st.p, copy(st.mix), vals, copy(st.u), :legendre)
end

"""
    from_legendre(st; map = :cdf) -> MixState in the :hermite frame

Mirror of `from_hermite`: a uniform orbit structure read into the Gaussian
frame through Φ⁻¹((1+x)/2).  Added 2026-09-10.

Use it ONLY as a donor of orbit-type mixes, never as a warm incumbent.  The
rule-level version of this map was measured on 2026-09-10 and is a non-basin
in this direction just as `from_hermite` is in the other: eleven solves at
d=2 — p31/p33/p35 controls where a Gaussian rule is already banked and the
uniform one is 6/24/48 nodes better, plus p37/p39 — converged from none of
its images, under either map, from every banked source.  What survives the
transfer is the ORBIT-TYPE MULTISET, which is what `seed_from_template`
takes; the values it carries are jittered and clamped there anyway.
"""
function from_legendre(st::MixState; map::Symbol = :cdf)
    st.basis === :legendre || throw(ArgumentError("from_legendre needs a :legendre state"))
    map === :cdf || throw(ArgumentError("only the :cdf map is defined in this direction"))
    vals = [Float64[WeightTransforms.unit_to_gauss(clamp(x, -0.999999, 0.999999)) for x in v]
            for v in st.vals]
    return MixState(st.d, st.p, copy(st.mix), vals, copy(st.u), :hermite)
end

"""
    continue_from_hermite(st; alpha0, ratio, tol, log_io, halves) -> (state, alpha_reached, ok)

Homotopy from the Gaussian to the uniform weight through the symmetric
Jacobi family (1−x²)^α on [−1,1]: for large α the weight is ≈ exp(−α x²),
so the Hermite rule scaled by 1/√(2α) solves the α-problem to high accuracy;
α is then stepped down geometrically (`ratio`) to 0, each step warm-started
from the previous solution.  A failed step is bisected (up to 6 times)
before giving up, in which case the last converged state and its α are
returned with ok = false.  Unlike the quantile maps (`from_hermite`), which
measured as non-basins for the uniform equations, every step here is a
small perturbation of a solved problem — the standard way to carry a
structure across a family of weights.  Positive weights are structural
(w = eᵘ); the value box keeps nodes inside the cube.
"""
function continue_from_hermite(st::MixState; alpha0::Real = 0.0, ratio::Float64 = 0.7,
                               tol::Float64 = 1e-12, log_io = nothing,
                               alpha_end::Real = 0.0, vmax::Float64 = 1.0,
                               maxiter::Int = 600, x0::Float64 = 0.3)
    st.basis === :hermite || throw(ArgumentError("needs a :hermite state"))
    d, p = st.d, st.p
    halves = any(T.par != 0 for T in st.mix)
    ymax = maximum(abs, vcat(st.vals...); init = 1.0)
    # start where the outermost node sits at x = y/√(2α) ≈ x0 (the smaller
    # x0, the more Gaussian the α-weight is at the nodes, the better the
    # scaled Hermite rule solves the first problem)
    α = alpha0 > 0 ? Float64(alpha0) : max(ymax^2 / (2 * x0^2), 8.0)
    cur = MixState(d, p, copy(st.mix), [v ./ sqrt(2α) for v in st.vals], copy(st.u), :legendre)
    δ, ok = solve!(cur, Work(d, p; halves, basis = :legendre, alpha = α); tol, vmax, maxiter)
    log_io === nothing || println(log_io, "  continuation: α=$(round(α, sigdigits=4)) start δ=$(round(δ, sigdigits=3)) ok=$ok")
    ok || return (cur, α, false)
    αe = Float64(alpha_end)
    while α > αe
        # next target: geometric step, snapping to alpha_end once close
        αn = α * ratio
        αn < αe + 0.02 && (αn = αe)
        step_ok = false
        for _ in 1:6
            cand = copy(cur)
            δ, ok = solve!(cand, Work(d, p; halves, basis = :legendre, alpha = αn); tol, vmax, maxiter)
            if ok
                cur = cand; α = αn; step_ok = true
                log_io === nothing || println(log_io, "  continuation: α=$(round(α, sigdigits=4)) δ=$(round(δ, sigdigits=3)) max|v|=$(round(maximum(abs, vcat(cur.vals...); init = 0.0), sigdigits=4))")
                break
            end
            αn = (α + αn) / 2                      # bisect the step
            log_io === nothing || println(log_io, "  continuation: step to α=$(round(αn * 2 - α, sigdigits=4)) failed (δ=$(round(δ, sigdigits=3))), retrying at α=$(round(αn, sigdigits=4))")
        end
        step_ok || return (cur, α, false)
    end
    return (cur, α, true)
end

"""
    continue_to_hermite(st; alpha1, ratio, tol, x0, log_io) -> (state, alpha_reached, ok)

Homotopy from the uniform weight to the Gaussian one — `continue_from_hermite`
run backwards.  α climbs through the symmetric Jacobi family (1−x²)^α on
[−1,1] from 0 (uniform), each step warm-started from the previous solution and
bisected up to 6 times on failure; once α is large enough that the outermost
node sits at |x| ≤ `x0`, the weight is ≈ exp(−α x²) there, so scaling the
nodes by √(2α) lands a Gaussian rule, which is then polished in the :hermite
frame.  Returns the Hermite state, the α it reached, and whether the final
Hermite solve converged.

Added 2026-09-10 (user: "make sure you also build the Le->Gh route", then
"both") after the rule-level quantile map measured as a NON-BASIN in this
direction, exactly as `from_hermite` did in the other one:
`hermite_from_legendre.jl` failed all eleven solves it attempted at d=2 —
p31/p33/p35 (where a Gaussian rule is already banked and the uniform one is
6/24/48 nodes better) and p37/p39, under both the cdf and node maps, from
every banked uniform source.  This is the construction the forward docstring
recommends instead: every step is a small perturbation of a solved problem.

Cost note: the first step out of α = 0 is additive (`alpha1`), since a
geometric step cannot leave zero; from there the ratio is multiplicative and
> 1.  Reaching |x| ≤ 0.3 needs α ≈ y²/(2x₀²) with y the outermost Hermite
node, i.e. α ≈ 300 at p = 37 — about 18 steps at the default ratio.
"""
function continue_to_hermite(st::MixState; alpha1::Real = 0.5, ratio::Float64 = 1 / 0.7,
                             tol::Float64 = 1e-12, x0::Float64 = 0.3,
                             alpha_max::Real = 1e6, maxiter::Int = 4000,
                             tries::Int = 8, log_io = nothing)
    # maxiter 4000, not the forward direction's 600 (measured 2026-09-10):
    # climbing out of α = 0 the steps that "failed" at 600 were sitting at
    # δ = 1e-4…2e-6 and still descending, so every one of them was an
    # iteration limit, not a lost basin — d2 p41 bisected 0.5 → 0.03 with each
    # attempt converging further than the last.  Going down from large α the
    # steps are cheap; coming up they are not.
    st.basis === :legendre || throw(ArgumentError("needs a :legendre state"))
    ratio > 1 || throw(ArgumentError("ratio must exceed 1 — α climbs here"))
    d, p = st.d, st.p
    halves = any(T.par != 0 for T in st.mix)
    cur = copy(st)
    # the α = 0 problem must actually be solved at this state before stepping
    δ, ok = solve!(cur, Work(d, p; halves, basis = :legendre, alpha = 0.0);
                   tol, vmax = 1.0, maxiter)
    log_io === nothing || println(log_io, "  continuation: α=0 start δ=$(round(δ, sigdigits=3)) ok=$ok")
    ok || return (cur, 0.0, false)
    α = 0.0
    while true
        αn = α == 0 ? Float64(alpha1) : α * ratio
        step_ok = false
        for _ in 1:tries
            cand = copy(cur)
            # Scaling predictor (2026-09-10, measured).  The Jacobi(α,α)
            # weight on [-1,1] has variance 1/(2α+3), so the branch we want
            # CONTRACTS as α rises — σ(α) = 1/√(2α+3), which is 1/√(2α) in
            # the Gaussian limit, exactly the factor the forward direction
            # applies when it starts.  Without this the corrector simply
            # finds the nearest solution and the nodes never move: max|x|
            # sat at 0.997→0.9987 through every successful step at every
            # slack, and the continuation died at α ≈ 0.4–0.5 holding nodes
            # against the edge, where (1-x²)^α collapses and their weights
            # must blow up to compensate.  Rescaling first puts the warm
            # start on the contracting branch.
            for v in cand.vals
                v .*= sqrt((2α + 3) / (2αn + 3))
            end
            δ, ok = solve!(cand, Work(d, p; halves, basis = :legendre, alpha = αn);
                           tol, vmax = 1.0, maxiter)
            if ok
                cur = cand; α = αn; step_ok = true
                log_io === nothing || println(log_io,
                    "  continuation: α=$(round(α, sigdigits=4)) δ=$(round(δ, sigdigits=3)) " *
                    "max|x|=$(round(maximum(abs, vcat(cur.vals...); init = 0.0), sigdigits=4))")
                break
            end
            αn = (α + αn) / 2                      # bisect back toward the solved α
            log_io === nothing || println(log_io,
                "  continuation: step to α=$(round(αn * 2 - α, sigdigits=4)) failed " *
                "(δ=$(round(δ, sigdigits=3))), retrying at α=$(round(αn, sigdigits=4))")
        end
        step_ok || return (cur, α, false)
        maximum(abs, vcat(cur.vals...); init = 0.0) ≤ x0 && break
        α ≥ alpha_max && return (cur, α, false)
    end
    # hand off: the α-weight is ≈ exp(−α x²) at these nodes, so y = x√(2α)
    her = MixState(d, p, copy(cur.mix), [v .* sqrt(2α) for v in cur.vals],
                   copy(cur.u), :hermite)
    δ, ok = solve!(her, Work(d, p; halves, basis = :hermite); tol, maxiter)
    log_io === nothing || println(log_io,
        "  handoff at α=$(round(α, sigdigits=4)): hermite δ=$(round(δ, sigdigits=3)) ok=$ok")
    return (her, α, ok)
end

# ---------------------------------------------------------------------------
# structure-space hop (2026-08-16)
# ---------------------------------------------------------------------------
"""
    reshape!(st, rng, types) -> Bool

Structure-space hop: replace one orbit by a type of comparable-or-LARGER size
(up to 2x), or append a fresh light orbit.  Either move may raise the node
count; the caller re-solves, re-eliminates, and accepts only a strictly lower
final count — so the non-monotonicity lives inside the move and banking is
untouched.  This is the escape hatch `eliminate!` lacks: its moves are
monotone (drop/zero/merge, swap only to smaller types), and at high p the
logs show incumbents frozen for 20+ segments because no downhill-only path
exists from their structure.  Returns false if no applicable move was found
(caller falls back to a value jitter).
"""
function reshape!(st::MixState, rng::AbstractRNG, types::Vector{OType})
    if rand(rng) < 0.4 || length(st.mix) < 2
        # append: a light fresh orbit barely perturbs the warm start but gives
        # the follow-up elimination a different descent path to specialize
        nonorigin = [T for T in types if T.k > 0]
        T = nonorigin[rand(rng, 1:length(nonorigin))]
        v = random_values(rng, T.k, st.p; basis = st.basis)
        push!(st.mix, T)
        push!(st.vals, v)
        push!(st.u, minimum(st.u) - 1.0)
    else
        o = rand(rng, 1:length(st.mix))
        sz = max(st.mix[o].size, 1)
        opts = [T for T in types if sz ÷ 2 ≤ T.size ≤ 2sz &&
                !(T.k == 0 && any(S.k == 0 for S in st.mix))]
        isempty(opts) && return false
        T2 = opts[rand(rng, 1:length(opts))]
        st.mix[o] = T2
        st.vals[o] = random_values(rng, T2.k, st.p; basis = st.basis)
        st.u[o] += log(sz / max(T2.size, 1))   # preserve the orbit's total mass
    end
    return true
end

# ---------------------------------------------------------------------------
# search driver
# ---------------------------------------------------------------------------
"""
    orbit_search(d, p; seconds, rng, best_nodes, verify, on_improve, log_io)

Random-restart + greedy-reduction search.  `verify(nodes, weights)` must
return the monomial exactness error; a state is only reported through
`on_improve(st, δ, nodes, weights, exactness)` if that error is ≤ `extol`.
Campaign callers pass the RELATIVE (backward-error) form of the check with
extol = 1e-9: the absolute form amplifies a machine-precision residual by
the raw moment magnitude (p-1)!! and rejects provably exact rules for p ≥ 17
(see verify_exactness in DesignedQuadrature.jl).
Returns the best verified node count found (or `best_nodes` if never beaten).

Refine-first knobs (2026-08-16, all defaulting to the pre-existing behavior;
measured at d=4 p≥21: 357 cold restarts, zero within 10% of the incumbent —
at high p the caller should shift the budget onto the incumbent):
  * `warm_prob` — chance that a restart AFTER the first re-seeds from the
    warm structure with a jitter drawn from `hop_jitter` (restart 1 keeps its
    traditional near-exact warm solve at σ = 0.01);
  * `hop_jitter` — (lo, hi) band the per-hop value jitter σ is drawn from;
  * `u_jitter` — hops also perturb the log-weights with σ_u = u_jitter·σ
    (0 = value-space only, the historical behavior);
  * `reshape_prob` — chance a hop mutates the orbit STRUCTURE via `reshape!`
    instead of jittering values (see that docstring; acceptance unchanged).
Every restart log line carries its seed class — warm / tmpl(d,p) / rand — so
the win rate per class is measurable per case (grep the seg logs).
"""
function orbit_search(d::Int, p::Int; seconds::Real = 300,
                      rng::AbstractRNG = Random.default_rng(),
                      best_nodes::Int = typemax(Int),
                      verify = nothing, on_improve = nothing,
                      warm::Union{Nothing,MixState} = nothing,
                      templates::Vector{MixState} = MixState[],
                      template_prob::Float64 = 0.45,
                      warm_prob::Float64 = 0.0,
                      hops_budget::Int = 3,
                      hop_jitter::Tuple{Float64,Float64} = (0.08, 0.23),
                      u_jitter::Float64 = 0.0,
                      reshape_prob::Float64 = 0.0,
                      factor_range = (1.6, 2.4),
                      halves::Bool = false,
                      basis::Symbol = :hermite,
                      tol::Float64 = 1e-12, extol::Float64 = 1e-8,
                      log_io = stdout, progress = nothing)
    check_basis(basis)
    W = Work(d, p; halves, basis)
    types = orbit_types(d; halves)
    # jitters are specified in Hermite units; the uniform frame is narrower
    js = jscale(basis, p)
    hopσ() = (hop_jitter[1] + (hop_jitter[2] - hop_jitter[1]) * rand(rng)) * js
    warm === nothing || warm.basis === basis ||
        throw(ArgumentError("warm state is $(warm.basis), search is $basis"))
    all(t.basis === basis for t in templates) ||
        throw(ArgumentError("a template's basis differs from the search basis $basis"))
    deadline = time() + seconds
    restart = 0
    best_run = typemax(Int)
    while time() < deadline
        restart += 1
        local st, tag
        if warm !== nothing && (restart == 1 || rand(rng) < warm_prob)
            st = copy(warm)
            # restart 1 re-solves the incumbent almost exactly; warm_prob
            # re-seeds start further out on the jitter ladder
            σ = restart == 1 ? 0.01 * js : hopσ()
            for v in st.vals
                v .+= σ .* randn(rng, length(v))
            end
            u_jitter > 0 && restart > 1 &&
                (st.u .+= (u_jitter * σ) .* randn(rng, length(st.u)))
            tag = "warm"
        elseif !isempty(templates) && rand(rng) < template_prob
            donor = templates[rand(rng, 1:length(templates))]
            st = seed_from_template(donor, d, p, rng;
                                    factor = 1.2 + 0.4 * rand(rng), types)
            tag = "tmpl($(donor.d),$(donor.p))"
        else
            fac = factor_range[1] + rand(rng) * (factor_range[2] - factor_range[1])
            # mostly uniform type sampling: heavy bias starves the initial
            # solve of genericity (0/8 convergence in d=4); elimination and
            # swap moves do the specializing instead
            st = random_state(d, p, rng; factor = fac, bias = rand(rng)^2 * 0.8, types, basis)
            tag = "rand"
        end
        δ, ok = solve!(st, W; tol)
        if !ok
            restart % 20 == 0 &&
                println(log_io, "  restart $restart [$tag]: no initial convergence (δ=$(round(δ, sigdigits=3)))")
            continue
        end
        eliminate!(st, W, rng; tol, log_io = nothing)
        # basin hopping: strong value jitter (or a structure reshape) +
        # re-elimination escapes local minima of the greedy reduction
        hops = hops_budget
        while hops > 0
            cand = copy(st)
            if !(reshape_prob > 0 && rand(rng) < reshape_prob &&
                 reshape!(cand, rng, types))
                σ = hopσ()
                for v in cand.vals
                    v .+= σ .* randn(rng, length(v))
                end
                u_jitter > 0 &&
                    (cand.u .+= (u_jitter * σ) .* randn(rng, length(cand.u)))
            end
            _, hok = solve!(cand, W; tol)
            if hok
                eliminate!(cand, W, rng; tol, log_io = nothing)
                if mix_nodes(cand) < mix_nodes(st)
                    st = cand
                    continue          # hop budget resets only on improvement
                end
            end
            hops -= 1
        end
        δ, ok = solve!(st, W; tol)
        n = mix_nodes(st)
        if n < best_run || restart ≤ 5 || restart % 25 == 0
            println(log_io, "  restart $restart [$tag]: $(n) nodes  δ=$(round(δ, sigdigits=3))  [$(describe(st))]")
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
                println(log_io, "  *** improvement: $n nodes via $tag (exactness $(round(ex, sigdigits=3)))")
                flush(log_io)
            else
                println(log_io, "  (candidate $n nodes [$tag] failed monomial verify: $(round(ex, sigdigits=3)))")
            end
        end
    end
    return best_nodes
end

end # module
