#=
LaguerreDQ.jl -- designed quadrature under S_d symmetry (coordinate
permutations only) for the product GAMMA weight on the orthant [0,∞)^d,

    w(x) = ∏_i x_i^a e^{-x_i} / Γ(a+1),      a > -1,

a = 0 the exponential weight (χ²₂/2), a = k/2 − 1 the χ²_k weight up to the
scale factor 2 (2026-08-29, PROPOSAL_nonnormal_native_search.md §1 item 2).

Why.  The package offers this family only as quantile-transformed Gaussian
rules, which are not exact, and tensor Gauss–Laguerre is the only exact
option (((p+1)/2)^d nodes).  A native designed rule is new ground: the
weight has no sign symmetry, so the group is S_d, there is no Möller-type
bound, and the support has a boundary at 0.

Ansatz.  A rule is a union of S_d orbits.  An orbit type is (mults, z): z
coordinates are 0 (on the boundary — allowed), the rest split into groups of
equal value, group sizes mults[1..k].  Free parameters: the k distinct
NONNEGATIVE values and one weight.  Orbit = distinct permutations of the
pattern: size = d!/(z!·∏ mults!).  No sign flips, no central pair.

Conditions.  One per sorted α (permutation invariance), for EVERY α with
|α| ≤ p — no parity kills anything here: Σ_nodes w ∏_i L̃_{α_i}(x_i) =
δ_{α=0}, with L̃_n the generalized Laguerre polynomials orthonormal for the
probability weight above (three-term recurrence α_n = 2n+a+1,
β_n = √(n(n+a))).  Count = #partitions of 0..p into ≤ d parts — e.g. 83 at
d=3 p=11 against B_3's 22 — so expect node counts above the Gaussian ones.

Solver, elimination, hops, templates: the SymmetricDQ/PermCentralDQ recipe.
Differences: the LM step is projected onto the box 0 ≤ v ≤ vmax (a value
sitting at 0 is a node on the boundary face), initial weights fall off like
e^{-Σx/2}, random values and jitters are scaled to the Gauss–Laguerre node
range, and same-d donors scale radially by (p+1)/(p'+1).

`from_hermite(st_B)` maps a B_d Gaussian state through the quantile map
y ↦ −log Φ(−y) (or the Gamma quantile): the map is not odd, so every B_d
orbit splits into ∏(m_j+1) S_d orbits (one per count of positive signs in
each value group, zeros going to the median).  Measured for the uniform
weight (METHODS.md §17), transformed Gaussian structures are not solution
basins; they are offered here as donors of orbit mixes only.
=#
module LaguerreDQ

using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "WeightTransforms.jl"))
using .WeightTransforms: gauss_to_exp, gauss_to_gamma, gauss_laguerre_nodes

# Weighted-basis formulation (2026-09-07, user request).  The moment conditions
#   Σ_o w_o Π_i L̃_{α_i}(x_oi) = δ_α
# are evaluated as  Σ_o w̃_o Π_i ψ_{α_i}(x_oi)  with the Laguerre FUNCTIONS
#   ψ_n(x) = L̃_n(x) e^{-x/2}   (bounded on [0,∞) for every n)
# and the orbit weight relative to the square root of the density,
#   w̃_o = w_o e^{Σ_i x_oi / 2} = exp(ũ_o),  ũ_o = u_o + Σ_j m_j v_oj / 2.
# Identical algebra, but the factors the solver multiplies are O(1) instead of
# e^{-55} × 4e5 (d2 p19 corner orbits), the LM variables are (vals, ũ) instead
# of (vals, u), and the Marquardt damping is scaled by the diagonal so that
# light orbits are not frozen by a uniform λ.  The STATE still stores u (the
# absolute log weight per node): sidecars, expand_rule, save/load and every
# move are untouched.
# Measured the same day (d2 p17 cold starts, 12 seeds, 600 and 10 000 LM
# iterations; d2 p19 drops from the 99-node state): the weighted form converges
# no more often and reaches no lower residuals than the old one — the Jacobian
# condition at a random start is 3e7 either way, and the LM stalls in the same
# flat valleys.  So it is OFF by default and selectable with LAGUERRE_WEIGHTED=1;
# the obstacle is the search geometry (basin failure), not the scaling.
const WEIGHTED = get(ENV, "LAGUERRE_WEIGHTED", "0") != "0"
const DEBUG = get(ENV, "LAGUERRE_DEBUG", "0") != "0"     # per-iteration LM trace on stderr

export OType, build_type, orbit_types, conditions, MixState, mix_nodes,
       mix_params, random_state, solve!, eliminate!, expand_rule,
       orbit_search, save_state, load_state, describe, Work, from_hermite,
       gauss_laguerre_rule, tensor_state

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
    size::Int            # orbit cardinality = #rows
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
    OType(copy(mults), z, length(mults), asg, length(perms))
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

# all orbit types; z = d is the origin (a single point, k = 0)
function orbit_types(d::Int)
    types = OType[]
    for z in 0:d, m in partitions_le(d - z, d)
        push!(types, build_type(m, z, d))
    end
    return types
end

# invariant conditions: sorted α, ALL degrees 0..p, padded; α = 0 first
conditions(d::Int, p::Int) =
    [vcat(λ, zeros(Int, d - length(λ))) for n in 0:p for λ in partitions_le(n, d)]

# ---------------------------------------------------------------------------
# mix state
# ---------------------------------------------------------------------------
mutable struct MixState
    d::Int
    p::Int
    mix::Vector{OType}
    vals::Vector{Vector{Float64}}   # per orbit, length k — NONNEGATIVE values
    u::Vector{Float64}              # log weight per orbit (weight per node)
    alpha::Float64                  # Laguerre parameter a of the weight
end

mix_nodes(st::MixState)  = sum(T.size for T in st.mix; init = 0)
mix_params(st::MixState) = sum(T.k + 1 for T in st.mix; init = 0)

Base.copy(st::MixState) = MixState(st.d, st.p, copy(st.mix),
                                   [copy(v) for v in st.vals], copy(st.u), st.alpha)

function describe(st::MixState)
    parts = String[]
    for (o, T) in enumerate(st.mix)
        sig = isempty(T.mults) ? "origin" :
              join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "")
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
    alpha::Float64
end
function Work(d::Int, p::Int; alpha::Real = 0.0)
    alpha > -1 || throw(ArgumentError("Laguerre parameter must exceed -1"))
    c = conditions(d, p)
    Work(c, length(c), Float64(alpha))
end

# node range of the weight at degree p: the largest Gauss–Laguerre node
node_range(p::Int, a::Float64) = gauss_laguerre_nodes(cld(p + 1, 2), a)[end]
# value box: comfortably beyond the Gauss–Laguerre range
basis_vmax(p::Int, a::Float64) = 1.6 * node_range(p, a) + 4.0
# jitters are specified in Hermite units (range √(2p+2)); rescale
jscale(p::Int, a::Float64) = node_range(p, a) / sqrt(2p + 2)

# orthonormal generalized Laguerre values and derivatives at x, n = 0..p
#   x p_n = β_{n+1} p_{n+1} + α_n p_n + β_n p_{n-1},  α_n = 2n+a+1, β_n = √(n(n+a))
function poly_cols!(H::Matrix{Float64}, Hd::Matrix{Float64}, col::Int,
                    x::Float64, p::Int, a::Float64)
    H[1, col] = 1.0; Hd[1, col] = 0.0
    if p == 0
        WEIGHTED && (H[1, col] = exp(-x / 2); Hd[1, col] = -H[1, col] / 2)
        return
    end
    β(n) = sqrt(n * (n + a))
    b1 = β(1)
    H[2, col] = (x - (a + 1)) / b1; Hd[2, col] = 1 / b1
    for n in 1:(p - 1)
        an = 2n + a + 1; bn = β(n); bn1 = β(n + 1)
        H[n+2, col]  = ((x - an) * H[n+1, col] - bn * H[n, col]) / bn1
        Hd[n+2, col] = (H[n+1, col] + (x - an) * Hd[n+1, col] - bn * Hd[n, col]) / bn1
    end
    if WEIGHTED                      # ψ_n = L̃_n e^{-x/2},  ψ_n' = (L̃_n' - L̃_n/2) e^{-x/2}
        sc = exp(-x / 2)
        @inbounds for n in 1:p+1
            Hd[n, col] = (Hd[n, col] - H[n, col] / 2) * sc
            H[n, col]  *= sc
        end
    end
end

# log of the orbit's LM weight variable: ũ = u + Σ_j m_j v_j / 2 when weighted, else u
orbit_lw(st::MixState, o::Int) = WEIGHTED ?
    st.u[o] + sum(st.mix[o].mults[j] * st.vals[o][j] for j in 1:st.mix[o].k; init = 0.0) / 2 :
    st.u[o]

# R (length C) and optionally J (C × N); returns ‖R‖
function eval_rj!(R::Vector{Float64}, J::Union{Nothing,Matrix{Float64}},
                  st::MixState, W::Work)
    d, p = st.d, st.p
    st.alpha == W.alpha ||
        throw(ArgumentError("state alpha $(st.alpha) ≠ work alpha $(W.alpha)"))
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
        w = exp(orbit_lw(st, o))          # w̃_o when WEIGHTED (see the note at the top)
        poly_cols!(H, Hd, 1, 0.0, p, W.alpha)
        for j in 1:T.k
            poly_cols!(H, Hd, j + 1, st.vals[o][j], p, W.alpha)
        end
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
                        dS[j] += pre[i] * suf[i+1] * Hd[α[i] + 1, j + 1]
                    end
                end
            end
            R[c] += w * S
            if J !== nothing
                for j in 1:T.k
                    J[c, off + j] += w * dS[j]
                end
                J[c, off + T.k + 1] += w * S      # d/du of exp(u)·(...)
            end
        end
        off += T.k + 1
    end
    R[1] -= 1.0
    return norm(R)
end

# ---------------------------------------------------------------------------
# Levenberg-Marquardt (box-projected) + minimum-norm Newton polish
# ---------------------------------------------------------------------------
function pack!(θ, st::MixState)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.k
            θ[off + j] = st.vals[o][j]
        end
        θ[off + T.k + 1] = orbit_lw(st, o)
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
        # 2026-09-07: the floor was -46.  The corner orbits of a d-fold Gauss-
        # Laguerre grid have log-weights down to -55 (d2 p19), -94 (d3 p21) and
        # -121 (d5 p17); clamping them at -46 destroys the grid's exactness
        # (residual floors 1e-8 .. 7e-2, exactly the logged "jittered seed did
        # not re-solve" values) and makes every tensor descent reject its first
        # move.  Float64 is fine down to about -700.
        lw = θ[off + T.k + 1]
        WEIGHTED && (lw -= sum(T.mults[j] * st.vals[o][j] for j in 1:T.k; init = 0.0) / 2)
        st.u[o] = clamp(lw, -690.0, 5.0)
        off += T.k + 1
    end
    return st
end

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

LM with Nielsen damping (dual form when conditions < unknowns), steps
projected onto the box 0 ≤ v ≤ vmax, then minimum-norm Newton polish.
"""
function solve!(st::MixState, W::Work; tol::Float64 = 1e-12,
                maxiter::Int = 600, λ0::Float64 = 1e-3,
                vmax::Float64 = basis_vmax(st.p, W.alpha))
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
                        # weighted: Marquardt scaling — damp each column relative
                        # to its own size (floored), so a light orbit's node can
                        # still move; unweighted: the uniform λ as before
                        A[i, i] += WEIGHTED ? λ * max(A[i, i] / colmax, 1e-6) : λ
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
            @inbounds for i in eachindex(θt)          # project onto the box
                isval[i] && (θt[i] = clamp(θt[i], 0.0, vmax))
            end
            dθ = θt .- θ
            unpack!(st2, θt)
            δn = eval_rj!(Rt, nothing, st2, W)
            pred = δ^2 - sum(abs2, R .+ J * dθ)
            ρ = pred > 0 ? (δ^2 - δn^2) / pred : -1.0
            DEBUG && @printf("    it δ=%.3e δn=%.3e pred=%.3e ρ=%.3f λ=%.3e ν=%.1e |dθ|=%.2e\n", δ, δn, pred, ρ, λ, ν, norm(dθ))
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
                (λ > 1e14 || ν > 1e8) && (DEBUG && println("    abort: λ=$λ ν=$ν"); break)
            end
        end
        accepted || break
    end

    if tol < δ < 1e-6        # 2026-09-07: never polish a state that is already at tol
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
# random initial states
# ---------------------------------------------------------------------------
function random_values(rng::AbstractRNG, k::Int, p::Int, a::Float64)
    rmax = node_range(p, a)
    sep = 0.02 * rmax
    v = Float64[]
    while length(v) < k
        # bias toward the bulk of the weight (its mass is at x ≲ a+1+√p)
        c = rmax * rand(rng)^1.5
        all(abs(c - x) > sep for x in v) && push!(v, c)
    end
    return v
end

# initial log-weight: the weight falls off like e^{-x}; half that in log space
init_u(T, v) = T.k == 0 ? 0.0 : -0.5 * sum(T.mults[j] * v[j] for j in 1:T.k)

function random_state(d::Int, p::Int, rng::AbstractRNG;
                      factor::Float64 = 1.3, types = orbit_types(d),
                      bias::Float64 = 1.0, alpha::Real = 0.0)
    a = Float64(alpha)
    C = length(conditions(d, p))
    target = ceil(Int, factor * C)
    nonorigin = [T for T in types if T.k > 0]
    wts = [((T.k + 1) / T.size)^bias for T in nonorigin]
    cw = cumsum(wts) ./ sum(wts)
    mix = OType[]
    if rand(rng) < 0.5
        push!(mix, types[findfirst(T -> T.k == 0, types)])
    end
    while sum(T.k + 1 for T in mix; init = 0) < target
        r = rand(rng)
        push!(mix, nonorigin[findfirst(≥(r), cw)])
    end
    vals = [random_values(rng, T.k, p, a) for T in mix]
    u = [init_u(T, vals[o]) for (o, T) in enumerate(mix)]
    st = MixState(d, p, mix, vals, u, a)
    tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
    st.u .-= log(tot)
    return st
end

# ---------------------------------------------------------------------------
# structure transfer
# ---------------------------------------------------------------------------
"""
    seed_from_template(donor, d, p, rng; factor) -> MixState

Same-d donors scale radially by (p+1)/(p_src+1) (Gauss–Laguerre ranges grow
linearly in p); (d-1)-donors embed with one more zero coordinate.  Filler
orbits to `factor` × #conditions.
"""
function seed_from_template(donor::MixState, d::Int, p::Int, rng::AbstractRNG;
                            factor::Float64 = 1.35, types = orbit_types(d))
    donor.d == d || donor.d == d - 1 ||
        throw(ArgumentError("donor must have dimension d or d-1"))
    a = donor.alpha
    scale = (p + 1) / (donor.p + 1)
    vmax = basis_vmax(p, a)
    js = jscale(p, a)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    for (o, T) in enumerate(donor.mix)
        z = T.z + (d - donor.d)
        push!(mix, build_type(copy(T.mults), z, d))
        v = clamp.(donor.vals[o] .* scale, 0.0, vmax)
        v .+= (0.03 * js) .* randn(rng, length(v))
        push!(vals, clamp.(v, 0.0, vmax))
        push!(u, donor.u[o])
    end
    C = length(conditions(d, p))
    target = ceil(Int, factor * C)
    nonorigin = [T for T in types if T.k > 0]
    while sum(T.k + 1 for T in mix; init = 0) < target
        T = nonorigin[rand(rng, 1:length(nonorigin))]
        push!(mix, T)
        v = random_values(rng, T.k, p, a)
        push!(vals, v)
        push!(u, init_u(T, v))
    end
    st = MixState(d, p, mix, vals, u, a)
    tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
    st.u .-= log(tot)
    return st
end

# ---------------------------------------------------------------------------
# tensor Gauss-Laguerre seed
# ---------------------------------------------------------------------------
"""
    gauss_laguerre_rule(m, a) -> (x, w)

The m-point generalized Gauss-Laguerre rule for the PROBABILITY weight
`x^a e^{-x} / Γ(a+1)` on [0,∞) (Golub-Welsch: nodes are the Jacobi-matrix
eigenvalues, weights the squared first components of its eigenvectors).
Weights sum to 1 and the rule is exact to degree 2m-1.
"""
function gauss_laguerre_rule(m::Int, a::Float64 = 0.0)
    m ≥ 1 || throw(ArgumentError("m must be positive"))
    m == 1 && return ([1.0 + a], [1.0])
    E = eigen(SymTridiagonal([2k + a + 1.0 for k in 0:m-1],
                             [sqrt(k * (k + a)) for k in 1:m-1]))
    ord = sortperm(E.values)
    return E.values[ord], vec(E.vectors[1, ord]) .^ 2
end

"""
    tensor_state(d, p; alpha, m) -> MixState

The d-fold tensor product of the m-point Gauss-Laguerre rule, written exactly
as an S_d orbit mix.  With `m = cld(p+1, 2)` the product rule is exact to
total degree 2m-1 ≥ p, so the returned state is a *solution* of the degree-p
conditions with strictly positive weights — a feasible starting point, which
is what random restarts have failed to find on the un-banked cells (a cold
`orbit_search` at d=2 p=17 logged ~97,000 restarts with zero convergences,
δ pinned at 1e-6..4e-6; STATUS_2026-09-05).  Feed it to `eliminate!` to
descend toward a minimal rule.

The grid is S_d-invariant and its weight is a symmetric product, so it
decomposes exactly into orbits: one per multiset {i_1 ≤ … ≤ i_d} of 1-D
indices, whose distinct permutations are precisely the grid points carrying
that multiset, all sharing the weight ∏_j w[i_j].  Cost is m^d nodes in
C(m+d-1, d) orbits.
"""
function tensor_state(d::Int, p::Int; alpha::Real = 0.0, m::Int = cld(p + 1, 2))
    d ≥ 1 || throw(ArgumentError("d must be positive"))
    2m - 1 ≥ p ||
        throw(ArgumentError("m = $m is exact only to degree $(2m - 1) < p = $p"))
    a = Float64(alpha)
    x, w = gauss_laguerre_rule(m, a)
    lw = log.(w)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    idx = ones(Int, d)                       # nondecreasing index multiset
    while true
        mults = Int[]; vv = Float64[]; lu = 0.0
        j = 1
        while j ≤ d                          # runs of equal indices = value groups
            t = j
            while t < d && idx[t + 1] == idx[j]
                t += 1
            end
            mul = t - j + 1
            push!(mults, mul); push!(vv, x[idx[j]]); lu += mul * lw[idx[j]]
            j = t + 1
        end
        push!(mix, build_type(mults, 0, d)); push!(vals, vv); push!(u, lu)
        i = d                                # next multiset, colex order
        while i ≥ 1 && idx[i] == m
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for t in (i + 1):d
            idx[t] = idx[i]
        end
    end
    return MixState(d, p, mix, vals, u, a)
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
        m2 = copy(T.mults); v2 = copy(st.vals[o])
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
               on_improve, deadline) -> st

Greedy reduction: attempt the ranked moves, reconverging warm (plus jittered
retries) after each; accept the first success.  When blocked, swap an orbit
for a strictly smaller type with fresh values.

`on_improve(st, δ)`, when given, is called after every accepted reduction, so
a long descent can bank its intermediate rules instead of losing them if the
worker is stopped.  `deadline` (an absolute `time()`) bounds the descent; the
state is left at the best reduction reached.
"""
function eliminate!(st::MixState, W::Work, rng::AbstractRNG;
                    tol::Float64 = 1e-12, jitter_tries::Int = 1,
                    swap_rounds::Int = 12, log_io = nothing,
                    on_improve = nothing, deadline::Float64 = Inf)
    types = orbit_types(st.d)
    js = jscale(st.p, W.alpha)
    while true
        time() < deadline || break
        improved = false
        for move in reduction_moves(st)
            mix_nodes(st) ≤ 1 && break
            time() < deadline || break
            cand = apply_move(st, move)
            mix_params(cand) == 0 && continue
            δ, ok = solve!(cand, W; tol)
            if !ok
                for _ in 1:jitter_tries
                    c2 = copy(cand)
                    for v in c2.vals
                        v .= abs.(v .+ (0.02 * js) .* randn(rng, length(v)))
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
            on_improve === nothing || on_improve(st, δ)
            break
        end
        if !improved
            wt = [exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix)]
            ord = sortperm(wt)
            for _ in 1:swap_rounds
                time() < deadline || break
                o = ord[min(1 + floor(Int, abs(randn(rng)) * 2), length(ord))]
                smaller = [T for T in types if T.size < st.mix[o].size &&
                           !(T.k == 0 && any(S.k == 0 for S in st.mix))]
                isempty(smaller) && continue
                T2 = smaller[rand(rng, 1:length(smaller))]
                cand = copy(st)
                cand.mix[o] = T2
                cand.vals[o] = random_values(rng, T2.k, st.p, W.alpha)
                cand.u[o] = st.u[o] + log(st.mix[o].size / max(T2.size, 1))
                δ, ok = solve!(cand, W; tol)
                ok || continue
                st.mix = cand.mix; st.vals = cand.vals; st.u = cand.u
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
"""Expand to the full node set (all distinct permutations), weights per node."""
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
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::MixState, δ::Float64)
    open(path, "w") do io
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(mix_nodes(st)) resid=$δ group=S alpha=$(st.alpha)")
        for (o, T) in enumerate(st.mix)
            println(io, join(T.mults, ","), "|", T.z, "|",
                    join(st.vals[o], ","), "|", st.u[o])
        end
    end
end

function load_state(path::AbstractString, d::Int, p::Int)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
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
        push!(mix, build_type(m, z, d))
        push!(vals, v)
        push!(u, parse(Float64, parts[4]))
    end
    return MixState(d, p, mix, vals, u, a)
end

# ---------------------------------------------------------------------------
# quantile-map donor from a Gaussian B_d state
# ---------------------------------------------------------------------------
"""
    from_hermite(mix, vals, u, d, p; alpha) -> MixState

Takes the raw fields of a SymmetricDQ B_d state (mults/z per orbit, unsigned
values, log-weights) and maps it through the Gamma(alpha+1) quantile of Φ:
each B_d orbit of type (mults, z) splits into ∏(m_j+1) S_d orbits, one per
choice of how many of each group's m_j coordinates are positive (mapped to
T(+v_j)) versus negative (T(−v_j)); zero coordinates go to T(0).  Weights
per node carry over.  Not a solution basin — a donor of orbit mixes.
"""
function from_hermite(mults::Vector{Vector{Int}}, zs::Vector{Int},
                      vals::Vector{Vector{Float64}}, u::Vector{Float64},
                      d::Int, p::Int; alpha::Real = 0.0)
    a = Float64(alpha)
    T(y) = a == 0 ? gauss_to_exp(y) : gauss_to_gamma(y, a + 1)
    mix = OType[]; nv = Vector{Float64}[]; nu = Float64[]
    for o in eachindex(mults)
        m = mults[o]; z = zs[o]; v = vals[o]
        k = length(m)
        # enumerate sign-count vectors s ∈ ∏ {0..m_j}
        ranges = [0:m[j] for j in 1:k]
        for s in Iterators.product(ranges...)
            groups = Tuple{Float64,Int}[]           # (value, multiplicity)
            for j in 1:k
                s[j] > 0 && push!(groups, (T(v[j]), s[j]))
                m[j] - s[j] > 0 && push!(groups, (T(-v[j]), m[j] - s[j]))
            end
            z > 0 && push!(groups, (T(0.0), z))
            # merge equal values, drop nothing (all values are > 0 here)
            sort!(groups; by = first)
            merged = Tuple{Float64,Int}[]
            for g in groups
                if !isempty(merged) && abs(merged[end][1] - g[1]) < 1e-12
                    merged[end] = (merged[end][1], merged[end][2] + g[2])
                else
                    push!(merged, g)
                end
            end
            push!(mix, build_type([g[2] for g in merged], 0, d))
            push!(nv, [g[1] for g in merged])
            push!(nu, u[o])
        end
    end
    return MixState(d, p, mix, nv, nu, a)
end

# ---------------------------------------------------------------------------
# search driver
# ---------------------------------------------------------------------------
"""
    orbit_search(d, p; seconds, rng, best_nodes, verify, on_improve, log_io, alpha, …)

Random-restart + greedy-reduction search over S_d orbit mixes for the
Gamma(alpha+1) product weight.  Same contract as SymmetricDQ.orbit_search
(warm incumbent, donor templates, hops, verify gate, progress callback).
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
                      factor_range = (1.6, 2.4),
                      alpha::Real = 0.0,
                      tol::Float64 = 1e-12, extol::Float64 = 1e-8,
                      log_io = stdout, progress = nothing)
    a = Float64(alpha)
    W = Work(d, p; alpha = a)
    types = orbit_types(d)
    js = jscale(p, a)
    hopσ() = (hop_jitter[1] + (hop_jitter[2] - hop_jitter[1]) * rand(rng)) * js
    warm === nothing || warm.alpha == a ||
        throw(ArgumentError("warm state has alpha $(warm.alpha), search has $a"))
    all(t.alpha == a for t in templates) ||
        throw(ArgumentError("a template's alpha differs from the search alpha $a"))
    deadline = time() + seconds
    restart = 0
    best_run = typemax(Int)
    while time() < deadline
        restart += 1
        local st, tag
        if warm !== nothing && (restart == 1 || rand(rng) < warm_prob)
            st = copy(warm)
            σ = restart == 1 ? 0.01 * js : hopσ()
            for v in st.vals
                v .= abs.(v .+ σ .* randn(rng, length(v)))
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
            st = random_state(d, p, rng; factor = fac, bias = rand(rng)^2 * 0.8, types, alpha = a)
            tag = "rand"
        end
        δ, ok = solve!(st, W; tol)
        if !ok
            restart % 20 == 0 &&
                println(log_io, "  restart $restart [$tag]: no initial convergence (δ=$(round(δ, sigdigits=3)))")
            continue
        end
        eliminate!(st, W, rng; tol, log_io = nothing)
        hops = hops_budget
        while hops > 0
            cand = copy(st)
            σ = hopσ()
            for v in cand.vals
                v .= abs.(v .+ σ .* randn(rng, length(v)))
            end
            u_jitter > 0 &&
                (cand.u .+= (u_jitter * σ) .* randn(rng, length(cand.u)))
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
