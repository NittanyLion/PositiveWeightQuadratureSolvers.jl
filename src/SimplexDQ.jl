#=
SimplexDQ.jl -- designed quadrature under the simplex symmetry group
G = S_{d+1} × Z_2 acting on R^d, for the Gaussian weight N(0, I_d).

Why.  The campaign's largest single wins came from NEW symmetry groups
(F_4, SC, icosahedral), and d=5 has the worst node/Möller ratios in the bank.
Let e_0..e_d be the unit vectors from the origin to the vertices of a regular
simplex (Σ e_i = 0, e_i·e_j = -1/d).  G permutes the e_i (S_{d+1}) and
contains the central inversion (Z_2).  For d=3, G = O_h = B_3 and nothing is
gained; for d=4 (|G| = 240 vs |B_4| = 384) and d=5 (1440 vs 3840) G is NOT a
subgroup of B_d (it has elements of order 5 resp. 6, which B_d lacks) -- the
d=5 analog of the icosahedral move that gave d=3 its only at-bound rules.
Stroud-Secrest's 32-node d=5 p=5 rule (banked as hermite_d5_p5_n32.csv, found
by the SC search) IS G-invariant: ± the 6 simplex vertices (12 nodes) plus the
20-point orbit of (a,a,a,-a,-a,-a) -- 4 unknowns for the 4 G-invariant
conditions at p=5.

Frame.  The frame is chosen so that coordinate permutations of R^d are in G:
e_0 = -(1/√d)·1 and e_i = a f_i + b 1 (i = 1..d) with a = √((d+1)/d),
b = (1/√d - a)/d.  Then S_d × Z_2 (the SC group, PermCentralDQ.jl) is a
subgroup of G, so a G-invariant rule is SC-invariant in this frame and the SC
condition set (one condition per SORTED exponent vector with |α| even,
orthonormal Hermite products) is complete for it -- Sobolev's theorem.  Only
#{monomials in the power sums p_2..p_{d+1} of even degree ≤ p} of those
conditions are independent (the true G-invariant count, `invariant_count`);
the redundant ones are kept because they cost little and avoid a numerical
rank selection.  Why not an explicit invariant basis (power-sum monomials,
orthonormalized)?  Because at p ≥ 13 the Gram matrix of that basis is
Hilbert-like (condition ≫ 1e14), so the orthonormalizing transform destroys
the accuracy the 1e-11 exactness gate needs.  Hermite products evaluated by
the three-term recurrence on the enumerated orbit rows are stable, and the
cost is the same design as SC's: rows = distinct permutations of the value
pattern over d+1 slots instead of d.

Ansatz.  A point is x = Σ_i c_i e_i = E c with c ∈ R^{d+1}; adding a
constant to c does nothing (E·1 = 0), so c is normalized to Σ_i m_i c_i = 0.
An orbit TYPE is the multiplicity pattern of equal c-values:
  * non-sym: a partition (m_1..m_k) of d+1 into k ≥ 2 groups; free values
    v_1..v_{k-1}, the last one fixed by the normalization; the orbit is the
    distinct permutations of the pattern AND their negatives (size 2·rows).
  * sym: the multiset {c_i} equals {-c_i}: pairs of groups (q_j, ±v_j) plus a
    middle group of m_0 zeros (2 Σ q_j + m_0 = d+1); free values v_1..v_r; the
    orbit is centrally symmetric on its own and must not be doubled (size =
    rows; half the rows are kept and the parity factor 2 restored).  The
    2-group partition (m, m) is always sym (v_2 = -v_1) and is listed only
    there; the origin is the sym type with no pairs.
One log-weight per orbit.  Everything else -- LM with Nielsen damping and the
dual form, minimum-norm Newton polish, greedy elimination (drop / merge /
zero / symmetrize / swap), basin hopping, donor templates, sidecar persistence
-- mirrors PermCentralDQ.jl, and the API names are kept parallel.
=#
module SimplexDQ

using LinearAlgebra, Random, Printf

export OType, build_type, orbit_types, conditions, invariant_count, Work,
       MixState, mix_nodes, mix_params, random_state, solve!, eliminate!,
       expand_rule, orbit_search, save_state, load_state, describe,
       simplex_frame, seed_from_template, brute_residual, group_generators

# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------
"""
    simplex_frame(d) -> E (d × (d+1))

Columns are the unit vectors to the vertices of a regular simplex: column 1
is -(1/√d)·1, column i+1 is a f_i + b 1, so coordinate permutations of R^d
permute columns 2..d+1.  E'E = (1 + 1/d) I - (1/d) 11', E·1 = 0.
"""
function simplex_frame(d::Int)
    a = sqrt((d + 1) / d)
    b = (1 / sqrt(d) - a) / d
    E = fill(b, d, d + 1)
    E[:, 1] .= -1 / sqrt(d)
    for i in 1:d
        E[i, i + 1] += a
    end
    return E
end

"""Generators of G as d×d matrices: a coordinate transposition, the simplex
reflection swapping e_0 and e_1 (through the hyperplane ⊥ e_0 - e_1), and
-I.  Used by the tests to check that expanded node sets are G-invariant."""
function group_generators(d::Int)
    E = simplex_frame(d)
    P = Matrix{Float64}(I, d, d)
    if d ≥ 2
        P[1, 1] = 0; P[2, 2] = 0; P[1, 2] = 1; P[2, 1] = 1
    end
    u = E[:, 1] - E[:, 2]
    Rf = Matrix{Float64}(I, d, d) - 2 * (u * u') / dot(u, u)
    return [P, Rf, -Matrix{Float64}(I, d, d)]
end

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

"""
    OType

`sym == false`: `mults` is a partition of d+1 into k ≥ 2 groups (any order),
`nfree = k-1`, labels 1..k.  `sym == true`: `mults` are the pair
multiplicities q_1..q_r, `m0` the middle multiplicity, `nfree = r`, labels
±j and 0.  `M[:, :, r]` maps the free values to the coordinates of row r
(y = M_r v); `signfac` is 2 for every type except the origin (1), whose
single row is its own negative.  `size` is the full orbit cardinality.
"""
struct OType
    sym::Bool
    mults::Vector{Int}
    m0::Int
    nfree::Int
    asg::Matrix{Int}           # rows × (d+1) labels
    M::Array{Float64,3}        # d × nfree × rows
    signfac::Float64
    size::Int
end

function build_type(mults::Vector{Int}, sym::Bool, m0::Int, d::Int)
    n = d + 1
    E = simplex_frame(d)
    if !sym
        m0 == 0 || throw(ArgumentError("non-sym types have no middle group"))
        sum(mults) == n || throw(ArgumentError("mults must partition d+1"))
        k = length(mults)
        k ≥ 2 || throw(ArgumentError("a non-sym type needs ≥ 2 groups (the origin is sym)"))
        pattern = Int[]
        for (j, m) in enumerate(mults), _ in 1:m
            push!(pattern, j)
        end
        perms = distinct_perms(pattern)
        rows = length(perms)
        asg = Matrix{Int}(undef, rows, n)
        M = zeros(d, k - 1, rows)
        A = zeros(n, k - 1)
        for (r, pm) in enumerate(perms)
            asg[r, :] .= pm
            fill!(A, 0.0)
            for i in 1:n
                lab = pm[i]
                if lab < k
                    A[i, lab] = 1.0
                else
                    for j in 1:(k - 1)
                        A[i, j] = -mults[j] / mults[k]
                    end
                end
            end
            M[:, :, r] .= E * A
        end
        return OType(false, copy(mults), 0, k - 1, asg, M, 2.0, 2 * rows)
    else
        r_ = length(mults)
        2 * sum(mults; init = 0) + m0 == n ||
            throw(ArgumentError("2·Σ pairs + middle must equal d+1"))
        if r_ == 0                                   # the origin
            asg = zeros(Int, 1, n)
            return OType(true, Int[], m0, 0, asg, zeros(d, 0, 1), 1.0, 1)
        end
        pattern = Int[]
        for (j, q) in enumerate(mults)
            for _ in 1:q
                push!(pattern, j)
            end
            for _ in 1:q
                push!(pattern, -j)
            end
        end
        append!(pattern, zeros(Int, m0))
        perms = distinct_perms(pattern)
        total = length(perms)
        keep = [pm for pm in perms if pm < -pm]      # one of each ± pair
        length(keep) * 2 == total || error("sym orbit did not split into ± pairs")
        rows = length(keep)
        asg = Matrix{Int}(undef, rows, n)
        M = zeros(d, r_, rows)
        A = zeros(n, r_)
        for (r, pm) in enumerate(keep)
            asg[r, :] .= pm
            fill!(A, 0.0)
            for i in 1:n
                lab = pm[i]
                lab == 0 && continue
                A[i, abs(lab)] = sign(lab)
            end
            M[:, :, r] .= E * A
        end
        return OType(true, copy(mults), m0, r_, asg, M, 2.0, total)
    end
end

nonsym_type(mults::Vector{Int}, d::Int) = build_type(mults, false, 0, d)
sym_type(pairs::Vector{Int}, m0::Int, d::Int) = build_type(pairs, true, m0, d)
origin_type(d::Int) = build_type(Int[], true, d + 1, d)

isorigin(T::OType) = T.sym && T.nfree == 0

"""All orbit types in dimension d, the origin first."""
function orbit_types(d::Int)
    n = d + 1
    types = OType[origin_type(d)]
    for s in 1:(n ÷ 2), q in partitions_le(s, s)
        push!(types, sym_type(q, n - 2s, d))
    end
    for m in partitions_le(n, n)
        length(m) ≥ 2 || continue
        (length(m) == 2 && m[1] == m[2]) && continue   # always sym: listed above
        push!(types, nonsym_type(m, d))
    end
    return types
end

# SC condition set: sorted alpha with |alpha| even, padded; alpha = 0 first.
conditions(d::Int, p::Int) =
    [vcat(λ, zeros(Int, d - length(λ)))
     for n in 0:2:p for λ in partitions_le(n, d)]

"""Number of independent G-invariant conditions: monomials in the power sums
p_2..p_{d+1} of even total degree ≤ p."""
function invariant_count(d::Int, p::Int)
    cnt = 0
    for n in 0:2:p
        # partitions of n into parts from {2, ..., d+1}
        cnt += count(λ -> all(x -> x ≥ 2, λ), partitions_le(n, n, d + 1))
    end
    return cnt
end

# ---------------------------------------------------------------------------
# mix state
# ---------------------------------------------------------------------------
mutable struct MixState
    d::Int
    p::Int
    mix::Vector{OType}
    vals::Vector{Vector{Float64}}   # per orbit, length nfree
    u::Vector{Float64}              # log weight per orbit (weight per node)
end

mix_nodes(st::MixState)  = sum(T.size for T in st.mix; init = 0)
mix_params(st::MixState) = sum(T.nfree + 1 for T in st.mix; init = 0)
has_origin(st::MixState) = any(isorigin, st.mix)

Base.copy(st::MixState) = MixState(st.d, st.p, copy(st.mix),
                                   [copy(v) for v in st.vals], copy(st.u))

function typesig(T::OType)
    if isorigin(T)
        return "0"
    elseif T.sym
        return join(string.(T.mults) .* "±", "") * (T.m0 > 0 ? "|$(T.m0)" : "")
    else
        return join(T.mults, "+")
    end
end

function describe(st::MixState)
    parts = String[]
    for T in st.mix
        push!(parts, "$(typesig(T))($(T.size))")
    end
    return "$(mix_nodes(st)) nodes = " * join(parts, " ")
end

# full c-pattern values of an orbit (non-sym: k values incl. the derived one)
function full_values(T::OType, v::Vector{Float64})
    if T.sym
        return copy(v)
    else
        k = length(T.mults)
        vk = -sum(T.mults[j] * v[j] for j in 1:(k - 1); init = 0.0) / T.mults[k]
        return vcat(v, vk)
    end
end

# ---------------------------------------------------------------------------
# residual and Jacobian
# ---------------------------------------------------------------------------
struct Work
    d::Int
    p::Int
    conds::Vector{Vector{Int}}
    C::Int          # number of conditions evaluated (SC count)
    CG::Int         # number of independent G-invariant conditions
end
Work(d::Int, p::Int) = (c = conditions(d, p); Work(d, p, c, length(c), invariant_count(d, p)))

function hermite_col!(H::Matrix{Float64}, col::Int, x::Float64, p::Int)
    H[1, col] = 1.0
    p ≥ 1 && (H[2, col] = x)
    for n in 1:(p - 1)
        H[n+2, col] = (x * H[n+1, col] - sqrt(n) * H[n, col]) / sqrt(n + 1)
    end
end

# R (length C) and optionally J (C × N); returns ‖R‖
function eval_rj!(R::Vector{Float64}, J::Union{Nothing,Matrix{Float64}},
                  st::MixState, W::Work)
    d, p = st.d, st.p
    fill!(R, 0.0)
    J === nothing || fill!(J, 0.0)
    kmax = maximum(T.nfree for T in st.mix; init = 0)
    H   = zeros(p + 1, d)
    y   = zeros(d)
    f   = zeros(d)
    g   = zeros(d)
    pre = zeros(d + 1)
    suf = zeros(d + 1)
    Sacc = zeros(W.C)
    dS   = zeros(W.C, max(kmax, 1))
    off = 0
    for (o, T) in enumerate(st.mix)
        w = exp(st.u[o])
        v = st.vals[o]
        nf = T.nfree
        fill!(Sacc, 0.0)
        (J !== nothing && nf > 0) && fill!(dS, 0.0)
        for r in axes(T.asg, 1)
            # coordinates of this row: y = M_r v
            @inbounds for i in 1:d
                s = 0.0
                for j in 1:nf
                    s += T.M[i, j, r] * v[j]
                end
                y[i] = s
            end
            for i in 1:d
                hermite_col!(H, i, y[i], p)
            end
            for (c, α) in enumerate(W.conds)
                @inbounds for i in 1:d
                    f[i] = H[α[i] + 1, i]
                end
                pre[1] = 1.0
                @inbounds for i in 1:d
                    pre[i+1] = pre[i] * f[i]
                end
                Sacc[c] += pre[d+1]
                if J !== nothing && nf > 0
                    suf[d+1] = 1.0
                    @inbounds for i in d:-1:1
                        suf[i] = suf[i+1] * f[i]
                    end
                    @inbounds for i in 1:d
                        g[i] = α[i] == 0 ? 0.0 :
                               pre[i] * suf[i+1] * sqrt(α[i]) * H[α[i], i]
                    end
                    @inbounds for j in 1:nf
                        s = 0.0
                        for i in 1:d
                            s += g[i] * T.M[i, j, r]
                        end
                        dS[c, j] += s
                    end
                end
            end
        end
        fac = w * T.signfac
        @inbounds for c in 1:W.C
            R[c] += fac * Sacc[c]
        end
        if J !== nothing
            @inbounds for c in 1:W.C
                for j in 1:nf
                    J[c, off + j] += fac * dS[c, j]
                end
                J[c, off + nf + 1] += fac * Sacc[c]    # d/du of exp(u)·(...)
            end
        end
        off += nf + 1
    end
    R[1] -= 1.0
    return norm(R)
end

"""Brute-force residual on the expanded node set (validation only): the same
conditions evaluated node by node, no orbit structure used."""
function brute_residual(st::MixState, W::Work)
    nodes, w = expand_rule(st)
    d, p = st.d, st.p
    R = zeros(W.C)
    H = zeros(p + 1, d)
    for s in axes(nodes, 1)
        for i in 1:d
            hermite_col!(H, i, nodes[s, i], p)
        end
        for (c, α) in enumerate(W.conds)
            R[c] += w[s] * prod(H[α[i] + 1, i] for i in 1:d)
        end
    end
    R[1] -= 1.0
    return R
end

# ---------------------------------------------------------------------------
# Levenberg-Marquardt + minimum-norm Newton polish (as PermCentralDQ)
# ---------------------------------------------------------------------------
function pack!(θ, st::MixState)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.nfree
            θ[off + j] = st.vals[o][j]
        end
        θ[off + T.nfree + 1] = st.u[o]
        off += T.nfree + 1
    end
    return θ
end

function unpack!(st::MixState, θ)
    off = 0
    for (o, T) in enumerate(st.mix)
        for j in 1:T.nfree
            st.vals[o][j] = θ[off + j]
        end
        st.u[o] = clamp(θ[off + T.nfree + 1], -46.0, 5.0)
        off += T.nfree + 1
    end
    return st
end

function value_mask(st::MixState)
    mask = falses(mix_params(st))
    off = 0
    for T in st.mix
        for j in 1:T.nfree
            mask[off + j] = true
        end
        off += T.nfree + 1
    end
    return mask
end

"""
    solve!(st, W; tol, maxiter) -> (residual, converged)

Converge the mix state in place.  `tol` is on the condition residual (SC
condition set); after the main loop an undamped minimum-norm Newton polish
drives it toward machine precision.
"""
function solve!(st::MixState, W::Work; tol::Float64 = 1e-12,
                maxiter::Int = 600, λ0::Float64 = 1e-3, vmax::Float64 = 15.0)
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
            if any(i -> isval[i] && abs(θt[i]) > vmax, eachindex(θt))
                λ *= ν; ν *= 2.0; continue
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
    return (δ, δ ≤ tol)
end

# ---------------------------------------------------------------------------
# random initial states
# ---------------------------------------------------------------------------
# k signed, separated magnitudes in the Hermite node range
function random_signed(rng::AbstractRNG, k::Int, p::Int)
    vmax = sqrt(2p + 2)
    v = Float64[]
    while length(v) < k
        c = (0.15 + rand(rng) * vmax) * (rand(rng) < 0.5 ? -1 : 1)
        all(abs(c - x) > 0.12 for x in v) && push!(v, c)
    end
    return v
end

"""Random free values for type `T`: non-sym types get k separated values
centered to Σ m_j v_j = 0 (the last is then the derived one); sym types get
r nonzero magnitudes distinct in |·|."""
function random_values(rng::AbstractRNG, T::OType, p::Int)
    if T.sym
        vmax = sqrt(2p + 2)
        v = Float64[]
        while length(v) < T.nfree
            c = 0.15 + rand(rng) * vmax
            all(abs(c - abs(x)) > 0.12 for x in v) && push!(v, c)
        end
        return v
    else
        k = length(T.mults)
        v = random_signed(rng, k, p)
        v .-= sum(T.mults[j] * v[j] for j in 1:k) / sum(T.mults)
        return v[1:(k - 1)]
    end
end

# squared radius of the orbit's representative node
function radius2(T::OType, v::Vector{Float64})
    T.nfree == 0 && return 0.0
    y = T.M[:, :, 1] * v
    return dot(y, y)
end

function normalize_mass!(st::MixState)
    tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
    st.u .-= log(tot)
    return st
end

function random_state(d::Int, p::Int, rng::AbstractRNG;
                      factor::Float64 = 1.3, types = orbit_types(d),
                      bias::Float64 = 1.0, origin_prob::Float64 = 0.5)
    CG = invariant_count(d, p)
    target = ceil(Int, factor * CG)
    nonorigin = [T for T in types if !isorigin(T)]
    wts = [((T.nfree + 1) / T.size)^bias for T in nonorigin]
    cw = cumsum(wts) ./ sum(wts)
    mix = OType[]
    rand(rng) < origin_prob && push!(mix, types[findfirst(isorigin, types)])
    while sum(T.nfree + 1 for T in mix; init = 0) < target
        r = rand(rng)
        push!(mix, nonorigin[findfirst(≥(r), cw)])
    end
    vals = [random_values(rng, T, p) for T in mix]
    u = [-0.25 * radius2(T, vals[o]) for (o, T) in enumerate(mix)]
    return normalize_mass!(MixState(d, p, mix, vals, u))
end

# ---------------------------------------------------------------------------
# structure transfer: seed a target case from a donor case's best structure
# ---------------------------------------------------------------------------
"""
    seed_from_template(donor, d, p, rng; factor) -> MixState

Same-d donors: values scaled by √((2p+1)/(2p_src+1)), jittered.
(d-1)-donors: each type gets one more slot with c = 0 (non-sym: an extra
singleton group; sym: middle + 1) -- a structural template only, since the
(d-1)-simplex frame does not embed isometrically in the d-simplex frame.
Filler orbits are added to `factor` × #invariant conditions.
"""
function seed_from_template(donor::MixState, d::Int, p::Int, rng::AbstractRNG;
                            factor::Float64 = 1.35, types = orbit_types(d))
    donor.d == d || donor.d == d - 1 ||
        throw(ArgumentError("donor must have dimension d or d-1"))
    scale = sqrt((2p + 1) / (2 * donor.p + 1))
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    for (o, T) in enumerate(donor.mix)
        if donor.d == d
            T2 = T
            v = copy(donor.vals[o])
        elseif T.sym
            T2 = sym_type(copy(T.mults), T.m0 + 1, d)
            v = copy(donor.vals[o])
        else
            T2 = nonsym_type(vcat(T.mults, 1), d)
            v = full_values(T, donor.vals[o])   # the new slot's value 0 is derived
        end
        if isorigin(T2) && any(isorigin, mix)
            continue
        end
        v = sign.(v) .* clamp.(abs.(v) .* scale, 0.05, 14.0)
        v .+= 0.03 .* randn(rng, length(v))
        push!(mix, T2); push!(vals, v); push!(u, donor.u[o])
    end
    CG = invariant_count(d, p)
    target = ceil(Int, factor * CG)
    nonorigin = [T for T in types if !isorigin(T)]
    while sum(T.nfree + 1 for T in mix; init = 0) < target
        T = nonorigin[rand(rng, 1:length(nonorigin))]
        v = random_values(rng, T, p)
        push!(mix, T); push!(vals, v); push!(u, -0.25 * radius2(T, v))
    end
    return normalize_mass!(MixState(d, p, mix, vals, u))
end

# ---------------------------------------------------------------------------
# reduction moves
# ---------------------------------------------------------------------------
# rebuild an orbit from a full (centered) value pattern; collapses to sym
# when the partition is (m, m) and to the origin when one group is left.
# Returns (type, free values) or nothing when the result would be a second
# origin (caller drops the orbit instead).
function orbit_from_pattern(mults::Vector{Int}, vfull::Vector{Float64},
                            d::Int, origin_exists::Bool)
    k = length(mults)
    if k == 1
        origin_exists && return nothing
        return (origin_type(d), Float64[])
    elseif k == 2 && mults[1] == mults[2]
        return (sym_type([mults[1]], 0, d), [vfull[1]])
    else
        return (nonsym_type(copy(mults), d), vfull[1:(k - 1)])
    end
end

function apply_move(st::MixState, move)
    kind, o, arg = move
    st2 = copy(st)
    T = st.mix[o]
    drop!(s) = (deleteat!(s.mix, o); deleteat!(s.vals, o); deleteat!(s.u, o))
    others_have_origin = any(isorigin(st.mix[q]) for q in eachindex(st.mix) if q != o)
    if kind === :drop
        drop!(st2)
    elseif kind === :merge && !T.sym
        a, b = arg
        m2 = copy(T.mults); v2 = full_values(T, st.vals[o])
        vnew = (m2[a] * v2[a] + m2[b] * v2[b]) / (m2[a] + m2[b])
        m2[a] += m2[b]; v2[a] = vnew
        deleteat!(m2, b); deleteat!(v2, b)
        res = orbit_from_pattern(m2, v2, st.d, others_have_origin)
        if res === nothing
            drop!(st2)
        else
            st2.mix[o] = res[1]; st2.vals[o] = res[2]
        end
    elseif kind === :merge && T.sym
        a, b = arg
        q2 = copy(T.mults); v2 = copy(st.vals[o])
        s = sign(v2[a] * v2[b]); s == 0 && (s = 1.0)
        vnew = (q2[a] * v2[a] + q2[b] * s * v2[b]) / (q2[a] + q2[b])
        q2[a] += q2[b]; v2[a] = vnew
        deleteat!(q2, b); deleteat!(v2, b)
        st2.mix[o] = sym_type(q2, T.m0, st.d); st2.vals[o] = v2
    elseif kind === :zero                       # sym: a pair into the middle
        a = arg
        q2 = copy(T.mults); v2 = copy(st.vals[o])
        m0 = T.m0 + 2 * q2[a]
        deleteat!(q2, a); deleteat!(v2, a)
        if isempty(q2) && others_have_origin
            drop!(st2)
        else
            st2.mix[o] = sym_type(q2, m0, st.d); st2.vals[o] = v2
        end
    elseif kind === :symm                       # non-sym -> sym
        pairs, middle = arg                     # pairs: (a, b) group indices; middle: groups
        vfull = full_values(T, st.vals[o])
        q = Int[]; v2 = Float64[]
        for (a, b) in pairs
            push!(q, T.mults[a]); push!(v2, (vfull[a] - vfull[b]) / 2)
        end
        m0 = sum(T.mults[g] for g in middle; init = 0)
        st2.mix[o] = sym_type(q, m0, st.d); st2.vals[o] = v2
    end
    return st2
end

# best symmetrization of a non-sym orbit: greedy matching of equal-mult
# groups with values closest to opposite; the rest go to the middle.
function symm_candidate(T::OType, v::Vector{Float64})
    vfull = full_values(T, v)
    k = length(T.mults)
    free = trues(k)
    pairs = Tuple{Int,Int}[]
    cost = 0.0
    while true
        best = (Inf, 0, 0)
        for a in 1:k, b in (a + 1):k
            (free[a] && free[b] && T.mults[a] == T.mults[b]) || continue
            c = abs(vfull[a] + vfull[b])
            c < best[1] && (best = (c, a, b))
        end
        best[2] == 0 && break
        push!(pairs, (best[2], best[3]))
        free[best[2]] = false; free[best[3]] = false
        cost = max(cost, best[1])
    end
    isempty(pairs) && return nothing
    middle = [g for g in 1:k if free[g]]
    for g in middle
        cost = max(cost, abs(vfull[g]))
    end
    return (cost, (pairs, middle))
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
        T.nfree == 0 && continue
        if T.sym
            jm = argmin(abs.(v))
            push!(moves, (:zero, o, jm)); push!(pr, abs(v[jm]))
            if T.nfree ≥ 2
                best = (Inf, 0, 0)
                for a in 1:T.nfree, b in (a + 1):T.nfree
                    gp = abs(abs(v[a]) - abs(v[b]))
                    gp < best[1] && (best = (gp, a, b))
                end
                push!(moves, (:merge, o, (best[2], best[3]))); push!(pr, best[1])
            end
        else
            vfull = full_values(T, v)
            k = length(vfull)
            best = (Inf, 0, 0)
            for a in 1:k, b in (a + 1):k
                gp = abs(vfull[a] - vfull[b])
                gp < best[1] && (best = (gp, a, b))
            end
            push!(moves, (:merge, o, (best[2], best[3]))); push!(pr, best[1])
            sc = symm_candidate(T, v)
            if sc !== nothing
                push!(moves, (:symm, o, sc[2])); push!(pr, sc[1])
            end
        end
    end
    o1 = sortperm(pr[1:n0])
    o2 = n0 .+ sortperm(pr[(n0+1):end])
    return moves[vcat(o1, o2)]
end

"""
    eliminate!(st, W, rng; tol, jitter_tries, swap_rounds, log_io) -> st

Greedy reduction: repeatedly attempt the ranked moves, reconverging from the
warm start (plus jittered retries) after each; accept the first success.
"""
function eliminate!(st::MixState, W::Work, rng::AbstractRNG;
                    tol::Float64 = 1e-12, jitter_tries::Int = 1,
                    swap_rounds::Int = 12, log_io = nothing)
    types = orbit_types(st.d)
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
                        v .+= 0.02 .* randn(rng, length(v))
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
                smaller = [T for T in types if T.size < st.mix[o].size &&
                           !(isorigin(T) && has_origin(st))]
                isempty(smaller) && continue
                T2 = smaller[rand(rng, 1:length(smaller))]
                cand = copy(st)
                cand.mix[o] = T2
                cand.vals[o] = random_values(rng, T2, st.p)
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
"""Expand to the full node set: every row and (except for the origin) its
negation, weights per node.  Coincident nodes (degenerate configurations,
e.g. a value hitting 0 or two values coinciding) are NOT merged here -- the
runner dedupes before banking."""
function expand_rule(st::MixState)
    d = st.d
    N = mix_nodes(st)
    nodes = zeros(N, d)
    w = zeros(N)
    row = 0
    for (o, T) in enumerate(st.mix)
        wo = exp(st.u[o])
        v = st.vals[o]
        for r in axes(T.asg, 1)
            row += 1
            for i in 1:d
                s = 0.0
                for j in 1:T.nfree
                    s += T.M[i, j, r] * v[j]
                end
                nodes[row, i] = s
            end
            w[row] = wo
            if T.signfac == 2.0
                row += 1
                nodes[row, :] .= .-nodes[row-1, :]
                w[row] = wo
            end
        end
    end
    @assert row == N
    return nodes, w
end

function save_state(path::AbstractString, st::MixState, δ::Float64)
    open(path, "w") do io
        println(io, "# d=$(st.d) p=$(st.p) nodes=$(mix_nodes(st)) resid=$δ group=SX")
        for (o, T) in enumerate(st.mix)
            println(io, T.sym ? "s" : "n", "|", join(T.mults, ","), "|", T.m0, "|",
                    join(st.vals[o], ","), "|", st.u[o])
        end
    end
end

function load_state(path::AbstractString, d::Int, p::Int)
    mix = OType[]; vals = Vector{Float64}[]; u = Float64[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        parts = split(line, "|")
        sym = parts[1] == "s"
        m = isempty(parts[2]) ? Int[] : parse.(Int, split(parts[2], ","))
        m0 = parse(Int, parts[3])
        v = isempty(parts[4]) ? Float64[] : parse.(Float64, split(parts[4], ","))
        push!(mix, build_type(m, sym, m0, d))
        push!(vals, v)
        push!(u, parse(Float64, parts[5]))
    end
    return MixState(d, p, mix, vals, u)
end

# ---------------------------------------------------------------------------
# search driver
# ---------------------------------------------------------------------------
"""
    orbit_search(d, p; seconds, rng, best_nodes, verify, on_improve, log_io)

Random-restart + greedy-reduction search over G-orbit mixes.  Same contract
as PermCentralDQ.orbit_search: `verify(nodes, weights)` returns the
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
                      tol::Float64 = 1e-12, extol::Float64 = 1e-8,
                      log_io = stdout, progress = nothing)
    W = Work(d, p)
    println(log_io, "  conditions: $(W.C) evaluated (S_d×Z_2 set), $(W.CG) independent G-invariants")
    deadline = time() + seconds
    restart = 0
    best_run = typemax(Int)
    while time() < deadline
        restart += 1
        local st
        if warm !== nothing && restart == 1
            st = copy(warm)
            for v in st.vals
                v .+= 0.01 .* randn(rng, length(v))
            end
        elseif !isempty(templates) && rand(rng) < template_prob
            donor = templates[rand(rng, 1:length(templates))]
            st = seed_from_template(donor, d, p, rng;
                                    factor = 1.2 + 0.4 * rand(rng))
        else
            fac = factor_range[1] + rand(rng) * (factor_range[2] - factor_range[1])
            st = random_state(d, p, rng; factor = fac, bias = rand(rng)^2 * 0.8)
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
                v .+= (0.08 + 0.15 * rand(rng)) .* randn(rng, length(v))
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
