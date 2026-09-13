#=
SpectralInit.jl -- spectral initialization for the designed-quadrature
searches (Vioreanu-Rokhlin / Bremer-Gimbutas-Rokhlin; adopted 2026-08-31
per litcheck/STRATEGIES_NOT_YET_EMPLOYED.md item 1).

Candidate nodes come from the approximate joint spectrum of the truncated
multiplication operators on P_s: in the orthonormal product basis of the
target weight, M_k is the compression of "multiply by x_k" onto
span{phi_alpha : |alpha| <= s}.  Before truncation the M_k commute and
their joint spectrum supports a quadrature; after truncation they nearly
commute, and the Rayleigh-quotient points
    x_i = (v_i' M_1 v_i, ..., v_i' M_d v_i)
over the eigenvectors v_i of a random positive combination A = sum t_k M_k
cluster near quadrature configurations -- a principled, non-random start
(the engine behind the published record generalized-Gaussian rules).
Candidates are ranked by the joint-eigenvector residual
    r_i = sum_k || M_k v_i - x_ik v_i ||^2
(small r = well-localized point); the n best are returned with
least-squares weights on the degree-<=p orthonormal conditions (clamped
positive -- the solver's exp-weight parametrization needs w > 0; the LM
polish is what makes them good).

In d = 1, M_1 is the truncated Jacobi matrix, the eigenvectors are exact,
and the candidates reproduce the Gauss nodes (validated ~1e-13 for all
three bases; see spectral_start.jl).

Frames match the V2 solver: :hermite on R^d, :legendre on [-1,1]^d (the
bank's [0,1]^d is the affine image), :laguerre on [0,inf)^d with the
generalized weight x^a e^{-x}/Gamma(a+1) (candidates only -- the free
solver does not speak :laguerre; feed them to the S_d orbit tools).

Only stdlib (LinearAlgebra, Random): the cluster and OSPool images carry
no packages.
=#
module SpectralInit

using LinearAlgebra, Random

export spectral_nodes

# orthonormal three-term recurrence  x phi_n = b_{n+1} phi_{n+1} + a_n phi_n + b_n phi_{n-1}
# (same coefficients WeightTransforms uses for its Golub-Welsch tridiagonals)
recur_a(n::Int, basis::Symbol, alpha::Float64) =
    basis === :laguerre ? 2n + alpha + 1 : 0.0
recur_b(n::Int, basis::Symbol, alpha::Float64) =
    basis === :hermite  ? sqrt(Float64(n)) :
    basis === :legendre ? n / sqrt(4.0 * n^2 - 1) :
    basis === :laguerre ? sqrt(n * (n + alpha)) :
    throw(ArgumentError("basis must be :hermite, :legendre or :laguerre"))

# all multi-indices with |alpha| <= s (order irrelevant; used consistently)
function multi_indices(d::Int, s::Int)
    out = Vector{Vector{Int}}()
    idx = zeros(Int, d)
    function rec(pos::Int, left::Int)
        if pos > d
            push!(out, copy(idx))
            return
        end
        for v in 0:left
            idx[pos] = v
            rec(pos + 1, left - v)
        end
        idx[pos] = 0
    end
    rec(1, s)
    return out
end

# 1-D orthonormal values phi_0..phi_m at x, into P[1..m+1]
function phi_values!(P::AbstractVector{Float64}, x::Float64, m::Int,
                     basis::Symbol, a::Float64)
    P[1] = 1.0
    m == 0 && return P
    P[2] = (x - recur_a(0, basis, a)) / recur_b(1, basis, a)
    @inbounds for n in 2:m
        P[n+1] = ((x - recur_a(n - 1, basis, a)) * P[n] -
                  recur_b(n - 1, basis, a) * P[n-1]) / recur_b(n, basis, a)
    end
    return P
end

# least-squares weights on the degree-<=p orthonormal conditions (target
# delta_{alpha,0}), clamped positive and renormalized -- a warm start for
# the exp-weight solver, not a finished rule
function ls_weights(nodes::Matrix{Float64}, p::Int, basis::Symbol, a::Float64)
    n, d = size(nodes)
    conds = multi_indices(d, p)
    Pv = zeros(n, p + 1, d)
    buf = zeros(p + 1)
    for i in 1:n, k in 1:d
        phi_values!(buf, nodes[i, k], p, basis, a)
        @views Pv[i, :, k] .= buf
    end
    Phi = zeros(length(conds), n)
    for (ci, al) in enumerate(conds), i in 1:n
        c = 1.0
        @inbounds for k in 1:d
            c *= Pv[i, al[k] + 1, k]
        end
        Phi[ci, i] = c
    end
    e = zeros(length(conds))
    e[findfirst(al -> sum(al) == 0, conds)] = 1.0
    w = Phi \ e
    w .= max.(w, 1e-8)
    w ./= sum(w)
    return w
end

"""
    spectral_nodes(d, p, n; basis = :hermite, alpha = 0.0, rng) ->
        (nodes = n x d, weights, level, resid)

Candidate nodes for an n-node degree-p rule of the given weight, from the
truncated-multiplication-operator spectrum at the smallest level s with
C(d+s, d) >= n.  Each call draws a fresh random combination A = sum t_k M_k,
so repeated calls give distinct (equally principled) candidate sets.
`resid` is the per-node joint-eigenvector residual (ascending); `weights`
are positive least-squares starters.
"""
function spectral_nodes(d::Int, p::Int, n::Int; basis::Symbol = :hermite,
                        alpha::Real = 0.0,
                        rng::AbstractRNG = Random.default_rng())
    a = Float64(alpha)
    s = 0
    while binomial(d + s, d) < n
        s += 1
    end
    A_ind = multi_indices(d, s)
    N = length(A_ind)
    pos = Dict{Vector{Int},Int}(v => i for (i, v) in enumerate(A_ind))
    M = [zeros(N, N) for _ in 1:d]
    up = zeros(Int, d)
    for (i, al) in enumerate(A_ind), k in 1:d
        M[k][i, i] = recur_a(al[k], basis, a)
        if sum(al) < s
            copyto!(up, al)
            up[k] += 1
            j = pos[up]
            b = recur_b(al[k] + 1, basis, a)
            M[k][i, j] = b
            M[k][j, i] = b
        end
    end
    t = abs.(randn(rng, d)) .+ 0.1
    t ./= sum(t)
    F = eigen(Symmetric(sum(t[k] .* M[k] for k in 1:d)))
    X = zeros(N, d)
    r = zeros(N)
    tmp = zeros(N)
    for i in 1:N
        v = @view F.vectors[:, i]
        for k in 1:d
            xi = dot(v, M[k], v)
            X[i, k] = xi
            mul!(tmp, M[k], v)
            tmp .-= xi .* v
            r[i] += sum(abs2, tmp)
        end
    end
    keep = sortperm(r)[1:n]
    nodes = X[keep, :]
    return (nodes = nodes, weights = ls_weights(nodes, p, basis, a),
            level = s, resid = r[keep])
end

end # module
