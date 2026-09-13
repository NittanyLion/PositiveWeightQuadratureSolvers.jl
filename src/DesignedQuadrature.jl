#=
DesignedQuadrature.jl

A Julia translation of the MATLAB "Designed Quadrature" code in ../DesignedQuadrature
(Keshavarzzadeh, Kirby & Narayan, "Numerical integration in multiple dimensions with
designed quadrature", SIAM J. Sci. Comput. 2018).

The solver finds n_s nodes and positive weights in d dimensions such that all
orthogonal-basis polynomials of total degree ≤ p are integrated exactly, by a
regularized Gauss-Newton iteration on the moment residuals with penalty terms for
the node-box and weight-positivity constraints.

Two bases / weight functions are supported:
  :legendre  – uniform weight; nodes returned on [0,1]^d, weights sum to 1
               (this reproduces the MATLAB code, which uses Jacobi α=β=0)
  :hermite   – standard Gaussian weight N(0,I_d); nodes on ℝ^d, weights sum to 1
               (the probabilists' Hermite recurrence that is present but commented
               out in pol_mul_jacobi.m)

Only stdlib dependencies (LinearAlgebra, Random).
=#
module DesignedQuadrature

using LinearAlgebra, Random

export designed_quadrature, total_degree_indices, verify_exactness, lhsdesign, basis_target, elaplace_smoment,
       find_rule, save_rule, load_rule

# ---------------------------------------------------------------------------
# Multi-indices of total degree ≤ p in d dimensions, in the same order as the
# MATLAB total_degree_indices.m ("traveling ones-man"): graded by total degree,
# and within each degree in decreasing lexicographic order, e.g. d=2, p=2:
# (0,0) (1,0) (0,1) (2,0) (1,1) (0,2).
# ---------------------------------------------------------------------------
function total_degree_indices(d::Int, p::Int)
    d ≥ 1 && p ≥ 0 || throw(ArgumentError("need d ≥ 1 and p ≥ 0"))
    rows = Vector{NTuple{d,Int}}()
    idx = zeros(Int, d)
    function rec!(pos::Int, rem::Int)
        if pos == d
            idx[d] = rem
            push!(rows, NTuple{d,Int}(idx))
            return
        end
        for v in rem:-1:0
            idx[pos] = v
            rec!(pos + 1, rem - v)
        end
    end
    for q in 0:p
        rec!(1, q)
    end
    a = Matrix{Int}(undef, length(rows), d)
    for (i, r) in enumerate(rows), k in 1:d
        a[i, k] = r[k]
    end
    return a
end


# ---------------------------------------------------------------------------
# :elaplace (2026-09-08, PROPOSAL_scale_mixtures.md): the ELLIPTICAL multivariate
# Laplace X = √S·Z, S ~ 2·Exp(1), Z ~ N(0, I_d).  Every marginal is Laplace(0,1)
# (density ½e^{-|x|}, even moments (2k)!), E S^k = 2^k k!, and the law is NOT a
# product, so the exactness targets in the product basis are not e_0 (see
# basis_target).  The 1-D tables use the orthonormal polynomials of the
# Laplace(0,1) marginal — symmetric three-term recurrence x·π_n = √β_{n+1} π_{n+1}
# + √β_n π_{n-1}, β from the Hankel Cholesky of (2k)! in 1024-bit arithmetic
# (scratch laplace_recur.jl) — because the Hermite basis explodes at this
# weight's outer nodes: h_21(37) ≈ 1e23 against π_21(37) ≈ 2e7.
# ---------------------------------------------------------------------------
const ELAPLACE_BETA = [2.0, 10.0, 21.6, 39.733333333333334, 60.961968680089484, 89.19093651868727, 120.0750662209188, 158.37847232233295, 198.93507207394833, 247.2987869940132, 297.5398198028537, 355.95350469324956, 415.88801905013145, 484.3436523392879, 553.9788255916748, 632.469928372759, 711.8116500371966, 800.3328339268859, 889.3860608272848, 987.9327434912548, 1086.7017302273332, 1195.269945986698, 1303.7584020662655, 1422.3446701108796, 1540.555871373476, 1669.1571007473299, 1797.0939709432926, 1935.7073900156097, 2073.3725621409526, 2221.9956649613464, 2369.3915284159043, 2528.0220330589573, 2685.1507706047955, 2853.7865862443437, 3020.6502034534783, 3199.2894039320836]
elaplace_smoment(k::Int) = k == 0 ? 1.0 : prod(2.0 * j for j in 1:k)      # E S^k = 2^k k!

# Exactness targets μ_a = E[Π_k π_{a_k}(X_k)] for the multi-indices in `aind`
# (rows), in the basis of the weight: e_0 for the product weights (orthonormal
# product bases), closed-form for :elaplace via the monomial coefficients of the
# π_n (BigFloat) and E X^i = E S^{|i|/2} Π (i_k−1)!! on all-even i.
function basis_target(basis::Symbol, aind::AbstractMatrix{Int})
    n_terms, d = size(aind)
    μ = zeros(n_terms); n_terms ≥ 1 && (μ[1] = 1.0)
    basis === :elaplace || return μ
    pmax = maximum(aind)
    prec = precision(BigFloat)
    setprecision(BigFloat, 1024)
    try
        N = max(pmax, 1)
        mom = [iseven(k) ? factorial(big(k)) : big(0) for k in 0:2N+1]
        H = [mom[i+j-1] for i in 1:N+1, j in 1:N+1]
        R = cholesky(Hermitian(H)).U
        β = [(R[k+1, k+1] / R[k, k])^2 for k in 1:N]
        # monomial coefficients c[n+1][i+1] of π_n
        c = [zeros(BigFloat, pmax + 1) for _ in 0:pmax]
        c[1][1] = 1
        if pmax ≥ 1
            c[2][2] = 1 / sqrt(β[1])
            for n in 2:pmax          # π_n = (x π_{n-1} − √β_{n-1} π_{n-2}) / √β_n
                for i in 0:n-1; c[n+1][i+2] += c[n][i+1]; end
                for i in 0:n-2; c[n+1][i+1] -= sqrt(β[n-1]) * c[n-1][i+1]; end
                c[n+1] ./= sqrt(β[n])
            end
        end
        dfact(i) = i ≤ 1 ? big(1) : prod(big(j) for j in (i-1):-2:1)
        ES(k) = big(2)^k * factorial(big(k))
        for r in 1:n_terms
            a = @view aind[r, :]
            if any(isodd, a); μ[r] = 0.0; continue; end
            # convolve over coordinates by half-degree of the monomial i
            acc = Dict{Int,BigFloat}(0 => big(1))
            for k in 1:d
                nxt = Dict{Int,BigFloat}()
                for (h, v) in acc, i in 0:2:a[k]
                    coef = c[a[k]+1][i+1]
                    coef == 0 && continue
                    nxt[h + i ÷ 2] = get(nxt, h + i ÷ 2, big(0)) + v * coef * dfact(i)
                end
                acc = nxt
            end
            μ[r] = Float64(sum(v * ES(h) for (h, v) in acc))
        end
    finally
        setprecision(BigFloat, prec)
    end
    return μ
end

# ---------------------------------------------------------------------------
# 1D orthogonal polynomial tables.
# P[s, n+1] and Pd[s, n+1] hold the degree-n polynomial and its derivative at
# x[s], for n = 0..p.  Legendre matches pol_mul_jacobi.m at α = β = 0; Hermite
# is the orthonormal probabilists' recurrence (h_n = He_n / √(n!)).
# ---------------------------------------------------------------------------
function poly_tables!(P::AbstractMatrix, Pd::AbstractMatrix,
                      x::AbstractVector, basis::Symbol)
    n_s, cols = size(P)
    pmax = cols - 1
    @inbounds for s in 1:n_s
        P[s, 1] = 1.0
        Pd[s, 1] = 0.0
    end
    pmax == 0 && return nothing
    @inbounds for s in 1:n_s
        P[s, 2] = x[s]
        Pd[s, 2] = 1.0
    end
    if basis === :legendre
        @inbounds for n in 2:pmax, s in 1:n_s
            xs = x[s]
            P[s, n+1]  = ((2n - 1) * xs * P[s, n] - (n - 1) * P[s, n-1]) / n
            Pd[s, n+1] = ((2n - 1) * (P[s, n] + xs * Pd[s, n]) - (n - 1) * Pd[s, n-1]) / n
        end
    elseif basis === :hermite
        # orthonormal recurrence h_n = (x·h_{n-1} - √(n-1)·h_{n-2})/√n: the raw
        # He_n have norm √(n!) (≈7e9 at n=21), which wrecks the Gauss-Newton
        # conditioning at high degree; normalizing keeps every condition O(1)
        @inbounds for n in 2:pmax
            a = 1.0 / sqrt(n)
            b = sqrt(n - 1.0)
            for s in 1:n_s
                xs = x[s]
                P[s, n+1]  = a * (xs * P[s, n] - b * P[s, n-1])
                Pd[s, n+1] = a * (P[s, n] + xs * Pd[s, n] - b * Pd[s, n-1])
            end
        end
    elseif basis === :laguerre
        # orthonormal Laguerre (alpha = 0, weight e^{-x} on [0,∞)), 2026-09-06:
        # x·l_n = (n+1)·l_{n+1} + (2n+1)·l_n + n·l_{n-1}, l_0 = 1, l_1 = x - 1.
        # The V2 solver's free-elimination route needs the same tables for
        # the exponential weight that LaguerreDQ.poly_cols! builds per orbit.
        @inbounds for s in 1:n_s
            P[s, 2]  = x[s] - 1.0
            Pd[s, 2] = 1.0
        end
        @inbounds for n in 2:pmax, s in 1:n_s
            xs = x[s]
            P[s, n+1]  = ((xs - (2n - 1)) * P[s, n] - (n - 1) * P[s, n-1]) / n
            Pd[s, n+1] = (P[s, n] + (xs - (2n - 1)) * Pd[s, n] - (n - 1) * Pd[s, n-1]) / n
        end
    elseif basis === :elaplace
        # orthonormal Laplace(0,1)-marginal recurrence (see ELAPLACE_BETA)
        @inbounds for s in 1:n_s
            P[s, 2]  = x[s] / sqrt(ELAPLACE_BETA[1])
            Pd[s, 2] = 1.0 / sqrt(ELAPLACE_BETA[1])
        end
        @inbounds for n in 2:pmax
            bm = sqrt(ELAPLACE_BETA[n-1]); a = 1.0 / sqrt(ELAPLACE_BETA[n])
            for s in 1:n_s
                xs = x[s]
                P[s, n+1]  = a * (xs * P[s, n] - bm * P[s, n-1])
                Pd[s, n+1] = a * (P[s, n] + xs * Pd[s, n] - bm * Pd[s, n-1])
            end
        end
    else
        throw(ArgumentError("unknown basis $basis (use :legendre, :hermite, :laguerre or :elaplace)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Moment residual R and Jacobian J (top block).
# For multi-index a: R_i = Σ_s w_s Π_k P_k(x_sk, a_k), and the sensitivities
# w.r.t. node coordinates and weights.  Uses prefix/suffix products so a row
# costs O(n_s·d) instead of recomputing recurrences per index as MATLAB does.
# ---------------------------------------------------------------------------
function moment_block!(R::AbstractVector, J::AbstractMatrix,
                       aind::Matrix{Int}, T::Array{Float64,3}, Td::Array{Float64,3},
                       w::AbstractVector, pre::Vector{Float64}, suf::Vector{Float64})
    n_terms, d = size(aind)
    n_s = length(w)
    @inbounds for i in 1:n_terms
        Ri_dot_w = 0.0
        for s in 1:n_s
            # prefix/suffix products of the 1D values v_k = P_k(x_sk, a_ik)
            acc = 1.0
            for k in 1:d
                pre[k] = acc
                acc *= T[s, aind[i, k] + 1, k]
            end
            c = acc                       # full product Π_k v_k
            acc = 1.0
            for k in d:-1:1
                suf[k] = acc
                acc *= T[s, aind[i, k] + 1, k]
            end
            for j in 1:d
                cd = pre[j] * suf[j] * Td[s, aind[i, j] + 1, j]
                J[i, (j - 1) * n_s + s] = w[s] * cd
            end
            J[i, d * n_s + s] = c
            Ri_dot_w += w[s] * c
        end
        R[i] = Ri_dot_w
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Penalty constraints (identical to cons_computation.m / cons_w_computation.m).
# Node box: quadratic penalty outside |x| ≤ 1 - 1e-6.  Weights: quadratic
# penalty below 1e-6.
# ---------------------------------------------------------------------------
function node_box_penalty(x::Real, par::Real)
    lim = 1.0 - 1e-6
    ax = abs(x)
    ax - lim ≥ 0 || return (0.0, 0.0)
    return (par * (ax - lim)^2, par * 2 * (ax - lim) * sign(x))
end

# one-sided box for the exponential weight (2026-09-06): nodes live on the
# orthant x ≥ 0; penalize x below a hair above the boundary, like the
# Legendre box keeps a hair inside ±1.
function node_halfspace_penalty(x::Real, par::Real)
    lim = 1e-6
    x - lim ≥ 0 && return (0.0, 0.0)
    return (par * (x - lim)^2, par * 2 * (x - lim))
end

function weight_penalty(w::Real, par::Real; thr::Real = 1e-6)
    x = w - thr
    x ≥ 0 && return (0.0, 0.0)
    return (par * x^2, par * 2 * x)   # 0.25(x-|x|)² = x² and its derivative for x<0
end

# Tikhonov ladder from generator.m (default 1000 covers the δ ≥ 2000 case that
# MATLAB leaves undefined on a first iteration).
function tikhonov_parameter(δ::Real)
    dtikh = 1000.0
    δ < 2000  && (dtikh = 1000.0)
    δ < 500   && (dtikh = 500.0)
    δ < 200   && (dtikh = 100.0)
    δ < 50    && (dtikh = 30.0)
    δ < 10    && (dtikh = 10.0)
    δ < 1     && (dtikh = 5.0)
    δ < 0.5   && (dtikh = 1.0)
    δ < 0.1   && (dtikh = 0.1)
    δ < 0.01  && (dtikh = 0.05)
    δ < 0.001 && (dtikh = 0.01)
    δ < 1e-4  && (dtikh = 0.001)
    δ < 1e-5  && (dtikh = 5e-4)
    # finer rungs than generator.m (which stops at 5e-4): without them the
    # damping dominates the terminal descent of large rules and stalls the
    # iteration around ‖R‖ ~ 1e-8 (seen at d=2, p=21)
    δ < 1e-6  && (dtikh = 1e-4)
    δ < 1e-8  && (dtikh = 1e-5)
    δ < 1e-10 && (dtikh = 1e-6)
    return dtikh
end

# ---------------------------------------------------------------------------
# Latin hypercube sample on [0,1]^d (one stratified point per row-slice, like
# MATLAB's lhsdesign without the maximin polish).
# ---------------------------------------------------------------------------
function lhsdesign(rng::AbstractRNG, n_s::Int, d::Int)
    L = Matrix{Float64}(undef, n_s, d)
    for k in 1:d
        perm = randperm(rng, n_s)
        for s in 1:n_s
            L[s, k] = (perm[s] - rand(rng)) / n_s
        end
    end
    return L
end
lhsdesign(n_s::Int, d::Int) = lhsdesign(Random.default_rng(), n_s, d)

# Standard normal quantile (Acklam's rational approximation, |rel err| < 1.2e-9;
# only used to spread the Gauss-Hermite initial guess, so this accuracy is ample).
function norminvcdf(u::Float64)
    0.0 < u < 1.0 || throw(DomainError(u, "quantile needs u in (0,1)"))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    e = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow
        q = sqrt(-2 * log(u))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((e[1]*q + e[2])*q + e[3])*q + e[4])*q + 1)
    elseif u ≤ phigh
        q = u - 0.5
        r = q * q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) * q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else
        q = sqrt(-2 * log1p(-u))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((e[1]*q + e[2])*q + e[3])*q + e[4])*q + 1)
    end
end

# ---------------------------------------------------------------------------
# Main solver.
#
#   designed_quadrature(d, p, n_s; basis, init, rng, tol, maxiter, verbose)
#
# init: optional n_s×d matrix of Latin-hypercube values on (0,1) (e.g. the
# recorded lhc.csv from the MATLAB run) — otherwise one is drawn from rng.
# stall_window: give up after this many iterations without ‖R‖ improving by
# 0.1% (0 disables; the MATLAB code has no such cutoff and runs to maxiter).
# stopflag: an atomic Bool that, when set by another thread, aborts this solve
# (used by find_rule to stop losing seeds once one start has converged).
#
# Returns a NamedTuple (nodes, weights, residual, iterations, converged, status).
# nodes is n_s×d: on [0,1]^d for :legendre (uniform probability weight, as the
# MATLAB XW), on ℝ^d for :hermite (standard Gaussian weight).
# ---------------------------------------------------------------------------
function designed_quadrature(d::Int, p::Int, n_s::Int;
                             basis::Symbol = :legendre,
                             init::Union{Nothing,AbstractMatrix} = nothing,
                             rng::AbstractRNG = Random.default_rng(),
                             tol::Float64 = 1e-9,
                             maxiter::Int = 5000,
                             stall_window::Int = 300,
                             stopflag::Union{Nothing,Threads.Atomic{Bool}} = nothing,
                             progress_file::Union{Nothing,String} = nothing,
                             warn_underdetermined::Bool = true,
                             symmetric::Bool = false,
                             verbose::Bool = false)
    # symmetric mode solves for n_s centrally symmetric node PAIRS: each unknown
    # "node" is a pair representative x_j carrying the pair's total weight, so
    # for the surviving even-total-degree conditions the residual has the same
    # algebraic form as the standard problem (odd conditions vanish identically
    # under x → -x and are dropped).  The returned rule is the expanded 2·n_s
    # nodes ±x_j with weight w_j/2 each.  This halves the system and searches
    # exactly the manifold where high-degree rules naturally live.
    aind = total_degree_indices(d, p)
    if symmetric
        keep = [iseven(sum(@view aind[i, :])) for i in 1:size(aind, 1)]
        aind = aind[keep, :]
    end
    n_terms = size(aind, 1)
    warn_underdetermined && n_terms > (d + 1) * n_s &&
        @warn "fewer unknowns ($((d+1)*n_s)) than moment conditions ($n_terms); convergence needs a symmetric solution"

    lhs = init === nothing ? lhsdesign(rng, n_s, d) : Matrix{Float64}(init)
    size(lhs) == (n_s, d) || throw(ArgumentError("init must be n_s×d = $n_s×$d"))

    # initial nodes: [-1,1]^d for Legendre, Gaussian spread for Hermite
    b = basis === :hermite ? norminvcdf.(lhs) : lhs .* 2 .- 1
    w = fill(n_terms / n_s, n_s)

    # Hermite has no node box (domain is all of ℝ^d); only weight positivity.
    use_box = basis === :legendre
    n_cons = use_box ? (d + 1) * n_s : n_s
    N = (d + 1) * n_s              # unknowns
    M = n_terms + n_cons           # augmented residual length

    x = vcat(vec(b), w)
    R = zeros(M)
    J = zeros(M, N)
    T  = Array{Float64,3}(undef, n_s, p + 1, d)
    Td = Array{Float64,3}(undef, n_s, p + 1, d)
    pre = zeros(d); suf = zeros(d)

    δ = Inf
    iters = 0
    status = :maxiter
    Rmom = view(R, 1:n_terms)
    δbest = Inf
    since_improve = 0

    while iters < maxiter
        if stopflag !== nothing && stopflag[]
            status = :aborted
            break
        end
        iters += 1
        for k in 1:d
            @views b[:, k] .= x[(k-1)*n_s+1 : k*n_s]
            @views poly_tables!(T[:, :, k], Td[:, :, k], b[:, k], basis)
        end
        w .= @view x[d*n_s+1 : end]

        fill!(J, 0.0)
        moment_block!(Rmom, J, aind, T, Td, w, pre, suf)

        # penalty parameter, as in generator.m: 1/‖R - n_terms·e₁‖ floored at 1000
        s2 = (Rmom[1] - n_terms)^2
        for i in 2:n_terms
            s2 += Rmom[i]^2
        end
        parm = max(1 / sqrt(s2), 1000.0)

        row = n_terms
        if use_box
            for k in 1:d, s in 1:n_s
                row += 1
                R[row], J[row, (k-1)*n_s+s] = node_box_penalty(b[s, k], parm)
            end
        end
        for s in 1:n_s
            row += 1
            R[row], J[row, d*n_s+s] = weight_penalty(w[s], parm)
        end
        R[1] -= 1.0                   # RHS: the zeroth moment must equal 1

        δ = norm(R)
        if δ ≤ tol
            status = :converged
            break
        end
        if δ > 1e30
            status = :diverged
            verbose && @warn "residual blew up; regularization too small"
            break
        end
        if δ < 0.999 * δbest
            δbest = δ
            since_improve = 0
        elseif stall_window > 0 && (since_improve += 1) > stall_window
            status = :stalled
            verbose && @warn "no residual improvement in $stall_window iterations"
            break
        end
        if progress_file !== nothing && iters % 10 == 0
            open(io -> print(io, iters, " ", δbest), progress_file, "w")
        end

        dtikh = tikhonov_parameter(δ)
        F = try                                       # thin SVD suffices
            svd(J)
        catch e
            e isa LinearAlgebra.LAPACKException || rethrow()
            # gesdd occasionally fails on ill-conditioned iterates; QR-based
            # gesvd is slower but far more robust
            try
                svd(J; alg = LinearAlgebra.QRIteration())
            catch e2
                e2 isa LinearAlgebra.LAPACKException || rethrow()
                status = :diverged
                break
            end
        end
        step = F.V * ((F.U' * R) ./ (F.S .+ dtikh))
        x .-= step

        # Newton decrement (convergence stall test from generator.m)
        t = dot(step, J' * R)
        ndecr = t > 0 ? sqrt(t) : 0.0
        if ndecr > 0 && δ / ndecr > 1e4
            status = :stalled
            verbose && @warn "Newton decrement / ‖R‖ = $(ndecr/δ) is tiny; increase n_s"
            break
        end
        verbose && iters % 25 == 0 && println("  iter $iters:  ‖R‖ = $δ")
    end

    # Polish: a stalled-but-close iterate sits near the solution manifold, where
    # any Tikhonov damping throttles the terminal Newton contraction.  Undamped
    # minimum-norm steps (pseudo-inverse with a rank cutoff) close the gap.
    # The weight floor is relaxed to 1e-10 here (with a penalty stiff enough
    # that converged weights stay strictly positive): at high degree the
    # outermost nodes genuinely need weights below the search-phase 1e-6 floor,
    # which otherwise puts a hard ~1e-9 floor on the residual itself.
    if status !== :converged && δ < 1e-6
        xbest = copy(x)
        Rbest = copy(R)
        Jbest = copy(J)
        δpol = Inf
        cutfacs = (1e-10, 1e-12, 1e-14)   # rank cutoff, tightened on stagnation
        ci = 1
        newton_step!(x, Jm, Rv, cut) = begin
            F = svd(Jm)
            thresh = cut * F.S[1]
            x .-= F.V * [σ > thresh ? u / σ : 0.0 for (u, σ) in zip(F.U' * Rv, F.S)]
        end
        for _ in 1:40
            for k in 1:d
                @views b[:, k] .= x[(k-1)*n_s+1 : k*n_s]
                @views poly_tables!(T[:, :, k], Td[:, :, k], b[:, k], basis)
            end
            w .= @view x[d*n_s+1 : end]
            fill!(J, 0.0)
            moment_block!(Rmom, J, aind, T, Td, w, pre, suf)
            row = n_terms
            if use_box
                for k in 1:d, s in 1:n_s
                    row += 1
                    R[row], J[row, (k-1)*n_s+s] = node_box_penalty(b[s, k], 1000.0)
                end
            end
            for s in 1:n_s
                row += 1
                R[row], J[row, d*n_s+s] = weight_penalty(w[s], 1e10; thr = 1e-10)
            end
            R[1] -= 1.0
            δnew = norm(R)
            if δnew < δpol
                δpol = δnew
                xbest .= x
                Rbest .= R
                Jbest .= J
                if δnew ≤ tol
                    status = :converged
                    break
                end
                newton_step!(x, J, R, cutfacs[ci])
            else
                ci += 1                   # this cutoff stagnated; tighten it
                ci > length(cutfacs) && break
                x .= xbest
                newton_step!(x, Jbest, Rbest, cutfacs[ci])
            end
        end
        if δpol < Inf
            δ = δpol
            x .= xbest
        end
        for k in 1:d
            @views b[:, k] .= x[(k-1)*n_s+1 : k*n_s]
        end
        w .= @view x[d*n_s+1 : end]
    end

    if symmetric
        b = vcat(b, .-b)
        w = vcat(w ./ 2, w ./ 2)
    end
    nodes = basis === :legendre ? b ./ 2 .+ 0.5 : copy(b)
    return (nodes = nodes, weights = copy(w), residual = δ,
            iterations = iters, converged = status === :converged, status = status)
end

# ---------------------------------------------------------------------------
# Exactness check: worst-case quadrature error over all monomials x^a, |a| ≤ p.
#   :legendre — against ∫_{[0,1]^d} x^a dx = Π 1/(a_k+1)
#   :hermite  — against E[x^a] under N(0,I) = Π (a_k-1)!! (0 for odd a_k)
#   :laguerre — against E[x^a] under the product Gamma(alpha+1, 1) weight on
#               [0,∞)^d, Π (alpha+1)(alpha+2)…(alpha+a_k) (rising factorial);
#               alpha = 0 is the exponential weight (added 2026-08-29)
# ---------------------------------------------------------------------------
function verify_exactness(nodes::AbstractMatrix, w::AbstractVector, p::Int;
                          basis::Symbol = :legendre, relative::Bool = false,
                          alpha::Real = 0.0)
    # relative = true reports each monomial's error as a backward error,
    # |q - exact| / Σ_s |w_s| Π_k |x_sk|^a_k, i.e. relative to the mass the
    # summation actually moves.  The absolute form is meaningless for Gauss
    # weights once p ≥ 17: raw moments reach (p-1)!! (3.4e7 at p = 19), so a
    # PROVABLY exact rule shows absolute error ~1e-6 from roundoff alone
    # (measured 2026-08-12 on the 5-fold tensor of 10-pt Gauss-Hermite),
    # while good rules sit at relative 1e-15..1e-12 across all p.
    d = size(nodes, 2)
    gaussmoment(n) = isodd(n) ? 0.0 : (n == 0 ? 1.0 : prod(1.0:2.0:(n-1)))
    gammamoment(n) = n == 0 ? 1.0 : prod(alpha + 1.0 + j for j in 0:n-1)
    basis in (:legendre, :hermite, :laguerre, :elaplace) ||
        throw(ArgumentError("unknown basis $basis (use :legendre, :hermite, :laguerre or :elaplace)"))
    worst = 0.0
    for a in eachrow(total_degree_indices(d, p))
        q = sum(prod(nodes[s, k]^a[k] for k in 1:d) * w[s] for s in eachindex(w))
        exact = basis === :legendre ? prod(1.0 / (a[k] + 1) for k in 1:d) :
                basis === :laguerre ? prod(gammamoment(a[k]) for k in 1:d) :
                basis === :elaplace ? elaplace_smoment(sum(a) ÷ 2) * prod(gaussmoment(a[k]) for k in 1:d) :
                                      prod(gaussmoment(a[k]) for k in 1:d)
        err = abs(q - exact)
        if relative
            s_abs = sum(abs(w[s]) * prod(abs(nodes[s, k])^a[k] for k in 1:d)
                        for s in eachindex(w))
            err /= max(s_abs, 1.0)
        end
        worst = max(worst, err)
    end
    return worst
end

# ---------------------------------------------------------------------------
# Parallel multi-start search for a rule.
#
#   find_rule(d, p; basis, n_s, seeds, tol, maxiter, stall_window, verbose)
#
# For each candidate size (n_s may be an Int, a range, or nothing for an
# automatic range starting just below ceil(n_terms/(d+1)) — symmetric rules can
# exist slightly below the naive unknowns ≥ conditions bound), all seeds are
# attempted in parallel across Julia threads (start with julia -t N); the first
# convergence aborts the remaining starts.  Returns a NamedTuple like
# designed_quadrature plus n_s and the winning seed (rerun that seed
# single-threadedly to reproduce the identical rule), or nothing.
# ---------------------------------------------------------------------------
function find_rule(d::Int, p::Int;
                   basis::Symbol = :legendre,
                   n_s::Union{Nothing,Int,AbstractVector{Int}} = nothing,
                   seeds = 1:64,
                   tol::Float64 = 1e-12,
                   maxiter::Int = 2000,
                   stall_window::Int = 300,
                   symmetric::Bool = false,
                   verbose::Bool = false)
    # in symmetric mode n_s counts node PAIRS and only even-degree conditions
    # bind, so the automatic range is sized from that smaller count
    n_terms = symmetric ?
        count(iseven, vec(sum(total_degree_indices(d, p), dims = 2))) :
        binomial(p + d, d)
    ns_candidates = n_s isa Int ? (n_s:n_s) :
                    n_s !== nothing ? n_s :
                    begin
                        lo = max(1, ceil(Int, n_terms / (d + 1)) - 1)
                        lo:(lo + max(10, lo ÷ 3))
                    end
    seedvec = collect(seeds)
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)      # search parallelism lives at the seed level
    try
        for ns in ns_candidates
            found = Threads.Atomic{Bool}(false)
            results = Vector{Any}(nothing, length(seedvec))
            Threads.@threads :dynamic for k in eachindex(seedvec)
                found[] && continue
                r = try
                    designed_quadrature(d, p, ns; basis,
                                        rng = MersenneTwister(seedvec[k]),
                                        tol, maxiter, stall_window, symmetric,
                                        stopflag = found,
                                        warn_underdetermined = false)
                catch e   # a numerically doomed seed must not kill the search
                    verbose && @warn "seed $(seedvec[k]) failed" exception = e
                    nothing
                end
                if r !== nothing && r.converged
                    results[k] = r
                    found[] = true
                end
            end
            k = findfirst(!isnothing, results)
            if k !== nothing
                verbose && (println("found $basis d=$d p=$p rule: n_s=$(length(results[k].weights)), seed $(seedvec[k])"); flush(stdout))
                return (; results[k]..., n_s = length(results[k].weights), seed = seedvec[k])
            end
            verbose && (println("no $basis d=$d p=$p rule at n_s=$ns ($(length(seedvec)) seeds)"); flush(stdout))
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Plain-text persistence: one node per line, "x_1,...,x_d,w", full precision.
# ---------------------------------------------------------------------------
function save_rule(path::AbstractString, rule)
    open(path, "w") do io
        for s in axes(rule.nodes, 1)
            println(io, join(string.([rule.nodes[s, :]; rule.weights[s]]), ","))
        end
    end
    return path
end

function load_rule(path::AbstractString)
    rows = [parse.(Float64, split(line, ',')) for line in eachline(path)]
    M = permutedims(reduce(hcat, rows))
    return (nodes = M[:, 1:end-1], weights = M[:, end])
end

end # module
