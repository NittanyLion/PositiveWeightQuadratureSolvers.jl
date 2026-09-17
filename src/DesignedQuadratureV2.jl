#=
DesignedQuadratureV2.jl

Experimental successor to DesignedQuadrature.jl (which it includes and reuses,
unmodified).  Same problem — nodes + positive weights matching all orthogonal-
basis moments up to total degree p — but a rebuilt solver:

  1. Adaptive Levenberg-Marquardt damping (Nielsen gain-ratio update) with a
     step acceptance test, replacing the hard-coded Tikhonov ladder and the
     unconditional `x -= step` of the original.
  2. Geodesic acceleration (Transtrum & Sethna 2012): a cheap second-order
     correction along the step, finite-differenced from one extra residual
     evaluation, reusing the iteration's SVD.
  3. Weights parameterized as w = exp(u): positivity is structural, the
     weight-penalty rows and their moving barrier parameter disappear, and the
     system is a clean zero-residual nonlinear least-squares problem.
  4. Continuation drivers: `node_elimination` (converge a generously sized rule
     once, then repeatedly drop the lightest pair and re-converge from the warm
     start) and `degree_continuation` (warm-start degree p from a converged
     degree p' < p rule padded with fresh pairs).
  5. Racing support: any solve can be checkpointed (`pairs` in the result +
     `init_pairs` warm start + `save_pairs`/`load_pairs`), so a driver can run
     many seeds for a short budget, keep the best fraction by residual, and
     resume only the survivors (successive halving) — see race_one.jl.
  6. The damped step is computed by QR of the stacked system [J; √λ·I]
     (`linsolve = :qr`, default) instead of a full SVD — about half the cost
     per factorization; `linsolve = :svd` keeps the old path, which amortizes
     better when many λ retries hit the same Jacobian.

Solver entry point: `designed_quadrature_v2` — same contract as
`designed_quadrature` plus `init_pairs` (warm start in solver coordinates) and
a `pairs` field in the result (symmetric mode: pair representatives + full pair
weights, i.e. exactly what `init_pairs` accepts).
=#

include(joinpath(@__DIR__, "DesignedQuadrature.jl"))

module DesignedQuadratureV2

using LinearAlgebra, Random
using ..DesignedQuadrature: total_degree_indices, poly_tables!, moment_block!, node_halfspace_penalty, basis_target, ELAPLACE_BETA,
                            node_box_penalty, lhsdesign, norminvcdf,
                            verify_exactness, save_rule, load_rule

# polish: systems with at least this many unknowns try the dual-Cholesky
# Newton steps before any SVD (2026-09-17; SYMQ_BIG_POLISH_N overrides)
const BIG_POLISH_N = parse(Int, get(ENV, "SYMQ_BIG_POLISH_N", "12000"))

export designed_quadrature_v2, node_elimination, degree_continuation,
       verify_exactness, save_rule, load_rule, save_pairs, load_pairs

# ---------------------------------------------------------------------------
# Polynomial values only (no derivatives) — for cheap trial-step residuals.
# ---------------------------------------------------------------------------
function poly_values!(P::AbstractMatrix, x::AbstractVector, basis::Symbol)
    n_s, cols = size(P)
    pmax = cols - 1
    @inbounds for s in 1:n_s
        P[s, 1] = 1.0
    end
    pmax == 0 && return nothing
    @inbounds for s in 1:n_s
        P[s, 2] = x[s]
    end
    if basis === :legendre
        @inbounds for n in 2:pmax, s in 1:n_s
            P[s, n+1] = ((2n - 1) * x[s] * P[s, n] - (n - 1) * P[s, n-1]) / n
        end
    elseif basis === :laguerre   # orthonormal, alpha = 0 (2026-09-06; see poly_tables!)
        @inbounds for s in 1:n_s
            P[s, 2] = x[s] - 1.0
        end
        @inbounds for n in 2:pmax, s in 1:n_s
            P[s, n+1] = ((x[s] - (2n - 1)) * P[s, n] - (n - 1) * P[s, n-1]) / n
        end
    elseif basis === :elaplace   # orthonormal Laplace-marginal recurrence (2026-09-08)
        @inbounds for s in 1:n_s
            P[s, 2] = x[s] / sqrt(ELAPLACE_BETA[1])
        end
        @inbounds for n in 2:pmax
            bm = sqrt(ELAPLACE_BETA[n-1]); a = 1.0 / sqrt(ELAPLACE_BETA[n])
            for s in 1:n_s
                P[s, n+1] = a * (x[s] * P[s, n] - bm * P[s, n-1])
            end
        end
    else  # :hermite (orthonormal)
        @inbounds for n in 2:pmax
            a = 1.0 / sqrt(n)
            b = sqrt(n - 1.0)
            for s in 1:n_s
                P[s, n+1] = a * (x[s] * P[s, n] - b * P[s, n-1])
            end
        end
    end
    return nothing
end

function moment_resid!(R::AbstractVector, aind::Matrix{Int},
                       T::Array{Float64,3}, w::AbstractVector)
    n_terms, d = size(aind)
    n_s = length(w)
    @inbounds for i in 1:n_terms
        acc = 0.0
        for s in 1:n_s
            c = 1.0
            for k in 1:d
                c *= T[s, aind[i, k] + 1, k]
            end
            acc += w[s] * c
        end
        R[i] = acc
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Main solver: adaptive-LM + geodesic acceleration on the exp-weight system.
# ---------------------------------------------------------------------------
function designed_quadrature_v2(d::Int, p::Int, n_s::Int;
                                basis::Symbol = :hermite,
                                init::Union{Nothing,AbstractMatrix} = nothing,
                                init_pairs::Union{Nothing,Tuple} = nothing,
                                rng::AbstractRNG = Random.default_rng(),
                                tol::Float64 = 1e-12,
                                maxiter::Int = 3000,
                                stall_window::Int = 150,
                                lambda0::Float64 = 1e-3,
                                accel::Bool = true,
                                linsolve::Symbol = :auto,
                                trace::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
                                abort_iters::Int = 0,
                                abort_resid::Float64 = Inf,
                                polish::Bool = true,
                                stopflag::Union{Nothing,Threads.Atomic{Bool}} = nothing,
                                progress_file::Union{Nothing,String} = nothing,
                                checkpoint_file::Union{Nothing,String} = nothing,
                                checkpoint_secs::Float64 = 600.0,
                                symmetric::Bool = true,
                                scale::Bool = get(ENV, "SYMQ_LM_SCALE", "1") ≠ "0",
                                polish_gate::Float64 = 1e-12,
                                verbose::Bool = false)
    basis === :laguerre && symmetric &&
        throw(ArgumentError("the exponential weight has no ±x pairing: use symmetric = false with basis = :laguerre"))
    aind = total_degree_indices(d, p)
    if symmetric
        keep = [iseven(sum(@view aind[i, :])) for i in 1:size(aind, 1)]
        aind = aind[keep, :]
    end
    n_terms = size(aind, 1)
    tgt = basis_target(basis, aind)      # e_0 for product weights; closed-form for :elaplace

    # --- initial point -----------------------------------------------------
    local b0::Matrix{Float64}, w0::Vector{Float64}
    if init_pairs !== nothing
        b0 = Matrix{Float64}(init_pairs[1])
        w0 = Vector{Float64}(init_pairs[2])
        size(b0) == (n_s, d) && length(w0) == n_s ||
            throw(ArgumentError("init_pairs must be ($n_s×$d nodes, $n_s weights)"))
    else
        lhs = init === nothing ? lhsdesign(rng, n_s, d) : Matrix{Float64}(init)
        b0 = basis === :hermite  ? norminvcdf.(lhs) :
             basis === :elaplace ? (u -> -sign(u - 0.5) * log1p(-2abs(u - 0.5))).(lhs) :
             basis === :laguerre ? -log.(1 .- lhs) : lhs .* 2 .- 1   # Exp(1) quantile for the orthant
        w0 = fill(n_terms / n_s, n_s)   # same start as v1, for comparability
    end

    use_box  = basis === :legendre
    use_half = basis === :laguerre          # orthant x ≥ 0 (2026-09-06)
    n_cons = (use_box || use_half) ? d * n_s : 0
    N = (d + 1) * n_s
    M = n_terms + n_cons

    b = copy(b0)
    u = log.(max.(w0, 1e-300))
    w = exp.(u)

    R  = zeros(M);  Rt = zeros(M);  Rh = zeros(M)
    J  = zeros(M, N)
    Dc = ones(N)                              # column scales, scaled polish only
    T  = Array{Float64,3}(undef, n_s, p + 1, d)
    Td = Array{Float64,3}(undef, n_s, p + 1, d)
    Tt = Array{Float64,3}(undef, n_s, p + 1, d)
    pre = zeros(d); suf = zeros(d)
    bt = similar(b); ut = similar(u); wt = similar(w)

    Rmom  = view(R, 1:n_terms)
    Rtmom = view(Rt, 1:n_terms)
    Rhmom = view(Rh, 1:n_terms)

    # residual at (nodes, weights) into (Rout via Tbuf), returns ‖Rout‖
    function eval_resid!(Rout, Rmomv, Tbuf, bn, wn)
        for k in 1:d
            @views poly_values!(Tbuf[:, :, k], bn[:, k], basis)
        end
        moment_resid!(Rmomv, aind, Tbuf, wn)
        row = n_terms
        if use_box
            for k in 1:d, s in 1:n_s
                row += 1
                Rout[row], _ = node_box_penalty(bn[s, k], 1000.0)
            end
        elseif use_half
            for k in 1:d, s in 1:n_s
                row += 1
                Rout[row], _ = node_halfspace_penalty(bn[s, k], 1000.0)
            end
        end
        Rmomv .-= tgt
        return norm(Rout)
    end

    # full residual + Jacobian at current (b, u); J in (node, u) coordinates
    function eval_full!()
        for k in 1:d
            @views poly_tables!(T[:, :, k], Td[:, :, k], b[:, k], basis)
        end
        w .= exp.(u)
        fill!(J, 0.0)
        moment_block!(Rmom, J, aind, T, Td, w, pre, suf)
        row = n_terms
        if use_box
            for k in 1:d, s in 1:n_s
                row += 1
                pen, dpen = node_box_penalty(b[s, k], 1000.0)
                R[row] = pen
                J[row, (k-1)*n_s+s] = dpen
            end
        elseif use_half
            for k in 1:d, s in 1:n_s
                row += 1
                pen, dpen = node_halfspace_penalty(b[s, k], 1000.0)
                R[row] = pen
                J[row, (k-1)*n_s+s] = dpen
            end
        end
        Rmom .-= tgt
        @inbounds for s in 1:n_s          # chain rule: ∂R/∂u_s = ∂R/∂w_s · w_s
            col = d * n_s + s
            for i in 1:M
                J[i, col] *= w[s]
            end
        end
        return norm(R)
    end

    δ = eval_full!()
    δbest = δ
    since_improve = 0
    t_ckpt = time()
    λ = lambda0 * maximum(sum(abs2, J, dims = 1))
    ν = 2.0
    iters = 0
    status = :maxiter
    # :auto — always factor on the smaller side.  Wide system (M ≤ N): dual
    # form, an M×M Cholesky of JJᵀ+λI.  Tall system (M > N): normal form, an
    # N×N Cholesky of JᵀJ+λI (measured 2–6× faster per iteration than the
    # stacked QR at identical convergence; both squarings carry the same
    # conditioning, and the λ floor keeps them safe).
    if linsolve === :auto
        linsolve = M ≤ N ? :dual : :chol
    end

    v  = zeros(N); acc = zeros(N); dx = zeros(N); Jv = zeros(M); fvv = zeros(M)
    h = 0.1                                  # geodesic finite-difference fraction
    Aq   = linsolve === :qr ? zeros(M + N, N) : zeros(0, 0)   # stacked [J; √λ·I]
    Raug = linsolve === :qr ? zeros(M + N) : zeros(0)
    G    = linsolve === :dual ? zeros(M, M) : zeros(0, 0)     # J·Jᵀ
    Gλ   = linsolve === :dual ? zeros(M, M) : zeros(0, 0)
    ytmp = linsolve === :dual ? zeros(M) : zeros(0)
    H    = linsolve === :chol ? zeros(N, N) : zeros(0, 0)     # JᵀJ (M > N side)
    Hλ   = linsolve === :chol ? zeros(N, N) : zeros(0, 0)
    gtmp = linsolve === :chol ? zeros(N) : zeros(0)

    while iters < maxiter
        if stopflag !== nothing && stopflag[]
            status = :aborted; break
        end
        iters += 1
        if δ ≤ tol
            status = :converged; break
        end

        # Factor once per Jacobian on the SVD path (λ retries are then cheap);
        # the QR path factors [J; √λ·I] per λ trial — cheaper when most
        # iterations accept on the first try, which adaptive λ makes typical.
        local F
        if linsolve === :svd
            F = try
                svd(J)
            catch e
                e isa LinearAlgebra.LAPACKException || rethrow()
                try svd(J; alg = LinearAlgebra.QRIteration())
                catch e2
                    e2 isa LinearAlgebra.LAPACKException || rethrow()
                    status = :diverged; break
                end
            end
        end
        UtR = linsolve === :svd ? F.U' * R : Float64[]
        σ = linsolve === :svd ? F.S : Float64[]
        linsolve === :qr && (@views Aq[1:M, :] .= J)
        linsolve === :dual && mul!(G, J, J')   # reused across all λ trials
        # JᵀJ, reused across all λ trials.  Upper triangle only (all that
        # cholesky!(Symmetric(Hλ)) reads), by syrk on the moment rows; each
        # box/orthant penalty row holds a single entry, in column row − n_terms,
        # so its contribution is one diagonal term.  2026-09-17: against the
        # dense mul!(H, J', J) this is ≈ 4× fewer flops on a Le pair system,
        # where the penalty rows are half of M (d5 p21: 1.5e14 → 3.7e13).
        if linsolve === :chol
            @views BLAS.syrk!('U', 'T', 1.0, J[1:n_terms, :], 0.0, H)
            @inbounds for c in 1:n_cons
                H[c, c] += J[n_terms + c, c]^2
            end
        end

        # damped least-squares solve: dst .= -(JᵀJ+λI)⁻¹ Jᵀ rhs
        # The dual identity (JᵀJ+λI)⁻¹Jᵀ = Jᵀ(JJᵀ+λI)⁻¹ is exact, not an
        # approximation; it just moves the factorization to the smaller side.
        solve_damped! = function (dst, rhs, λv, Fq)
            if linsolve === :svd
                Utr = rhs === R ? UtR : F.U' * rhs
                dst .= .- (F.V * ((σ .* Utr) ./ (σ .^ 2 .+ λv)))
            elseif linsolve === :dual
                ytmp .= Fq \ rhs
                mul!(dst, J', ytmp)
                dst .= .- dst
            elseif linsolve === :chol
                mul!(gtmp, J', rhs)
                copyto!(dst, gtmp)
                ldiv!(Fq, dst)
                dst .= .- dst
            else
                Raug[1:M] .= rhs
                dst .= .- (Fq \ Raug)
            end
            return dst
        end

        accepted = false
        qr_failed = false
        λ_entry = λ
        # Pass 1 uses the geodesic correction under Transtrum's acceptance
        # criterion (too large a correction ⇒ reject and damp harder, which is
        # what makes acceleration safe).  If that exhausts its λ trials without
        # a step, pass 2 retries plain LM from the entry damping, so a large
        # curvature term can never hard-stall the solve on its own.
        for pass in 1:(accel ? 2 : 1)
        use_accel = accel && pass == 1
        for _ in 1:30                        # λ trials on this Jacobian
            Fq = nothing
            if linsolve === :qr
                sλ = sqrt(λ)
                @inbounds for i in 1:N
                    Aq[M+i, i] = sλ
                end
                Fq = try
                    qr(Aq)
                catch e
                    e isa LinearAlgebra.LAPACKException || rethrow()
                    qr_failed = true; break
                end
            elseif linsolve === :dual
                copyto!(Gλ, G)
                @inbounds for i in 1:M
                    Gλ[i, i] += λ
                end
                Fq = try
                    cholesky!(Symmetric(Gλ))
                catch e
                    e isa LinearAlgebra.PosDefException || rethrow()
                    λ *= ν; ν *= 2.0          # too ill-conditioned: damp harder
                    λ > 1e14 && (qr_failed = true; break)
                    continue
                end
            elseif linsolve === :chol
                copyto!(Hλ, H)
                @inbounds for i in 1:N
                    Hλ[i, i] += λ
                end
                Fq = try
                    cholesky!(Symmetric(Hλ))
                catch e
                    e isa LinearAlgebra.PosDefException || rethrow()
                    λ *= ν; ν *= 2.0          # too ill-conditioned: damp harder
                    λ > 1e14 && (qr_failed = true; break)
                    continue
                end
            end
            solve_damped!(v, R, λ, Fq)
            dx .= v
            if use_accel
                mul!(Jv, J, v)
                @views begin
                    bt .= b .+ h .* reshape(v[1:d*n_s], n_s, d)
                    ut .= u .+ h .* v[d*n_s+1:end]
                end
                wt .= exp.(min.(ut, 30.0))
                eval_resid!(Rh, Rhmom, Tt, bt, wt)
                fvv .= (2 / h^2) .* (Rh .- R .- h .* Jv)
                solve_damped!(acc, fvv, λ, Fq)
                if norm(acc) ≤ 0.75 * norm(v)
                    dx .+= 0.5 .* acc
                else
                    λ *= ν; ν *= 2.0         # curvature too strong: shrink trust region
                    ν > 1e8 && break
                    continue
                end
            end
            @views begin
                bt .= b .+ reshape(dx[1:d*n_s], n_s, d)
                ut .= min.(u .+ dx[d*n_s+1:end], 30.0)
            end
            wt .= exp.(ut)
            δnew = eval_resid!(Rt, Rtmom, Tt, bt, wt)
            # gain ratio against the linear model along v
            mul!(Jv, J, v)
            pred = δ^2 - sum(abs2, R .+ Jv)
            ρ = pred > 0 ? (δ^2 - δnew^2) / pred : -1.0
            if δnew < δ && ρ > 1e-4
                b .= bt; u .= ut; w .= exp.(u)
                copyto!(R, Rt)
                δ = δnew
                λ *= max(1/3, 1 - (2ρ - 1)^3)
                ν = 2.0
                accepted = true
                break
            else
                λ *= ν; ν *= 2.0
                (λ > 1e14 || ν > 1e8) && break
            end
        end
        (accepted || qr_failed) && break
        λ = λ_entry; ν = 2.0                 # pass 2: plain LM from entry damping
        end

        if !accepted
            status = qr_failed ? :diverged : :stalled
            verbose && @warn "no acceptable step at ‖R‖=$δ (λ=$λ)"
            break
        end

        # refresh R, J at the accepted point (R already current; J needed fresh)
        δ = eval_full!()
        if δ ≤ tol
            status = :converged; break
        end
        if δ < 0.999 * δbest
            δbest = δ; since_improve = 0
        elseif stall_window > 0 && (since_improve += 1) > stall_window
            status = :stalled
            verbose && @warn "no residual improvement in $stall_window iterations"
            break
        end
        trace !== nothing && push!(trace, (Float64(iters), δbest))
        # hopelessness cutoff: a solve still far from the quadratic basin at
        # abort_iters essentially never reaches tol — stop paying for it
        if abort_iters > 0 && iters ≥ abort_iters && δbest > abort_resid
            status = :stalled
            break
        end
        if progress_file !== nothing && iters % 10 == 0
            open(io -> println(io, iters, " ", δbest), progress_file, "w")
        end
        # In-solve checkpoint (2026-09-17): the d5 p17 pair solve was killed
        # before its first convergence on every attempt since 09-05, and lost
        # everything each time.  Written to a temporary and renamed, so a kill
        # mid-write leaves the previous checkpoint intact.
        if checkpoint_file !== nothing && time() - t_ckpt ≥ checkpoint_secs
            save_pairs(checkpoint_file * ".tmp", (b, w))
            mv(checkpoint_file * ".tmp", checkpoint_file; force = true)
            t_ckpt = time()
        end
        verbose && iters % 25 == 0 && println("  iter $iters:  ‖R‖ = $δ  λ = $λ")
    end

    # --- polish: undamped minimum-norm Newton steps -------------------------
    # The main loop stops the moment δ crosses tol; near the solution manifold
    # undamped steps are in the quadratic regime and cost a handful of
    # iterations to reach the machine-precision floor (~1e-15), which is what
    # makes the rule accurate on raw monomials, not just on the orthonormal
    # basis (the two differ by ~√(p!) ≈ 6e6 at p=17).
    if polish && δ < 1e-6
        bbest = copy(b); ubest = copy(u); δpol = δ
        cutfacs = (1e-10, 1e-12, 1e-14)
        ci = 1
        # 2026-09-17, for the d5 p17–p21 pair systems (N = 14k–35k), where this
        # loop — not the LM iterations — was most of an elimination step:
        #  · the normal-equation work arrays are dead from here on; release them
        #    before the factorizations below need the memory;
        #  · a rejected step leaves (b, u), hence J and R, where they were, so the
        #    retry at the next cutoff REUSES the factorization instead of
        #    recomputing an identical SVD (same numbers, up to three SVDs fewer
        #    per solve, at every size);
        #  · with no box/orthant penalty active the penalty rows of J and R are
        #    exactly zero, and the SVD of the moment block alone is the same
        #    minimum-norm step at half the rows (Le: M = 2·n_terms);
        #  · N ≥ BIG_POLISH_N only: try Tikhonov–Newton steps through the dual
        #    Cholesky of J Jᵀ + εI first (one syrk + one n_terms² Cholesky, ~10×
        #    cheaper than the SVD and n_terms² memory); steps are kept only while
        #    they at least halve ‖R‖, and the SVD path below finishes the job
        #    unless the cheap one already sits on the floor.
        if N ≥ BIG_POLISH_N
            H = zeros(0, 0); Hλ = zeros(0, 0); G = zeros(0, 0); Gλ = zeros(0, 0)
            Aq = zeros(0, 0); GC.gc()
            Gp = zeros(n_terms, n_terms)
            for _ in 1:40
                eval_full!()
                (n_cons > 0 && any(!iszero, @view R[n_terms+1:M])) && break
                @views BLAS.syrk!('U', 'N', 1.0, J[1:n_terms, :], 0.0, Gp)
                ε = 1e-11 * maximum(Gp[i, i] for i in 1:n_terms)
                @inbounds for i in 1:n_terms
                    Gp[i, i] += ε
                end
                Fc = try
                    cholesky!(Symmetric(Gp))
                catch e
                    e isa LinearAlgebra.PosDefException || rethrow()
                    break
                end
                @views dxp = J[1:n_terms, :]' * (Fc \ R[1:n_terms])
                @views begin
                    b .-= reshape(dxp[1:d*n_s], n_s, d)
                    u .-= dxp[d*n_s+1:end]
                end
                w .= exp.(u)
                δnew = eval_resid!(Rt, Rtmom, Tt, b, w)
                if δnew < 0.5 * δpol
                    δpol = δnew
                    bbest .= b; ubest .= u
                    iters += 1
                else
                    δnew < δpol && (δpol = δnew; bbest .= b; ubest .= u)
                    b .= bbest; u .= ubest; w .= exp.(u)
                    break
                end
            end
            Gp = zeros(0, 0); GC.gc()
        end
        # big systems skip the SVD path when the cheap steps already put the RULE
        # through the bank's relative gate with a decade to spare — the gate, not
        # ‖R‖, is what the bank accepts (p21: one SVD is ~70 G and half an hour)
        big_done = false
        if N ≥ BIG_POLISH_N
            wb = exp.(ubest)
            bx = symmetric ? vcat(bbest, .-bbest) : bbest
            wx = symmetric ? vcat(wb ./ 2, wb ./ 2) : wb
            big_done = verify_exactness(basis === :legendre ? bx ./ 2 .+ 0.5 : bx, wx, p;
                                        basis = basis, relative = true) ≤ polish_gate
        end
        Fp = nothing; UtRp = Float64[]
        for _ in 1:40
            big_done && break
            if Fp === nothing
                eval_full!()
                pen_on = n_cons > 0 && any(!iszero, @view R[n_terms+1:M])
                Fp = try
                    pen_on ? svd(J) : svd!(J[1:n_terms, :])
                catch e
                    e isa LinearAlgebra.LAPACKException || rethrow()
                    break
                end
                UtRp = pen_on ? Fp.U' * R : Fp.U' * view(R, 1:n_terms)
            end
            thresh = cutfacs[ci] * Fp.S[1]
            dxp = Fp.V * [σi > thresh ? ui / σi : 0.0
                          for (ui, σi) in zip(UtRp, Fp.S)]
            @views begin
                b .-= reshape(dxp[1:d*n_s], n_s, d)
                u .-= dxp[d*n_s+1:end]
            end
            w .= exp.(u)
            δnew = eval_resid!(Rt, Rtmom, Tt, b, w)
            if δnew < δpol
                δpol = δnew
                bbest .= b; ubest .= u
                iters += 1
                Fp = nothing                  # moved: factor afresh
                δnew ≤ 1e-15 && break
            else
                b .= bbest; u .= ubest; w .= exp.(u)
                ci += 1                       # same point: keep Fp, tighter cutoff
                ci > length(cutfacs) && break
            end
        end
        Fp = nothing
        b .= bbest; u .= ubest; w .= exp.(u)
        δ = δpol
    end

    # --- scaled polish (2026-09-11, METHODS §29; user: "wire the scaled solver")
    # The polish above measures convergence on the orthonormal residual and
    # truncates its SVD at 1e-10…1e-14 of σ₁.  Where a rule's weights span many
    # decades — GH d=2 p ≥ 35, whose tail nodes sit 1e-22…1e-25 below the centre
    # — the directions that move the light nodes fall under that cutoff, so the
    # polish returns at ‖R‖ ~ 1e-12 with the bank's relative gate still at 1e-8
    # (measured: the p37 grid minus its lightest node, 9.1e-13 vs 1.46e-8).
    # This pass runs only when the relative gate is still above `polish_gate`
    # after the old polish, so every rule that polish already finishes is
    # untouched.  It takes Tikhonov-filtered steps in column-scaled coordinates
    # (J̃ = J D⁻¹, D = column norms), trying several filters per SVD and keeping
    # the one with the best gate, and it never returns a worse gate than it was given.
    # Scaling the main LM loop the same way was tried and rejected: cold
    # symmetric solves went from 16/16 to 0–2/16 (scratchpad/v2scaled/regress.jl).
    # (not at N ≥ BIG_POLISH_N: its full-J SVD is the 100 GB object the polish
    #  above was rewritten to avoid, and Le rules do not need it)
    if polish && scale && δ < 1e-6 && N < BIG_POLISH_N
        rule_gate = function (bb, ww)
            bx = symmetric ? vcat(bb, .-bb) : bb
            wx = symmetric ? vcat(ww ./ 2, ww ./ 2) : ww
            verify_exactness(basis === :legendre ? bx ./ 2 .+ 0.5 : bx, wx, p;
                             basis = basis, relative = true)
        end
        g0 = rule_gate(b, w)
        if !(g0 ≤ polish_gate)
            # steps are chosen and kept by the GATE, not by ‖R‖: at the Float64
            # floor ‖R‖ keeps creeping down while the relative gate drifts back
            # up (measured 7.4e-12 → 1.31e-11 on the p37 n360 drop), and the gate
            # is what the bank accepts
            gbest = g0; bgb = copy(b); ugb = copy(u); λf = 1e-6
            # never trade the solver's own convergence for gate: a candidate may not
            # leave ‖R‖ above what it was on entry (or tol), so `converged` keeps
            # meaning what callers such as node_elimination rely on
            δcap = max(δ, tol)
            bc = similar(b); uc = similar(u)
            for _ in 1:12                     # bounded: each pass is one more SVD
                eval_full!()
                @inbounds for col in 1:N
                    c = norm(view(J, :, col))
                    Dc[col] = c > 1e-300 ? c : 1.0
                    ic = 1.0 / Dc[col]
                    for i in 1:M
                        J[i, col] *= ic
                    end
                end
                Fs = try
                    svd(J)
                catch e
                    e isa LinearAlgebra.LAPACKException || rethrow()
                    break
                end
                UtRs = Fs.U' * R
                gpass = gbest; λ_best = 0.0
                for λc in (λf * 1e-4, λf * 1e-2, λf, λf * 1e2)
                    z = Fs.V * ((Fs.S .* UtRs) ./ (Fs.S .^ 2 .+ λc))
                    z ./= Dc
                    @views begin
                        bc .= b .- reshape(z[1:d*n_s], n_s, d)
                        uc .= min.(u .- z[d*n_s+1:end], 30.0)
                    end
                    δc = eval_resid!(Rt, Rtmom, Tt, bc, exp.(uc))
                    δc ≤ δcap || continue
                    gc = rule_gate(bc, exp.(uc))
                    if gc < gpass
                        gpass = gc; λ_best = λc
                        bgb .= bc; ugb .= uc
                    end
                end
                λ_best == 0.0 && break
                gain = gpass / gbest
                gbest = gpass
                b .= bgb; u .= ugb; w .= exp.(u)
                λf = clamp(λ_best, 1e-16, 1e-2)
                iters += 1
                gbest ≤ 0.1 * polish_gate && break
                # keep going only while a pass buys at least 10 %: a rule far
                # from any solution creeps here for dozens of SVDs and still
                # fails the gate
                gain > 0.9 && break
            end
            # b, u hold the best-gate iterate (or the entry point, if no pass
            # improved it); the residual is recomputed below
        end
    end

    δ = eval_resid!(R, Rmom, T, b, w)
    status === :converged && δ > tol && (status = :stalled)  # paranoia
    δ ≤ tol && (status = :converged)

    pairs = (copy(b), copy(w))
    if symmetric
        bx = vcat(b, .-b)
        wx = vcat(w ./ 2, w ./ 2)
    else
        bx = copy(b); wx = copy(w)
    end
    nodes = basis === :legendre ? bx ./ 2 .+ 0.5 : bx
    return (nodes = nodes, weights = wx, residual = δ, iterations = iters,
            converged = status === :converged, status = status, pairs = pairs)
end

# ---------------------------------------------------------------------------
# Node elimination: converge once at a generous size, then repeatedly remove
# the lightest pair and re-converge from the warm start.  Every converged size
# is optionally saved; returns the smallest converged rule.
# ---------------------------------------------------------------------------
function node_elimination(d::Int, p::Int;
                          basis::Symbol = :hermite,
                          symmetric::Bool = true,
                          m_start::Int,
                          m_stop::Int = 1,
                          rng::AbstractRNG = Random.default_rng(),
                          tol::Float64 = 1e-12,
                          maxiter_first::Int = 4000,
                          maxiter_step::Int = 1500,
                          stall_first::Int = 200,
                          stall_step::Int = 100,
                          tries::Int = 4,
                          batch0::Int = 1,
                          save_prefix::Union{Nothing,String} = nothing,
                          progress_file::Union{Nothing,String} = nothing,
                          init_pairs::Union{Nothing,Tuple} = nothing,
                          checkpoint_file::Union{Nothing,String} = nothing,
                          tag::String = "elim",
                          log_io::IO = stdout)
    t0 = time()
    # init_pairs (2026-08-29): warm start the FIRST solve from given pair
    # representatives + pair weights in solver coordinates (m_start must
    # match their count) — e.g. a quantile-transformed rule of another weight
    init_pairs === nothing || length(init_pairs[2]) == m_start ||
        throw(ArgumentError("init_pairs has $(length(init_pairs[2])) pairs, m_start=$m_start"))
    # Resume the first solve from its own checkpoint (2026-09-17) when one of
    # the right size is on disk; the file name carries cell, size and seed.
    if checkpoint_file !== nothing && isfile(checkpoint_file)
        ck = try load_pairs(checkpoint_file) catch; nothing end
        if ck !== nothing && length(ck[2]) == m_start && all(isfinite, ck[1]) && all(>(0), ck[2])
            init_pairs = ck
            println(log_io, "$tag: resuming the m=$m_start solve from $(basename(checkpoint_file))")
            flush(log_io)
        end
    end
    r = designed_quadrature_v2(d, p, m_start; basis, symmetric, rng, tol,
                               maxiter = maxiter_first, stall_window = stall_first,
                               init_pairs, progress_file, checkpoint_file)
    if !r.converged
        println(log_io, "$tag: initial solve at m=$m_start FAILED " *
                        "(status=$(r.status), ‖R‖=$(r.residual), $(round(Int, time()-t0))s)")
        flush(log_io)
        return nothing
    end
    println(log_io, "$tag: m=$m_start converged  iters=$(r.iterations)  " *
                    "‖R‖=$(round(r.residual, sigdigits=3))  t=$(round(Int, time()-t0))s")
    flush(log_io)
    best = r
    save_prefix !== nothing && save_rule("$(save_prefix)_n$(length(r.weights)).csv", r)

    b, w = r.pairs
    m = m_start
    batch = max(1, batch0)
    while m > m_stop
        order = sortperm(w)
        success = false
        # Batch phase: while many pairs still carry negligible weight, dropping
        # them one at a time wastes a full re-convergence per pair.  Drop the
        # `batch` lightest at once and halve the batch whenever that fails,
        # falling through to the single-drop candidate search at batch == 1.
        while batch > 1 && m - batch ≥ m_stop
            keep = order[(batch+1):end]
            wdrop = sum(w[order[1:batch]])
            ts = time()
            rt = designed_quadrature_v2(d, p, m - batch; basis, symmetric, tol,
                                        init_pairs = (b[keep, :], w[keep]),
                                        maxiter = maxiter_step,
                                        stall_window = stall_step,
                                        progress_file)
            if rt.converged
                b, w = rt.pairs
                m -= batch
                best = rt
                println(log_io, "$tag: m=$m converged  (batch drop of $batch, " *
                                "total w=$(round(wdrop, sigdigits=3)); iters=$(rt.iterations), " *
                                "$(round(Int, time()-ts))s, total $(round(Int, time()-t0))s)")
                flush(log_io)
                save_prefix !== nothing && save_rule("$(save_prefix)_n$(length(rt.weights)).csv", rt)
                success = true
                break
            else
                println(log_io, "$tag: batch drop of $batch to m=$(m-batch) failed " *
                                "(status=$(rt.status), ‖R‖=$(round(rt.residual, sigdigits=3)), " *
                                "$(round(Int, time()-ts))s) — halving batch")
                flush(log_io)
                batch = batch ÷ 2
            end
        end
        success && continue
        for c in order[1:min(tries, length(order))]
            keep = setdiff(1:m, c)
            wdrop = w[c]
            ts = time()
            rt = designed_quadrature_v2(d, p, m - 1; basis, symmetric, tol,
                                        init_pairs = (b[keep, :], w[keep]),
                                        maxiter = maxiter_step,
                                        stall_window = stall_step,
                                        progress_file)
            if rt.converged
                b, w = rt.pairs
                m -= 1
                best = rt
                println(log_io, "$tag: m=$m converged  (dropped w=$(round(wdrop, sigdigits=3)); " *
                                "iters=$(rt.iterations), $(round(Int, time()-ts))s, total $(round(Int, time()-t0))s)")
                flush(log_io)
                save_prefix !== nothing && save_rule("$(save_prefix)_n$(length(rt.weights)).csv", rt)
                success = true
                break
            else
                println(log_io, "$tag: m=$(m-1) attempt (drop #$c, w=$(round(wdrop, sigdigits=3))) failed " *
                                "(status=$(rt.status), ‖R‖=$(round(rt.residual, sigdigits=3)), $(round(Int, time()-ts))s)")
                flush(log_io)
            end
        end
        success || break
    end
    println(log_io, "$tag: done — smallest converged rule has $(length(best.weights)) nodes " *
                    "(m=$m pairs), total $(round(Int, time()-t0))s")
    flush(log_io)
    return best
end

# ---------------------------------------------------------------------------
# Degree continuation: warm-start degree p from a converged symmetric rule at
# a lower degree, padding with fresh low-weight pairs up to m_target.
# `pairs` is (b, w) in solver coordinates (pair representatives, pair weights).
# ---------------------------------------------------------------------------
function degree_continuation(d::Int, p::Int, pairs::Tuple, m_target::Int;
                             basis::Symbol = :hermite,
                             rng::AbstractRNG = Random.default_rng(),
                             tol::Float64 = 1e-12,
                             maxiter::Int = 4000,
                             stall_window::Int = 200,
                             pad_weight::Float64 = 1e-4,
                             progress_file::Union{Nothing,String} = nothing)
    b0, w0 = Matrix{Float64}(pairs[1]), Vector{Float64}(pairs[2])
    m0 = length(w0)
    m_target ≥ m0 || throw(ArgumentError("m_target=$m_target < warm-start pairs=$m0"))
    n_add = m_target - m0
    if n_add > 0
        lhs = lhsdesign(rng, n_add, d)
        badd = basis === :hermite  ? norminvcdf.(lhs) :
               basis === :elaplace ? (u -> -sign(u - 0.5) * log1p(-2abs(u - 0.5))).(lhs) :
               basis === :laguerre ? -log.(1 .- lhs) : lhs .* 2 .- 1
        b0 = vcat(b0, badd)
        w0 = vcat(w0, fill(pad_weight, n_add))
    end
    return designed_quadrature_v2(d, p, m_target; basis, symmetric = true, tol,
                                  init_pairs = (b0, w0), maxiter, stall_window,
                                  progress_file)
end

# Checkpoints for racing: pair representatives + pair weights, save_rule format.
function save_pairs(path::AbstractString, pairs::Tuple)
    b, w = pairs
    open(path, "w") do io
        for s in axes(b, 1)
            println(io, join(string.([b[s, :]; w[s]]), ","))
        end
    end
    return path
end

function load_pairs(path::AbstractString)
    r = load_rule(path)
    return (r.nodes, r.weights)
end

# Recover symmetric pair representatives (b, w_pair) from an expanded rule
# (2m nodes, ±x with w/2 each), e.g. one loaded from CSV.
function pairs_from_rule(nodes::AbstractMatrix, weights::AbstractVector)
    n = size(nodes, 1)
    iseven(n) || throw(ArgumentError("expanded symmetric rule must have an even node count"))
    used = falses(n)
    b = Vector{Vector{Float64}}()
    w = Float64[]
    for i in 1:n
        used[i] && continue
        xi = @view nodes[i, :]
        found = 0
        for j in i+1:n
            used[j] && continue
            if isapprox(collect(xi), -collect(@view nodes[j, :]); atol = 1e-10)
                found = j; break
            end
        end
        found == 0 && throw(ArgumentError("node $i has no antipodal partner"))
        used[i] = used[found] = true
        push!(b, collect(xi))
        push!(w, weights[i] + weights[found])
    end
    return (permutedims(reduce(hcat, b)), w)
end

export pairs_from_rule

end # module
