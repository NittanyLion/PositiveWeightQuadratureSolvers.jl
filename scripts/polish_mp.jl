# Extended-precision polish of one banked rule (2026-09-11, METHODS §29.5–29.7).
#
#   julia -t 16 polish_mp.jl <rules/…/family_dD_pP_nN.csv> [--type x4|big] [--target 1e-68]
#                            [--maxit 10] [--memcap-gb 80] [--no-rewrite]
#                            [--svdcut 1e-13] [--stall 1.0] [--backtrack 0]
#
# --svdcut, --stall, --backtrack (also POLISH_MP_SVDCUT / _STALL / _BACKTRACK, and
# POLISH_MP_MAXIT) control the Newton loop; their defaults reproduce the 2026-09-11
# production run exactly.  --svdcut is the relative singular-value floor of the
# truncated step: the residual cannot fall below the part of R that lives in the
# DISCARDED left singular directions, which is what stalled GH d4 p19 / d5 p13 /
# d5 p17 at 1e-26 … 1e-28 (METHODS §29.9).  --stall S accepts a step whenever
# ‖R‖ < S·‖R_prev‖ (S = 1 is strict decrease); --backtrack K halves a rejected step
# up to K times before giving up, for cells whose first step overshoots.
#
# User: "match Diablo's precision for Le" (Diallo & Worku stop at ‖f − Vᵀw‖₂ < 1e-66;
# their rules measure 1e-68 … 1e-76 in our metric) and "you might as well do the
# extended-precision polishing for Gh also" — "don't impose Diablo or Fiesta precision
# on Gh": GH is held to Van Zandt's quad-precision listings, 32 correct digits.
#
# Defaults per family:
#   legendre  residual in BigFloat (384 bits), target relative error 1e-68, 80 digits out
#   hermite   residual in Float64x4 (~64 digits), target 1e-34, 40 digits out
#
# Method (pilot: scratchpad/prec/polish_mp.jl): minimum-norm Newton on the orthonormal
# moment system of DesignedQuadratureV2 (solver frame, u = log w, target e_0), with the
# residual in the extended type and the step from a Float64 truncated SVD of the
# column-scaled Jacobian — an inexact Newton that gains ~13 digits per iteration.
# Central (±pair) symmetry is used when every node has its antipode.  The verified
# relative gate (verify_exactness's backward error, all monomials, in the residual's
# type) decides success.
#
# Writes rules_mp/<name>.mp.csv (bank frame, weights summing to 1, a # header with the
# source sha256, arithmetic, iterations and verified error), and, unless --no-rewrite,
# replaces the Float64 bank file by the polished rule rounded to Float64 — only when its
# Float64 gate is at least 2× better than the original's, weights stay positive and, for Le, every
# node stays inside the open cube.  Exit status 0 = target met, 2 = skipped (too large),
# 3 = target not met (nothing rewritten).
using MultiFloats, LinearAlgebra, Printf, Base.Threads, Dates, SHA
BLAS.set_num_threads(max(1, Threads.nthreads()))
const JL = joinpath(@__DIR__, "..", "src")
include(joinpath(JL, "DesignedQuadrature.jl"))
using .DesignedQuadrature: load_rule, total_degree_indices, verify_exactness

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i+1]
end
const FILE = abspath(ARGS[1])
const MM = match(r"^(legendre|hermite)_d(\d+)_p(\d+)_n(\d+)\.csv$", basename(FILE))
MM === nothing && error("not a canonical legendre/hermite bank file: $(basename(FILE))")
const FAM = String(MM[1]); const D = parse(Int, MM[2]); const P = parse(Int, MM[3])
setprecision(BigFloat, 384)
const TNAME  = argval("--type", FAM == "legendre" ? "big" : "x4")
const T      = TNAME == "big" ? BigFloat : Float64x4
const TARGET = parse(Float64, argval("--target", FAM == "legendre" ? "1e-68" : "1e-34"))
# --digits (2026-09-20): 80 for a Gaussian rule needs --type big --target 1e-68 with it
const DIGITS = parse(Int, argval("--digits", FAM == "legendre" ? "80" : "40"))
optval(flag, env, default) = argval(flag, get(ENV, env, default))
const MAXIT  = parse(Int, optval("--maxit", "POLISH_MP_MAXIT", "10"))
const SVDCUT = parse(Float64, optval("--svdcut", "POLISH_MP_SVDCUT", "1e-13"))
const STALL  = parse(Float64, optval("--stall", "POLISH_MP_STALL", "1.0"))
const BACKTRACK = parse(Int, optval("--backtrack", "POLISH_MP_BACKTRACK", "0"))
const MEMCAP = parse(Float64, argval("--memcap-gb", "80"))
const REWRITE = "--no-rewrite" ∉ ARGS
const OUTDIR = joinpath(dirname(dirname(FILE)), "rules_mp")   # rules/x.csv → rules_mp/
say(s) = (println(Dates.format(now(), "HH:MM:SS"), "  ", s); flush(stdout))

# --- orthonormal 1-D basis and derivative, degree 0..P ------------------------
function basis!(φ::AbstractVector{S}, dφ::AbstractVector{S}, t::S) where S
    φ[1] = one(S); dφ[1] = zero(S)
    P ≥ 1 && (φ[2] = t; dφ[2] = one(S))
    if FAM == "hermite"
        for n in 2:P
            φ[n+1] = (t * φ[n] - sqrt(S(n - 1)) * φ[n-1]) / sqrt(S(n))
            dφ[n+1] = sqrt(S(n)) * φ[n]
        end
    else   # Legendre P_n scaled by √(2n+1): orthonormal for dt/2 on [-1,1]
        for n in 2:P
            φ[n+1] = ((2n - 1) * t * φ[n] - (n - 1) * φ[n-1]) / n
            dφ[n+1] = dφ[n-1] + (2n - 1) * φ[n]
        end
        for n in 0:P
            c = sqrt(S(2n + 1)); φ[n+1] *= c; dφ[n+1] *= c
        end
    end
end

# --- load, solver frame, ±pair detection by hashing rounded antipodes ----------
r = load_rule(FILE)
X64 = Matrix{Float64}(r.nodes); w64 = Vector{Float64}(r.weights)
n = length(w64)
Tsol = FAM == "legendre" ? 2 .* X64 .- 1 : X64
key(v) = Tuple(round.(Int, v .* 1e8))
idx = Dict{NTuple{D,Int},Vector{Int}}()
for s in 1:n
    push!(get!(idx, key(Tsol[s, :]), Int[]), s)
end
partner = zeros(Int, n)
for s in 1:n
    partner[s] ≠ 0 && continue
    if maximum(abs, Tsol[s, :]) < 1e-12
        partner[s] = s; continue
    end
    for s2 in get(idx, key(-Tsol[s, :]), Int[])
        (s2 == s || partner[s2] ≠ 0) && continue
        if maximum(abs.(Tsol[s2, :] .+ Tsol[s, :])) < 1e-9 && abs(w64[s2] - w64[s]) ≤ 1e-9 * w64[s]
            partner[s] = s2; partner[s2] = s; break
        end
    end
end
const SYM = all(>(0), partner)
const reps = SYM ? [s for s in 1:n if partner[s] ≥ s] : collect(1:n)
const iscenter = SYM ? [partner[s] == s for s in reps] : falses(n)
const A = let A0 = total_degree_indices(D, P)
    SYM ? A0[[iseven(sum(A0[i, :])) for i in 1:size(A0, 1)], :] : A0
end
const NR = length(reps); const M = size(A, 1)
const FREEX = [!c for c in iscenter]
const XCOLS = [(j, k) for j in 1:NR for k in 1:D if FREEX[j]]
const NPAR = length(XCOLS) + NR
memgb = 8 * M * NPAR * 4.5 / 2^30          # J, scaled copy, U, V (measured ≈ 4–5 J)
say(@sprintf("%s  n=%d  %s: %d reps (%d center)  m=%d  N=%d  type %s  target %.0e  est. %.1f GB",
             basename(FILE), n, SYM ? "±pairs" : "unreduced", NR, count(iscenter), M, NPAR, TNAME, TARGET, memgb))
say(@sprintf("        maxit %d  svdcut %.0e  stall %.3g  backtrack %d", MAXIT, SVDCUT, STALL, BACKTRACK))
if memgb > MEMCAP
    say("SKIPPED: estimated $(round(memgb, digits = 1)) GB > memcap $(MEMCAP) GB")
    exit(2)
end

# --start <twin.mp.csv> (2026-09-20): begin from an existing extended-precision twin of THIS
# bank file instead of from its Float64 rounding.  A Gaussian rule can be rotated, so the
# solutions form a manifold; a polish restarted from the rounded file converges to a point
# ~1e-16 (up to 6e-12, GH d3 p31) away from the twin's, and its Float64 rounding then differs
# from the bank file in hundreds of entries.  Started from the 40-digit twin the step is ~1e-35
# and the rounding — hence the bank file and the Float64 deposit — is unchanged.
const START = argval("--start", "")
const XSTART, WSTART = if START == ""
    (nothing, nothing)
else
    srows = [split(l, ",") for l in eachline(START) if !startswith(l, "#") && !isempty(strip(l)) && !startswith(l, "x")]
    length(srows) == n || error("--start has $(length(srows)) rows, the bank file has $n")
    Xs = [parse(BigFloat, rw[k]) for rw in srows, k in 1:D]; ws = [parse(BigFloat, rw[D+1]) for rw in srows]
    maximum(abs.(Float64.(Xs) .- X64)) < 1e-9 || error("--start is not the rule in the bank file (row order or values differ)")
    (FAM == "legendre" ? 2 .* Xs .- 1 : Xs, ws)
end
xstart(S, s) = XSTART === nothing ? S.(Tsol[s, :]) : S.(XSTART[s, :])
wstart(S, s) = WSTART === nothing ? S(w64[s]) : S(WSTART[s])
function start_params(S)
    Xr = zeros(S, NR, D); wr = zeros(S, NR)
    for (j, s) in enumerate(reps)
        if iscenter[j] || !SYM
            iscenter[j] || (Xr[j, :] .= xstart(S, s))
            wr[j] = wstart(S, s)
        else
            s2 = partner[s]
            Xr[j, :] .= (xstart(S, s) .- xstart(S, s2)) ./ 2
            wr[j] = wstart(S, s) + wstart(S, s2)
        end
    end
    Xr, log.(wr)
end

# --- residual (and Float64 Jacobian) ------------------------------------------
function tables(X::AbstractMatrix{S}) where S
    Φ = [Matrix{S}(undef, NR, P + 1) for _ in 1:D]; dΦ = [Matrix{S}(undef, NR, P + 1) for _ in 1:D]
    @threads for j in 1:NR
        φ = zeros(S, P + 1); dφ = zeros(S, P + 1)
        for k in 1:D
            basis!(φ, dφ, X[j, k]); Φ[k][j, :] .= φ; dΦ[k][j, :] .= dφ
        end
    end
    Φ, dΦ
end
const COLX = Dict((j, k) => c for (c, (j, k)) in enumerate(XCOLS))
function resid(X::AbstractMatrix{S}, u; jac = false) where S
    w = exp.(u); Φ, dΦ = tables(X)
    R = zeros(S, M); J = jac ? zeros(S, M, NPAR) : zeros(S, 0, 0)
    @threads for i in 1:M
        acc = zero(S)
        for j in 1:NR
            v = w[j]
            for k in 1:D; v *= Φ[k][j, A[i, k] + 1]; end
            acc += v
            if jac
                J[i, length(XCOLS) + j] = v
                if FREEX[j]
                    for k in 1:D
                        g = w[j] * dΦ[k][j, A[i, k] + 1]
                        for k2 in 1:D; k2 == k || (g *= Φ[k2][j, A[i, k2] + 1]); end
                        J[i, COLX[(j, k)]] = g
                    end
                end
            end
        end
        R[i] = acc - (all(iszero, view(A, i, :)) ? one(S) : zero(S))
    end
    R, J
end

# --- expanded rule in the bank frame, and the relative gate in type S ----------
# Rows come out in the BANK FILE's order (2026-09-20).  Until then a ±pair-reduced rule was
# written in pair order (node, antipode, …), which is the bank's order only when the bank file
# was itself rewritten from such a twin; GH d2 p29 n153, whose 40-digit twin was made
# "unreduced", came back from the 80-digit run as the same 153 rows permuted.
function expand(X, u, S)
    Xe = zeros(S, n, D); ws = zeros(S, n)
    for (j, s) in enumerate(reps)
        wj = exp(S(u[j]))
        if !SYM
            Xe[s, :] .= S.(X[j, :]); ws[s] = wj
        elseif iscenter[j]
            ws[s] = wj                                   # the center node: coordinates stay zero
        else
            s2 = partner[s]
            Xe[s, :] .= S.(X[j, :]);  ws[s] = wj / 2
            Xe[s2, :] .= -S.(X[j, :]); ws[s2] = wj / 2
        end
    end
    FAM == "legendre" ? (Xe .+ 1) ./ 2 : Xe, ws
end
dfact(k, S) = (r = one(S); for j in 1:2:(k-1); r *= j; end; r)
function relgate(X, u, S)
    Xe, we = expand(X, u, S); ne = length(we)
    Aall = total_degree_indices(D, P); ma = size(Aall, 1)
    Pw = [Matrix{S}(undef, ne, P + 1) for _ in 1:D]
    @threads for s in 1:ne
        for k in 1:D
            Pw[k][s, 1] = one(S)
            for e in 1:P; Pw[k][s, e+1] = Pw[k][s, e] * Xe[s, k]; end
        end
    end
    errs = zeros(Float64, ma)
    @threads for i in 1:ma
        acc = zero(S); aa = zero(S)
        for s in 1:ne
            t = we[s]; for k in 1:D; t *= Pw[k][s, Aall[i, k] + 1]; end
            acc += t; aa += abs(t)
        end
        a = view(Aall, i, :)
        ex = FAM == "legendre" ? prod(one(S) / (ak + 1) for ak in a) :
             (any(isodd, a) ? zero(S) : prod(dfact(ak, S) for ak in a))
        errs[i] = Float64(abs(acc - ex) / max(aa, one(S)))
    end
    maximum(errs)
end

# --- Newton: extended residual, Float64 scaled truncated-SVD step --------------
t_start = time()
Xr, u = start_params(T)
R, _ = resid(Xr, u)
g0_f64 = verify_exactness(X64, w64, P; basis = Symbol(FAM), relative = true)
say(@sprintf("it  0  ‖R‖ %.2e  (Float64 gate of the bank file %.2e)", Float64(norm(R)), g0_f64))
scaled_jacobian(X, uu) = begin
    J = resid(Float64.(X), Float64.(uu); jac = true)[2]
    c = [max(norm(view(J, :, j)), 1e-300) for j in 1:NPAR]
    J ./= c'
    J, c
end
# svd! uses LAPACK's divide-and-conquer driver (gesdd!), which can fail to converge —
# LAPACKException(1), legendre_d5_p17_n7200 on 2026-09-11 — and took the whole cell down.
# Fall back to the slower but more robust QR-iteration driver (gesvd!).  svd! destroys its
# argument, so the fallback rebuilds the Jacobian rather than holding a second copy of a
# multi-GB matrix through the common path.
function scaled_svd(X, uu)
    J, c = scaled_jacobian(X, uu)
    try
        return svd!(J), c
    catch err
        err isa LinearAlgebra.LAPACKException || rethrow()
        say("       gesdd! failed ($err) — retrying with the gesvd! (QR iteration) driver")
    end
    J = nothing; GC.gc()
    J2, c2 = scaled_jacobian(X, uu)
    svd!(J2; alg = LinearAlgebra.QRIteration()), c2
end

gate = Inf; its = 0; nprev = Float64(norm(R))
for it in 1:MAXIT
    global Xr, u, R, gate, its, nprev
    t0 = time()
    F, c = scaled_svd(Xr, u)
    keep = F.S .> SVDCUT * F.S[1]
    nkeep = count(keep); σrel = F.S[nkeep] / F.S[1]
    R64 = Float64.(R)
    Uk = F.U[:, keep]; UtR = Uk' * R64; Uk = nothing
    z = F.V[:, keep] * ((-UtR) ./ F.S[keep])
    # the truncated step can only remove the part of R inside the kept left singular
    # directions; what is left in the discarded ones is the floor this iteration can reach
    dropped = sqrt(max(0.0, norm(R64)^2 - norm(UtR)^2))
    F = nothing
    Xn = Xr; un = u; Rn = R; nn = Inf; step = 1.0; okstep = false
    for b in 0:BACKTRACK
        δ = T.((step .* z) ./ c)
        Xt = copy(Xr); ut = u .+ δ[length(XCOLS)+1:end]
        for (col, (j, k)) in enumerate(XCOLS); Xt[j, k] += δ[col]; end
        Rt, _ = resid(Xt, ut)
        nt = Float64(norm(Rt))
        if nt < STALL * nprev
            Xn, un, Rn, nn, okstep = Xt, ut, Rt, nt, true; break
        end
        nn = nt
        b < BACKTRACK && say(@sprintf("it %2d  ‖R‖ would be %.2e at step %.3g — halving", it, nt, step))
        step /= 2
    end
    if !okstep
        say(@sprintf("it %2d  ‖R‖ would be %.2e (no decrease; rank %d/%d, σ_min/σ_1 %.1e, R outside the kept range %.2e) — stop",
                     it, nn, nkeep, min(M, NPAR), σrel, dropped)); break
    end
    Xr, u, R, nprev, its = Xn, un, Rn, nn, it
    say(@sprintf("it %2d  ‖R‖ %.2e  rank %d/%d  σ_min/σ_1 %.1e  R outside %.2e  step %.3g  (%.0f s)",
                 it, nn, nkeep, min(M, NPAR), σrel, dropped, step, time() - t0))
    if nn < 1e6 * TARGET || it == MAXIT
        gate = relgate(Xr, u, T)
        say(@sprintf("       relative gate (%s) %.2e", TNAME, gate))
        gate ≤ TARGET && break
    end
end
isfinite(gate) || (gate = relgate(Xr, u, T))
ok = gate ≤ TARGET
# POLISH_MP_GRADCHECK=1: rebuild the Jacobian in the RESIDUAL's type (not Float64) and
# report how far the stopping point is from a least-squares stationary point, as the
# largest |cos ∠(R, column j of J)|.  ≈ 0 means Jᵀ R = 0 to the residual's own precision:
# the stop is a real stationary point of ‖R‖ — there is no exact rule nearby and no amount
# of extra iteration, cut or precision in the STEP can go lower.  O(1) would mean the floor
# is an artifact of the Float64 step and a better linear solve would keep descending.
if !ok && get(ENV, "POLISH_MP_GRADCHECK", "0") == "1"
    JT = resid(Xr, u; jac = true)[2]
    nR = norm(R)
    cosmax = maximum(1:NPAR) do j
        cj = view(JT, :, j); nj = norm(cj)
        nj > 0 ? Float64(abs(dot(cj, R)) / (nj * nR)) : 0.0
    end
    JT = nothing
    say(@sprintf("stationarity: max |cos∠(R, J col)| = %.2e with J in %s  (≈0 ⇒ genuine least-squares stationary point)", cosmax, TNAME))
end
Xe, we = expand(Xr, u, T)
positive = all(>(0), we)
wall = time() - t_start
say(@sprintf("RESULT %s  %s  gate %.2e (target %.0e)  iterations %d  wall %.0f s  maxrss %.1f GB",
             basename(FILE), ok && positive ? "MET" : "NOT MET", gate, TARGET, its, wall, Sys.maxrss() / 2^30))
(ok && positive) || exit(3)

# --- write the high-precision twin -------------------------------------------
mkpath(OUTDIR)
out = joinpath(OUTDIR, replace(basename(FILE), ".csv" => ".mp.csv"))
src_sha = bytes2hex(open(sha256, FILE))
fmt(v) = @sprintf("%.*e", DIGITS - 1, BigFloat(v))
tmp = out * ".tmp"
open(tmp, "w") do io
    println(io, "# source: ", relpath(FILE, dirname(OUTDIR)), "  sha256 ", src_sha)
    println(io, "# frame: ", FAM == "legendre" ? "[0,1]^$D, uniform weight" : "standard normal N(0,I_$D)", "; weights sum to 1")
    println(io, @sprintf("# polish: polish_mp.jl, residual %s, %d Newton iterations, %s, %s",
                         TNAME, its, SYM ? "±pair-reduced" : "unreduced", gethostname()))
    println(io, @sprintf("# verified max relative monomial error (all degree ≤ %d, %s): %.3e", P, TNAME, gate))
    println(io, "# digits: ", DIGITS, "  date: ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    for s in 1:length(we)
        println(io, join((fmt(Xe[s, k]) for k in 1:D), ","), ",", fmt(we[s]))
    end
end
mv(tmp, out; force = true)
say("wrote $(relpath(out, dirname(OUTDIR)))")

# --- rewrite the Float64 bank file, only if strictly no worse ----------------
if REWRITE
    X2 = Float64.(Xe); w2 = Float64.(we)
    g2 = verify_exactness(X2, w2, P; basis = Symbol(FAM), relative = true)
    inside = FAM ≠ "legendre" || all(0 .< X2 .< 1)
    # at least 2× better: a tie only churns the file through Dropbox and every rsync
    if g2 < 0.5 * g0_f64 && all(>(0), w2) && inside
        tmpb = FILE * ".polish.tmp"
        open(tmpb, "w") do io
            for s in 1:length(w2)
                println(io, join(string.([X2[s, :]; w2[s]]), ","))
            end
        end
        mv(tmpb, FILE; force = true)
        say(@sprintf("bank file rewritten: Float64 gate %.2e → %.2e", g0_f64, g2))
    else
        say(@sprintf("bank file kept: rounded Float64 gate %.2e vs original %.2e (positive %s, inside %s)",
                     g2, g0_f64, all(>(0), w2), inside))
    end
end
