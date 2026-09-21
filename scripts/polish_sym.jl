# Extended-precision polish of a B_d-symmetric Gaussian rule IN ITS ORBIT FRAME (2026-09-19).
#
#   julia -t 16 polish_sym.jl <rules/hermite_dD_pP_nN.csv> [--target 1e-34] [--maxit 20] [--no-rewrite]
#
# Why a second polisher.  polish_mp.jl works in the free-node (±pair) frame with a Float64
# Jacobian.  For a rule found under full B_d symmetry that frame is singular at the solution:
# the symmetry-breaking directions are (near-)null directions of J, κ(J) ≈ 1e16, and the
# inexact Newton floors at 1e-27 … 1e-30 (GH d4 p19 n1505, d5 p13 n1135; METHODS §29.9), or
# the free-frame Jacobian is simply too large (GH d5 p21 n13199).  In the orbit frame the
# unknowns are the distinct positive values and one weight per orbit, and by Sobolev's theorem
# the conditions are one per partition of even exponents — a system of a few dozen to a few
# hundred unknowns, so the WHOLE Newton iteration, linear algebra included, runs in BigFloat.
#
# Steps: detect the orbits of the bank file (complete B_d orbits with equal weights, or exit 4);
# Newton in BigFloat(384) on the invariant conditions, minimum-norm / least-squares step from
# the normal equations of the column-scaled Jacobian; expand in the bank file's node order;
# verify the relative gate over ALL monomials of degree ≤ p in Float64x4 (the same functional
# as polish_mp.jl); write rules_mp/<name>.mp.csv (40 digits) in polish_mp.jl's format and,
# unless --no-rewrite, replace the Float64 bank file by the rounded twin when its Float64 gate
# is no worse than the original's (the deposit ships a twin only if it rounds to the bank rule).
# Exit 0 = target met, 3 = not met (nothing written), 4 = not a union of complete B_d orbits.
using MultiFloats, LinearAlgebra, Printf, Base.Threads, Dates, SHA
const JL = get(ENV, "POLISH_SYM_JL", joinpath(@__DIR__, "..", "src"))
include(joinpath(JL, "DesignedQuadrature.jl"))
using .DesignedQuadrature: load_rule, total_degree_indices, verify_exactness

argval(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
const FILE = abspath(ARGS[1])
const MM = match(r"^hermite_d(\d+)_p(\d+)_n(\d+)\.csv$", basename(FILE))
MM === nothing && error("not a canonical hermite bank file: $(basename(FILE))")
const D = parse(Int, MM[1]); const P = parse(Int, MM[2])
setprecision(BigFloat, 384)
const TARGET = parse(Float64, argval("--target", "1e-34"))
const MAXIT = parse(Int, argval("--maxit", "20"))
const REWRITE = "--no-rewrite" ∉ ARGS
# --digits 80 (2026-09-20; with --target 1e-68): the gate is then evaluated in BigFloat(384),
# because Float64x4 floors near 1e-62 and cannot certify 1e-68.
const DIGITS = parse(Int, argval("--digits", "40"))
const GNAME = DIGITS > 40 ? "BigFloat(384)" : "x4"
const OUTDIR = joinpath(dirname(dirname(FILE)), "rules_mp")
say(s) = (println(Dates.format(now(), "HH:MM:SS"), "  ", s); flush(stdout))

# --- orbit detection ------------------------------------------------------------
r = load_rule(FILE)
const X64 = Matrix{Float64}(r.nodes); const w64 = Vector{Float64}(r.weights)
const n = length(w64)
const ZTOL = 1e-10; const VTOL = 1e-7
okey(x) = Tuple(round.(Int, sort(abs.(x)) .* 1e6))
groups = Dict{NTuple{D,Int},Vector{Int}}()
for s in 1:n
    push!(get!(groups, okey(X64[s, :]), Int[]), s)
end
struct Orbit
    nodes::Vector{Int}          # rows of the bank file
    k::Int                      # distinct positive values
    mults::Vector{Int}          # their multiplicities
    z::Int                      # zero coordinates
    g::Matrix{Int}              # n_orbit × D: value index of each coordinate (0 = zero)
    perms::Vector{Vector{Int}}  # distinct arrangements of the value indices
end
function distinct_arrangements(v::Vector{Int})
    out = Vector{Vector{Int}}(); seen = Set{Vector{Int}}()
    function rec(cur, rest)
        if isempty(rest)
            cur ∈ seen || (push!(seen, copy(cur)); push!(out, copy(cur))); return
        end
        used = Set{Int}()
        for i in eachindex(rest)
            rest[i] ∈ used && continue
            push!(used, rest[i])
            rec(vcat(cur, rest[i]), vcat(rest[1:i-1], rest[i+1:end]))
        end
    end
    rec(Int[], v); out
end
orbits = Orbit[]; v0 = Vector{Vector{BigFloat}}(); lw0 = BigFloat[]
for (_, idx) in sort(collect(groups); by = kv -> minimum(kv[2]))
    a = sort(abs.(X64[idx[1], :]))
    vals = Float64[]; z = 0
    for t in a
        if t < ZTOL; z += 1
        elseif isempty(vals) || t - vals[end] > VTOL * max(1.0, t); push!(vals, t); end
    end
    k = length(vals)
    g = zeros(Int, length(idx), D)
    for (row, s) in enumerate(idx), c in 1:D
        t = abs(X64[s, c])
        t < ZTOL && continue
        j = argmin(abs.(vals .- t))
        abs(vals[j] - t) ≤ VTOL * max(1.0, t) || (say("node $s does not fit its orbit's values"); exit(4))
        g[row, c] = j
    end
    mults = [count(==(j), g[1, :]) for j in 1:k]
    size_expected = factorial(D) ÷ (factorial(z) * prod(factorial.(mults))) * 2^(D - z)
    rows = Set(Tuple(X64[s, :] .≥ 0) .=> Tuple(g[row, :]) for (row, s) in enumerate(idx))
    wspread = (maximum(w64[idx]) - minimum(w64[idx])) / maximum(w64[idx])
    if length(idx) ≠ size_expected || length(rows) ≠ length(idx) || wspread > 1e-9 ||
       any(count(==(j), g[row, :]) ≠ mults[j] for row in 1:length(idx) for j in 1:k)
        say(@sprintf("not a complete B_%d orbit: %d nodes, expected %d, weight spread %.1e (first row %d)",
                     D, length(idx), size_expected, wspread, idx[1]))
        exit(4)
    end
    # starting values: orbit means in BigFloat (averages out the Float64 noise)
    vb = [sum(BigFloat(abs(X64[s, c])) for (row, s) in enumerate(idx) for c in 1:D if g[row, c] == j) /
          (mults[j] * length(idx)) for j in 1:k]
    push!(orbits, Orbit(idx, k, mults, z, g, distinct_arrangements(vcat(zeros(Int, z), [j for j in 1:k for _ in 1:mults[j]]))))
    push!(v0, vb); push!(lw0, log(sum(BigFloat.(w64[idx])) / length(idx)))
end
const ORB = orbits; const NO = length(ORB)
const VOFF = cumsum([0; [o.k for o in ORB]])          # value columns of orbit o: VOFF[o]+1 …
const NV = VOFF[end]; const NPAR = NV + NO

# --- invariant conditions: partitions with even parts, |α| ≤ P, at most D parts -------
function even_partitions(d, p)
    out = Vector{Vector{Int}}()
    rec(cur, left, maxpart) = begin
        push!(out, vcat(cur, zeros(Int, d - length(cur))))
        length(cur) == d && return
        for a in 2:2:min(maxpart, left); rec(vcat(cur, a), left - a, a); end
    end
    rec(Int[], p, p); out
end
const ALPHA = even_partitions(D, P - (isodd(P) ? 1 : 0)); const M = length(ALPHA)
say(@sprintf("%s  n=%d  B_%d orbit frame: %d orbits, %d values + %d weights = %d unknowns, %d invariant conditions",
             basename(FILE), n, D, NO, NV, NO, NPAR, M))

# orthonormal probabilists' Hermite h_0..h_P and derivatives at t
function hermite!(h::Vector{S}, dh::Vector{S}, t::S) where S
    h[1] = one(S); dh[1] = zero(S)
    P ≥ 1 && (h[2] = t; dh[2] = one(S))
    for m in 2:P
        h[m+1] = (t * h[m] - sqrt(S(m - 1)) * h[m-1]) / sqrt(S(m))
        dh[m+1] = sqrt(S(m)) * h[m]
    end
end
function resjac(v::Vector{Vector{BigFloat}}, lw::Vector{BigFloat}; jac = true)
    S = BigFloat
    R = zeros(S, M); R[1] = -one(S)                  # ALPHA[1] is the zero partition
    J = jac ? zeros(S, M, NPAR) : zeros(S, 0, 0)
    contrib = Vector{Any}(undef, NO)
    @threads for o in 1:NO
        ob = ORB[o]; k = ob.k
        H = Matrix{S}(undef, k + 1, P + 1); Hd = zeros(S, k + 1, P + 1)
        h = zeros(S, P + 1); dh = zeros(S, P + 1)
        hermite!(h, dh, zero(S)); H[1, :] .= h                      # row 1: the zero coordinate
        for j in 1:k; hermite!(h, dh, v[o][j]); H[j+1, :] .= h; Hd[j+1, :] .= dh; end
        Sα = zeros(S, M); dSα = zeros(S, M, k)
        f = Vector{S}(undef, D)
        for a in 1:M, σ in ob.perms
            α = ALPHA[a]
            for i in 1:D; f[i] = H[σ[i]+1, α[i]+1]; end
            Sα[a] += prod(f)
            if jac
                for i in 1:D
                    σ[i] == 0 && continue
                    t = Hd[σ[i]+1, α[i]+1]
                    for i2 in 1:D; i2 == i || (t *= f[i2]); end
                    dSα[a, σ[i]] += t
                end
            end
        end
        contrib[o] = (Sα, dSα)
    end
    for o in 1:NO
        ob = ORB[o]; Sα, dSα = contrib[o]
        c = exp(lw[o]) * big(2)^(D - ob.z)           # per-node weight × sign multiplicity
        R .+= c .* Sα
        if jac
            J[:, NV+o] .= c .* Sα
            for j in 1:ob.k; J[:, VOFF[o]+j] .= c .* dSα[:, j]; end
        end
    end
    R, J
end

# --- expand in the bank file's node order, relative gate in Float64x4 ------------------
function expand(v, lw)
    Xe = zeros(BigFloat, n, D); we = zeros(BigFloat, n)
    for o in 1:NO, (row, s) in enumerate(ORB[o].nodes)
        we[s] = exp(lw[o])
        for c in 1:D
            j = ORB[o].g[row, c]
            j == 0 || (Xe[s, c] = X64[s, c] < 0 ? -v[o][j] : v[o][j])
        end
    end
    we ./= sum(we)
    Xe, we
end
dfact(k, S) = (r = one(S); for j in 1:2:(k-1); r *= j; end; r)
function relgate(XeB, weB)
    S = DIGITS > 40 ? BigFloat : Float64x4
    Xe = S.(XeB); we = S.(weB); ne = length(we)
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
        ex = any(isodd, a) ? zero(S) : prod(dfact(ak, S) for ak in a)
        errs[i] = Float64(abs(acc - ex) / max(aa, one(S)))
    end
    maximum(errs)
end

# --- Newton, everything in BigFloat ------------------------------------------------------
t_start = time()
v = deepcopy(v0); lw = copy(lw0)
g0_f64 = verify_exactness(X64, w64, P; basis = :hermite, relative = true)
R, J = resjac(v, lw)
say(@sprintf("it  0  ‖R‖ %.2e  (Float64 gate of the bank file %.2e)", Float64(norm(R)), g0_f64))
its = 0
for it in 1:MAXIT
    global v, lw, R, J, its
    t0 = time()
    c = [max(norm(view(J, :, j)), big"1e-300") for j in 1:NPAR]
    Js = J ./ c'
    sv = svdvals(Float64.(Js))
    # minimum-norm (M ≤ NPAR) or least-squares (M > NPAR) step from the normal equations; a
    # tiny ridge keeps a rank-deficient system solvable without moving a regular one
    z = if M ≤ NPAR
        G = Js * Js'; λ = big"1e-90" * tr(G) / M
        Js' * (cholesky(Hermitian(G + λ * I)) \ (-R))
    else
        G = Js' * Js; λ = big"1e-90" * tr(G) / NPAR
        cholesky(Hermitian(G + λ * I)) \ (-(Js' * R))
    end
    δ = z ./ c
    step = big(1.0); accepted = false; nprev = norm(R)
    for b in 0:8
        vt = [v[o] .+ step .* δ[VOFF[o]+1:VOFF[o]+ORB[o].k] for o in 1:NO]
        lt = lw .+ step .* δ[NV+1:end]
        Rt, _ = resjac(vt, lt; jac = false)
        if norm(Rt) < nprev
            v, lw = vt, lt; accepted = true; break
        end
        step /= 2
    end
    accepted || (say(@sprintf("it %2d  no decrease — stop", it)); break)
    R, J = resjac(v, lw); its = it
    say(@sprintf("it %2d  ‖R‖ %.2e  σ_min/σ_1 %.1e (rank %d/%d at 1e-13)  step %.3g  (%.0f s)", it, Float64(norm(R)),
                 sv[end] / sv[1], count(>(1e-13 * sv[1]), sv), min(M, NPAR), Float64(step), time() - t0))
    norm(R) < big"1e-100" && break
end
Xe, we = expand(v, lw)
shift = maximum(abs.(Float64.(Xe) .- X64)); wshift = maximum(abs.(Float64.(we) .- w64) ./ w64)
say(@sprintf("moved from the bank file: max |Δx| %.2e, max rel |Δw| %.2e", shift, wshift))
gate = relgate(Xe, we)
say(@sprintf("       relative gate (%s, all monomials of degree ≤ %d) %.2e", GNAME, P, gate))
ok = gate ≤ TARGET && all(>(0), we)
say(@sprintf("RESULT %s  %s  gate %.2e (target %.0e)  iterations %d  wall %.0f s  maxrss %.1f GB",
             basename(FILE), ok ? "MET" : "NOT MET", gate, TARGET, its, time() - t_start, Sys.maxrss() / 2^30))
ok || exit(3)

# --- write the twin (polish_mp.jl's format) ----------------------------------------------
mkpath(OUTDIR)
out = joinpath(OUTDIR, replace(basename(FILE), ".csv" => ".mp.csv"))
src_sha = bytes2hex(open(sha256, FILE))
fmt(x) = @sprintf("%.*e", DIGITS - 1, x)
tmp = out * ".tmp"
open(tmp, "w") do io
    println(io, "# source: ", relpath(FILE, dirname(OUTDIR)), "  sha256 ", src_sha)
    println(io, "# frame: standard normal N(0,I_$D); weights sum to 1")
    println(io, @sprintf("# polish: polish_sym.jl, B_%d orbit frame (%d orbits, %d unknowns, %d invariant conditions), Newton and linear algebra in BigFloat(384), %d iterations, %s",
                         D, NO, NPAR, M, its, gethostname()))
    println(io, @sprintf("# verified max relative monomial error (all degree ≤ %d, %s): %.3e", P, GNAME, gate))
    println(io, "# digits: ", DIGITS, "  date: ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    for s in 1:n
        println(io, join((fmt(Xe[s, k]) for k in 1:D), ","), ",", fmt(we[s]))
    end
end
mv(tmp, out; force = true)
say("wrote $(relpath(out, dirname(OUTDIR)))")

# --- rewrite the Float64 bank file so that the twin rounds to it -------------------------
if REWRITE
    X2 = Float64.(Xe); w2 = Float64.(we)
    # compare the two Float64 rules by their error evaluated in x4 (the deposit's err_float64
    # functional): verify_exactness in Float64 arithmetic is rounding noise at this level
    g0_f64 = relgate(BigFloat.(X64), BigFloat.(w64))
    g2 = relgate(BigFloat.(X2), BigFloat.(w2))
    # The orbit frame is regular but has weakly determined directions, so the exact rule can sit
    # 1e-12 away from a bank rule whose error is already at rounding level (GH d4 p19: 2.0e-12,
    # 5e-16).  make_publish.jl ships a twin only if it rounds to the bank rule within 1e-12, so
    # a bank file that far from its twin is replaced by the twin's rounding as long as that is
    # equally good (within 2×, which is rounding luck at 1e-16); otherwise polish_mp.jl's rule.
    far = max(shift, maximum(abs.(w2 .- w64))) > 5e-13
    if all(>(0), w2) && (g2 < 0.5 * g0_f64 || (far && g2 ≤ 2 * g0_f64))
        tmpb = FILE * ".polish.tmp"
        open(tmpb, "w") do io
            for s in 1:n
                println(io, join(string.([X2[s, :]; w2[s]]), ","))
            end
        end
        mv(tmpb, FILE; force = true)
        say(@sprintf("bank file rewritten: Float64 gate %.2e → %.2e", g0_f64, g2))
    else
        say(@sprintf("bank file kept: rounded Float64 gate %.2e vs original %.2e (positive %s)", g2, g0_f64, all(>(0), w2)))
    end
end
