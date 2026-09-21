# Extended-precision polish with ALL of the iteration in BigFloat (2026-09-20).
#
#   julia -t 8 polish_lm_big.jl <rules/hermite_dD_pP_nN.csv> [--start twin.mp.csv] [--target 1e-68] [--digits 80] [--maxit 40]
#
# Why a third polisher.  polish_mp.jl takes its step from a Float64 SVD of the Jacobian.  A
# rule on a solution manifold that is NOT made of complete B_d orbits (GH d4 p9 n116, the
# S_4×Z_2 rule: rank 280 of 290) has singular values below what Float64 resolves; the residual
# along those directions is never corrected and the iteration floors — at 1.1e-63 for that
# cell, short of the 1e-68 the 80-digit deposit needs — whatever --svdcut, --stall and --maxit
# are.  polish_sym.jl exits 4 on it.  Here the Jacobian, the normal equations and the solve
# are all BigFloat, which is affordable only for small rules: the unknowns are every node
# coordinate and log-weight (no ±pair reduction), the equations every orthonormal Hermite
# product of degree ≤ p, and the step is a Levenberg step with a damping far below the
# resolved spectrum, so null directions simply do not move.  Never rewrites the bank file;
# writes rules_mp/<name>.mp.csv in polish_mp.jl's format.  Exit 0 = target met, 3 = not met.
using LinearAlgebra, Printf, Dates, SHA, Base.Threads
const JL = joinpath(@__DIR__, "..", "src")
include(joinpath(JL, "DesignedQuadrature.jl"))
using .DesignedQuadrature: load_rule, total_degree_indices
argval(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
const FILE = abspath(ARGS[1])
const MM = match(r"^hermite_d(\d+)_p(\d+)_n(\d+)\.csv$", basename(FILE))
MM === nothing && error("not a canonical hermite bank file: $(basename(FILE))")
const D = parse(Int, MM[1]); const P = parse(Int, MM[2])
setprecision(BigFloat, 640)
const TARGET = parse(Float64, argval("--target", "1e-68"))
const DIGITS = parse(Int, argval("--digits", "80"))
const MAXIT = parse(Int, argval("--maxit", "40"))
const OUTDIR = joinpath(dirname(dirname(FILE)), "rules_mp")
say(s) = (println(Dates.format(now(), "HH:MM:SS"), "  ", s); flush(stdout))

r = load_rule(FILE)
X = BigFloat.(Matrix{Float64}(r.nodes)); u = log.(BigFloat.(Vector{Float64}(r.weights))); n = length(u)
# --start <twin.mp.csv>: begin from an extended-precision twin of this bank file (see polish_mp.jl:
# a cold restart slides along the rotation manifold and the Float64 rounding moves with it)
let st = argval("--start", "")
    if st != ""
        rows = [split(l, ",") for l in eachline(st) if !startswith(l, "#") && !isempty(strip(l)) && !startswith(l, "x")]
        length(rows) == n || error("--start has $(length(rows)) rows, the bank file has $n")
        Xs = [parse(BigFloat, rw[k]) for rw in rows, k in 1:D]
        maximum(abs.(Float64.(Xs) .- Float64.(X))) < 1e-9 || error("--start is not the rule in the bank file")
        global X = Xs; global u = log.([parse(BigFloat, rw[D+1]) for rw in rows])
    end
end
const A = total_degree_indices(D, P); const M = size(A, 1)
n * (D + 1) ≤ 4000 || (say("SKIPPED: $(n * (D + 1)) unknowns is too many for an all-BigFloat solve"); exit(2))

# orthonormal probabilists' Hermite values and derivatives, degree 0..P
function basis(t)
    φ = Vector{BigFloat}(undef, P + 1); dφ = similar(φ)
    φ[1] = 1; dφ[1] = 0
    P ≥ 1 && (φ[2] = t; dφ[2] = 1)
    for k in 2:P
        φ[k+1] = (t * φ[k] - sqrt(BigFloat(k - 1)) * φ[k-1]) / sqrt(BigFloat(k))
        dφ[k+1] = sqrt(BigFloat(k)) * φ[k]
    end
    φ, dφ
end
function system(X, u)
    w = exp.(u)
    R = zeros(BigFloat, M); J = zeros(BigFloat, M, n * (D + 1))
    B = [basis(X[s, k]) for s in 1:n, k in 1:D]
    @threads for i in 1:M
        for s in 1:n
            t = w[s]; for k in 1:D; t *= B[s, k][1][A[i, k]+1]; end
            R[i] += t
            J[i, n*D+s] = t                                    # ∂/∂u_s
            for k in 1:D
                g = w[s]
                for k2 in 1:D; g *= k2 == k ? B[s, k2][2][A[i, k2]+1] : B[s, k2][1][A[i, k2]+1]; end
                J[i, (s-1)*D+k] = g
            end
        end
    end
    R[findfirst(i -> all(==(0), A[i, :]), 1:M)] -= 1            # E ψ_0 = 1, every other E ψ_α = 0
    R, J
end
dfact(k) = (r = BigFloat(1); for j in 1:2:(k-1); r *= j; end; r)
function relgate(X, w)                                          # the deposit's functional, in BigFloat
    errs = zeros(BigFloat, M)
    @threads for i in 1:M
        acc = BigFloat(0); aa = BigFloat(0)
        for s in 1:n
            t = w[s]; for k in 1:D; t *= X[s, k]^A[i, k]; end
            acc += t; aa += abs(t)
        end
        ex = any(isodd, A[i, :]) ? BigFloat(0) : prod(dfact(a) for a in A[i, :])
        errs[i] = abs(acc - ex) / max(aa, 1)
    end
    maximum(errs)
end

say("$(basename(FILE))  n=$n  unknowns $(n * (D + 1))  equations $M  BigFloat($(precision(BigFloat)))  target $(TARGET)")
t0 = time(); its = 0
R, J = system(X, u); nr = norm(R)
say(@sprintf("it  0  ‖R‖ %.2e", nr))
while its < MAXIT && nr > BigFloat(TARGET) / 1000
    global X, u, R, J, nr, its
    H = J' * J
    λ = maximum(diag(H)) * BigFloat(10)^-90                     # far below the resolved spectrum
    dx = -((H + λ * I) \ (J' * R))
    Xn = X .+ reshape(dx[1:n*D], D, n)'; un = u .+ dx[n*D+1:end]
    Rn, Jn = system(Xn, un); nn = norm(Rn)
    its += 1
    say(@sprintf("it %2d  ‖R‖ %.2e  ‖dx‖ %.2e", its, nn, norm(dx)))
    nn < nr || break
    X, u, R, J, nr = Xn, un, Rn, Jn, nn
end
w = exp.(u); w ./= sum(w)
gate = relgate(X, w)
ok = gate ≤ TARGET && all(>(0), w)
say(@sprintf("RESULT %s  %s  gate %.2e (target %.0e)  iterations %d  wall %.0f s",
             basename(FILE), ok ? "MET" : "NOT MET", gate, TARGET, its, time() - t0))
ok || exit(3)
mkpath(OUTDIR); out = joinpath(OUTDIR, replace(basename(FILE), ".csv" => ".mp.csv"))
fmt(v) = @sprintf("%.*e", DIGITS - 1, v)
open(out * ".tmp", "w") do io
    println(io, "# source: ", relpath(FILE, dirname(OUTDIR)), "  sha256 ", bytes2hex(open(sha256, FILE)))
    println(io, "# frame: standard normal N(0,I_$D); weights sum to 1")
    println(io, @sprintf("# polish: polish_lm_big.jl, Levenberg step with Jacobian and solve in BigFloat(%d), %d iterations, unreduced, %s",
                         precision(BigFloat), its, gethostname()))
    println(io, @sprintf("# verified max relative monomial error (all degree ≤ %d, big): %.3e", P, gate))
    println(io, "# digits: ", DIGITS, "  date: ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    for s in 1:n; println(io, join((fmt(X[s, k]) for k in 1:D), ","), ",", fmt(w[s])); end
end
mv(out * ".tmp", out; force = true); say("wrote rules_mp/$(basename(out))")
