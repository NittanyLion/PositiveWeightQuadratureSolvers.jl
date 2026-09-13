# LaguerrePolish.jl — extended-precision polish of an orthant rule for the product
# Gamma(alpha+1, 1) weight, in the RELATIVE monomial form the bank's gate measures
# (2026-09-07).
#
# Why: the descents solve in the orthonormal Laguerre basis to δ ≤ 1e-12, but the
# bank's gate is the relative monomial error at 1e-11, and the change of basis
# x^n = n! Σ_k (-1)^k C(n,k) L̃_k amplifies roundoff by up to 2^p — 5e5 at p = 19.
# So a genuinely converged 99-node d2 p19 rule shows ex ≈ 8e-9 and is thrown away
# ("failed re-verify"), segment after segment.  A few Gauss–Newton steps in
# BigFloat on the monomial system itself take the same rule to ex ≈ 1e-15.
#
# polish(nodes, w, p; alpha, digits = 50, maxiter = 8) -> (nodes, w, ok, ex_before, ex_after)
#   nodes  n × d Float64 (≥ 0), w Float64 (> 0).  Coordinates that are exactly 0
#   stay 0 (boundary nodes); weights are moved through their logs so they stay
#   positive.  Returns Float64 arrays.  ok = false (inputs returned unchanged)
#   when the system is too large for the cap, a step leaves the orthant, or the
#   residual does not improve.
# cost_ok(n, d, p): the Gauss–Newton step costs C²·N BigFloat multiplications
#   (C = C(d+p, d) equations, N = n(d+1) unknowns); LAGUERRE_POLISH_CAP (default
#   1e9, ~ a minute) bounds it.  Above the cap the caller banks nothing and logs
#   the rule for an offline polish (polish_bigfloat.jl basis=laguerre).
module LaguerrePolish
using LinearAlgebra

const CAP = parse(Float64, get(ENV, "LAGUERRE_POLISH_CAP", "1e9"))

function exponents(d, p)
    out = Vector{Vector{Int}}()
    for idx in CartesianIndices(ntuple(_ -> p + 1, d))
        a = collect(Tuple(idx)) .- 1
        sum(a) ≤ p && push!(out, a)
    end
    out
end

nequations(d, p) = binomial(d + p, d)
cost_ok(n, d, p) = (C = nequations(d, p); N = n*(d + 1); Float64(C)^2*N ≤ CAP)

function polish(nodes::AbstractMatrix, w::AbstractVector, p::Int; alpha = 0.0, digits = 50, maxiter = 8)
    n, d = size(nodes)
    cost_ok(n, d, p) || return (Matrix(nodes), Vector(w), false, NaN, NaN)
    EX = exponents(d, p); C = length(EX)
    old = precision(BigFloat)
    setprecision(BigFloat, ceil(Int, digits*3.33))
    try
        X = BigFloat.(nodes); lw = log.(BigFloat.(w)); α = BigFloat(alpha)
        # exact relative moments M_a = Π_k Π_{j<a_k} (α+1+j)
        gm(a) = a == 0 ? one(BigFloat) : prod(α + 1 + j for j in 0:a-1)
        M = [prod(gm(a[k]) for k in 1:d) for a in EX]
        free = [X[s, k] > 0 for s in 1:n, k in 1:d]          # boundary coordinates stay put
        colx = Dict{Tuple{Int,Int},Int}(); N = 0
        for s in 1:n, k in 1:d
            free[s, k] && (N += 1; colx[(s, k)] = N)
        end
        Nw = N; N += n                                          # + one log-weight per node
        pw = Array{BigFloat}(undef, n, d, p + 1)               # x_sk^j
        R = zeros(BigFloat, C); J = zeros(BigFloat, C, N)
        function assemble!()
            for s in 1:n, k in 1:d
                pw[s, k, 1] = 1
                for j in 1:p; pw[s, k, j+1] = pw[s, k, j]*X[s, k]; end
            end
            fill!(J, 0)
            for (c, a) in enumerate(EX)
                acc = zero(BigFloat)
                for s in 1:n
                    t = exp(lw[s])
                    for k in 1:d; t *= pw[s, k, a[k]+1]; end
                    t /= M[c]
                    acc += t
                    J[c, Nw + s] = t
                    for k in 1:d
                        a[k] == 0 && continue
                        haskey(colx, (s, k)) || continue
                        J[c, colx[(s, k)]] = t*a[k]/X[s, k]
                    end
                end
                R[c] = acc - 1
            end
            maximum(abs, R)
        end
        ex0 = assemble!(); ex = ex0
        μ = BigFloat(10)^(-2digits)
        for it in 1:maxiter
            ex < BigFloat(10)^(-(digits - 8)) && break
            δ = C ≤ N ? -(J' * ((J*J' + μ*I) \ R)) : -((J'J + μ*I) \ (J'R))
            Xn = copy(X); lwn = copy(lw)
            for ((s, k), c) in colx; Xn[s, k] += δ[c]; end
            for s in 1:n; lwn[s] += δ[Nw + s]; end
            any(Xn .< 0) && return (Matrix(nodes), Vector(w), false, Float64(ex0), Float64(ex))
            X, lw = Xn, lwn
            exn = assemble!()
            exn < ex || return (Matrix(nodes), Vector(w), false, Float64(ex0), Float64(ex))
            ex = exn
        end
        return (Float64.(X), Float64.(exp.(lw)), true, Float64(ex0), Float64(ex))
    finally
        setprecision(BigFloat, old)
    end
end

# ---------------------------------------------------------------------------
# Symmetry-reduced polish (2026-09-08).
#
# Why: polish() above works on the full node set with every monomial, C(d+p,d)
# equations × n(d+1) unknowns, and its cost cap put it out of reach of exactly
# the rules that need it — every tensor-descent rule at d3 p ≥ 19 and d4 p ≥ 19
# was logged "above LAGUERRE_POLISH_CAP" (502 times by 09-08), so the bank
# froze at the grid while the descent went on.  Measured the same morning: the
# Float64 LM in the orthonormal Laguerre basis is at its roundoff floor (δ
# 1e-12 at tol 1e-14..1e-16, no movement) while the relative monomial error
# sits at 3e-8 … 1e-6.  The descent's rules are S_d-symmetric (S_{d-1}×S_1 for
# the product ansatz), so one condition per α modulo the group and one
# parameter per distinct coordinate value per orbit is the same system at a
# fraction of the size: d3 p19 is 314 × 714 instead of 1540 × 3752, d4 p23
# about 1.2k × 3.5k instead of 17.5k × 76k.  The relative monomial residual
# has only positive terms, so Float64 Gauss–Newton on it converges to ~1e-14;
# BigFloat is the fallback, not the default.
#
# polish_sym(nodes, w, p; alpha, perms, maxiter, digits) -> (nodes, w, ok, ex0, ex1)
#   perms: the symmetry group as coordinate permutations (default S_d).  Nodes
#   are grouped into orbits under it; an orbit whose input nodes do not match
#   its regenerated points (a non-symmetric rule) makes the call return
#   ok = false unchanged.  digits = 0 works in Float64; digits > 0 in BigFloat.
# polish_auto(nodes, w, p; alpha, perms, gate) -> same tuple: Float64 first,
#   then BigFloat(30 digits) from the Float64 result if the gate is not met.
# ---------------------------------------------------------------------------

function all_perms(d::Int)
    out = Vector{Int}[]
    rec(pre, rest) = isempty(rest) ? push!(out, copy(pre)) :
        (for (i, r) in enumerate(rest); rec(vcat(pre, r), vcat(rest[1:i-1], rest[i+1:end])); end)
    rec(Int[], collect(1:d))
    out
end
# S_{d-1} × S_1: the last coordinate is not exchangeable (LaguerreProductDQ)
prod_perms(d::Int) = [vcat(σ, d) for σ in all_perms(d - 1)]

canon(v::AbstractVector, perms) = maximum(Tuple(v[σ]) for σ in perms)

function moment_conditions(d::Int, p::Int, perms)
    seen = Set{NTuple{d,Int}}(); out = Vector{Int}[]
    for idx in CartesianIndices(ntuple(_ -> p + 1, d))
        a = collect(Tuple(idx)) .- 1
        sum(a) ≤ p || continue
        c = canon(a, perms)
        c ∈ seen && continue
        push!(seen, c); push!(out, collect(c))
    end
    out
end

# orbit decomposition of an expanded rule: per orbit the assignment rows
# (0 = structural zero, j = parameter group), values, log-weight
struct SymOrbit
    asg::Matrix{Int}
    k::Int
end

function polish_sym(nodes::AbstractMatrix, w::AbstractVector, p::Int;
                    alpha = 0.0, perms = nothing, maxiter::Int = 12, digits::Int = 0,
                    verbose::Bool = false)
    n, d = size(nodes)
    perms === nothing && (perms = all_perms(d))
    # coordinate classes under the group (indices the group can exchange)
    cls = collect(1:d)
    for σ in perms, i in 1:d
        a, b = cls[i], cls[σ[i]]
        a == b && continue
        lo, hi = minmax(a, b)
        cls[cls .== hi] .= lo
    end
    rnd(x) = round(x; digits = 12) + 0.0
    # group nodes into orbits by canonical form
    groups = Dict{NTuple{d,Float64},Vector{Int}}()
    for s in 1:n
        push!(get!(groups, canon(rnd.(nodes[s, :]), perms), Int[]), s)
    end
    orbits = SymOrbit[]; vals = Vector{Float64}[]; lw = Float64[]
    for (key, members) in groups
        rep = collect(key)
        gid = Dict{Tuple{Int,Float64},Int}(); gidx = zeros(Int, d); vs = Float64[]
        for i in 1:d
            rep[i] > 0 || continue
            gk = (cls[i], rep[i])
            haskey(gid, gk) || (push!(vs, rep[i]); gid[gk] = length(vs))
            gidx[i] = gid[gk]
        end
        rows = unique([Tuple(gidx[σ]) for σ in perms])
        length(rows) == length(members) ||
            (verbose && println("polish_sym: orbit with $(length(members)) nodes regenerates $(length(rows)) points — not symmetric"); 
             return (Matrix(nodes), Vector(w), false, NaN, NaN))
        asg = Matrix{Int}(undef, length(rows), d)
        for (r, t) in enumerate(rows); asg[r, :] .= collect(t); end
        push!(orbits, SymOrbit(asg, length(vs))); push!(vals, vs)
        push!(lw, log(sum(w[members]) / length(members)))
    end
    conds = moment_conditions(d, p, perms); C = length(conds)
    N = sum(o.k + 1 for o in orbits)
    T = digits > 0 ? BigFloat : Float64
    old = precision(BigFloat)
    digits > 0 && setprecision(BigFloat, ceil(Int, digits * 3.33))
    try
        α = T(alpha)
        gm(a) = a == 0 ? one(T) : prod(α + 1 + j for j in 0:a-1)
        M = [prod(gm(c[i]) for i in 1:d) for c in conds]
        V = [T.(v) for v in vals]; U = T.(lw)
        kmax = maximum(o.k for o in orbits; init = 0)
        pw = zeros(T, kmax, p + 1)
        A = zeros(Int, kmax)
        R = zeros(T, C); J = zeros(T, C, N)
        function assemble!(V, U, withJ::Bool)
            fill!(R, 0); withJ && fill!(J, 0)
            off = 0
            for (o, ob) in enumerate(orbits)
                wo = exp(U[o]); v = V[o]
                for j in 1:ob.k
                    pw[j, 1] = 1
                    for e in 1:p; pw[j, e+1] = pw[j, e] * v[j]; end
                end
                for (c, a) in enumerate(conds)
                    S = zero(T)
                    dS = withJ ? zeros(T, ob.k) : T[]
                    for r in axes(ob.asg, 1)
                        fill!(A, 0); dead = false
                        for i in 1:d
                            j = ob.asg[r, i]
                            if j == 0
                                a[i] > 0 && (dead = true; break)
                            else
                                A[j] += a[i]
                            end
                        end
                        dead && continue
                        t = one(T)
                        for j in 1:ob.k; t *= pw[j, A[j] + 1]; end
                        S += t
                        if withJ
                            for j in 1:ob.k
                                A[j] > 0 && (dS[j] += A[j] * t / v[j])
                            end
                        end
                    end
                    R[c] += wo * S / M[c]
                    if withJ
                        for j in 1:ob.k; J[c, off + j] += wo * dS[j] / M[c]; end
                        J[c, off + ob.k + 1] += wo * S / M[c]
                    end
                end
                off += ob.k + 1
            end
            R .-= 1
            maximum(abs, R)
        end
        ex0 = assemble!(V, U, true); ex = ex0
        target = digits > 0 ? T(10)^(-(digits - 6)) : 1e-14
        for it in 1:maxiter
            ex ≤ target && break
            μ = (digits > 0 ? T(10)^(-digits) : 1e-14) * max(maximum(abs, J), one(T))^2
            δ = C ≤ N ? -(J' * ((J * J' + μ * I) \ R)) : -((J' * J + μ * I) \ (J' * R))
            step = one(T); accepted = false
            for _ in 1:6
                Vn = [copy(v) for v in V]; Un = copy(U); off = 0; okstep = true
                for (o, ob) in enumerate(orbits)
                    for j in 1:ob.k
                        Vn[o][j] += step * δ[off + j]
                        Vn[o][j] > 0 || (okstep = false)
                    end
                    Un[o] += step * δ[off + ob.k + 1]
                    off += ob.k + 1
                end
                if okstep
                    exn = assemble!(Vn, Un, true)
                    if exn < ex
                        V, U, ex = Vn, Un, exn; accepted = true; break
                    end
                end
                step /= 2
            end
            verbose && println("  it $it: ex $ex")
            accepted || (assemble!(V, U, true); break)
        end
        # rebuild the Float64 rule
        out = zeros(n, d); ow = zeros(n); row = 0
        for (o, ob) in enumerate(orbits)
            wo = Float64(exp(U[o]))
            for r in axes(ob.asg, 1)
                row += 1
                for i in 1:d
                    j = ob.asg[r, i]
                    out[row, i] = j == 0 ? 0.0 : Float64(V[o][j])
                end
                ow[row] = wo
            end
        end
        return (out, ow, ex < ex0, Float64(ex0), Float64(ex))
    finally
        setprecision(BigFloat, old)
    end
end

function polish_auto(nodes, w, p; alpha = 0.0, perms = nothing, gate = 1e-12, verbose = false)
    n1, w1, ok1, e0, e1 = polish_sym(nodes, w, p; alpha, perms, verbose)
    ok1 || return (n1, w1, false, e0, e1)
    e1 ≤ gate && return (n1, w1, true, e0, e1)
    n2, w2, ok2, _, e2 = polish_sym(n1, w1, p; alpha, perms, digits = 30, maxiter = 6, verbose)
    return ok2 && e2 < e1 ? (n2, w2, true, e0, e2) : (n1, w1, true, e0, e1)
end

end # module
