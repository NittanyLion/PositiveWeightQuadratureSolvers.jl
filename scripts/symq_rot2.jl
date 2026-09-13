# D_k rotational-orbit search for planar Gaussian rules (2026-09-13, user: "ok re
# rotational orbit ansatz"; STATUS 2026-09-13, METHODS §32).
#
# The standard normal weight on R² is invariant under the whole orthogonal
# group, so ANY dihedral group D_k (rotations by 2π/k plus a reflection) is an
# admissible symmetry ansatz — unlike the square, which only offers D_4.  The
# best classical planar Gaussian rules live in exactly this class (Haegemans–
# Piessens 1976: hexagonal, p=11, n=25), and the B_2 = D_4 orbit search never
# beat free elimination on the GH d=2 bank.  This tool searches D_k-invariant
# rules for one k at a time.
#
# Orbit types under D_k (reflection about the x-axis):
#   center      1 node                        unknowns: u            (w = e^u)
#   mirror-A    k nodes at angles 2πj/k       unknowns: r, u
#   mirror-B    k nodes at angles π/k + 2πj/k unknowns: r, u
#   generic     2k nodes at ±θ + 2πj/k        unknowns: r, θ, u
#
# Conditions (Sobolev): a D_k-invariant rule is exact to degree p iff it is
# exact on the D_k-invariant polynomials of degree ≤ p, which are spanned by
# r^{2a} Re(z^{kb}) = r^{2a+kb} cos(kbθ), 2a + kb ≤ p.  In s = r²/2 ~ Exp(1)
# the orthonormal version is
#   φ_{a,b}(s, θ) = c_{a,b} s^{kb/2} L_a^{(kb)}(s) cos(kbθ),
#   c_{a,b} = sqrt((b ≡ 0 ? 1 : 2) · a! / Γ(a + kb + 1)),
# with E[φ_{a,b}] = δ_{a0} δ_{b0}, so the residual is Σ_o |o| w_o φ_{a,b}(o) − δ.
# Every node of an orbit gives the same φ value (φ is invariant), so an orbit
# costs one evaluation.  C_k(p) = #{(a, b) : 2a + kb ≤ p} conditions:
# k=6, p=31 → 51; k=4, p=31 → 72.
#
# Counting heuristic: the cheapest orbits per unknown are the center (1 node
# per unknown) and the mirror orbits (k/2 per unknown), so a D_k rule near the
# counting bound needs ≈ (k/2)·C_k(p) nodes: k=6, p=31 → ~153 against the
# free-node bound (p+1)(p+2)/6 = 176 and Möller's 144.  Whether such rules
# EXIST is what the search decides; the count says only where to look.
#
# usage: julia symq_rot2.jl <k> <p> <minutes> [seed]        (d = 2, basis hermite)
#        julia symq_rot2.jl --selftest                        (HP76 hexagonal rule)
#
# Banks rules/hermite_d2_p{p}_n{n}.csv (plain x1,x2,w rows) whenever a rule
# smaller than the bank's best passes verify_exactness ≤ 1e-11 (relative, all
# monomials) with every weight positive; lineage line via symq_lineage.jl.
# Log symq/rot2_d2p{p}_k{k}.log; progress symq/prog_rot2_d2p{p}_k{k}.txt.

include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
using .DesignedQuadrature
using LinearAlgebra, Random, Printf, Dates

symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

BLAS.set_num_threads(1)

const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const EXTOL = 1e-11
const D     = 2

# --------------------------------------------------------------------------- #
# invariant basis
# --------------------------------------------------------------------------- #
struct Cond
    a::Int
    b::Int
    c::Float64          # normalization
end

# c² = (1 or 2) · a! / (a + kb)!  — an integer ratio, so no log-gamma needed
conditions(k, p) = [Cond(a, b, sqrt((b ≡ 0 ? 1.0 : 2.0) / prod(Float64, (a + 1):(a + k * b); init = 1.0)))
                    for b ∈ 0:(p ÷ k) for a ∈ 0:((p - k * b) ÷ 2)]

# generalized Laguerre L_a^{(α)}(s) by the three-term recurrence
function laguerre(a::Int, α::Float64, s::Float64)
    a ≡ 0 && return 1.0
    l0 = 1.0; l1 = 1.0 + α - s
    for n ∈ 1:(a - 1)
        l0, l1 = l1, ((2n + 1 + α - s) * l1 - (n + α) * l0) / (n + 1)
    end
    l1
end

# φ_{a,b} at polar (r, θ) — r ≥ 0
function phi(cd::Cond, k::Int, r::Float64, θ::Float64)
    s = r * r / 2
    α = k * cd.b
    v = cd.c * laguerre(cd.a, Float64(α), s)
    α > 0 && (v *= s^(α / 2) * cos(α * θ))
    v
end

# --------------------------------------------------------------------------- #
# structures and parameter vectors
# --------------------------------------------------------------------------- #
# kind: 0 center, 1 mirror-A, 2 mirror-B, 3 generic
struct Structure
    c::Int
    mA::Int
    mB::Int
    g::Int
end
nnodes(k, st::Structure) = st.c + k * (st.mA + st.mB) + 2k * st.g
nunk(st::Structure)      = st.c + 2 * (st.mA + st.mB) + 3 * st.g
orbits(st::Structure)    = vcat(fill(0, st.c), fill(1, st.mA), fill(2, st.mB), fill(3, st.g))
Base.show(io::IO, st::Structure) = print(io, "c$(st.c) A$(st.mA) B$(st.mB) g$(st.g)")

# unpack x into per-orbit (r, θ, u), using actual polar coordinates of the
# representative node so a negative r or a wandering θ stays consistent
function unpack(x, kinds, k)
    out = Vector{NTuple{3,Float64}}(undef, length(kinds)); i = 1
    for (o, kind) ∈ enumerate(kinds)
        if kind ≡ 0
            out[o] = (0.0, 0.0, x[i]); i += 1
        elseif kind ≡ 1 || kind ≡ 2
            r = x[i]; u = x[i + 1]; i += 2
            θ0 = kind ≡ 1 ? 0.0 : π / k
            out[o] = (abs(r), r < 0 ? θ0 + π : θ0, u)
        else
            r = x[i]; θ = x[i + 1]; u = x[i + 2]; i += 3
            out[o] = (abs(r), r < 0 ? θ + π : θ, u)
        end
    end
    out
end
osize(kind, k) = kind ≡ 0 ? 1 : kind ≡ 3 ? 2k : k

function residual!(F, x, kinds, k, conds)
    prm = unpack(x, kinds, k)
    fill!(F, 0.0)
    for (o, kind) ∈ enumerate(kinds)
        r, θ, u = prm[o]
        m = osize(kind, k) * exp(u)
        for (i, cd) ∈ enumerate(conds)
            F[i] += m * phi(cd, k, r, θ)
        end
    end
    F[1] -= 1.0                      # conds[1] is (a,b) = (0,0)
    F
end

function jacobian!(Jm, x, kinds, k, conds, Fp, Fm)
    n = length(x)
    for j ∈ 1:n
        h = 1e-6 * max(1.0, abs(x[j]))
        xj = x[j]
        x[j] = xj + h; residual!(Fp, x, kinds, k, conds)
        x[j] = xj - h; residual!(Fm, x, kinds, k, conds)
        x[j] = xj
        @views Jm[:, j] .= (Fp .- Fm) ./ (2h)
    end
    Jm
end

# Levenberg–Marquardt; returns (x, ‖F‖∞, converged)
function lm!(x, kinds, k, conds; maxit = 400, tol = 1e-15, accept = 1e-12)
    m = length(conds); n = length(x)
    F = zeros(m); Fp = zeros(m); Fm = zeros(m); Ft = zeros(m)
    Jm = zeros(m, n); λ = 1e-3
    residual!(F, x, kinds, k, conds); f = norm(F)
    xt = similar(x)
    for it ∈ 1:maxit
        maximum(abs, F) ≤ tol && return x, maximum(abs, F), true   # at the FD-Jacobian floor
        jacobian!(Jm, x, kinds, k, conds, Fp, Fm)
        JtJ = Jm' * Jm; g = Jm' * F
        dg = max.(diag(JtJ), 1e-12)
        improved = false
        for _ ∈ 1:12
            A = JtJ + λ * Diagonal(dg)
            δ = try
                -(cholesky(Symmetric(A)) \ g)
            catch
                -(A + 1e-10I) \ g
            end
            xt .= x .+ δ
            residual!(Ft, xt, kinds, k, conds); ft = norm(Ft)
            if ft < f
                x .= xt; F .= Ft; f = ft; λ = max(λ / 3, 1e-15); improved = true
                break
            else
                λ *= 4
            end
        end
        improved || return x, maximum(abs, F), maximum(abs, F) ≤ accept   # stalled: accept if inside the gate
        (f < 1e-9 && λ > 1e-3) && (λ = 1e-6)
    end
    x, maximum(abs, F), maximum(abs, F) ≤ accept
end

# --------------------------------------------------------------------------- #
# nodes ↔ parameters
# --------------------------------------------------------------------------- #
function build_nodes(x, kinds, k)
    prm = unpack(x, kinds, k)
    X = Float64[]; Y = Float64[]; W = Float64[]
    for (o, kind) ∈ enumerate(kinds)
        r, θ, u = prm[o]; w = exp(u)
        if kind ≡ 0
            push!(X, 0.0); push!(Y, 0.0); push!(W, w)
        else
            angs = kind ≡ 3 ? [θ, -θ] : [θ]
            for a0 ∈ angs, j ∈ 0:(k - 1)
                a = a0 + 2π * j / k
                push!(X, r * cos(a)); push!(Y, r * sin(a)); push!(W, w)
            end
        end
    end
    hcat(X, Y), W
end

# merge coincident nodes (a generic orbit that collapsed onto a mirror line, or
# two orbits that met): summed weights, so n can only fall
function dedupe(nodes, w; tol = 1e-8)
    n = length(w); keep = trues(n); w2 = copy(w)
    for i ∈ 1:n
        keep[i] || continue
        for j ∈ (i + 1):n
            keep[j] || continue
            if abs(nodes[i, 1] - nodes[j, 1]) ≤ tol && abs(nodes[i, 2] - nodes[j, 2]) ≤ tol
                w2[i] += w2[j]; keep[j] = false
            end
        end
    end
    nodes[keep, :], w2[keep]
end

# decompose a rule into D_k orbits; returns (structure, x) or nothing
function extract(nodes, w, k; tol = 1e-7)
    n = length(w)
    r = hypot.(nodes[:, 1], nodes[:, 2])
    θ = atan.(nodes[:, 2], nodes[:, 1])
    per = 2π / k
    θc = mod.(θ, per); θc = min.(θc, per .- θc)          # fold to [0, π/k]
    used = falses(n)
    kinds = Int[]; prms = Vector{Vector{Float64}}()
    order = sortperm(r)
    for i ∈ order
        used[i] && continue
        if r[i] ≤ tol
            push!(kinds, 0); push!(prms, [log(w[i])]); used[i] = true; continue
        end
        grp = [j for j ∈ 1:n if !used[j] && abs(r[j] - r[i]) ≤ tol * max(1, r[i]) && abs(θc[j] - θc[i]) ≤ tol && abs(w[j] - w[i]) ≤ tol * max(1e-300, w[i])]
        kind = θc[i] ≤ tol ? 1 : abs(θc[i] - π / k) ≤ tol ? 2 : 3
        length(grp) ≡ osize(kind, k) || return nothing
        used[grp] .= true
        push!(kinds, kind)
        push!(prms, kind ≡ 3 ? [r[i], θc[i], log(w[i])] : [r[i], log(w[i])])
    end
    # reorder to structure order: center, A, B, generic
    ord = sortperm(kinds)
    st = Structure(count(x -> x ≡ 0, kinds), count(x -> x ≡ 1, kinds), count(x -> x ≡ 2, kinds), count(x -> x ≡ 3, kinds))
    st, vcat(prms[ord]...)
end

# --------------------------------------------------------------------------- #
# bank
# --------------------------------------------------------------------------- #
function best_rule(p)
    best = typemax(Int); path = ""
    for f ∈ readdir(RULES)
        m = match(Regex("^hermite_d2_p$(p)_n(\\d+)\\.csv\$"), f)
        m ≡ nothing && continue
        nn = parse(Int, m[1])
        nn < best && (best = nn; path = joinpath(RULES, f))
    end
    best, path
end
function read_rule(path)
    rows = [parse.(Float64, split(l, ',')) for l ∈ eachline(path) if !isempty(strip(l)) && !startswith(l, "#")]
    hcat([r[1] for r ∈ rows], [r[2] for r ∈ rows]), [r[3] for r ∈ rows]
end

# --------------------------------------------------------------------------- #
# self-test: the Haegemans–Piessens hexagonal rule is D_6 with c1 A2 B2
# --------------------------------------------------------------------------- #
function selftest()
    path = joinpath(RULES, "hermite_d2_p11_n25.csv")
    nodes, w = read_rule(path)
    k = 6; p = 11
    conds = conditions(k, p)
    println("k=$k p=$p: $(length(conds)) invariant conditions")
    ex = extract(nodes, w, k)
    ex ≡ nothing && error("selftest: rule is not D_6-symmetric under the extractor")
    st, x = ex
    println("structure $st  n=$(nnodes(k, st))  unknowns=$(nunk(st))")
    F = zeros(length(conds)); residual!(F, x, orbits(st), k, conds)
    println("invariant residual ‖F‖∞ = ", maximum(abs, F))
    nodes2, w2 = build_nodes(x, orbits(st), k)
    e = verify_exactness(nodes2, w2, p; basis = :hermite, relative = true)
    println("rebuilt rule: n=$(length(w2))  Σw-1=$(sum(w2)-1)  verify_exactness=$e")
    # perturb and re-solve
    x2 = x .* (1 .+ 1e-3 .* randn(MersenneTwister(1), length(x)))
    x2, fmax, ok = lm!(x2, orbits(st), k, conds)
    n2, w3 = build_nodes(x2, orbits(st), k)
    e2 = verify_exactness(n2, w3, p; basis = :hermite, relative = true)
    println("perturbed 1e-3 and re-solved: converged=$ok ‖F‖∞=$fmax verify=$e2")
    ok2 = maximum(abs, F) < 1e-12 && e < 1e-13 && ok && e2 < 1e-12
    println(ok2 ? "SELFTEST PASS" : "SELFTEST FAIL")
    ok2
end

# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
function structures(k, C; slack = 4)
    out = Structure[]
    for g ∈ 0:cld(C, 3), c ∈ 0:1
        # m = mA + mB with U = c + 2m + 3g ∈ [C, C + slack]
        for m ∈ 0:cld(C, 2)
            U = c + 2m + 3g
            C ≤ U ≤ C + slack || continue
            for mA ∈ 0:m
                push!(out, Structure(c, mA, m - mA, g))
            end
        end
    end
    out
end

function random_start(rng, st, k, p)
    kinds = orbits(st)
    R = sqrt(2p + 3.0)
    x = Float64[]
    nm = nnodes(k, st)
    for kind ∈ kinds
        if kind ≡ 0
            push!(x, log(1 / nm))
        else
            r = R * sqrt(rand(rng)) * (0.6 + 0.4 * rand(rng))
            u = -r^2 / 2 + log(2 / (R^2 * nm)) + log(1 + 0.5 * rand(rng))
            push!(x, r)
            kind ≡ 3 && push!(x, (π / k) * (0.1 + 0.8 * rand(rng)))
            push!(x, u)
        end
    end
    # normalize weights to sum 1
    nds, w = build_nodes(x, kinds, k)
    sh = log(sum(w)); i = 1
    for kind ∈ kinds
        i += kind ≡ 0 ? 0 : kind ≡ 3 ? 2 : 1
        x[i] -= sh; i += 1
    end
    x
end

function main()
    k = parse(Int, ARGS[1]); p = parse(Int, ARGS[2]); minutes = parse(Float64, ARGS[3])
    seed = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 1
    k ≥ 2 || error("k ≥ 2")
    rng = MersenneTwister(1009 * seed + 31 * k + p)
    logio = open(joinpath(SYMQ, "rot2_d2p$(p)_k$(k).log"), "a")
    prog  = joinpath(SYMQ, "prog_rot2_d2p$(p)_k$(k).txt")
    conds = conditions(k, p); C = length(conds)
    best0, bpath = best_rule(p)
    best = best0
    # diagnostic/sandbox: search below a chosen count instead of the bank best
    haskey(ENV, "SYMQ_ROT2_BEST") && (best = parse(Int, ENV["SYMQ_ROT2_BEST"]))
    deadline = time() + 60minutes
    println(logio, "=== $(now()) rot2 k=$k p=$p seed=$seed: $C invariant conditions, bank best $best0, budget $minutes min")
    println(logio, "counting bound (mirror orbits): n ≈ $(round(Int, k * C / 2)); free-node bound $((p+1)*(p+2)÷6)")
    flush(logio)
    write(prog, "d=2 p=$p  start $best0, run-best $best0, none banked (rot2 k=$k r0)\n")
    attempt = 0

    function bank!(nodes, w, tag)
        nodes, w = dedupe(nodes, w)
        n = length(w)
        all(w .> 0) || return false
        n < best || return false
        ex = verify_exactness(nodes, w, p; basis = :hermite, relative = true)
        if ex > EXTOL
            println(logio, "  n=$n candidate failed the monomial gate ($ex) — not banked"); flush(logio)
            return false
        end
        path = joinpath(RULES, "hermite_d2_p$(p)_n$(n).csv")
        open(path, "w") do io
            for i ∈ 1:n
                println(io, join(string.([nodes[i, 1], nodes[i, 2], w[i]]), ","))
            end
        end
        symq_lineage!(basename(path), "none",
                      "D_k rotational-orbit search k=$k, $tag (symq_rot2.jl seed $seed)"; rules = RULES)
        best = n
        println(logio, "BANKED $path (exactness $ex) [$tag]"); flush(logio)
        write(prog, "d=2 p=$p  start $best0, run-best $n, banked $n (rot2 k=$k r$attempt)\n")
        true
    end

    # orbit elimination from a converged D_k rule: drop one orbit (lightest
    # first), reconverge from the survivors with a little jitter, bank every
    # verified improvement, repeat while it works.  Used on the bank best when
    # that happens to be D_k-symmetric, and after every cold success — descent
    # from a solved rule is far cheaper than a fresh cold start at each n
    # (measured 2026-09-13: cold starts converge in ~1-2% of tries).
    function descend!(x, kinds, tag)
        improved = true
        while improved && time() < deadline
            improved = false
            prm = unpack(x, kinds, k)
            order = sortperm([osize(kinds[o], k) * exp(prm[o][3]) for o ∈ eachindex(kinds)])
            for o ∈ order
                time() < deadline || break
                kinds2 = deleteat!(copy(kinds), o)
                st2 = Structure(count(x -> x ≡ 0, kinds2), count(x -> x ≡ 1, kinds2),
                                count(x -> x ≡ 2, kinds2), count(x -> x ≡ 3, kinds2))
                nunk(st2) ≥ C - 2 || continue
                idx = Int[]; i = 1
                for (oo, kind) ∈ enumerate(kinds)
                    len = kind ≡ 0 ? 1 : kind ≡ 3 ? 3 : 2
                    oo ≢ o && append!(idx, i:(i + len - 1)); i += len
                end
                n0 = nnodes(k, Structure(count(x -> x ≡ 0, kinds), count(x -> x ≡ 1, kinds),
                                         count(x -> x ≡ 2, kinds), count(x -> x ≡ 3, kinds)))
                for rr ∈ 1:4
                    attempt += 1
                    x2 = x[idx] .* (1 .+ 1e-3 * rr .* randn(rng, length(idx)))
                    x2, fmax, ok = lm!(x2, kinds2, k, conds)
                    ok || continue
                    nds, ww = build_nodes(x2, kinds2, k)
                    if bank!(nds, ww, "$tag, orbit drop from n=$n0")
                        x = x2; kinds = kinds2; improved = true; break
                    end
                end
                improved && break
            end
        end
        x, kinds
    end

    # warm start: orbit elimination from the bank best if it is D_k-symmetric
    if !isempty(bpath)
        nodes, w = read_rule(bpath)
        ex = extract(nodes, w, k)
        if ex ≢ nothing
            st, x = ex
            println(logio, "bank best is D_$k-symmetric: $st — trying orbit elimination"); flush(logio)
            descend!(x, orbits(st), "bank best")
        else
            println(logio, "bank best (n=$best0) is not D_$k-symmetric; cold structures only"); flush(logio)
        end
    end

    # cold structures, descending n below the current best; restarts grow per pass
    sts = structures(k, C)
    R = 6; pass = 0
    while time() < deadline
        pass += 1
        cand = sort([st for st ∈ sts if nnodes(k, st) < best], by = st -> (-nnodes(k, st), nunk(st)))
        isempty(cand) && (println(logio, "no structure below n=$best; done"); break)
        println(logio, "pass $pass: $(length(cand)) structures below n=$best, $R restarts each"); flush(logio)
        success = false; nconv = 0; nfail = 0
        for st ∈ cand
            time() < deadline || break
            nnodes(k, st) < best || continue
            kinds = orbits(st)
            for rr ∈ 1:R
                time() < deadline || break
                attempt += 1
                x = random_start(rng, st, k, p)
                x, fmax, ok = lm!(x, kinds, k, conds)
                ok || (nfail += 1; continue)
                nconv += 1
                nds, ww = build_nodes(x, kinds, k)
                if bank!(nds, ww, "cold structure $st")
                    success = true
                    descend!(x, kinds, "cold structure $st")   # ride the find down
                    break
                end
            end
        end
        println(logio, "  pass $pass: $nconv solves converged, $nfail did not"); flush(logio)
        success || (R = min(2R, 400))
    end
    println(logio, "=== $(now()) done: best $best (started $best0), $attempt solves"); flush(logio)
    close(logio)
end

if "--selftest" ∈ ARGS
    exit(selftest() ? 0 : 1)
else
    main()
end
