# Icosahedral orbit ansatz for d = 3 Gaussian cubature.
#
# WHY THIS EXISTS.  Every d=3 rule in the bank that reaches (or nearly reaches)
# Möller's bound is icosahedral -- p=5 n=13 (attains 13) and p=9 n=45 (Möller
# 43) -- and BOTH were transcribed from the literature, not found by our
# solvers.  They cannot be found by ours: the icosahedral group has 5-fold
# axes, the hyperoctahedral group B_3 used by symq_run.jl has axes of order
# 2, 3, 4 only, so B_3 provably cannot express these rules.  Measured
# 2026-08-13 (n/Möller by structure): icosahedral 1.00 and 1.05; B_3 degrades
# 1.04 -> 1.48 as p grows; unstructured free-elim output is worse still.
#
# THE ANSATZ.  A rule is an optional origin node plus a set of SHELLS.  Each
# shell is one orbit of the full icosahedral group I_h (order 120) scaled to a
# radius r, all of whose nodes share one weight.  Orbit sizes are 12
# (icosahedron vertices), 20 (dodecahedron), 30 (icosidodecahedron), 60 (on a
# mirror plane) and 120 (generic).  All are centrally symmetric, as Möller's
# setting requires.
#
# THE SOLVE is separable: for FIXED radii the exactness conditions are LINEAR
# in the weights, so the weights come from a least-squares solve and the only
# genuinely nonlinear unknowns are the radii (one per shell).  A degree-25 rule
# needs at most ~12 shells, so the outer problem is a <=12-dimensional
# least-squares -- against ~3000 node coordinates for the equivalent free
# search.  That is the whole point of the ansatz.
module IcosahedralDQ

using LinearAlgebra, Random

export ico_group, ico_orbit, ICO_DIRS, build_rule, solve_shells, orbit_size

# ---------------------------------------------------------------------------
# the group
# ---------------------------------------------------------------------------
const φ = (1 + sqrt(5)) / 2

"12 icosahedron vertices: cyclic permutations of (0, ±1, ±φ), normalized."
function ico_vertices()
    V = Vector{Vector{Float64}}()
    for (a, b) in ((0.0, 1.0), (0.0, -1.0))
        for s in (1.0, -1.0)
            push!(V, [a, b, s * φ]); push!(V, [b, s * φ, a]); push!(V, [s * φ, a, b])
        end
    end
    unique!(v -> round.(v, digits = 10), V)
    return [v / norm(v) for v in V]
end

"""
    ico_group() -> Vector{Matrix{Float64}}

The 120 elements of I_h (60 rotations and their negatives).  Built by mapping
one (vertex, neighbour) frame onto every other: 12 vertices x 5 neighbours = 60
rotations, which is the order of the icosahedral rotation group.
"""
function ico_group()
    V = ico_vertices()
    # the 5 nearest vertices to v are its neighbours
    function neighbours(v)
        d = [(dot(v, w), w) for w in V if norm(w - v) > 1e-9]
        sort!(d, by = first, rev = true)
        return [w for (_, w) in d[1:5]]
    end
    frame(a, b) = (e1 = a; e2 = b - dot(b, e1) * e1; e2 /= norm(e2); hcat(e1, e2, cross(e1, e2)))
    M0 = frame(V[1], neighbours(V[1])[1])
    G = Matrix{Float64}[]
    for w in V, n in neighbours(w)
        R = frame(w, n) * M0'
        any(H -> maximum(abs.(H - R)) < 1e-9, G) || push!(G, R)
    end
    length(G) == 60 || error("icosahedral rotation group came out with $(length(G)) elements, expected 60")
    return vcat(G, [-R for R in G])          # adjoin the inversion -> I_h
end

const GROUP = ico_group()

"Distinct points of the I_h orbit of `g` (a 3-vector), as a matrix."
function ico_orbit(g::AbstractVector)
    pts = Vector{Vector{Float64}}()
    for R in GROUP
        q = R * g
        any(p -> norm(p - q) < 1e-9, pts) || push!(pts, q)
    end
    return reduce(vcat, [q' for q in pts])
end

orbit_size(g::AbstractVector) = size(ico_orbit(g), 1)

# canonical generator directions, one per special orbit type
const ICO_DIRS = let V = ico_vertices()
    v = V[1]
    nb = [w for (_, w) in sort([(dot(v, w), w) for w in V if norm(w - v) > 1e-9],
                               by = first, rev = true)[1:5]]
    # a face of the icosahedron is (v, n1, n2) with n1, n2 mutually adjacent
    n1 = nb[1]
    n2 = nb[findfirst(w -> dot(n1, w) > 0.4 && norm(w - n1) > 1e-9, nb)]
    f = v + n1 + n2; f /= norm(f)                 # 3-fold axis  -> 20 points
    e = v + n1;      e /= norm(e)                 # 2-fold axis  -> 30 points
    m = v + 0.35 * n1; m /= norm(m)               # mirror plane -> 60 points
    Dict(12 => v, 20 => f, 30 => e, 60 => m)
end

# ---------------------------------------------------------------------------
# exactness conditions for the Gaussian N(0, I_3)
# ---------------------------------------------------------------------------
gaussmoment(n::Int) = isodd(n) ? 0.0 : (n == 0 ? 1.0 : prod(1.0:2.0:(n - 1)))

"Exponent triples of total degree <= p that are not killed by central symmetry."
function conditions(p::Int)
    c = NTuple{3,Int}[]
    for a in 0:2:p, b in 0:2:(p - a), g in 0:2:(p - a - b)
        push!(c, (a, b, g))
    end
    return c
end

"""
    build_rule(types, radii; origin_weight = nothing)

Expand shell specs into (nodes, shell_index).  `types` are orbit sizes
(12/20/30/60) or explicit generator directions; `radii` the matching radii.
An origin shell is requested with type 1.
"""
function build_rule(types::Vector, radii::Vector{Float64})
    nodes = Matrix{Float64}(undef, 0, 3); owner = Int[]
    for (k, t) in enumerate(types)
        if t == 1
            nodes = vcat(nodes, zeros(1, 3)); push!(owner, k)
        else
            g = t isa Integer ? ICO_DIRS[t] : t
            O = ico_orbit(g) .* radii[k]
            nodes = vcat(nodes, O); append!(owner, fill(k, size(O, 1)))
        end
    end
    return nodes, owner
end

"""
    residual_vec(p, types, radii) -> (rvec, weights)

Variable projection: for fixed radii the weights are the solution of a linear
least-squares problem, and the returned residual vector is what remains of the
(relatively scaled) exactness conditions after that solve.
"""
function residual_vec(p::Int, types::Vector, radii::Vector{Float64})
    nodes, owner = build_rule(types, radii)
    C = conditions(p)
    ns = length(types)
    A = zeros(length(C), ns)
    for (i, (a, b, g)) in enumerate(C), s in 1:size(nodes, 1)
        A[i, owner[s]] += nodes[s, 1]^a * nodes[s, 2]^b * nodes[s, 3]^g
    end
    m = [gaussmoment(a) * gaussmoment(b) * gaussmoment(g) for (a, b, g) in C]
    scale = [max(abs(v), 1.0) for v in m]
    As = A ./ scale; ms = m ./ scale
    w = As \ ms
    return As * w - ms, w
end

"solve_shells(p, types, radii) -> (rms residual, weights)"
function solve_shells(p::Int, types::Vector, radii::Vector{Float64})
    r, w = residual_vec(p, types, radii)
    return norm(r) / sqrt(length(r)), w
end

"""
    fit(p, types; tries, rng, rmax) -> named tuple or nothing

Multistart Levenberg-Marquardt over the RADII only (weights eliminated by the
linear solve above -- variable projection).  Accepts only all-positive weights.
"""
function fit(p::Int, types::Vector; tries::Int = 200, rng = Random.default_rng(),
             rmax::Float64 = 6.0, tol::Float64 = 1e-13, maxiter::Int = 300,
             deadline::Float64 = Inf)
    ns = length(types)
    free = [k for k in 1:ns if types[k] != 1]          # the origin has no radius
    isempty(free) && return nothing
    best = nothing
    for _ in 1:tries
        time() > deadline && break
        r = ones(Float64, ns)
        r[free] .= sort(rand(rng, length(free)) .* rmax .+ 0.2)
        λ = 1e-3
        R0, w0 = residual_vec(p, types, r)
        f0 = norm(R0)
        for _ in 1:maxiter
            # finite-difference Jacobian of the projected residual VECTOR
            J = zeros(length(R0), length(free)); h = 1e-6
            for (j, k) in enumerate(free)
                rp = copy(r); rp[k] += h
                J[:, j] = (residual_vec(p, types, rp)[1] - R0) / h
            end
            improved = false
            for _ in 1:20                          # λ trials on this Jacobian
                step = -(J' * J + λ * I) \ (J' * R0)
                rn = copy(r); rn[free] .+= step
                if all(rn[free] .> 0.05)
                    R1, w1 = residual_vec(p, types, rn)
                    if norm(R1) < f0
                        r, R0, w0, f0 = rn, R1, w1, norm(R1)
                        λ = max(λ / 3, 1e-12); improved = true
                        break
                    end
                end
                λ *= 10
                λ > 1e10 && break
            end
            (!improved || f0 / sqrt(length(R0)) < tol) && break
        end
        frms = f0 / sqrt(length(R0))
        if frms < 1e-11 && minimum(w0) > 0
            nodes, _ = build_rule(types, r)
            cand = (residual = frms, radii = r, weights = w0, n = size(nodes, 1))
            (best === nothing || cand.residual < best.residual) && (best = cand)
            best.residual < tol && break
        end
    end
    return best
end

end # module
