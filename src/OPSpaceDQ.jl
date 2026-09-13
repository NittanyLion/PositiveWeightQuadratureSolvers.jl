# Orthogonal-polynomial-space construction of d=2 cubature (HP76 method,
# implemented from first principles for the Gaussian weight).
#
# THE LESSON THIS ENCODES.  Our solvers search node coordinates -- ~10^2..10^5
# unknowns.  HP76 searches a TWO-parameter family: nodes are common zeros of
#
#   φ1 = P_{5,1} + μ1 P_{3,3} = xy ψ1(x², y²)
#   φ2 = P_{1,5} + μ2 P_{3,3} = xy ψ2(x², y²)
#   φ3 = P_{6,0} + λ1 P_{4,2} + λ2 P_{2,4} + λ3 P_{0,6} = ψ3(x², y²)
#
# where the P are the degree-6 orthogonal polynomials of the weight and the
# λ_i(μ1, μ2) are forced by requiring ψ3 to vanish at the 4 common zeros of
# ψ1, ψ2.  Everything below is DERIVED from the weight's moments -- nothing
# is transcribed from the paper -- so agreement with the published Table 6
# is a genuine two-sided check.
#
# Weight convention inside this module: e^{-(x²+y²)} (the paper's), moments
# ν_{2a,2b} = Γ(a+1/2)Γ(b+1/2).  Convert results to N(0, I) at the end via
# x -> x√2, w -> w/π.
module OPSpaceDQ

using LinearAlgebra

export family, rule25, to_gaussian

# ν_{p,q} for e^{-(x²+y²)}: 1-D factor Γ((p+1)/2) for even p, 0 for odd
gm(n::Int) = isodd(n) ? 0.0 : gamma_half(n)
gamma_half(n) = n == 0 ? sqrt(pi) : ((n - 1) / 2) * gamma_half(n - 2)
ν(p::Int, q::Int) = gm(p) * gm(q)

# ---------------------------------------------------------------------------
# degree-6 orthogonal polynomials by Gram-Schmidt against all lower monomials.
# A polynomial is stored as Dict{(a,b) => coeff}.
# ---------------------------------------------------------------------------
inner(u::Dict, v::Dict) = sum(cu * cv * ν(au + av, bu + bv)
                              for ((au, bu), cu) in u, ((av, bv), cv) in v)

function op(a::Int, b::Int)
    m = a + b
    P = Dict((a, b) => 1.0)
    # subtract projections onto ALL monomials of lower total degree (their
    # span is degree-graded, so one pass of least-squares suffices)
    lower = [(i, j) for t in 0:(m - 1) for i in 0:t for j in (t - i,)]
    # solve the normal equations for the projection coefficients
    G = [ν(l1[1] + l2[1], l1[2] + l2[2]) for l1 in lower, l2 in lower]
    r = [ν(a + l[1], b + l[2]) for l in lower]
    c = G \ r
    for (k, l) in enumerate(lower)
        abs(c[k]) > 1e-13 && (P[l] = get(P, l, 0.0) - c[k])
    end
    return P
end

# ψ(X, Y) for an even/odd-structured P: divide out xy (for P51-type) or keep
# even (for P60-type), mapping x² -> X, y² -> Y
function toψ(P::Dict; sortxy::Bool)
    ψ = Dict{Tuple{Int,Int},Float64}()
    for ((a, b), c) in P
        abs(c) < 1e-13 && continue
        if sortxy
            (isodd(a) && isodd(b)) || error("expected odd-odd monomial, got ($a,$b)")
            ψ[((a - 1) ÷ 2, (b - 1) ÷ 2)] = c
        else
            (iseven(a) && iseven(b)) || error("expected even-even monomial, got ($a,$b)")
            ψ[(a ÷ 2, b ÷ 2)] = c
        end
    end
    return ψ
end

ψeval(ψ::Dict, X, Y) = sum(c * X^a * Y^b for ((a, b), c) in ψ)

lincomb(u::Dict, α, v::Dict) = begin
    w = copy(u)
    for (k, c) in v
        w[k] = get(w, k, 0.0) + α * c
    end
    w
end

# the fixed basis, built once
const P51 = op(5, 1); const P15 = op(1, 5); const P33 = op(3, 3)
const P60 = op(6, 0); const P42 = op(4, 2); const P24 = op(2, 4); const P06 = op(0, 6)
const Ψ51 = toψ(P51, sortxy = true);  const Ψ15 = toψ(P15, sortxy = true)
const Ψ33 = toψ(P33, sortxy = true)
const Ψ60 = toψ(P60, sortxy = false); const Ψ42 = toψ(P42, sortxy = false)
const Ψ24 = toψ(P24, sortxy = false); const Ψ06 = toψ(P06, sortxy = false)

# ---------------------------------------------------------------------------
# the 4 common zeros of ψ1, ψ2 in the (X, Y) = (x², y²) plane.
# ψ1 is LINEAR in Y and ψ2 is LINEAR in X (structure of P51/P15), so
# elimination gives a quartic in X solved by companion matrix.
# ---------------------------------------------------------------------------
function common_zeros(μ1, μ2)
    ψ1 = lincomb(Ψ51, μ1, Ψ33)          # coeffs on X²,X,Y,XY,1
    ψ2 = lincomb(Ψ15, μ2, Ψ33)          # coeffs on Y²,Y,X,XY,1
    g(ψ, k) = get(ψ, k, 0.0)
    # ψ1 = A(X) + B(X)·Y,  A = c20 X² + c10 X + c00,  B = c01 + c11 X
    A(X) = g(ψ1, (2, 0)) * X^2 + g(ψ1, (1, 0)) * X + g(ψ1, (0, 0))
    B(X) = g(ψ1, (0, 1)) + g(ψ1, (1, 1)) * X
    # substitute Y = -A/B into ψ2 = d02 Y² + d01 Y + d11 XY + d10 X + d00,
    # multiply by B²  ->  quartic in X
    d02 = g(ψ2, (0, 2)); d01 = g(ψ2, (0, 1)); d11 = g(ψ2, (1, 1))
    d10 = g(ψ2, (1, 0)); d00 = g(ψ2, (0, 0))
    # polynomial arithmetic in X (dense, low degree)
    padd(u, v) = [get(u, i, 0.0) + get(v, i, 0.0) for i in 1:max(length(u), length(v))]
    pmul(u, v) = begin
        w = zeros(length(u) + length(v) - 1)
        for (i, a) in enumerate(u), (j, b) in enumerate(v)
            w[i + j - 1] += a * b
        end
        w
    end
    Ap = [g(ψ1, (0, 0)), g(ψ1, (1, 0)), g(ψ1, (2, 0))]
    Bp = [g(ψ1, (0, 1)), g(ψ1, (1, 1))]
    q = padd(pmul([d02], pmul(Ap, Ap)),
             padd(pmul([-d01], pmul(Ap, Bp)),
                  padd(pmul([-d11, 0.0][[2, 1]], pmul(Ap, Bp)),  # -d11·X·A·B
                       pmul(padd([d00], [0.0, d10]), pmul(Bp, Bp)))))
    while length(q) > 1 && abs(q[end]) < 1e-10 * maximum(abs.(q))
        pop!(q)
    end
    deg = length(q) - 1
    (deg < 1 || !all(isfinite, q)) && return nothing
    C = zeros(deg, deg)                              # companion matrix
    C[:, end] = -q[1:deg] ./ q[end]
    for i in 2:deg
        C[i, i - 1] = 1.0
    end
    Xr = eigvals(C)
    out = Tuple{ComplexF64,ComplexF64}[]
    for X in Xr
        Y = -(g(ψ1, (2, 0)) * X^2 + g(ψ1, (1, 0)) * X + g(ψ1, (0, 0))) /
             (g(ψ1, (0, 1)) + g(ψ1, (1, 1)) * X)
        push!(out, (X, Y))
    end
    return out
end

# λ1, λ2, λ3 from ψ3(X_i, Y_i) = 0 at the common zeros (4 eqs, 3 unknowns,
# consistent by the theory; solved least-squares, consistency checked)
function lambdas(zs)
    A = reduce(vcat, [ComplexF64[ψeval(Ψ42, X, Y) ψeval(Ψ24, X, Y) ψeval(Ψ06, X, Y)]
                      for (X, Y) in zs])
    r = ComplexF64[-ψeval(Ψ60, X, Y) for (X, Y) in zs]
    λ = A \ r
    resid = norm(A * λ - r)
    return real.(λ), resid
end

# real roots of a real cubic (coefficients ascending), via companion matrix
function realroots(coef)
    all(isfinite, coef) || return Float64[]
    c = copy(coef)
    while length(c) > 1 && abs(c[end]) < 1e-12 * maximum(abs.(c))
        pop!(c)
    end
    deg = length(c) - 1
    deg < 1 && return Float64[]
    C = zeros(deg, deg)
    C[:, end] = -c[1:deg] ./ c[end]
    for i in 2:deg
        C[i, i - 1] = 1.0
    end
    return [real(z) for z in eigvals(C) if abs(imag(z)) < 1e-8]
end

"""
    family(μ1, μ2) -> named tuple or nothing

The 28-point (generically) degree-11 formula of the family at (μ1, μ2), in
the e^{-(x²+y²)} convention: full-symmetry orbits of 4 generic points, plus
x-axial and y-axial pairs from ψ3's sections.  Returns nodes, weights, the
consistency residual of the λ solve, and the max exactness violation.
"""
function family(μ1, μ2)
    zs = common_zeros(μ1, μ2)
    zs === nothing && return nothing
    λ, λres = lambdas(zs)
    ψ3 = lincomb(lincomb(lincomb(Ψ60, λ[1], Ψ42), λ[2], Ψ24), λ[3], Ψ06)
    g(k) = get(ψ3, k, 0.0)
    # axial sections: ψ3(X,0) and ψ3(0,Y) are cubics
    Xax = realroots([g((0, 0)), g((1, 0)), g((2, 0)), g((3, 0))])
    Yax = realroots([g((0, 0)), g((0, 1)), g((0, 2)), g((0, 3))])
    # assemble real nodes
    orbs = Vector{Matrix{Float64}}()
    for (X, Y) in zs
        (abs(imag(X)) < 1e-9 && abs(imag(Y)) < 1e-9) || return nothing
        (real(X) > 0 && real(Y) > 0) || return nothing
        x, y = sqrt(real(X)), sqrt(real(Y))
        push!(orbs, [x y; -x y; x -y; -x -y])
    end
    for X in Xax
        X > 1e-12 || (X > -1e-8 ? continue : return nothing)
        x = sqrt(X)
        push!(orbs, [x 0.0; -x 0.0])
    end
    for Y in Yax
        Y > 1e-12 || (Y > -1e-8 ? continue : return nothing)
        y = sqrt(Y)
        push!(orbs, [0.0 y; 0.0 -y])
    end
    origin = (abs(g((0, 0))) < 1e-9)     # constant term gone -> origin node
    n = sum(size(o, 1) for o in orbs) + (origin ? 1 : 0)
    nodes = vcat(orbs...)
    origin && (nodes = vcat(nodes, [0.0 0.0]))
    owner = vcat([fill(k, size(o, 1)) for (k, o) in enumerate(orbs)]...,
                 origin ? [length(orbs) + 1] : Int[])
    # weights from even-even exactness ≤ 11
    conds = [(a, b) for a in 0:2:11 for b in 0:2:(11 - a)]
    nw = maximum(owner)
    M = zeros(length(conds), nw)
    for (i, (a, b)) in enumerate(conds), s in 1:size(nodes, 1)
        M[i, owner[s]] += nodes[s, 1]^a * nodes[s, 2]^b
    end
    rhs = [ν(a, b) for (a, b) in conds]
    scale = max.(abs.(rhs), 1.0)
    w = (M ./ scale) \ (rhs ./ scale)
    exres = maximum(abs.((M * w - rhs) ./ scale))
    W = [w[owner[s]] for s in 1:size(nodes, 1)]
    return (nodes = nodes, weights = W, orbitw = w, n = n,
            λ = λ, λres = λres, exres = exres, origin = origin)
end

"convert an e^{-(x²+y²)} rule to the project's N(0, I) convention"
to_gaussian(nodes, w) = (nodes .* sqrt(2), w ./ pi)

end # module
