#=
WeightTransforms.jl -- coordinate-wise quantile maps between the Gaussian
weight and the other product weights the solvers search natively
(2026-08-29, PROPOSAL_nonnormal_native_search.md §3).

T = F⁻¹ ∘ Φ, applied per coordinate, sends a Gaussian rule to a rule with the
target's node count, positive weights summing to one and nodes in the
target's high-mass region — an "almost solved" configuration that a native
LM solve turns into an exactly-solved one.  For a symmetric target T is odd
about the center, so a B_d (or S_d × Z_2) orbit maps to an orbit of the same
type: the whole orbit decomposition of a Hermite sidecar carries over, and
the native search starts from a structurally correct ansatz at the right n.

Only stdlib: erf/erfc come from the MPFR library that ships with Julia
(BigFloat), called directly — SpecialFunctions.jl is not a dependency of this
project and must not become one (the cluster and OSPool images carry no
packages).  Accuracy is full double precision; the maps are evaluated once
per warm start, so BigFloat cost is irrelevant.
=#
module WeightTransforms

using LinearAlgebra

export erf64, erfc64, normcdf, normccdf, norminv,
       gauss_to_unit, unit_to_gauss, gauss_to_exp, gauss_to_gamma,
       gauss_hermite_nodes, gauss_legendre_nodes, gauss_laguerre_nodes,
       node_map, monotone_interp

const RM = Base.MPFR.ROUNDING_MODE

# --- 1-D Gauss rules (Golub–Welsch), the anchors of the node-matching maps --
# probabilists' Hermite (weight N(0,1)): β_k = √k
gauss_hermite_nodes(m::Int) = m == 1 ? [0.0] :
    sort(eigvals(SymTridiagonal(zeros(m), [sqrt(k) for k in 1:m-1])))
# Legendre on [-1,1]: β_k = k/√(4k²-1)
gauss_legendre_nodes(m::Int) = m == 1 ? [0.0] :
    sort(eigvals(SymTridiagonal(zeros(m), [k / sqrt(4k^2 - 1.0) for k in 1:m-1])))
# generalized Laguerre, weight x^a e^{-x} on [0,∞): α_k = 2k+a+1, β_k = √(k(k+a))
gauss_laguerre_nodes(m::Int, a::Real = 0.0) = m == 1 ? [1.0 + a] :
    sort(eigvals(SymTridiagonal([2k + a + 1.0 for k in 0:m-1],
                                [sqrt(k * (k + a)) for k in 1:m-1])))

"""
    monotone_interp(xs, ys) -> f

Monotone piecewise-cubic Hermite interpolant (Fritsch–Carlson) through the
strictly increasing knots (xs, ys); beyond the last knot it continues with
the end slope.  Used to build node-matching quantile maps.
"""
function monotone_interp(xs::Vector{Float64}, ys::Vector{Float64})
    n = length(xs)
    n == length(ys) && n ≥ 2 || throw(ArgumentError("need ≥ 2 knots"))
    h = diff(xs); Δ = diff(ys) ./ h
    m = zeros(n)
    m[1] = Δ[1]; m[n] = Δ[end]
    for i in 2:n-1
        m[i] = (Δ[i-1] * Δ[i] ≤ 0) ? 0.0 :
               3 * (h[i-1] + h[i]) / ((2h[i] + h[i-1]) / Δ[i-1] + (h[i] + 2h[i-1]) / Δ[i])
    end
    function f(x::Float64)
        x ≤ xs[1] && return ys[1] + m[1] * (x - xs[1])
        x ≥ xs[n] && return ys[n] + m[n] * (x - xs[n])
        i = searchsortedlast(xs, x)
        t = (x - xs[i]) / h[i]
        h00 = (1 + 2t) * (1 - t)^2; h10 = t * (1 - t)^2
        h01 = t^2 * (3 - 2t);       h11 = t^2 * (t - 1)
        return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i+1] + h11 * h[i] * m[i+1]
    end
    return f
end

"""
    node_map(p, target; alpha) -> (y -> t)

Coordinate map that sends the positive 1-D Gauss–Hermite nodes of the degree-p
rule (m = ⌈(p+1)/2⌉ points) to the corresponding nodes of the target's Gauss
rule — `:legendre` (on [-1,1]) or `:laguerre` (generalized, parameter
`alpha`) — and interpolates monotonically in between.  For the symmetric
:legendre target the map is odd (0 ↦ 0), so orbits keep their types; beyond
the outermost Hermite node it approaches the boundary ±1 exponentially with
matching slope, never crossing it.  This puts the outer orbits of a designed
Hermite rule at the outer Gauss–Legendre node (≈0.97 at p=19) instead of on
the face of the cube (0.99996 under the plain cdf map 2Φ−1), which is where
the cdf map leaves the Legendre LM stuck against the box.
"""
function node_map(p::Int, target::Symbol; alpha::Real = 0.0)
    m = cld(p + 1, 2)
    gh = gauss_hermite_nodes(m)
    if target === :legendre
        gl = gauss_legendre_nodes(m)
        pos = gh .> 1e-12
        xs = [0.0; gh[pos]]; ys = [0.0; gl[pos]]
        f = monotone_interp(xs, ys)
        xm, ym = xs[end], ys[end]
        # tail: ym + (1-ym)(1 - e^{-s(y-xm)}), slope-matched at xm
        slope = (f(xm) - f(xm - 1e-6)) / 1e-6
        s = max(slope, 1e-6) / (1 - ym)
        g(y::Real) = begin
            a = abs(Float64(y))
            t = a ≤ xm ? f(a) : ym + (1 - ym) * (1 - exp(-s * (a - xm)))
            return copysign(min(t, 1 - 1e-12), Float64(y))
        end
        return g
    elseif target === :laguerre
        # full (signed) Hermite node set ↦ Laguerre node set, both ascending;
        # the map is not odd: −y lands near 0, +y in the tail
        gla = gauss_laguerre_nodes(m, alpha)
        f = monotone_interp(gh, gla)
        xm, ym = gh[end], gla[end]
        slope = (f(xm) - f(xm - 1e-6)) / 1e-6
        x1, y1 = gh[1], gla[1]
        slope1 = (f(x1 + 1e-6) - f(x1)) / 1e-6
        # left tail decays to 0 exponentially, right tail continues linearly
        h(y::Real) = begin
            z = Float64(y)
            z ≥ x1 ? (z ≤ xm ? f(z) : ym + slope * (z - xm)) :
                     y1 * exp(-(slope1 / y1) * (x1 - z))
        end
        return h
    else
        throw(ArgumentError("node_map target must be :legendre or :laguerre"))
    end
end

function erf_big(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_erf, :libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, RM[])
    return z
end
function erfc_big(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_erfc, :libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, RM[])
    return z
end
function gamma_big(a::BigFloat)
    z = BigFloat()
    ccall((:mpfr_gamma, :libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, RM[])
    return z
end
# upper incomplete gamma Γ(a, x) (MPFR ≥ 4.0)
function gamma_inc_big(a::BigFloat, x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_gamma_inc, :libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          z, a, x, RM[])
    return z
end

erf64(x::Real)  = Float64(erf_big(big(Float64(x))))
erfc64(x::Real) = Float64(erfc_big(big(Float64(x))))

const SQRT2 = sqrt(big(2))
"""Φ(x), the standard normal cdf (via erfc, so both tails are accurate)."""
normcdf(x::Real)  = Float64(erfc_big(-big(Float64(x)) / SQRT2) / 2)
"""1 − Φ(x) = Φ(−x)."""
normccdf(x::Real) = Float64(erfc_big(big(Float64(x)) / SQRT2) / 2)

"""Φ⁻¹(u) for u ∈ (0,1): Newton on the BigFloat cdf from a double-precision
start (Acklam's rational approximation)."""
function norminv(u::Real)
    0 < u < 1 || throw(DomainError(u, "norminv needs u in (0,1)"))
    u == 0.5 && return 0.0
    # Acklam (2003), |rel. err| < 1.15e-9 — then two Newton steps in BigFloat
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    dd = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425; phigh = 1 - plow
    x = if u < plow
        q = sqrt(-2log(u))
        (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
            ((((dd[1]*q+dd[2])*q+dd[3])*q+dd[4])*q+1)
    elseif u ≤ phigh
        q = u - 0.5; r = q*q
        (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
            (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2log(1 - u))
        -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
            ((((dd[1]*q+dd[2])*q+dd[3])*q+dd[4])*q+1)
    end
    xb = big(x); ub = big(Float64(u))
    for _ in 1:3
        F = erfc_big(-xb / SQRT2) / 2
        f = exp(-xb^2 / 2) / sqrt(2 * big(pi))
        xb -= (F - ub) / f
    end
    return Float64(xb)
end

# --- the maps -------------------------------------------------------------

"""Gaussian coordinate y → 2Φ(y) − 1 = erf(y/√2) ∈ (−1, 1): the centered
coordinate of the uniform weight on [−1,1] (the :legendre solvers' frame;
the banked [0,1]^d rule is x/2 + 1/2).  Odd, so B_d orbits map to B_d orbits."""
gauss_to_unit(y::Real) = erf64(Float64(y) / sqrt(2.0))
"""Inverse of gauss_to_unit (t ∈ (−1,1))."""
unit_to_gauss(t::Real) = norminv((1 + Float64(t)) / 2)

"""Gaussian coordinate y → Exp(1) quantile of Φ(y): −log(1 − Φ(y)) = −log Φ(−y).
NOT odd: y and −y land at different points, so a B_d orbit splits into
several S_d orbits (see LaguerreDQ.from_hermite)."""
gauss_to_exp(y::Real) = -Float64(log(erfc_big(big(Float64(y)) / SQRT2) / 2))

"""Gaussian coordinate y → Gamma(shape a, scale 1) quantile of Φ(y), by Newton
on the BigFloat regularized incomplete gamma.  a = 1 is gauss_to_exp."""
function gauss_to_gamma(y::Real, a::Real)
    a == 1 && return gauss_to_exp(y)
    ab = big(Float64(a))
    u = erfc_big(-big(Float64(y)) / SQRT2) / 2        # Φ(y)
    G = gamma_big(ab)
    # start: Wilson–Hilferty
    ζ = Float64(y)
    x0 = a * max(1 - 2 / (9a) + ζ * sqrt(2 / (9a)), 1e-3)^3
    xb = big(max(x0, 1e-8))
    for _ in 1:60
        P = 1 - gamma_inc_big(ab, xb) / G           # regularized lower
        f = xb^(ab - 1) * exp(-xb) / G
        step = (P - u) / f
        xn = xb - step
        xn ≤ 0 && (xn = xb / 2)
        done = abs(xn - xb) ≤ 1e-30 * max(abs(xb), 1)
        xb = xn
        done && break
    end
    return Float64(xb)
end

end # module
