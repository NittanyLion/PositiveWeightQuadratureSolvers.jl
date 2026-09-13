#!/usr/bin/env julia
#
# compare_rules.jl - settle "same rule or a distinct rule at the same count?"
# for cells where our banked rule ties a published one.
#
# Companion to build_literature_best.jl; fills the tie_rule_identity column of
# literature_best.tsv.  Run with:  julia --threads=2 compare_rules.jl
#
# ---------------------------------------------------------------------------
# NORMALIZATION - this is where such comparisons usually go wrong, so it is
# stated explicitly and asserted at run time for every pair.
#
#   Uniform (Le).  Both sides are put on [0,1]^d with weights summing to 1
#   (the bank frame).  litcheck/festa_sommariva/smr*.csv is already in that
#   frame (their README: converted from [-1,1]^2 with weights summing to 4).
#   No conversion is applied to either side; the assertions below check it.
#
#   Gaussian (GH).  The bank is N(0, I_d) with weights summing to 1.  Stroud's
#   E_n^{r^2} rules are for w(x) = exp(-x'x) with total mass V = pi^(n/2).
#   The conversion applied to STROUD's rule is
#         x -> x * sqrt(2),      w -> w / pi^(n/2),
#   after which both sides are N(0, I_d) with unit total weight.
#
# ---------------------------------------------------------------------------
# METHOD.  Two independent comparisons, reported separately:
#
#   1. Isometry-invariant fingerprints, which need no group at all:
#        - the sorted multiset of weights;
#        - the sorted multiset of all pairwise node distances.
#      Both are invariant under EVERY isometry of R^d.  If either differs by
#      more than the tolerance the rules are distinct, and no search over a
#      symmetry group can rescue them.  This is the strong direction.
#
#   2. An explicit alignment over the symmetry group of the domain:
#        - Le, d = 2: the 8 symmetries of the square, about (1/2, 1/2);
#        - Le, d = 3: the 48 symmetries of the cube, about (1/2,...,1/2);
#        - GH, d = 5: the 3840 signed permutations B_5, about the origin.
#      For each group element the nodes are matched nearest-neighbor; the
#      matching must be a bijection, and the score is the largest node
#      displacement, together with the largest weight difference over matched
#      pairs.  The reported mismatch is the minimum of that score over the group.
#
# VERDICT.  same-rule if the best alignment mismatch <= SAME_TOL; distinct if
# the fingerprints disagree by more than SAME_TOL; otherwise unestablished -
# the code refuses to force a verdict.

using Printf, LinearAlgebra   # standard library only, so the check is reproducible anywhere

"All d! permutations of 1:d (avoids a Combinatorics dependency)."
function allperms(d::Int)
    d == 1 && return [[1]]
    out = Vector{Vector{Int}}()
    for p ∈ allperms(d - 1), i ∈ 1:d
        q = copy(p); insert!(q, i, d); push!(out, q)
    end
    out
end

const ROOT = get(ENV, "PWQS_RESEARCH_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR", joinpath(ROOT, "project", "julia", "rules"))
const FS = joinpath(ROOT, "project", "litcheck", "festa_sommariva")
# Van Zandt's listings, already converted to N(0,I) with weights summing to 1 by
# julia/symq/convert_vz.jl, so no conversion is applied on this side either.
const VZ = joinpath(ROOT, "project", "julia", "symq", "vz_rules")
# Burkardt's transcriptions of Stroud's E_n^{r^2} families, in the exp(-x'x) frame.
const SM = joinpath(ROOT, "project", "litcheck", "stroud_m")

const SAME_TOL = 1e-8     # two rules this close are the same rule
const DIST_TOL = 1e-6     # fingerprints further apart than this are distinct

# ---------------------------------------------------------------- io
"Read a rule file: one node per line, `x1,...,xd,w`. Returns (X, w) with X d x n."
function readrule(path)
    rows = Vector{Vector{Float64}}()
    for line ∈ eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(rows, parse.(Float64, split(s, ',')))
    end
    m = reduce(hcat, rows)          # (d+1) x n
    return m[1:end-1, :], m[end, :]
end

# ---------------------------------------------------------------- Stroud E_n^{r^2}:7-1
"""
Stroud 1971 `E_n^{r^2}:7-1`, 2^n + 2n^2 + 1 nodes, degree 7, for w = exp(-x'x).
At n = 5 the printed radius formula is 0/0; the limit is r^2 = 9/4 (see
litcheck/LITERATURE_SWEEP_2026-08-13.md and litcheck/verify_stroud71_n5.py,
whose construction this reimplements in Julia).
Returned in the N(0,I) convention: x -> x*sqrt(2), w -> w/pi^(n/2).
"""
function stroud_7_1(n::Int)
    r2 = n == 5 ? 9 / 4 : (3 * (8 - n) - (n - 2) * sqrt(3 * (8 - n))) / (2 * (5 - n))
    s2 = (3n - 2 * sqrt(3 * (8 - n))) / (2 * (3n - 8))
    t2 = (6 + sqrt(3 * (8 - n))) / 2
    r, s, t = sqrt(r2), sqrt(s2), sqrt(t2)
    V = sqrt(pi^n)
    b = (8 - n) * V / 8 / r^6
    c = V / 2^(n + 3) / s^6
    d = V / 16 / t^6
    a = V - 2n * b - 2^n * c - 2n * (n - 1) * d

    pts = Vector{Vector{Float64}}()
    wts = Float64[]
    push!(pts, zeros(n)); push!(wts, a)
    for i ∈ 1:n, sg ∈ (-1, 1)
        p = zeros(n); p[i] = sg * r; push!(pts, p); push!(wts, b)
    end
    for signs ∈ Iterators.product(ntuple(_ -> (-1, 1), n)...)
        push!(pts, collect(Float64, signs) .* s); push!(wts, c)
    end
    for i ∈ 1:n, j ∈ (i+1):n, si ∈ (-1, 1), sj ∈ (-1, 1)
        p = zeros(n); p[i] = si * t; p[j] = sj * t
        push!(pts, p); push!(wts, d)
    end
    X = reduce(hcat, pts) .* sqrt(2)          # -> N(0, I)
    return X, wts ./ (pi^(n / 2))
end

# ---------------------------------------------------------------- Burkardt's Stroud files
"""
Read one of Burkardt's `en_r2_*` CSVs (`litcheck/stroud_m/`). Those are in
Stroud's `w = exp(-x'x)` frame with total mass `pi^(n/2)`; this returns the same
rule in N(0,I) with unit mass, via `x -> x*sqrt(2)`, `w -> w/pi^(n/2)`.
"""
function burkardt_en_r2(path, n)
    X, w = readrule(path)
    size(X, 1) == n || error("expected $n coordinate columns in $path")
    X .* sqrt(2), w ./ (pi^(n / 2))
end

"""
Stroud `E_n^{r^2}` rule 3-1, the axial `2N`-point degree-3 rule (Burkardt's
`en_r2_03_1.m`, "the Stroud rule 3.1", `O = 2*N`). Constructed rather than read,
because the degree-3 sources are not in the repo's `stroud_m` copy; in N(0,I) the
nodes are `+-sqrt(d) e_i` with equal weights `1/(2d)`. Verified by moment check
below rather than trusted.
"""
function stroud_3_1(d::Int)
    X = zeros(d, 2d)
    for i ∈ 1:d
        X[i, 2i-1] = sqrt(d)
        X[i, 2i] = -sqrt(d)
    end
    X, fill(1 / (2d), 2d)
end

"""
Generate a Gaussian rule and VERIFY it is exact to `deg` before letting it be used
as a comparison target. A silently wrong construction would masquerade as a
"distinct" rule, which is the one failure mode this whole file exists to avoid.
"""
function checked(gen, deg)
    X, w = gen()
    e = gauss_maxrelerr(X, w, deg)
    e < 1e-12 || error("generated rule is not exact to degree $deg (worst relative error $e)")
    (X, w)
end

"Worst relative error over every monomial of total degree <= p against N(0,I) moments."
function gauss_maxrelerr(X, w, p)
    d = size(X, 1)
    worst = 0.0
    for a ∈ Iterators.product(ntuple(_ -> 0:p, d)...)
        sum(a) ≤ p || continue
        ex = 1.0
        for ai ∈ a
            if isodd(ai)
                ex = 0.0
                break
            end
            for k ∈ 1:2:(ai-1)
                ex *= k
            end
        end
        got = sum(w[j] * prod(X[i, j]^a[i] for i ∈ 1:d) for j ∈ eachindex(w))
        worst = max(worst, abs(got - ex) / max(abs(ex), 1.0))
    end
    worst
end

# ---------------------------------------------------------------- symmetry groups
"Signed permutations of R^d (the group B_d), as (perm, signs) pairs."
function bd_group(d::Int)
    g = Tuple{Vector{Int},Vector{Int}}[]
    for p ∈ allperms(d), s ∈ Iterators.product(ntuple(_ -> (-1, 1), d)...)
        push!(g, (collect(p), collect(s)))
    end
    g
end

"Apply a signed permutation about center c."
function applyg(X, (perm, signs), c)
    Y = similar(X)
    for j ∈ axes(X, 2), i ∈ eachindex(perm)
        Y[i, j] = c[i] + signs[i] * (X[perm[i], j] - c[perm[i]])
    end
    Y
end

# ---------------------------------------------------------------- comparisons
"Sorted multiset of all pairwise node distances - invariant under every isometry."
function distance_fingerprint(X)
    n = size(X, 2)
    v = Float64[]
    sizehint!(v, n * (n - 1) ÷ 2)
    for i ∈ 1:n-1, j ∈ i+1:n
        push!(v, norm(@view(X[:, i]) .- @view(X[:, j])))
    end
    sort!(v)
end

"Largest node displacement under a bijective nearest-neighbor matching, or Inf."
function alignment_score(A, wa, B, wb)
    n = size(A, 2)
    used = falses(n)
    worst_x = 0.0
    worst_w = 0.0
    for i ∈ 1:n
        best, bj = Inf, 0
        for j ∈ 1:n
            used[j] && continue
            dij = norm(@view(A[:, i]) .- @view(B[:, j]))
            if dij < best
                best, bj = dij, j
            end
        end
        bj == 0 && return (Inf, Inf)
        used[bj] = true
        worst_x = max(worst_x, best)
        worst_w = max(worst_w, abs(wa[i] - wb[bj]))
    end
    (worst_x, worst_w)
end

"""
Candidate orthogonal maps carrying some node of B onto a fixed outermost node of
A, for d = 2.

Why this exists: the GAUSSIAN weight exp(-x'x) is invariant under the whole
orthogonal group O(d), not merely under the signed permutations B_d, so two
Gaussian rules that differ by a rotation are THE SAME RULE. (The uniform weight
on the cube is not rotation invariant - only the cube's own symmetry group
preserves the domain - which is why this is applied to the Gaussian cases only.)
Without it, a rotated copy of a published rule reads as "distinct", which is a
bug in the comparison rather than a fact about the rules.

The candidate set is finite and exact: fix the outermost node of A, and for every
node of B at the same radius and weight take the rotation carrying it there, with
and without a reflection.
"""
function o2_candidates(A, wa, B, wb; tol = 1e-9)
    Ms = Matrix{Float64}[]
    rA = [norm(@view A[:, i]) for i ∈ axes(A, 2)]
    rB = [norm(@view B[:, j]) for j ∈ axes(B, 2)]
    i0 = argmax(rA)
    θA = atan(A[2, i0], A[1, i0])
    for j ∈ axes(B, 2)
        (abs(rB[j] - rA[i0]) ≤ tol * max(1, rA[i0]) && abs(wb[j] - wa[i0]) ≤ tol) || continue
        θB = atan(B[2, j], B[1, j])
        for s ∈ (1, -1)
            φ = θA - s * θB
            push!(Ms, [cos(φ) -sin(φ); sin(φ) cos(φ)] * [1.0 0.0; 0.0 s])
        end
    end
    Ms
end

"Best alignment of B onto A over a list of orthogonal matrices about the origin."
function best_alignment_M(A, wa, B, wb, Ms)
    bx, bw = Inf, Inf
    for M ∈ Ms
        x, w = alignment_score(A, wa, M * B, wb)
        if x < bx
            bx, bw = x, w
        end
    end
    (bx, bw)
end

"Best alignment of B onto A over the group G (acting about center c)."
function best_alignment(A, wa, B, wb, G, c)
    bx, bw = Inf, Inf
    for g ∈ G
        x, w = alignment_score(A, wa, applyg(B, g, c), wb)
        if x < bx
            bx, bw = x, w
        end
    end
    (bx, bw)
end

function verdict(fp_gap, align_x)
    align_x ≤ SAME_TOL && return "same-rule"
    fp_gap > DIST_TOL && return "distinct"
    return "unestablished"
end

# ---------------------------------------------------------------- the cases
struct Case
    label::String
    ours::String          # path
    theirs::String        # path, or "" when generated
    gen::Union{Nothing,Function}
    source::String
    d::Int
    center::Vector{Float64}
    cube::Bool            # true = uniform weight on [0,1]^d, false = Gaussian on R^d
end

cases = Case[
    Case("Le d2 p9  n17",  joinpath(RULES, "legendre_d2_p9_n17.csv"),
         joinpath(FS, "smr09.csv"), nothing, "festa_sommariva2012 (SMR09)", 2, [0.5, 0.5], true),
    Case("Le d2 p11 n24",  joinpath(RULES, "legendre_d2_p11_n24.csv"),
         joinpath(FS, "smr11.csv"), nothing, "festa_sommariva2012 (SMR11)", 2, [0.5, 0.5], true),
    Case("Le d2 p17 n54",  joinpath(RULES, "legendre_d2_p17_n54.csv"),
         joinpath(FS, "smr17.csv"), nothing, "festa_sommariva2012 (SMR17)", 2, [0.5, 0.5], true),
    Case("Le d2 p23 n96",  joinpath(RULES, "legendre_d2_p23_n96.csv"),
         joinpath(FS, "smr23.csv"), nothing, "festa_sommariva2012 (SMR23)", 2, [0.5, 0.5], true),
    Case("GH d5 p7  n83",  joinpath(RULES, "hermite_d5_p7_n83.csv"),
         "", () -> checked(() -> stroud_7_1(5), 7), "stroud1971 (E_n^{r^2}:7-1)", 5, zeros(5), false),
    # Second batch, 2026-09-11: Van Zandt's quad-precision listings reached the
    # repo (julia/symq/vz_rules/, converted to N(0,I) with unit mass by
    # convert_vz.jl), which unblocks the Encyclopaedia-locked ties wherever he
    # lists a rule at our node count.  These three are his transcriptions of the
    # classical planar rules, not his own new rules - his listings include known
    # rules for comparison - so the credit at these cells belongs to Stroud.
    Case("GH d2 p3  n4",   joinpath(RULES, "hermite_d2_p3_n4.csv"),
         joinpath(VZ, "vz_d2_p3_n4.csv"), nothing,
         "van_zandt2019 listing of the classical 4-node rule (Stroud E_n^{r^2} 3-1, order 2N, axial)", 2, zeros(2), false),
    Case("GH d2 p5  n7",   joinpath(RULES, "hermite_d2_p5_n7.csv"),
         joinpath(VZ, "vz_d2_p5_n7.csv"), nothing,
         "van_zandt2019 listing of the classical 7-node rule (Stroud E_n^{r^2} 5-4, order 2^(N+1)-1)", 2, zeros(2), false),
    Case("GH d2 p7  n12",  joinpath(RULES, "hermite_d2_p7_n12.csv"),
         joinpath(VZ, "vz_d2_p7_n12.csv"), nothing,
         "van_zandt2019 listing of the classical 12-node degree-7 rule (no Stroud number: 7-1/7-2/7-3 have order 13/24/17 at N=2)", 2, zeros(2), false),
    # Third batch, 2026-09-11: a sweep of the still-unestablished GH ties against
    # Burkardt's stroud_m copies, by ORDER FORMULA. Only same-degree rules are
    # compared - en_r2_09_1 at N=3 also has 77 nodes, but it is a degree-9 rule
    # and our 77-node cell is degree 11, so it is NOT a counterpart.
    #   3-1: 2N          -> 6, 8, 10 at d = 3, 4, 5  (degree 3)
    #   5-1: N^2+N+2     -> 22 at d = 4              (degree 5)
    #   7-1: 2^N+2N^2+1  -> 27, 49 at d = 3, 4       (degree 7)
    Case("GH d3 p3  n6",   joinpath(RULES, "hermite_d3_p3_n6.csv"),
         "", () -> checked(() -> stroud_3_1(3), 3), "stroud1971 (E_n^{r^2} 3-1, order 2N, axial)", 3, zeros(3), false),
    Case("GH d4 p3  n8",   joinpath(RULES, "hermite_d4_p3_n8.csv"),
         "", () -> checked(() -> stroud_3_1(4), 3), "stroud1971 (E_n^{r^2} 3-1, order 2N, axial)", 4, zeros(4), false),
    Case("GH d5 p3  n10",  joinpath(RULES, "hermite_d5_p3_n10.csv"),
         "", () -> checked(() -> stroud_3_1(5), 3), "stroud1971 (E_n^{r^2} 3-1, order 2N, axial)", 5, zeros(5), false),
    Case("GH d4 p5  n22",  joinpath(RULES, "hermite_d4_p5_n22.csv"),
         "", () -> checked(() -> burkardt_en_r2(joinpath(SM, "en_r2_05_1_n4_o1.csv"), 4), 5),
         "stroud1971 (E_n^{r^2} 5-1, order N^2+N+2 = 22 at N=4)", 4, zeros(4), false),
    Case("GH d3 p7  n27 o1", joinpath(RULES, "hermite_d3_p7_n27.csv"),
         "", () -> checked(() -> burkardt_en_r2(joinpath(SM, "en_r2_07_1_n3_o1.csv"), 3), 7),
         "stroud1971 (E_n^{r^2} 7-1 option 1, order 2^N+2N^2+1 = 27 at N=3)", 3, zeros(3), false),
    Case("GH d3 p7  n27 o2", joinpath(RULES, "hermite_d3_p7_n27.csv"),
         "", () -> checked(() -> burkardt_en_r2(joinpath(SM, "en_r2_07_1_n3_o2.csv"), 3), 7),
         "stroud1971 (E_n^{r^2} 7-1 option 2, order 2^N+2N^2+1 = 27 at N=3)", 3, zeros(3), false),
    Case("GH d4 p7  n49 o1", joinpath(RULES, "hermite_d4_p7_n49.csv"),
         "", () -> checked(() -> burkardt_en_r2(joinpath(SM, "en_r2_07_1_n4_o1.csv"), 4), 7),
         "stroud1971 (E_n^{r^2} 7-1 option 1, order 2^N+2N^2+1 = 49 at N=4)", 4, zeros(4), false),
    Case("GH d4 p7  n49 o2", joinpath(RULES, "hermite_d4_p7_n49.csv"),
         "", () -> checked(() -> burkardt_en_r2(joinpath(SM, "en_r2_07_1_n4_o2.csv"), 4), 7),
         "stroud1971 (E_n^{r^2} 7-1 option 2, order 2^N+2N^2+1 = 49 at N=4)", 4, zeros(4), false),
]

# Xiao-Gimbutas cube degree 7 is included only if the data is present; the
# 2026-08-30 sweep's copy lived in a session scratchpad and is gone.
const XG = joinpath(@__DIR__, "xg_cube_p7_n26.csv")
if isfile(XG)
    push!(cases, Case("Le d3 p7  n26", joinpath(RULES, "legendre_d3_p7_n26.csv"),
                      XG, nothing, "xiao_gimbutas2010", 3, fill(0.5, 3), true))
end

println("compare_rules.jl - tie identity checks\n")
println("Tolerances: same-rule <= ", SAME_TOL, ", distinct if fingerprint gap > ", DIST_TOL, "\n")

results = String[]
for c ∈ cases
    if !isfile(c.ours)
        println(c.label, ": SKIPPED, missing ", c.ours); continue
    end
    A, wa = readrule(c.ours)
    if c.gen ≡ nothing
        isfile(c.theirs) || (println(c.label, ": SKIPPED, missing ", c.theirs); continue)
        B, wb = readrule(c.theirs)
    else
        B, wb = c.gen()
    end

    # normalization assertions - both sides unit mass, and for Le both in [0,1]^d
    @assert abs(sum(wa) - 1) < 1e-10 "ours: weights do not sum to 1 ($(sum(wa)))"
    @assert abs(sum(wb) - 1) < 1e-10 "theirs: weights do not sum to 1 ($(sum(wb)))"
    if c.cube
        @assert all(-1e-12 .≤ A .≤ 1 + 1e-12) "ours: not on [0,1]^d"
        @assert all(-1e-12 .≤ B .≤ 1 + 1e-12) "theirs: not on [0,1]^d"
    end
    @assert size(A, 2) == size(B, 2) "node counts differ: $(size(A,2)) vs $(size(B,2))"

    # 1. isometry-invariant fingerprints
    wgap = maximum(abs.(sort(wa) .- sort(wb)))
    dgap = maximum(abs.(distance_fingerprint(A) .- distance_fingerprint(B)))
    fp_gap = max(wgap, dgap)

    # 2. explicit alignment over the domain's symmetry group
    G = bd_group(c.d)
    ax, aw = best_alignment(A, wa, B, wb, G, c.center)
    how = "B_$(c.d)"
    # Gaussian cells: extend to O(d), under which the weight is invariant.
    # Implemented at d = 2; at d >= 3 a B_d match has sufficed so far, and if one
    # ever fails there while the fingerprints agree, the rules are congruent and
    # the alignment search - not the rules - is what falls short.
    if !c.cube && c.d == 2
        rx, rw = best_alignment_M(A, wa, B, wb, o2_candidates(A, wa, B, wb))
        if rx < ax
            ax, aw, how = rx, rw, "rotation in O(2), outside B_2"
        end
    end

    v = verdict(fp_gap, ax)
    @printf("%-15s  n=%3d  |G|=%4d\n", c.label, size(A, 2), length(G))
    @printf("    weight multiset gap    %.3e\n", wgap)
    @printf("    distance multiset gap  %.3e\n", dgap)
    @printf("    best node displacement %.3e   (weight diff %.3e)  [via %s]\n", ax, aw, how)
    @printf("    VERDICT  %s:%s\n\n", v, c.source)
    push!(results, @sprintf("%s\t%s:%s\tnode displacement %.3e via %s, weight gap %.3e, distance-multiset gap %.3e",
                            c.label, v, c.source, ax, how, wgap, dgap))
end

println("=== summary ===")
foreach(println, results)
