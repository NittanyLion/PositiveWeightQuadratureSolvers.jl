using PositiveWeightQuadratureSolvers
using LinearAlgebra, Random, Test

const DATA = joinpath(@__DIR__, "data")

# Everything here is deliberately tiny (d ≤ 3, p ≤ 7): the point is that the
# moment system, the verification gate and the deposit reader are wired up, not
# that a search converges.  No solver restarts are run.

@testset "PositiveWeightQuadratureSolvers" begin

    @testset "moment system, d = 2, p = 5" begin
        A = total_degree_indices(2, 5)
        @test size(A, 2) == 2
        @test size(A, 1) == binomial(5 + 2, 2)          # 21 monomials
        @test all(sum(A, dims = 2) .≤ 5)
        @test A[1, :] == [0, 0]
        # exactness conditions of the pair ansatz: only even total degrees bind
        @test count(iseven, vec(sum(A, dims = 2))) == 9   # degrees 0, 2, 4
    end

    @testset "deposited GH rule: d = 2, p = 5, n = 7" begin
        f = joinpath(DATA, "hermite_d2_p5_q3_n7.csv")
        rule = load_deposit_rule(f)
        @test size(rule.nodes) == (7, 2)
        @test length(rule.weights) == 7
        @test all(rule.weights .> 0)                    # positive weights
        @test sum(rule.weights) ≈ 1 atol = 1e-14        # probabilists' normalization
        v = verify_deposit(f)
        @test (v.d, v.p, v.n) == (2, 5, 7)
        @test v.err < 1e-11                             # the project's gate
        # degree 7 is NOT claimed, and a 7-node rule cannot reach it
        @test verify_exactness(rule.nodes, rule.weights, 7;
                               basis = :hermite, relative = true) > 1e-11
    end

    @testset "deposited Le rule: d = 2, p = 5, n = 7" begin
        f = joinpath(DATA, "legendre_d2_p5_q3_n7.csv")
        rule = load_deposit_rule(f)
        @test size(rule.nodes) == (7, 2)
        @test all(rule.weights .> 0)
        @test sum(rule.weights) ≈ 1 atol = 1e-14
        @test all(0 .< rule.nodes .< 1)                 # interior of [0,1]^2
        v = verify_deposit(f)
        @test (v.d, v.p, v.n) == (2, 5, 7)
        @test v.err < 1e-11
    end

    @testset "save_rule / load_rule round trip" begin
        rule = load_deposit_rule(joinpath(DATA, "hermite_d2_p5_q3_n7.csv"))
        mktempdir() do dir
            path = save_rule(joinpath(dir, "roundtrip.csv"), rule)
            back = load_rule(path)
            @test back.nodes ≈ rule.nodes
            @test back.weights ≈ rule.weights
        end
    end

    @testset "gate rejects a perturbed rule" begin
        rule = load_deposit_rule(joinpath(DATA, "hermite_d2_p5_q3_n7.csv"))
        X = copy(rule.nodes); X[1, 1] += 1e-6
        @test verify_exactness(X, rule.weights, 5; basis = :hermite, relative = true) > 1e-11
    end

    @testset "one-dimensional Gauss rules are exact to degree 2m-1" begin
        # Sanity checks on the 1-D building blocks the ansätze and the tensor
        # fills use; m = 4 points are exact to degree 7.
        xh = gauss_hermite_nodes(4)
        Jh = SymTridiagonal(zeros(4), [sqrt(k) for k ∈ 1:3])
        wh = eigen(Jh).vectors[1, :] .^ 2
        wh ./= sum(wh)
        @test sort(eigen(Jh).values) ≈ xh
        @test verify_exactness(reshape(sort(xh), :, 1), wh[sortperm(eigen(Jh).values)], 7;
                               basis = :hermite, relative = true) < 1e-13

        # uniform weight: gauss_legendre_rule is on [-1,1]; the bank frame is [0,1]
        xl, wl = PositiveWeightQuadratureSolvers.LegendreProductDQ.gauss_legendre_rule(4)
        @test sum(wl) ≈ 1
        @test verify_exactness(reshape((xl .+ 1) ./ 2, :, 1), wl, 7;
                               basis = :legendre, relative = true) < 1e-13
    end

    @testset "spectral start produces d = 3 nodes" begin
        s = spectral_nodes(3, 5, 20; rng = MersenneTwister(1))
        @test size(s.nodes) == (20, 3)
        @test all(isfinite, s.nodes)
        @test length(s.weights) == 20
    end

    @testset "orbit-search aliases are distinct entry points" begin
        for f ∈ (b_orbit_search, sc_orbit_search, simplex_orbit_search,
                 laguerre_orbit_search)
            @test f isa Function
        end
        @test b_orbit_search ≢ sc_orbit_search
    end
end
