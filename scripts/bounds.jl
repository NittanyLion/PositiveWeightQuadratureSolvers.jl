# Print "p n_even bound_pairs bound_nodes" for a dimension and a list of degrees.
# Usage: julia bounds.jl <d> <p1> <p2> ...
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
using .DesignedQuadrature
d = parse(Int, ARGS[1])
for p in parse.(Int, ARGS[2:end])
    aind = total_degree_indices(d, p)
    n_even = count(iseven, vec(sum(aind, dims = 2)))
    bound = ceil(Int, n_even / (d + 1))
    println(p, " ", n_even, " ", bound, " ", 2bound)
end
