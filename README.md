# PositiveWeightQuadratureSolvers.jl

The solvers that produced the positive-weight interior cubature rules deposited
on Zenodo for the **Gaussian** weight (GH — the standard normal N(0, I_d) on
R^d, weights summing to 1) and the **uniform** weight (Le — Lebesgue measure on
[0,1]^d, weights summing to 1).

A rule here is a set of nodes x_1, …, x_n and strictly positive weights w_1, …,
w_n with

    Σ_s w_s x_s^a  =  ∫ x^a dμ(x)      for every monomial of total degree ≤ p,

with every node in the interior of the domain.  The object of the search is to
make n as small as possible for a given (d, p), so the quality measure reported
throughout is

    ρ = n^(1/d) / q,      q = (p+1)/2,

where ρ = 1 is a Gauss product grid of the same degree and smaller is better.

## Where the three parts of this project live

| part | where |
| --- | --- |
| **the rules** — node/weight data, per-cell metadata, extended-precision twins, checksums | Zenodo, one deposit per weight family (`publish/gh`, `publish/le` in the research tree) |
| **the method** — what the ansätze are, why each exists, what was measured | the arXiv paper |
| **the code** — this repository, with a DOI minted from a GitHub release | here |

The deposits are self-contained: each ships `rules/`, `rules_extended/`,
`summary.csv` with a column dictionary (`COLUMNS.md`), the credit and tie
accounting (`CREDIT_AND_TIES.md`), a precision note and `SHA256SUMS`.  Nothing
in this repository is needed to *use* a deposited rule — only to reproduce the
search that found it, or the verification that gated it.

## Install

```julia
] add https://github.com/NittanyLion/PositiveWeightQuadratureSolvers.jl
```

or, from a clone,

```julia
] dev /path/to/PositiveWeightQuadratureSolvers.jl
```

Julia 1.11 or later.  `MultiFloats` is the only non-stdlib dependency; it is
needed by `scripts/polish_mp.jl` (double-double and quad-double arithmetic).

## Verifying a deposited rule

Download a deposit, then:

```julia
using PositiveWeightQuadratureSolvers

v = verify_deposit("publish/gh/rules/hermite_d2_p19_q10_n68.csv")
# (d = 2, p = 19, n = 68, err = …)   err < 1e-11 is the project's gate
```

`verify_deposit` reads the deposit's comment header, infers the weight family
from the file name (`hermite_…` → Gaussian, `legendre_…` → uniform) and the
degree from its `_pP_` field, and reports the worst **relative** monomial error
over all monomials of total degree ≤ p.  Spelled out:

```julia
rule = load_deposit_rule("publish/le/rules/legendre_d3_p13_q7_n128.csv")
verify_exactness(rule.nodes, rule.weights, 13; basis = :legendre, relative = true)
@assert all(rule.weights .> 0)
@assert all(0 .< rule.nodes .< 1)          # interior of [0,1]^3
```

The error is relative — a backward error, `|q − exact| / Σ_s |w_s| Π_k |x_sk|^a_k`
— because for the Gaussian weight the raw moments reach (p−1)!!, so an
*exactly* correct rule shows absolute error of order 1e-6 at p = 19 from
roundoff alone.  The gate is 1.0e-11 in the relative measure; the deposits report
each cell's verified error, measured in 192-bit BigFloat, in `summary.csv`.

`load_rule` (the research tree's own reader, re-exported here) parses every line
as numbers and so reads a bare bank file, not a deposited one; `load_deposit_rule`
skips the `#` block and the `x1,…,xd,w` header.

## Modules

`src/` holds the solver modules, byte-identical to the research tree.  Section
numbers refer to the methods paper / `METHODS.md`, and the letter is the tag the
deposits' `method` column uses (legend in each deposit's `COLUMNS.md`).

| module | what it is | tag | §  |
| --- | --- | --- | --- |
| `DesignedQuadrature.jl` | the moment system itself: `total_degree_indices`, orthonormal polynomial tables, the Gauss/uniform/Gamma/elliptical-Laplace targets, the V1 Levenberg–Marquardt solver, `find_rule`, `verify_exactness`, `load_rule`/`save_rule` | — | 0, 1 |
| `DesignedQuadratureV2.jl` | adaptive Levenberg–Marquardt with **node elimination** and degree continuation, plus the column-scaled polish; the workhorse behind almost every `F` cell | `F` | 2, 29.4 |
| `SymmetricDQ.jl` | fully symmetric **B_d** orbit ansatz (sign flips × permutations) with orbit-constant weights, the `D_d` half-orbit variant, and the native uniform search (`basis = :legendre`) | `B` | 4, 15, 17 |
| `PermCentralDQ.jl` | **S_d × Z_2** ansatz — permutations × central inversion, the Stroud-type structures | `SC` | 6 |
| `SimplexDQ.jl` | **S_{d+1} × Z_2** simplex symmetry | — | 16 |
| `IcosahedralDQ.jl` | icosahedral group, d = 3 — an order-5 rotation axis is impossible in B_3, so a match is definitive | `I` | 8 |
| `OPSpaceDQ.jl` | construction in orthogonal-polynomial space, d = 2 | — | 10 |
| `LegendreProductDQ.jl` | product-orbit descent for the uniform weight, D_{d−1} × B_1 | — | 21 |
| `SpectralInit.jl` | spectral start: truncated multiplication operators on P_s give a node set to start from instead of a random one | — | 20 |
| `WeightTransforms.jl` | 1-D Gauss nodes (Golub–Welsch) and the monotone quantile maps that carry a rule between weights — Gaussian ↔ uniform ↔ exponential/Gamma | — | 17, 19 |
| `LaguerreDQ.jl` | native orthant search for the exponential / Gamma weight | — | 18 |
| `LaguerreProductDQ.jl` | product-orbit descent for the same weight | — | 18, 21 |
| `LaguerrePolish.jl` | extended-precision Gauss–Newton polish in the *relative monomial* form the gate measures, for orthant rules | — | 12, 22 |

The `F4` (24-cell) and `H4` (icosian) tags that appear in the GH deposit at d = 4
come from `PolytopeDQ4.jl`, a retired module (§9) that is not shipped; the rules
it found are in the deposit and `scripts/classify_methods.jl` still recognizes
their symmetry.  `L` marks a literature rule, transcribed or re-found.

The ansatz modules deliberately export the *same* names — `OType`,
`build_type`, `orbit_types`, `conditions`, `random_state`, `solve!`,
`eliminate!`, `orbit_search` — because they implement one interface for
different symmetry groups.  They are therefore reachable as submodules
(`PositiveWeightQuadratureSolvers.SymmetricDQ.orbit_search`) rather than
re-exported wholesale; four disambiguated aliases are exported for the search
entry points: `b_orbit_search`, `sc_orbit_search`, `simplex_orbit_search`,
`laguerre_orbit_search`.

## Drivers

`scripts/` holds the search drivers, run as command-line programs.  They all
read and write a bank directory and a sidecar directory:

| variable | default | holds |
| --- | --- | --- |
| `SYMQ_ROOT` | `pwd()` | the parent of the two below |
| `SYMQ_RULES_DIR` | `$SYMQ_ROOT/rules` | the rule bank, `<family>_d<D>_p<P>_n<N>.csv` |
| `SYMQ_SIDECAR_DIR` | `$SYMQ_ROOT/symq` | logs, per-cell best structures, progress files |
| `SYMQ_COMMS_DIR` | `$SYMQ_ROOT/comms` | donor structures (in the campaign, a shared folder; here just a directory) |

A driver banks a rule only when it verifies at the 1.0e-11 relative gate **and**
improves on the file already in the bank, so pointing these at an empty
directory is safe and starts from nothing.

```sh
mkdir -p run/rules run/symq
export SYMQ_ROOT=$PWD/run
julia --project=. scripts/symq_run.jl 2 5 1 1          # 1 minute, seed 1
```

| script | what it does |
| --- | --- |
| `symq_run.jl <d> <p> <minutes> [seed] [mode…]` | the B_d / D_d orbit search. Modes: `D` half-orbits, `L` the uniform weight natively, `explore` wider budgets, `legendredonor` opt-in Gaussian→uniform orbit-type donors |
| `symq_sc.jl <d> <p> <minutes> [seed] [mode…]` | the same contract for the S_d × Z_2 ansatz |
| `symq_simplex.jl <d> <p> <minutes> [seed]` | the same for S_{d+1} × Z_2 |
| `symq_ico.jl <d=3> <p> <minutes> [seed]` | the same for the icosahedral group |
| `symq_leg_product.jl <d> <p> <minutes> [seed]` | product-orbit descent, uniform weight |
| `symq_freeelim.jl` | free-node elimination: solve at n, drop the least useful node, re-solve — the chain that produced most `F` cells (§7) |
| `symq_pairelim.jl` | the same on the ±pair ansatz |
| `symq_ladder.jl <d> <p_from> <p_to> [family]` | degree ladder: continue a converged rule up in degree, rung by rung (§20, §29.3) |
| `symq_rot2.jl` | D_k rotational-orbit search for GH d = 2 (§32) |
| `symq_enum.jl <d> <p> …` | deterministic enumeration of B_d orbit structures — certifies a floor for the ansatz rather than searching (§5) |
| `spectral_start.jl` | spectral initialization, then elimination |
| `update_legendre.jl <d> <p> [seed]` | pair elimination for the uniform weight, with a staging directory; promotes staged rules at the start of every run (§17). For the big cells (d = 5, p ≥ 17; §36) it takes `SYMQ_BLAS_THREADS` (default 1), `SYMQ_PAIR_LADDER` (start sizes as multiples of the pair floor, default `1.20,1.45,1.75,2.10`), `SYMQ_PAIR_BATCH` (pairs dropped per step, halved on failure), `SYMQ_PAIR_WARM=1` (continue from the banked rule) and `SYMQ_PAIR_RNGOFF`; it checkpoints inside a solve and resumes, and `--promote-only` may run beside a live eliminator |
| `dw_warm.jl` | warm start from a *published* rule (pair elimination from someone else's node set). Every rule it banks gets a lineage line, and the paper marks such a rule as derived — see "Credit" below (§28.2, §28.5) |
| `tensor_fill.jl <family…>` | tensor products of Gauss rules, as a warm start and as a fallback count for an empty cell (tag `T`) |
| `monotone_bank.jl [--dry] [family…]` | a degree-(p+2) rule *is* a degree-p rule: copies the smaller one down whenever a column is not monotone, and ledgers the copy (§27) |
| `polish_mp.jl <bank file> [--start twin] [--type x4\|big] [--target …] [--digits …]` | extended-precision polish in the free-node (±pair) frame: minimum-norm Newton on the orthonormal moment system with the residual in `Float64x4` (~64 digits) or 384-bit `BigFloat` and the step from a Float64 SVD of the Jacobian. The deposited files are its `--type big --target 1e-68 --digits 80` output. `--start` begins from an existing twin instead of from the rounded bank file: a Gaussian rule can be rotated, so a cold restart slides along the solution manifold and the Float64 rounding moves with it, while a warm start leaves it unchanged. Rows are written in the bank file's order. The result depends on thread count and the LAPACK build in its last digits: a rerun reproduces a deposited file to its accuracy (~1e-70), not byte for byte |
| `polish_sym.jl <bank file> [--target …] [--digits …]` | the same polish in the **orbit frame**, for a Gaussian rule that is a union of complete B_d orbits: a few dozen unknowns, so the whole Newton iteration runs in `BigFloat`. Needed where the free frame is singular at the solution and `polish_mp.jl` floors (GH d4 p19, d5 p11, p13, p17, p19, p21). Deterministic: a rerun reproduces the deposited file byte for byte. Exit 4 on a rule that is not a union of complete orbits |
| `polish_lm_big.jl <bank file> [--start twin] [--target …] [--digits …]` | Levenberg step with the Jacobian and the linear solve in `BigFloat` as well, for a small rule neither of the above can finish — GH d4 p9 n116, the S_4×Z_2 rule, whose Jacobian has rank 280 of 290 and which is not made of complete B_d orbits. Affordable up to a few thousand unknowns |
| `classify_methods.jl` | tags each banked best rule `B` / `SC` / `I` / `F4` / `H4` / `F` by *testing its symmetry*, not by trusting the tool that made it — this is where the deposits' `method` column comes from (§14) |
| `compare_rules.jl` | settles "same rule, or a distinct rule at the same count?" against published rules: isometry-invariant fingerprints (sorted weights, sorted pairwise distances) plus an explicit alignment over the domain's symmetry group |
| `bounds.jl <d> <p…>` | Möller's lower bound on n for each degree — how far a cell could still fall |
| `symq_lineage.jl` | not a driver: the single lineage writer the banking tools include, one line per banked rule in `rules/lineage/<host>.tsv` (§28.6) |

`compare_rules.jl` needs the other side of each comparison — published node
listings converted into the bank frame — and those files are **not** in this
repository: they are other authors' data.  Point `PWQS_RESEARCH_ROOT` at a tree
holding them, or read the script for the conversion each source needs (the
normalization is stated at the top and asserted at run time for every pair).

## Credit

A rule this code found from scratch is ours.  A rule whose elimination chain
starts from someone else's published rule — what `dw_warm.jl` does — is
*derived*, and the deposits and the paper say so, per cell, in the `credit`
column; once such a rule is used for a different problem, no credit is owed.
Rules transcribed from the literature, and rules our search re-found that turn
out to be a published rule, are theirs. `scripts/symq_lineage.jl` is what makes
this checkable: every banking tool writes one line naming the parent rule at the
moment it banks.

## What is not here

**Campaign orchestration is not included, and deliberately so.** The rules were
found by a long-running campaign across three local machines and two HPC pools,
and the scheduling half of it is plumbing, not method: the seat allocator and
priority policy, the launcher and stop scripts, the memory guard, the
multi-machine bank syncing and donor exchange, the cluster and OSPool job
manifests, the status windows, the hourly notifications and digests, and the
deposit builder that assembles the Zenodo archives. None of it is in this
repository. What is here is everything needed to run one search on one machine
and to verify what it produces.

**A search is not a function of its inputs alone.** Every driver is a randomized
multi-start descent under a wall-clock budget: the rule you get depends on the
seed, on how long you let it run, on how many threads BLAS was given, and — for
the elimination chains — on which rule happened to be in the bank when it
started. Running `symq_run.jl 2 19 60 1` will not in general reproduce the
deposited 68-node rule; it will produce *a* verified rule for that cell, usually
worse, sometimes better. The deposits, not a rerun, are the record of what was
found. Where a cell's history is known it is in `rules/lineage/` and in the
deposit's `credit_evidence` column.

**No rule data ships here.** The two files in `test/data/` are single deposited
rules (d = 2, p = 5, n = 7, one per family), copied verbatim so the test suite
has something real to verify. The bank itself is on Zenodo.

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite is a smoke test by design — d ≤ 3, p ≤ 7, a few seconds, no search
restarts. It checks the moment system's size and parity structure, that both
deposited rules load, have positive weights summing to 1, lie in the interior
and pass the gate at their claimed degree and fail it one degree higher, that a
perturbed rule is rejected, that 1-D Gauss rules are exact to degree 2m−1, and
that the module wiring and the orbit-search aliases resolve.

## Citation

See `CITATION.cff`. The rule data has its own Zenodo DOI per weight family, and
the method its own arXiv identifier; cite whichever you used.

## License

MIT — see `LICENSE`.
