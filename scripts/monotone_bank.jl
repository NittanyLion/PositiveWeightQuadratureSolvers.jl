#=
monotone_bank.jl — make every column of the bank monotone in the degree
(2026-09-10, after the user asked how Le d=3 q=19 could hold MORE nodes than
q=20).

A rule exact to total degree p+2 is exact to degree p, so the best count at p
can never legitimately exceed the best count at p+2.  The bank is per cell and
nothing propagated downward, so a cell whose eliminator lagged (or whose
result sat unpromoted in a stage directory) could show a tensor fill of 4598
nodes next to a designed 3316 at the next degree.  This pass copies the (p+2)
rule into the p cell whenever it is smaller — the same file, so it passes the
same gate it already passed — and records the copy in
symq/downgraded_rules.tsv so a table can say "= the degree-(p+2) rule".  The
copy is add-only and never overwrites; an eliminator on the lower cell then
starts from it instead of from the fill.

  julia monotone_bank.jl [--dry] [hermite|legendre|laguerre|elaplace ...]
  SYMQ_RULES_DIR / SYMQ_SIDECAR_DIR override the bank and symq/ locations.
=#
using DelimitedFiles, Dates
Dates_now() = Dates.now()

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

const J     = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const DRY   = "--dry" ∈ ARGS
const FAMS  = filter(a -> a ∈ ("hermite", "legendre", "laguerre", "elaplace"), ARGS)
const LEDGER = joinpath(SYMQ, "downgraded_rules.tsv")

fams = isempty(FAMS) ? ["hermite", "legendre", "laguerre", "elaplace"] : FAMS

# best banked (n, file) per (fam, d, p)
best = Dict{Tuple{String,Int,Int},Tuple{Int,String}}()
for f in readdir(RULES)
    m = match(r"^([a-z]+)_d(\d)_p(\d+)_n(\d+)\.csv$", f)
    m ≡ nothing && continue
    fam = m.captures[1]; fam ∈ fams || continue
    d, p, n = parse.(Int, m.captures[2:4])
    if !haskey(best, (fam, d, p)) || n < best[(fam, d, p)][1]
        best[(fam, d, p)] = (n, f)
    end
end

ncopied = 0
for fam in fams, d in 1:5
    ps = sort([p for (f, dd, p) in keys(best) if f == fam && dd == d])
    # walk downward so a copy can cascade (p+4 → p+2 → p)
    for p in reverse(ps)
        haskey(best, (fam, d, p + 2)) || continue
        n_here, _ = best[(fam, d, p)]
        n_up, f_up = best[(fam, d, p + 2)]
        n_up < n_here || continue
        dst = "$(fam)_d$(d)_p$(p)_n$(n_up).csv"
        if isfile(joinpath(RULES, dst))
            println("  $fam d=$d p=$p: $dst already exists (bank still lists $n_here as best?)")
            continue
        end
        # the p+2 rule must be strictly interior / positive — it is banked, so it
        # passed the gate; re-check the two cheap invariants anyway
        A = readdlm(joinpath(RULES, f_up), ',')
        w = A[:, end]
        minimum(w) > 0 || (println("  $fam d=$d p=$p: SKIP, $f_up has a non-positive weight"); continue)
        abs(sum(w) - 1) < 1e-10 || (println("  $fam d=$d p=$p: SKIP, $f_up mass off by $(sum(w) - 1)"); continue)
        if DRY
            println("  would copy $f_up → $dst  ($n_here → $n_up)")
        else
            cp(joinpath(RULES, f_up), joinpath(RULES, dst))
            open(LEDGER, "a") do io
                println(io, dst, '\t', "= ", f_up, '\t', "was ", n_here, '\t', string(Dates_now()))
            end
            # the parent is fully known here: the same file, one degree up
            symq_lineage!(dst, f_up,
                          "monotone downgrade: the degree-$(p + 2) rule is also a degree-$p rule, copied down (monotone_bank.jl)";
                          rules = RULES)
            println("  copied $f_up → $dst  ($n_here → $n_up)")
        end
        best[(fam, d, p)] = (n_up, f_up)     # cascade: the cell below copies from the SAME source file
        global ncopied += 1
    end
end

println(DRY ? "dry run: $ncopied cell(s) would change" : "$ncopied cell(s) copied; ledger $LEDGER")
