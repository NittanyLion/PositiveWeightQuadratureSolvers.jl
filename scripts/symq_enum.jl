#=
symq_enum.jl -- deterministic structure enumeration for one (d, p).

Enumerates B_d orbit-type multisets in INCREASING total node count, from
Möller's proven bound up to (current best - 1), and attempts each with K
random-init LM solves.  A structure that converges is verified, banked, and
published as a donor; a structure that doesn't is recorded as refuted.  When
the whole range below the current best is exhausted, the run has produced
strong evidence (not proof: K inits can miss, and the parameter window
excludes wildly overdetermined structures) that the current rule is minimal
WITHIN the B_d ansatz.

Machines share the work through append-only ledgers:
  comms/enum/d{d}p{p}_m{id}.tsv   one ledger per machine id
Each machine only appends to its own ledger (no sync conflicts) but reads
every ledger for the case; structures are partitioned by a stable hash so
the machines never duplicate work, and each scavenges the others' partitions
once its own is done.

Since 2026-08-27 this also runs as an OSPool job array (kind `enum` in
ospool/job_wrapper.sh): N jobs, ids 0..N-1, each with its own ledger, which
the round merge appends into bank/symq/enum/ and pull_ospool.sh brings into
comms/enum/.  Environment overrides for that:
  SYMQ_ENUM_DIR     ledger directory   (default comms/enum)
  SYMQ_MACHINE_ID   this worker's id   (default ~/.config/symq_machine_id, else 0)
  SYMQ_ENUM_PARTS   partition count    (default 2 — the office/home pair)
The certification verdict (range exhausted → retired.tsv) is only reached by
a worker that has seen every structure done, i.e. after the ledgers merge —
an office launch_enum on the case reads them all and closes it.

usage: julia symq_enum.jl <d> <p> [minutes]
=#
include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "SymmetricDQ.jl"))
using .DesignedQuadrature, .SymmetricDQ
using Random, Printf, LinearAlgebra

# Self-describing rules (2026-09-11): every banked rule gets one lineage line at
# bank time, in rules/lineage/<host>.tsv.  Included defensively — the no-op
# fallback below stands if the helper is missing (a cluster sandbox) or half
# synced, because a provenance line must never cost a solve.
symq_lineage!(args...; kw...) = nothing
try include(joinpath(@__DIR__, "symq_lineage.jl")) catch end

BLAS.set_num_threads(1)

const d       = parse(Int, ARGS[1])
const p       = parse(Int, ARGS[2])
const minutes = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 6 * 60.0

const J = get(ENV, "SYMQ_ROOT", pwd())
const RULES = get(ENV, "SYMQ_RULES_DIR",   joinpath(J, "rules"))
const SYMQ  = get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq"))
const COMMS = get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms"))
const ENUMD = get(ENV, "SYMQ_ENUM_DIR", joinpath(COMMS, "enum"))
const DONORS = joinpath(COMMS, "donors")
mkpath(ENUMD); mkpath(DONORS)

myid = haskey(ENV, "SYMQ_MACHINE_ID") ? parse(Int, ENV["SYMQ_MACHINE_ID"]) :
       try parse(Int, strip(read(joinpath(homedir(), ".config/symq_machine_id"), String)))
       catch; 0 end
const NPART = max(1, parse(Int, get(ENV, "SYMQ_ENUM_PARTS", "2")))
# Partition key that is identical on every Julia version and machine (Base.hash
# of a String is only promised stable within one version; the office runs a
# nightly, the OSPool container 1.12.6).  Only work-sharing depends on it, never
# correctness — a mismatch would merely duplicate attempts.
stablehash(s::AbstractString) = foldl((h, c) -> h * 0x100000001b3 ⊻ UInt64(c),
                                      codeunits(s); init = 0xcbf29ce484222325)

function current_best()
    best = typemax(Int)
    for f in readdir(RULES)
        m = match(Regex("^hermite_d$(d)_p$(p)_n(\\d+)\\.csv\$"), f)
        m === nothing && continue
        best = min(best, parse(Int, m[1]))
    end
    return best
end

moller = let
    v = 0
    for line in eachline(joinpath(SYMQ, "moller.tsv"))
        c = split(line)
        length(c) == 3 && parse(Int, c[1]) == d && parse(Int, c[2]) == p &&
            (v = parse(Int, c[3]))
    end
    v > 0 ? v : error("no Möller entry for d=$d p=$p")
end

const W = SymmetricDQ.Work(d, p)
const C = W.C
types = sort(orbit_types(d), by = T -> (T.size, T.k))
tsig(T) = isempty(T.mults) ? "o" : join(T.mults, "+") * (T.z > 0 ? "|$(T.z)z" : "")

# ---- ledgers ----------------------------------------------------------------
ledger_path(id) = joinpath(ENUMD, "d$(d)p$(p)_m$(id).tsv")
function load_done()
    done = Dict{String,String}()
    pat = Regex("^d$(d)p$(p)_m\\d+\\.tsv\$")
    for name in readdir(ENUMD)
        occursin(pat, name) || continue
        f = joinpath(ENUMD, name)
        for line in eachline(f)
            c = split(line, '\t')
            length(c) ≥ 2 && (done[c[1]] = c[2])
        end
    end
    return done
end
myledger = open(ledger_path(myid), "a")
record(sig, status) = (println(myledger, sig, '\t', status); flush(myledger))

# ---- enumeration in ascending-node bands ------------------------------------
# counts[i] = number of orbits of types[i]; params window keeps the space sane
const PMIN = C - 4
const PMAX = C + 8
const MAXORB = 40

function structures_in_band(lo, hi)   # lo < nodes ≤ hi
    out = Tuple{String,Int,Int,Vector{Int}}[]   # (sig, nodes, params, counts)
    nt = length(types)
    counts = zeros(Int, nt)
    function rec(i, nodes, params, norb)
        if nodes > hi || params > PMAX || norb > MAXORB
            return
        end
        if i > nt
            if nodes > lo && params ≥ PMIN
                sig = join(["$(tsig(types[t]))x$(counts[t])"
                            for t in 1:nt if counts[t] > 0], ",")
                push!(out, (sig, nodes, params, copy(counts)))
            end
            return
        end
        T = types[i]
        cmax = T.k == 0 ? 1 : MAXORB          # at most one origin
        c = 0
        while true
            counts[i] = c
            rec(i + 1, nodes + c * T.size, params + c * (T.k + 1), norb + c)
            c += 1
            (c > cmax || nodes + c * T.size > hi || params + c * (T.k + 1) > PMAX) && break
        end
        counts[i] = 0
    end
    rec(1, 0, 0, 0)
    sort!(out, by = t -> (t[2], -t[3]))
    return out
end

# ---- solving one structure --------------------------------------------------
function attempt(counts, tries, rng)
    for t in 1:tries
        mix = SymmetricDQ.OType[]
        vals = Vector{Float64}[]; u = Float64[]
        for (i, c) in enumerate(counts), _ in 1:c
            T = types[i]
            push!(mix, T)
            v = SymmetricDQ.random_values(rng, T.k, p)
            push!(vals, v)
            push!(u, T.k == 0 ? 0.0 :
                     -0.25 * sum(T.mults[j] * v[j]^2 for j in 1:T.k))
        end
        st = SymmetricDQ.MixState(d, p, mix, vals, u)
        tot = sum(exp(st.u[o]) * st.mix[o].size for o in eachindex(st.mix))
        st.u .-= log(tot)
        δ, ok = SymmetricDQ.solve!(st, W)
        ok && return st, δ
    end
    return nothing, Inf
end

# ---- main loop --------------------------------------------------------------
function main()
    logio = open(joinpath(SYMQ, "enum_d$(d)p$(p).log"), "a")
    prog  = joinpath(SYMQ, "prog_enumd$(d)p$(p).txt")
    deadline = time() + minutes * 60
    rng = MersenneTwister(1234 * d + p + 7 * myid)
    best0 = current_best()
    println(logio, "=== enum d=$d p=$p  [$(moller), $(best0 - 1)]  C=$C  params [$PMIN,$PMAX]  machine $myid/$NPART  $(round(minutes))min ===")
    flush(logio)

    ntested = 0; nskip = 0
    found = false
    band = 16
    lo = moller - 1
    while !found && lo < best0 - 1 && time() < deadline
        hi = min(lo + band, best0 - 1)
        structs = structures_in_band(lo, hi)
        done = load_done()
        println(logio, "band ($lo, $hi]: $(length(structs)) structures, $(count(s -> haskey(done, s[1]), structs)) already done")
        flush(logio)
        for pass in 1:2   # pass 1 = my partition, pass 2 = scavenge the rest
            for (sig, nodes, params, counts) in structs
                time() > deadline && break
                found && break
                mine = (stablehash(sig) % NPART == myid % NPART)
                (pass == 1) == mine || continue
                haskey(done, sig) && (nskip += 1; continue)
                tries = params ≥ C ? 6 : 12
                st, δ = attempt(counts, tries, rng)
                ntested += 1
                if st !== nothing
                    nodes2 = SymmetricDQ.mix_nodes(st)
                    nds, w = SymmetricDQ.expand_rule(st)
                    # relative gate — see verify_exactness in DesignedQuadrature.jl
                    ex = verify_exactness(nds, w, p; basis = :hermite,
                                          relative = true)
                    if ex ≤ 1e-9 && nodes2 < current_best()
                        path = joinpath(RULES, "hermite_d$(d)_p$(p)_n$(nodes2).csv")
                        open(path, "w") do io
                            for i in 1:nodes2
                                println(io, join(string.([nds[i, :]; w[i]]), ","))
                            end
                        end
                        # no parent rule: this orbit signature was solved from scratch
                        symq_lineage!(basename(path), "none",
                                      "orbit-type enumeration, signature $sig solved from scratch (symq_enum.jl)";
                                      rules = RULES)
                        SymmetricDQ.save_state(joinpath(SYMQ, "best_d$(d)_p$(p).txt"), st, δ)
                        cp(joinpath(SYMQ, "best_d$(d)_p$(p).txt"),
                           joinpath(DONORS, "best_d$(d)_p$(p).txt"); force = true)
                        record(sig, "FOUND n=$nodes2 ex=$ex")
                        println(logio, "*** FOUND $nodes2 nodes: $sig  (exactness $ex)")
                        flush(logio)
                        found = true
                    else
                        record(sig, "converged-but-rejected ex=$ex")
                    end
                else
                    record(sig, "refuted t=$tries")
                end
                ntested % 25 == 0 && write(prog,
                    "d=$d p=$p  start $best0, run-best $best0, none banked (enum ($lo,$hi] $ntested tried)\n")
            end
            time() > deadline && break
            done = load_done()      # refresh before the scavenge pass
        end
        lo = hi
    end

    status = found ? "IMPROVED" :
             (lo ≥ best0 - 1 ? "range exhausted (B_d floor evidence at $(current_best()))" :
                               "budget reached in band ending $lo")
    if !found && lo ≥ best0 - 1
        # full range below the current best refuted: close the combo so the
        # campaign stops spending workers on it (purple in the status window)
        rt = joinpath(SYMQ, "retired.tsv")
        already = isfile(rt) && any(startswith(l, "$d $p ") for l in eachline(rt))
        already || open(rt, "a") do io
            println(io, "$d $p enum-certified-B$(d)-floor-$(current_best())")
        end
    end
    println(logio, "=== enum done: $status  tested $ntested, skipped $nskip ===")
    write(prog, "d=$d p=$p  start $best0, run-best $(current_best()), enum: $status\n")
    close(myledger); close(logio)
end

main()
