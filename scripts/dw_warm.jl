# Pair elimination on the uniform cube, warm-started from a Diallo–Worku rule
# (2026-09-10, user: "yes, bank them as derived and move it into the campaign").
#
#   julia dw_warm.jl <d> <p> [seed]
#
# DW's fully symmetric rules (symq/dw_rules/, arXiv:2601.14488) contain −I, so each is
# already a ±pair rule — except where it carries a center node, which is dropped and the
# rest reconverged.  From there: drop one pair, reconverge, repeat, down to the pair floor.
# METHODS §28.
#
# Resumable and shareable.  The start is the smallest gate-passing rule among DW's, this
# host's staged files and any banked rule for the cell below DW's count, and the bank is
# re-read before every step, so a relaunch — or the same cell running on another machine
# and arriving through the hourly sync — continues from the best rule known.  Every
# converged size is staged, lineage-recorded and, when it beats the bank, banked at once:
# a step at p ≥ 37 takes hours, and nothing may wait for the end of the run.
#
# Lineage (user: "keep track of how we got to each solution, e.g. solution A came from
# warm-starting from Diablo, solution B from A plus doing X"): one line per rule in
# rules/lineage/<host>.tsv — file, parent, how, when, host.  Per host because rules/ is
# rsynced with -au between machines, and a shared file appended on two machines would
# lose one side.  make_appendices.jl walks these chains for the credit marks.

include(joinpath(@__DIR__, "..", "src", "DesignedQuadrature.jl"))
include(joinpath(@__DIR__, "..", "src", "DesignedQuadratureV2.jl"))
using .DesignedQuadrature, .DesignedQuadratureV2
using LinearAlgebra, Random, Printf, Dates

BLAS.set_num_threads(1)    # one core per worker, like every campaign worker (CLAUDE.md 2026-09-09)

const J       = get(ENV, "SYMQ_ROOT", pwd())
const RULES   = get(ENV, "SYMQ_RULES_DIR", joinpath(J, "rules"))
const HOST    = replace(gethostname(), r"\..*" => "")
const STAGE   = joinpath(get(ENV, "SYMQ_SIDECAR_DIR", joinpath(J, "symq")), "dw_stage", HOST)
const DWDIR   = get(ENV, "SYMQ_DW_DIR", joinpath(J, "symq", "dw_rules"))   # env: test hook
const LINEAGE = joinpath(RULES, "lineage", "$HOST.tsv")
const EXTOL   = 1e-11            # the bank gate
const TRIES   = 4                # pair candidates per step, as node_elimination
const MARGIN  = 1e-6             # the solver's box penalty starts at |x| = 1 − 1e-6

const D    = parse(Int, ARGS[1])
const P    = parse(Int, ARGS[2])
const SEED = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 1
const CELL = "legendre_d$(D)_p$(P)"
mkpath(STAGE); mkpath(dirname(LINEAGE))

say(s) = (println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", s); flush(stdout))
nof(f) = parse(Int, match(r"_n(\d+)\.csv$", f).captures[1])

function lineage!(file, parent, how)
    open(LINEAGE, "a") do io
        println(io, join((file, parent, how, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), HOST), '\t'))
    end
end

function gate(path)
    r = load_rule(path)
    ex = verify_exactness(r.nodes, r.weights, P; basis = :legendre, relative = true)
    ok = ex ≤ EXTOL && minimum(r.weights) > 0 && abs(sum(r.weights) - 1) ≤ 1e-10 &&
         all(0 .< r.nodes .< 1)
    return ok, ex, r
end

# canonical names only: Dropbox "conflicted copy" files (2026-09-11 01:35, two hosts banking the
# same n) crashed the p29 chain in nof() — they are skipped like every other bank tool skips them
canon(f) = occursin(Regex("^" * CELL * raw"_n\d+\.csv$"), f)
bank_files() = [f for f in readdir(RULES) if canon(f)]
bank_best()  = (fs = bank_files(); isempty(fs) ? typemax(Int) : minimum(nof, fs))

# Bank a staged rule if it beats the cell's best (add-only: never overwrites a file).
function promote(path)
    n = nof(path); dst = joinpath(RULES, basename(path))
    (n < bank_best() && !isfile(dst)) || return false
    ok, ex, _ = gate(path)
    ok || (say(@sprintf("NOT banked %s: gate failed (relerr %.1e)", basename(path), ex)); return false)
    cp(path, dst)
    say(@sprintf("BANKED %s (relerr %.1e)", basename(dst), ex))
    return true
end

const dwfile = only(filter(f -> startswith(f, "dw_d$(D)_p$(P)_n"), readdir(DWDIR)))
const N_DW   = nof(dwfile)
const M_EVEN = sum(binomial(k + D - 1, D - 1) for k in 0:2:P-1)
const M_FLOOR = cld(M_EVEN, D + 1)                  # pairs; below this LM cannot converge

# leftovers from a killed run
for f in sort(filter(canon, readdir(STAGE)), by = nof)
    promote(joinpath(STAGE, f))
end

# ---- start point --------------------------------------------------------------------
cands = Tuple{Int,String,String}[]                  # (n, path, lineage name)
for f in filter(canon, readdir(STAGE))
    push!(cands, (nof(f), joinpath(STAGE, f), f))
end
for f in bank_files()
    nof(f) < N_DW && push!(cands, (nof(f), joinpath(RULES, f), f))
end
sort!(cands)

cur_pairs = nothing; cur_name = ""; cur_n = 0
for (nc, pathc, namec) in cands
    ok, _, rc = gate(pathc)
    ok || continue
    global cur_pairs = pairs_from_rule(2 .* rc.nodes .- 1, rc.weights)
    global cur_name = namec; global cur_n = nc
    say("resuming from $namec ($(dirname(pathc) == RULES ? "bank" : "stage"))")
    break
end

if cur_pairs === nothing
    rdw = load_rule(joinpath(DWDIR, dwfile))
    Xdw = clamp.(2 .* rdw.nodes .- 1, -1 + MARGIN, 1 - MARGIN); wdw = rdw.weights
    ctr = [i for i in axes(Xdw, 1) if all(abs.(Xdw[i, :]) .< 1e-12)]
    if isempty(ctr)
        global cur_pairs = pairs_from_rule(Xdw, wdw)
        global cur_name = "dw:$dwfile"; global cur_n = N_DW
        say("starting from Diallo–Worku $dwfile ($(length(cur_pairs[2])) pairs)")
    else
        keepdw = setdiff(axes(Xdw, 1), ctr)
        b0, w0 = pairs_from_rule(Xdw[keepdw, :], wdw[keepdw] ./ sum(wdw[keepdw]))
        say("Diallo–Worku $dwfile has $(length(ctr)) center node(s): dropping, reconverging at $(length(w0)) pairs")
        rt0 = designed_quadrature_v2(D, P, length(w0); basis = :legendre, symmetric = true, tol = 1e-12,
                                     init_pairs = (b0, w0), maxiter = 4000, stall_window = 200)
        rt0.converged || (say("center-dropped start did not reconverge (‖R‖=$(rt0.residual)) — giving up"); exit(1))
        name0 = "$(CELL)_n$(length(rt0.weights)).csv"; path0 = joinpath(STAGE, name0)
        save_rule(path0, rt0)
        lineage!(name0, "dw:$dwfile", "center node dropped, remaining ±pairs reconverged (dw_warm.jl)")
        promote(path0)
        global cur_pairs = rt0.pairs; global cur_name = name0; global cur_n = length(rt0.weights)
    end
end

say(@sprintf("%s: n=%d, DW %d, pair floor %d pairs (%d nodes), seed %d", CELL, cur_n, N_DW, M_FLOOR, 2M_FLOOR, SEED))

# ---- elimination --------------------------------------------------------------------
while true
    # another host's (or another tool's) smaller rule for this cell, arrived by sync
    bb = bank_best()
    if bb < cur_n - 1
        f = "$(CELL)_n$(bb).csv"
        ok, _, r = gate(joinpath(RULES, f))
        if ok
            global cur_pairs = pairs_from_rule(2 .* r.nodes .- 1, r.weights)
            global cur_name = f; global cur_n = bb
            say("adopted banked $f")
        end
    end
    b, w = cur_pairs; m = length(w)
    m - 1 < M_FLOOR && (say("at the pair floor ($m pairs) — done"); break)
    order = sortperm(w)
    pool = order[1:min(2TRIES, m)]
    picks = SEED == 1 ? pool[1:min(TRIES, end)] : shuffle(Xoshiro(SEED * 100_003 + m), pool)[1:min(TRIES, end)]
    stepped = false
    for c in picks
        keep = setdiff(1:m, c)
        t0 = time()
        rt = designed_quadrature_v2(D, P, m - 1; basis = :legendre, symmetric = true, tol = 1e-12,
                                    init_pairs = (b[keep, :], w[keep]), maxiter = 1500, stall_window = 100)
        if rt.converged && minimum(rt.weights) > 0
            name = "$(CELL)_n$(length(rt.weights)).csv"; path = joinpath(STAGE, name)
            save_rule(path, rt)
            lineage!(name, cur_name, @sprintf("pair elimination: dropped pair of weight %.3g, reconverged (dw_warm.jl seed %d)", w[c], SEED))
            say(@sprintf("m=%d converged (dropped w=%.3g, %d iters, %d s)", m - 1, w[c], rt.iterations, round(Int, time() - t0)))
            promote(path)
            global cur_pairs = rt.pairs; global cur_name = name; global cur_n = length(rt.weights)
            stepped = true
            break
        end
        say(@sprintf("m=%d: drop w=%.3g failed (‖R‖=%.2e, %d s)", m - 1, w[c], rt.residual, round(Int, time() - t0)))
    end
    stepped || (say("no candidate reconverged at m=$(m - 1) — chain ends at n=$cur_n"); break)
end
open(joinpath(STAGE, "done_d$(D)_p$(P)"), "w") do io
    println(io, cur_n)
end
