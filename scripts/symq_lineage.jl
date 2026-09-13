# The one lineage writer for every tool that banks a rule (2026-09-11, user:
# "make every future rule self-describing").
#
# Every worker records one line per banked rule, AT BANK TIME, in
#     rules/lineage/<host>.tsv      file \t parent \t how \t when \t host
# the format dw_warm.jl introduced on 2026-09-10 and that make_appendices.jl and
# make_publish.jl already read (they key on the bare file name, take the FIRST
# line for a file, and skip blank and '#' lines).  Before this, only dw_warm.jl
# wrote lineage, which is why the Zenodo deposits carry 56 `unrecorded` cells:
# a rule arrived in rules/ and nothing anywhere said who made it or from what.
#
# WHY ONE FILE PER HOST, NOT ONE SHARED FILE.  rules/ is rsynced -au between the
# three machines and Dropbox, so a single shared ledger appended on two machines
# would either lose one side or turn into a "conflicted copy" — which has bitten
# this project repeatedly (symq/policy.tsv, symq/method_cache.tsv, and the p29
# chain on 2026-09-11).  One writer per file means: this host only.  Within a
# host, many workers append concurrently, so a line is built in full and written
# with ONE write call — an O_APPEND write of a short line is not interleaved.
#
# WHY NOTHING HERE MAY THROW.  A worker's rule file matters more than its
# provenance line, and these workers run seven hours unattended overnight and on
# cluster nodes with no Dropbox and no sync tree.  So the entire body is wrapped:
# a missing directory, a read-only tree, a full disk or a concurrent write
# degrades to a warning and the solve continues.  Callers include this file
# defensively too (a no-op fallback first, then `try include`), so a missing or
# half-synced copy of this file cannot take a worker down either.
#
# HONESTY.  `parent` is the rule this one actually came from — the previous rung
# of an elimination chain, the (d, p-2) rule a ladder rung continued, the
# degree-(p+2) rule a monotone downgrade copied, the uniform rule a Gaussian map
# started from.  A cold search that converged from scratch records "none" and
# says in `how` which tool and seed made it.  Never invent a parent: a wrong
# lineage is worse than a missing one, because make_appendices.jl walks these
# chains to decide which rules the paper must mark as derived.

# file   — the banked rule file (a bare name, or a path; the name is recorded)
# parent — the rule it came from, or "none"/nothing for a cold search
# how    — one line, plain text: the tool, what it did, the seed
# rules  — the bank directory whose lineage/ subdirectory receives the line;
#          pass the caller's own RULES so SYMQ_RULES_DIR test hooks are honored
function symq_lineage!(file, parent, how; rules = nothing)
    try
        oneline(s) = replace(string(s), r"[\t\r\n]+" => " ")
        dir  = rules === nothing ?
               get(ENV, "SYMQ_RULES_DIR", joinpath(get(ENV, "SYMQ_ROOT", pwd()), "rules")) : string(rules)
        host = replace(gethostname(), r"\..*" => "")
        par  = parent === nothing ? "none" : oneline(parent)
        occursin('/', par) && (par = basename(par))   # a path would break the chain lookup
        isempty(strip(par)) && (par = "none")
        line = join((basename(oneline(file)), par, oneline(how),
                     Libc.strftime("%Y-%m-%dT%H:%M:%S", time()), host), '\t') * "\n"
        ldir = joinpath(dir, "lineage")
        isdir(ldir) || mkpath(ldir)
        open(joinpath(ldir, "$(host).tsv"), "a") do io
            write(io, line)          # one write: concurrent appends stay whole lines
        end
    catch e
        try
            @warn "lineage line not written (the rule itself is banked)" file exception = e
        catch
        end
    end
    return nothing
end
