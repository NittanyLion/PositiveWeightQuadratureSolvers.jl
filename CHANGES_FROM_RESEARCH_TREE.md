# Changes from the research tree

Every file in `src/` and `scripts/` was copied from the research tree of the
cubature-rule project:

* `src/*.jl` and `scripts/*.jl` except `compare_rules.jl` — from
  `project/julia/` of the project's sync folder;
* `scripts/compare_rules.jl` — from `publish/shared/`;
* `test/data/hermite_d2_p5_q3_n7.csv` and `test/data/legendre_d2_p5_q3_n7.csv` —
  from `publish/gh/rules/` and `publish/le/rules/`, the Zenodo deposits, byte for
  byte including their comment headers.

**No algorithmic change was made to any copied file.**  The only edits replace
machine-specific absolute paths.  `src/` is byte-identical to the research tree;
all edits are in `scripts/`.  Newly written files (`src/PositiveWeightQuadratureSolvers.jl`,
`test/runtests.jl`, `Project.toml`, `README.md`, `LICENSE`, `CITATION.cff`,
`.gitignore`, this file) are not listed below.

## The three substitutions

| research tree | here |
| --- | --- |
| `include("/home/joris/oldDesignedQuadrature/julia/<Module>.jl")` | `include(joinpath(@__DIR__, "..", "src", "<Module>.jl"))` |
| `const J = "/home/joris/oldDesignedQuadrature/julia"` (the project root, parent of `rules/` and `symq/`) | `const J = get(ENV, "SYMQ_ROOT", pwd())` |
| `"/home/joris/Dropbox/oldDesignedQuadrature-sync/comms[/donors]"` (the cross-machine donor exchange) | `get(ENV, "SYMQ_COMMS_DIR", joinpath(J, "comms"))`, `joinpath(…, "donors")` |

`RULES` and `SYMQ` keep their meaning — the rule bank and the sidecar/log
directory — and now read `SYMQ_RULES_DIR` / `SYMQ_SIDECAR_DIR` in **every**
driver, with `joinpath(J, "rules")` and `joinpath(J, "symq")` as defaults.  Most
drivers already honored those two variables; the ones listed as "ENV override
added" below hard-wired `joinpath(J, …)` and now match the rest.  Nothing else
about how a driver reads or writes the bank changed.

## Per file

### `scripts/`

| file | what was replaced |
| --- | --- |
| `bounds.jl` | 1 absolute module include → `@__DIR__`/`../src` |
| `classify_methods.jl` | `joinpath(@__DIR__, "rules")` and `joinpath(@__DIR__, "symq", …)` (4 sites) → new `J` / `RULES` / `SYMQ` constants read from `SYMQ_ROOT` / `SYMQ_RULES_DIR` / `SYMQ_SIDECAR_DIR`, defaulting to `pwd()`. The research copy sat *inside* the bank's parent directory, so `@__DIR__` was the project root there; in `scripts/` it is not. |
| `compare_rules.jl` | `const ROOT = "/dropbox/Dropbox/oldDesignedQuadrature-sync"` → `get(ENV, "PWQS_RESEARCH_ROOT", pwd())`; `RULES` now also honors `SYMQ_RULES_DIR`. The competitor node files it reads (`litcheck/festa_sommariva`, `litcheck/stroud_m`, `symq/vz_rules`, `xg_cube_p7_n26.csv`) are **not** shipped — see the README. |
| `dw_warm.jl` | 2 module includes; `const J` |
| `monotone_bank.jl` | `const J = @__DIR__` → `get(ENV, "SYMQ_ROOT", pwd())` (same reason as `classify_methods.jl`) |
| `polish_mp.jl` | `const JL = @__DIR__` → `joinpath(@__DIR__, "..", "src")`, which is what its one `include(joinpath(JL, "DesignedQuadrature.jl"))` needs |
| `spectral_start.jl` | 3 module includes; `const J`; ENV override added for `RULES` |
| `symq_enum.jl` | 2 module includes; `const J`; `COMMS`; ENV overrides added for `RULES`, `SYMQ` |
| `symq_freeelim.jl` | 2 module includes; `const J` |
| `symq_ico.jl` | 2 module includes; `const J`; ENV overrides added for `RULES`, `SYMQ` |
| `symq_ladder.jl` | 3 module includes; `const J`; `DONORS` default |
| `symq_leg_product.jl` | 3 module includes; `const J`; `DONORS` default |
| `symq_lineage.jl` | its own `RULES` fallback `joinpath(@__DIR__, "rules")` → `joinpath(get(ENV, "SYMQ_ROOT", pwd()), "rules")` (same reason as `classify_methods.jl`) |
| `symq_pairelim.jl` | 2 module includes; `const J` |
| `symq_rot2.jl` | 1 module include; `const J` |
| `symq_run.jl` | 2 module includes; `const J`; `COMMS`; ENV overrides added for `RULES`, `SYMQ` |
| `symq_sc.jl` | 2 module includes; `const J`; `COMMS`; ENV overrides added for `RULES`, `SYMQ` |
| `symq_simplex.jl` | 2 module includes; `const J`; `DONORS` default |
| `tensor_fill.jl` | 1 module include; `const J` |
| `update_legendre.jl` | 2 module includes; `const J` |

### `src/`

No edits.  In particular the modules' own `include(joinpath(@__DIR__, …))` lines
were already relative:

* `DesignedQuadratureV2.jl` includes `DesignedQuadrature.jl` at its own top
  level, so including V2 defines both modules;
* `SymmetricDQ.jl`, `PermCentralDQ.jl` and `LaguerreDQ.jl` each include
  `WeightTransforms.jl` *inside* their own module, so each carries a nested
  copy — left exactly as it was.

`bounds.jl` is listed in the project's notes among the modules but is a
command-line script, so it lives in `scripts/`.

## Not carried over

Comment text in the copied files still refers to the campaign's Dropbox donor
exchange, the three machines, the clusters and the hourly timers.  None of that
machinery is in this repository; the comments were left intact rather than
rewritten, because they record why a code path exists.
