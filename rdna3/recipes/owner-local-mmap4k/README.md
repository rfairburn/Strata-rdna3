# Experimental four-GPU owner-local mmap4K recipe

This is an opt-in configuration recipe for an **already working, pinned four-GPU
HIP profile**. It makes a new private profile and a native-only shell launcher.
It does not start anything, change an existing profile, build an engine, deploy
this checkout's frontend, download a model, or prepare a pack. The normal path,
existing configs and defaults are unchanged.

This is a narrowly tested workaround, not a stability fix or a claim that mmap is
faster by default. Keep the pinned profile, native binary, frontend and pack for
rollback. Do not use this recipe to replace a working frontend with a newer one.

## Prerequisites

- A pinned Linux/POSIX HIP setup already working with GPU order `0,1,3,2`, layer
  cuts `12,24,36`, and `parallel: 4`. The generator accepts one flat JSON profile,
  at most 1 MiB, with string `exe` and `cwd` absolute paths, a string-list `args`, and a
  string-to-string `env` object. The native executable and cwd must already
  exist. `gpu` and `layer_split` can be the comma-separated strings or integer
  lists. Other layouts are not inferred or converted.
- The qualified native binary had SHA256
  `5720b9747e2c46b8f014ffefff2b36caa353d752c9342cd3f140a0aa92400db1`.
  Pin and check the binary separately. This tool neither hashes nor runs it and
  cannot establish that another build is compatible.
- An **already prepared, validated, read-only padded pack**, with the correct
  expert stride and tail padding for this native mmap path. Reuse that pack;
  no large copy or repack is needed. A directory's existence is not proof of
  padding, payload equivalence or suitability. The generator only checks its
  path. It never reads, hashes, copies, changes, creates or appends to weights
  or bank payloads. Pack preparation and validation are outside this recipe.
- The working profile must have exactly one separate-token `--pack VALUE` and
  one `--prefill auto`, and no `--no-prefill-borrow`. `--mmap-experts` is a
  different FileExpertSource path and is rejected. Conflicting layer/batch
  flags, GPU-role/device overrides (including native `--gpu` and
  `--split-device` in separate-token or `--flag=value` form) and nonempty
  `user_args` are rejected rather than guessed. Duplicate JSON keys at any depth
  are rejected.
- `effort_position` must be absent or `"start"`. An older frontend can otherwise
  probe the executable for effort-at-end support; this wrapper only accepts
  `--serve` as its first argument.
- **Native PWD absence must already have been validated for the working setup.**
  The required `--ack-pwd-absent` acknowledges that prerequisite. It is not a
  general policy to unset arbitrary environment variables. The generator cannot
  inspect or attest the daemon's actual environment or core limits.
- Neither `LD_PRELOAD` nor `STRATA_CRASH_NUMERIC` may be set, even to an empty
  value, in the profile or inherited native environment. `PWD` must not be set
  in profile `env`. The wrapper rejects inherited observer variables and unsets
  the shell's `PWD` before exec. There is no crash observer or private library.

No VRAM admission guarantee is made for other hardware or profiles. In
particular, this recipe does not force context, MTP, cache or memory budgets to
match the measured host.

## Generate, inspect, then activate separately

Python 3.11 and its standard library are enough. Use private absolute paths in
place of these placeholders; the output's parent directory must already exist.

```sh
python3 -B rdna3/recipes/owner-local-mmap4k/generate.py \
  --profile /ABS/PRIVATE/pinned-profile.json \
  --pack /ABS/READONLY/validated-padded-pack \
  --output /ABS/PRIVATE/new-mmap4k-bundle \
  --ack-pwd-absent
```

The output directory must be new. Existing outputs, input collisions, symlinks
in path components, parent traversal, and outputs inside the pack are refused.
No profile, environment or argument values are printed. An I/O failure can leave
an incomplete private bundle; do not activate it or reuse its name.

The bundle contains:

| File | Purpose | Mode |
|---|---|---|
| `profile.original.json` | Exact original input bytes, for pinned rollback | `0600` |
| `profile.json` | Copied profile with only the changes below | `0600` |
| `native-exec.sh` | Native launcher, with a statically shell-quoted executable path | `0700` |

The directory is `0700`. **Never commit generated bundles.** Both profiles can
contain private paths, credentials and environment fields copied from the input.
Keep them in private storage, outside a source checkout. Do not publish diffs or
logs containing those values.

Inspect the generated copy privately before using your existing manager to
activate it. Keep that manager's original frontend, launch options and working
cwd. The generated profile preserves `server_source` when present; it does not
choose or install a frontend if the input does not name one. Do not use the
repository's general setup/build/run scripts to activate this recipe.

### Exact configuration delta

- `exe` becomes the new bundle's absolute `native-exec.sh` path.
- The sole `--pack` value becomes the supplied padded pack path.
- The sole `--prefill auto` becomes `--prefill 4096`.
- Exactly one `--no-prefill-borrow` is appended.
- Only these profile environment values are added/replaced:
  `STRATA_ARENA_MMAP=1`, `STRATA_DP_PREFILL=0`, `STRATA_PREFILL_TIMING=1`.

Everything else is preserved as JSON data, including unknown fields: original
log, cwd, frontend, GPU order, parallelism, cuts, MTP/context/cache/thread/math
settings, sampling, tools, templates, vision and budgets. The copy's whitespace
can change; the rollback copy is byte-identical. No model registration or alias
is created. No source stager, cuts, library paths or other flags are rewritten.

The [eight-line launcher template](native-exec.sh.in) refuses observer variables
and non-serve probes, sets both soft and hard child core limits to zero, unsets
`PWD`, then execs the original absolute native path with unchanged arguments.
It sets **only for that native exec**:

- `STRATA_HIPBLASLT_TUNING=''`: overrides an inherited tuning-table path.
- `STRATA_HIPBLASLT_TABLE=off`: selects the table-off/null state. Both Lt fields
  are needed; setting just one is not equivalent.
- `STRATA_PF_STEP_SYNC=1`: the existing named-stream step synchronization path.

These three values are not changed in profile `env`, which can be shared with
vision. They do not change the native image, parent limits, system core policy
or sysctls. There is no environment-sourced `NATIVE_BIN`, preload, crash handler,
service, HTTP/API call, process manager or automatic fallback here.

## Host validation and rollback

Generation and CPU CI are not runtime qualification. After separate activation
through the existing manager, validate on the target host before relying on it:

1. Confirm the pinned native image/frontend, real child environment, PWD absence,
   child core limits and all four intended GPUs, not just the JSON fields.
2. Require native `LOADED`, manager `READY`, full owned expert residency
   `24576` total / `6144` per owner, no prefill borrowing and no fallback. A pack
   path or successful generator exit cannot establish these conditions.
3. Check the real four-slot workload, long fresh prefills, deep cache replay,
   park/restore, vision and required features with the unchanged manager. Inspect
   actual logs privately; the recipe deliberately preserves the original log.

The generator never activates or restores a profile. For rollback, use the
existing manager to restore the original pinned profile from
`profile.original.json`, with its original native/frontend/pack. Keep that
rollback available even after one successful run.

## Observed bounds, not a general performance promise

The qualified host was a Threadripper PRO 3955WX (16 cores / 32 threads), 128 GiB
RAM, four RX 7900 XTX 24 GiB cards (`gfx1100`), ROCm core 10.0. The native image
above used 4K prefill, DP prefill off, no borrowing, full owned residency,
native Lt bypass, step sync and prefill timing. Its original four slots, MTP and
262144-token context were retained, not imposed by this generator.

One full instrumented eight-case run and one full run without the crash observer
completed normally. In the no-observer run:

| Case | Observation |
|---|---|
| 83,629 fresh prompt tokens, 64 output tokens | Prefill 10,728.2 ms |
| 167,216 fresh prompt tokens, 64 output tokens | Prefill 20,981.5 ms |
| Deep replay | 167,211 cached tokens plus 5 fresh tokens |
| Four-slot cohort: 20/22/24/26K prompts, 256 outputs each | All completed normally, about 19.1 s |
| Post-cohort deep replay / park-restore | 99.997% reuse; first token 0.770 s |

Four occupied native slots were observed (three decoding and one reading).
This does not guarantee smooth simultaneous four-GPU decode: time to first token
varied while later prefills ran. There was no matched same-day pinned comparison,
so these are not measured speedups or overhead percentages.

Earlier unsynchronized SIG11 and cohort SIG6 failures remain unresolved. Step
waits may mask races. One full observer-off success does not prove broad
repeatability, a root-cause fix, quality, numerical/bit parity, or a full
262144-token workload in all four slots. Source math is unchanged, but fallback
rounding and algorithm choices have not been proven bit-identical. Preserve the
pinned rollback; this is not a new default recommendation.

## CPU checks

```sh
python3 -B -m unittest discover -s rdna3/recipes/owner-local-mmap4k -p 'test_*.py' -v
```

Tests use synthetic profiles and temporary directories. They check the complete
allowed delta, byte-preserved backup, quoting, fail-closed wrapper text, shell
syntax, permissions, rejection cases and CLI acknowledgement. They never execute
the native fixture, read a model payload, use a GPU/network/API or inspect a live
process. The path-scoped `Owner-local mmap recipe` workflow runs the same tests on
Python 3.11, without installing dependencies. It does not validate host behavior.
