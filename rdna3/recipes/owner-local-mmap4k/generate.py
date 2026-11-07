#!/usr/bin/env python3
"""Copy a pinned four-GPU HIP profile into a private, opt-in mmap4K bundle."""

import argparse
import copy
import json
import os
from pathlib import Path
import shlex


MAX_PROFILE_BYTES = 1024 * 1024
ENV_DELTA = {
    "STRATA_ARENA_MMAP": "1",
    "STRATA_DP_PREFILL": "0",
    "STRATA_PREFILL_TIMING": "1",
}


def checked_path(value):
    """Reject symlinks in every component, including dangling final links."""
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("paths must not contain parent traversal")
    path = path.absolute()
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError("symlink paths are not supported")
    return path


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def invalid_constant(_):
    raise ValueError("non-finite JSON numbers are not supported")


def option(args, flag, required=True):
    positions = [i for i, value in enumerate(args) if value == flag]
    if any(value.startswith(flag + "=") for value in args):
        raise ValueError("use separate option and value tokens for " + flag)
    if not positions and not required:
        return None
    if len(positions) != 1:
        raise ValueError("expected exactly one " + flag)
    index = positions[0] + 1
    if index == len(args) or not args[index] or args[index].startswith("--"):
        raise ValueError("missing value for " + flag)
    return index


def is_sequence(value, expected):
    return value == ",".join(map(str, expected)) or (
        isinstance(value, list)
        and all(type(item) is int for item in value)
        and value == expected
    )


def validate(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("expected one flat profile object")
    if cfg.get("backend") != "hip":
        raise ValueError("backend must be hip")
    if not is_sequence(cfg.get("gpu"), [0, 1, 3, 2]):
        raise ValueError("gpu must name the pinned order 0,1,3,2")
    if not is_sequence(cfg.get("layer_split"), [12, 24, 36]):
        raise ValueError("layer_split must be 12,24,36")
    if type(cfg.get("parallel")) is not int or cfg["parallel"] != 4:
        raise ValueError("parallel must be 4")
    if cfg.get("effort_position", "start") != "start":
        raise ValueError("effort_position must be absent or start")
    if any(cfg.get(key) for key in ("user_args", "gpu_roles", "gpu_none", "split_skip_if_fits")):
        raise ValueError("profile has unsupported argument or GPU overrides")
    if not isinstance(cfg.get("cwd"), str) or not Path(cfg["cwd"]).is_absolute():
        raise ValueError("cwd must be the existing absolute working directory")
    if not checked_path(cfg["cwd"]).is_dir():
        raise ValueError("cwd must be the existing absolute working directory")
    args = cfg.get("args")
    if not isinstance(args, list) or not all(isinstance(arg, str) and "\0" not in arg for arg in args):
        raise ValueError("args must be a list of strings without NUL")
    for flag in ("--no-prefill-borrow", "--mmap-experts", "--peer-device", "--split-skip-if-fits",
                 "--gpu", "--split-device"):
        if any(arg == flag or arg.startswith(flag + "=") for arg in args):
            raise ValueError("unsupported existing option: " + flag)
    pack_index = option(args, "--pack")
    prefill_index = option(args, "--prefill")
    if args[prefill_index] != "auto":
        raise ValueError("input --prefill must be auto")
    for flag, expected in (("--layer-split", "12,24,36"), ("--batch", "4"), ("--slots", "4")):
        index = option(args, flag, required=False)
        if index is not None and args[index] != expected:
            raise ValueError("conflicting value for " + flag)
    if "--batch" in args and "--slots" in args:
        raise ValueError("ambiguous batch and slots options")
    env = cfg.get("env")
    if not isinstance(env, dict) or not all(
        isinstance(value, str) and "\0" not in value and "\0" not in key and "=" not in key
        for key, value in env.items()
    ):
        raise ValueError("env must be a string-to-string object")
    if any(key in env for key in ("LD_PRELOAD", "STRATA_CRASH_NUMERIC", "PWD")):
        raise ValueError("profile must not set observer variables or PWD, even empty")
    if "HIP_VISIBLE_DEVICES" in env and env["HIP_VISIBLE_DEVICES"] != "0,1,3,2":
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with the pinned GPU order")
    if any(key in env for key in ("ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")):
        raise ValueError("unsupported device visibility override")
    exe = cfg.get("exe")
    if not isinstance(exe, str) or not Path(exe).is_absolute() or any(ord(c) < 32 for c in exe):
        raise ValueError("exe must be an absolute native path without control characters")
    native = checked_path(exe)
    if not native.is_file() or not os.access(native, os.X_OK):
        raise ValueError("exe must be an existing executable file")
    return pack_index, prefill_index


def private_file(path, data, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(data)


def generate(profile, pack, output, *, ack_pwd_absent=False):
    if os.name != "posix":
        raise ValueError("this launcher requires a POSIX host")
    if not ack_pwd_absent:
        raise ValueError("explicit --ack-pwd-absent is required; see README prerequisites")
    source = checked_path(profile)
    padded_pack = checked_path(pack)
    destination = checked_path(output)
    if not source.is_file():
        raise ValueError("input must be an existing profile file")
    if not padded_pack.is_dir():
        raise ValueError("pack must be an already prepared, validated directory")
    if destination.exists():
        raise ValueError("output must be a new directory; refusing overwrite")
    if padded_pack == destination or padded_pack in destination.parents:
        raise ValueError("output must not be inside the read-only pack")
    if source.stat().st_size > MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 1 MiB")
    with source.open("rb") as stream:
        original = stream.read(MAX_PROFILE_BYTES + 1)
    if len(original) > MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 1 MiB")
    try:
        cfg = json.loads(original, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("input must be valid UTF JSON") from None
    pack_index, prefill_index = validate(cfg)
    candidate = copy.deepcopy(cfg)
    candidate["exe"] = str(destination / "native-exec.sh")
    candidate["args"][pack_index] = str(padded_pack)
    candidate["args"][prefill_index] = "4096"
    candidate["args"].append("--no-prefill-borrow")
    candidate["env"].update(ENV_DELTA)
    template = Path(__file__).with_name("native-exec.sh.in").read_text(encoding="ascii")
    launcher = template.replace("@NATIVE@", shlex.quote(cfg["exe"])).encode("utf-8")
    generated = (json.dumps(candidate, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    # mkdir is exclusive. The bundle is never activated or passed to a process.
    destination.mkdir(mode=0o700)
    os.chmod(destination, 0o700)
    private_file(destination / "profile.original.json", original, 0o600)
    private_file(destination / "native-exec.sh", launcher, 0o700)
    private_file(destination / "profile.json", generated, 0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="existing pinned flat JSON profile")
    parser.add_argument("--pack", required=True, help="already validated padded read-only pack directory")
    parser.add_argument("--output", required=True, help="new private bundle directory (parent must exist)")
    parser.add_argument("--ack-pwd-absent", action="store_true", help="confirm native PWD absence was validated on the working profile")
    args = parser.parse_args()
    try:
        generate(args.profile, args.pack, args.output, ack_pwd_absent=args.ack_pwd_absent)
    except ValueError as error:
        parser.exit(2, "error: " + str(error) + "\n")
    except OSError:
        parser.exit(2, "error: cannot access profile, pack, template or output; no activation performed\n")
    print("Private bundle created. Nothing started or activated. Keep the bundle out of Git.")


if __name__ == "__main__":
    main()
