"""CPU-only recipe checks. No native image, pack payload or service is run."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mmap_recipe", HERE / "generate.py")
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


@unittest.skipUnless(os.name == "posix", "POSIX launcher and permissions")
class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native = self.root / "native ' quoted $name; inert"
        self.native.write_bytes(b"Inert CPU fixture: never execute this file.\n")
        self.native.chmod(0o700)
        self.pack = self.root / "padded pack"
        self.pack.mkdir()
        self.source = self.root / "pinned.json"
        self.output = self.root / "candidate"
        self.cfg = {
            "exe": str(self.native), "cwd": str(self.root), "backend": "hip",
            "gpu": [0, 1, 3, 2], "layer_split": "12,24,36", "parallel": 4,
            "log": "original.log", "server_source": "original/frontend.py",
            "lib_dirs": ["original/libs"], "tokenizer": "original/tokenizer",
            "vision": {"exe": "original/vision", "env": {"KEEP": "vision"}},
            "sampling": {"temperature": 0.7, "seed": 42},
            "tools": {"template": "unchanged", "enabled": True},
            "extra": {"unknown": [None, False, "ASCII and \u4e2d\u6587"]},
            "env": {"KEEP": "unchanged", "STRATA_ARENA_MMAP": "0",
                    "STRATA_HIPBLASLT_TUNING": "original/table",
                    "STRATA_HIPBLASLT_TABLE": "original",
                    "STRATA_PF_STEP_SYNC": "0"},
            "args": ["--pack", "original/pack", "--prefill", "auto",
                     "--mtp", "original/mtp", "--max-context", "262144",
                     "--expert-cache", "24576", "--pool-workers", "16",
                     "--kv", "int8", "--spec", "4", "--vision",
                     "--conversation-cache-mib", "8192", "--vram-reserve-mib", "512"],
        }
        self.save()

    def save(self):
        self.source.write_text(json.dumps(self.cfg, indent=1) + "\n\n", encoding="utf-8")

    def generate(self, **kwargs):
        return recipe.generate(self.source, self.pack, self.output, ack_pwd_absent=True, **kwargs)

    def cli(self, *extra, ack=True):
        command = [sys.executable, "-B", str(HERE / "generate.py"),
                   "--profile", str(self.source), "--pack", str(self.pack),
                   "--output", str(self.output)]
        if ack:
            command.append("--ack-pwd-absent")
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=10)

    def prepare_native_stub(self):
        self.native.write_text(
            '#!' + sys.executable + ' -B\n'
            'import json, os, resource, sys\n'
            'keys = ("PWD", "STRATA_PREFILL_TIMING", "MALLOC_MMAP_THRESHOLD_",\n'
            '        "STRATA_HIPBLASLT_TUNING", "STRATA_HIPBLASLT_TABLE",\n'
            '        "STRATA_PF_STEP_SYNC", "STRATA_KFDQ", "KEEP")\n'
            'print(json.dumps({"argv": sys.argv[1:],\n'
            '                  "env": {k: os.environ[k] for k in keys if k in os.environ},\n'
            '                  "core": resource.getrlimit(resource.RLIMIT_CORE)}))\n',
            encoding="utf-8",
        )

    def stub_environment(self):
        env = dict(os.environ)
        for key in ("LD_PRELOAD", "STRATA_CRASH_NUMERIC", "STRATA_PREFILL_TIMING",
                    "MALLOC_MMAP_THRESHOLD_", "STRATA_KFDQ"):
            env.pop(key, None)
        env["PWD"] = str(self.root)
        return env

    def run_wrapper(self, args, env):
        return subprocess.run([str(self.output / "native-exec.sh"), *args],
                              env=env, capture_output=True, text=True, timeout=10)

    def rejected(self):
        original = self.source.read_bytes()
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertNotIn(str(self.native), result.stderr)
        self.assertEqual(result.stdout, "")

    def test_oversized_profile_is_rejected_before_open(self):
        self.source.write_bytes(b" " * (recipe.MAX_PROFILE_BYTES + 1))
        with patch.object(Path, "open", side_effect=AssertionError("oversized input read")):
            with self.assertRaisesRegex(ValueError, "^profile exceeds 1 MiB$"):
                self.generate()
        self.assertFalse(self.output.exists())

    def test_cli_delta_preserves_every_other_field_and_input_bytes(self):
        original = self.source.read_bytes()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = copy.deepcopy(self.cfg)
        expected["exe"] = str(self.output / "native-exec.sh")
        expected["args"][1] = str(self.pack)
        expected["args"][3] = "4096"
        expected["args"].append("--no-prefill-borrow")
        expected["env"].update(recipe.ENV_DELTA)
        actual = json.loads((self.output / "profile.json").read_bytes())
        self.assertEqual(actual, expected)
        self.assertNotIn("STRATA_PREFILL_TIMING", actual["env"])
        self.assertNotIn("MALLOC_MMAP_THRESHOLD_", actual["env"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual((self.output / "profile.original.json").read_bytes(), original)
        self.assertEqual(self.native.read_bytes(), b"Inert CPU fixture: never execute this file.\n")
        self.assertEqual(list(self.pack.iterdir()), [])
        self.assertEqual(set(p.name for p in self.output.iterdir()),
                         {"profile.json", "profile.original.json", "native-exec.sh"})

    def test_exact_wrapper_quoting_fail_closed_and_shell_syntax(self):
        self.generate()
        launcher = self.output / "native-exec.sh"
        expected = (
            '#!/bin/sh\n'
            '[ "${LD_PRELOAD+x}" != x ] || exit 125\n'
            '[ "${STRATA_CRASH_NUMERIC+x}" != x ] || exit 125\n'
            '[ "${1-}" = --serve ] || exit 125\n'
            'ulimit -S -c 0 || exit 125\n'
            'ulimit -H -c 0 || exit 125\n'
            'unset PWD STRATA_PREFILL_TIMING || exit 125\n'
            "MALLOC_MMAP_THRESHOLD_=131072 STRATA_HIPBLASLT_TUNING='' "
            "STRATA_HIPBLASLT_TABLE=off STRATA_PF_STEP_SYNC=1 exec "
            + shlex.quote(str(self.native)) + ' "$@"\n'
        )
        self.assertEqual(launcher.read_text(), expected)
        self.assertEqual(len(expected.splitlines()), 8)
        self.assertNotIn("NATIVE_BIN", expected)
        checked = subprocess.run(["/bin/sh", "-n", str(launcher)], capture_output=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_native_allocator_override_and_profiling_absence_preserve_shared_env(self):
        self.prepare_native_stub()
        args = ["--serve", "", "space and 'quote\"", "$inert;\\literal", "line\nbreak", "\u4e2d\u6587"]
        for shared_timing in (None, "1", "0", ""):
            for shared_allocator in (None, "262144"):
                with self.subTest(shared_timing=shared_timing, shared_allocator=shared_allocator):
                    for key, value in (("STRATA_PREFILL_TIMING", shared_timing),
                                       ("MALLOC_MMAP_THRESHOLD_", shared_allocator)):
                        self.cfg["env"].pop(key, None)
                        if value is not None:
                            self.cfg["env"][key] = value
                    self.save()
                    original = self.source.read_bytes()
                    self.output = self.root / ("candidate-" + str(shared_timing) + "-" + str(shared_allocator))
                    self.generate()
                    shared_env = json.loads((self.output / "profile.json").read_bytes())["env"]
                    expected = dict(self.cfg["env"], STRATA_ARENA_MMAP="1", STRATA_DP_PREFILL="0")
                    self.assertEqual(shared_env, expected)
                    inherited_cases = ((None, None), (None, "524288"), ("1", "524288"),
                                       ("0", "524288"), ("", "524288"))
                    for inherited_timing, inherited_allocator in inherited_cases:
                        with self.subTest(inherited_timing=inherited_timing,
                                          inherited_allocator=inherited_allocator):
                            env = self.stub_environment()
                            if inherited_allocator is not None:
                                env["MALLOC_MMAP_THRESHOLD_"] = inherited_allocator
                            if inherited_timing is not None:
                                env["STRATA_PREFILL_TIMING"] = inherited_timing
                                env["STRATA_KFDQ"] = "0"
                            env.update(shared_env)
                            before = dict(env)
                            result = self.run_wrapper(args, env)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            actual = json.loads(result.stdout)
                            child_env = {
                                "MALLOC_MMAP_THRESHOLD_": "131072",
                                "STRATA_HIPBLASLT_TUNING": "",
                                "STRATA_HIPBLASLT_TABLE": "off",
                                "STRATA_PF_STEP_SYNC": "1",
                                "KEEP": "unchanged",
                            }
                            if inherited_timing is not None:
                                child_env["STRATA_KFDQ"] = "0"
                            self.assertEqual(actual, {"argv": args, "env": child_env, "core": [0, 0]})
                            self.assertEqual(env, before)
                    self.assertEqual(self.source.read_bytes(), original)
                    self.assertEqual((self.output / "profile.original.json").read_bytes(), original)
                    self.assertEqual(json.loads((self.output / "profile.json").read_bytes())["env"], expected)

    def test_wrapper_guards_prevent_stub_execution(self):
        self.prepare_native_stub()
        self.generate()
        for key in ("LD_PRELOAD", "STRATA_CRASH_NUMERIC"):
            # Whitespace tests presence without asking the loader to load a library.
            for value in ("", " "):
                with self.subTest(key=key, value=value):
                    env = self.stub_environment()
                    env[key] = value
                    result = self.run_wrapper(["--serve"], env)
                    self.assertEqual(result.returncode, 125, result.stderr)
                    self.assertEqual(result.stdout, "")
        for args in ([], ["--version"], ["other", "--serve"]):
            with self.subTest(args=args):
                result = self.run_wrapper(args, self.stub_environment())
                self.assertEqual(result.returncode, 125, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_permissions_even_with_restrictive_umask(self):
        old = os.umask(0o777)
        try:
            self.generate()
        finally:
            os.umask(old)
        for name, mode in (("", 0o700), ("profile.json", 0o600),
                           ("profile.original.json", 0o600), ("native-exec.sh", 0o700)):
            self.assertEqual(stat.S_IMODE((self.output / name).stat().st_mode), mode)

    def test_no_overwrite_or_input_collision(self):
        self.generate()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(self.cli().returncode, 2)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)
        original = self.source.read_bytes()
        result = self.cli("--output", str(self.source))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.source.read_bytes(), original)

    def test_pwd_ack_required(self):
        self.assertEqual(self.cli(ack=False).returncode, 2)
        self.assertFalse(self.output.exists())

    def test_duplicate_keys_at_any_depth(self):
        for raw in ('{"backend":"hip","backend":"hip"}',
                    '{"vision":{"exe":"one","exe":"two"}}'):
            with self.subTest(raw=raw):
                self.source.write_text(raw)
                self.rejected()

    def test_wrong_json_shape_or_syntax(self):
        for raw in ('[]', 'null', '{', '{"env":NaN}', '{"env":Infinity}',
                    '{"profiles":{}}'):
            with self.subTest(raw=raw):
                self.source.write_text(raw)
                self.rejected()

    def test_conflicting_profile_fields(self):
        original = copy.deepcopy(self.cfg)
        for key, value in (("backend", "cuda"), ("gpu", [0, 1, 2, 3]),
                           ("gpu", [False, 1, 3, 2]), ("layer_split", "auto"),
                           ("parallel", "4"), ("parallel", True), ("parallel", 3),
                           ("args", "--pack original"), ("env", []),
                           ("env", {"KEEP": 1}), ("effort_position", "end"),
                           ("user_args", ["--prefill", "2048"]),
                           ("gpu_roles", {"0": "off"}), ("gpu_none", True),
                           ("split_skip_if_fits", True), ("cwd", "relative"),
                           ("exe", "relative/native"), ("exe", str(self.root / "missing"))):
            with self.subTest(key=key, value=value):
                self.cfg = copy.deepcopy(original)
                self.cfg[key] = value
                self.save()
                self.rejected()

    def test_observer_and_device_env_rejected_even_empty(self):
        for key in ("LD_PRELOAD", "STRATA_CRASH_NUMERIC", "PWD",
                    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            with self.subTest(key=key):
                self.cfg["env"][key] = ""
                self.save()
                self.rejected()
                del self.cfg["env"][key]

    def test_argument_ambiguity_and_conflicts(self):
        original = list(self.cfg["args"])
        variants = [original + ["--pack", "another"], original + ["--prefill", "auto"],
                    original + ["--pack=another"], original + ["--prefill=auto"],
                    original[2:], original[:2], ["--pack"],
                    ["--pack", "--prefill", "auto"],
                    ["--pack", "original", "--prefill", "4096"]]
        for flag in ("--no-prefill-borrow", "--mmap-experts", "--peer-device", "--split-skip-if-fits"):
            variants.extend([original + [flag], original + [flag + "=1"]])
        for flag in ("--layer-split", "--batch", "--slots"):
            variants.extend([original + [flag, "2"], original + [flag + "=4"]])
        variants.append(original + ["--batch", "4", "--slots", "4"])
        for args in variants:
            with self.subTest(args=args):
                self.cfg["args"] = args
                self.save()
                self.rejected()

    def test_native_device_overrides_rejected_before_output_creation(self):
        original = list(self.cfg["args"])
        for flag, conflicting, pinned in (("--gpu", "0,1,2,3", "0,1,3,2"),
                                          ("--split-device", "3,2,1", "1,3,2")):
            variants = ([flag, conflicting], [flag + "=" + conflicting],
                        [flag, pinned], [flag + "=" + pinned],
                        [flag, pinned, flag, conflicting],
                        [flag + "=" + pinned, flag + "=" + conflicting],
                        [flag, pinned, flag + "=" + conflicting])
            for suffix in variants:
                with self.subTest(flag=flag, suffix=suffix):
                    self.cfg["args"] = original + suffix
                    self.save()
                    self.rejected()

    def test_supported_equivalent_field_shapes_are_not_rewritten(self):
        self.cfg["gpu"] = "0,1,3,2"
        self.cfg["layer_split"] = [12, 24, 36]
        self.cfg["effort_position"] = "start"
        self.cfg["env"]["HIP_VISIBLE_DEVICES"] = "0,1,3,2"
        self.cfg["args"] += ["--layer-split", "12,24,36", "--batch", "4"]
        self.save()
        self.generate()
        actual = json.loads((self.output / "profile.json").read_bytes())
        for key in ("gpu", "layer_split", "parallel", "effort_position"):
            self.assertEqual(actual[key], self.cfg[key])
        self.assertEqual(actual["args"][-5:], ["--layer-split", "12,24,36", "--batch", "4", "--no-prefill-borrow"])

    def test_symlink_paths_including_ancestors_and_dangling_output(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for changes in ({"--profile": str(alias / self.source.name)},
                        {"--pack": str(alias / self.pack.name)},
                        {"--output": str(alias / "candidate")}):
            with self.subTest(changes=changes):
                flat = [item for pair in changes.items() for item in pair]
                self.assertEqual(self.cli(*flat).returncode, 2)
                self.assertFalse(self.output.exists())
        self.output.symlink_to(self.root / "missing")
        self.assertEqual(self.cli().returncode, 2)
        self.assertTrue(self.output.is_symlink())
        self.output.unlink()
        self.cfg["exe"] = str(alias / self.native.name)
        self.save()
        self.rejected()

    def test_no_output_inside_pack_or_missing_parent(self):
        for output in (self.pack / "candidate", self.root / "missing" / "candidate"):
            result = self.cli("--output", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
        self.assertEqual(list(self.pack.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
