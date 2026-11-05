"""Power helper tests use temporary files only, never real GPUs or root commands."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from serve import idle_power_helper as helper

# Synthetic PCI addresses, used only below a temporary fake sysfs root.
DEVICES = ["0000:01:00.0", "0000:02:00.0"]


def monitor_child(policies, read_fd, write_fd, lease):
    os.close(write_fd)
    try:
        helper.rollback_monitor(policies, read_fd, lease_s=lease)
    finally:
        os.close(read_fd)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only policy helper")
class Policies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for device in DEVICES:
            d = self.root / device
            d.mkdir()
            (d / "vendor").write_text("0x1002\n")
            (d / "power_dpm_force_performance_level").write_text("auto\n")
        self.policies = helper.Policies(DEVICES, root=self.root)

    def session(self, foreign=False):
        return helper.Session(self.policies, os.getuid(), foreign=lambda owner, uid: foreign)

    def test_low_and_active_preserve_other_files(self):
        extra = self.root / "unrelated"
        extra.write_text("unchanged")
        session = self.session()
        self.assertEqual(session.request({"command": "idle", "owner": 123})["state"], "low")
        self.assertEqual([self.policies.read(d) for d in DEVICES], ["low", "low"])
        self.assertTrue(session.request({"command": "active"})["normal"])
        self.assertTrue(self.policies.normal())
        self.assertEqual(extra.read_text(), "unchanged")
        self.assertEqual(self.policies.owned, set())

    def test_another_client_prevents_low(self):
        session = self.session(foreign=True)
        self.assertTrue(session.request({"command": "idle", "owner": 123})["normal"])
        self.assertTrue(self.policies.normal())
        self.assertEqual(self.policies.owned, set())

    def test_another_client_appearing_restores_low(self):
        session = self.session()
        session.request({"command": "idle", "owner": 123})
        session.foreign = lambda owner, uid: True
        session.heartbeat()
        self.assertTrue(self.policies.normal())

    def test_partial_low_failure_rolls_back_every_owned_device(self):
        session = self.session()
        original = Path.write_text
        def write(path, data, *args, **kwargs):
            if path == self.policies.path(DEVICES[1]) and data == "low\n":
                raise OSError("test device failure")
            return original(path, data, *args, **kwargs)
        with mock.patch.object(Path, "write_text", write):
            answer = session.request({"command": "idle", "owner": 123})
        self.assertTrue(answer["disabled"])
        self.assertTrue(answer["normal"])
        self.assertTrue(self.policies.normal())
        self.assertEqual(self.policies.owned, set())

    def test_external_manual_policy_is_not_overwritten(self):
        session = self.session()
        session.request({"command": "idle", "owner": 123})
        self.policies.path(DEVICES[0]).write_text("manual\n")
        answer = session.request({"command": "active"})
        self.assertFalse(answer["normal"])
        self.assertTrue(answer["disabled"])
        self.assertEqual(self.policies.read(DEVICES[0]), "manual")
        self.assertEqual(self.policies.read(DEVICES[1]), "auto")

    def test_bad_command_disables_and_restores(self):
        session = self.session()
        session.request({"command": "idle", "owner": 123})
        answer = session.request({"command": "write", "path": "/anything"})
        self.assertTrue(answer["normal"])
        self.assertTrue(answer["disabled"])
        self.assertTrue(self.policies.normal())

    def test_requires_auto_at_start(self):
        self.policies.path(DEVICES[0]).write_text("manual\n")
        with self.assertRaises(ValueError):
            helper.Policies(DEVICES, root=self.root)
        self.assertEqual(self.policies.read(DEVICES[0]), "manual")

    def test_device_allowlist_rejects_paths_duplicates_and_non_amd(self):
        for devices in ([], ["../../etc"], [DEVICES[0], DEVICES[0]], ["0000:01:00.9"]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                helper.Policies(devices, root=self.root)
        (self.root / DEVICES[0] / "vendor").write_text("0x10de\n")
        with self.assertRaises(ValueError):
            helper.Policies(DEVICES, root=self.root)
        with self.assertRaises(ValueError):
            self.policies.path("../../etc")

    def test_foreign_owner_metadata_is_conservative(self):
        kfd, proc = self.root / "kfd", self.root / "proc"
        kfd.mkdir()
        proc.mkdir()
        (kfd / "123").mkdir()
        (proc / "123").mkdir()
        # The excluded directory must never be treated as a compute client or descended into.
        (kfd / ".terraform" / "456").mkdir(parents=True)
        self.assertFalse(helper.foreign_compute(123, os.getuid(), kfd, proc))
        (kfd / "456").mkdir()
        self.assertTrue(helper.foreign_compute(123, os.getuid(), kfd, proc))
        self.assertTrue(helper.foreign_compute(None, os.getuid(), kfd, proc))
        self.assertTrue(helper.foreign_compute(True, os.getuid(), kfd, proc))
        self.assertTrue(helper.foreign_compute(123, os.getuid() + 1, kfd, proc))
        self.assertTrue(helper.foreign_compute(123, os.getuid(), self.root / "missing", proc))

    def check_guard(self, eof):
        if not hasattr(os, "fork"):
            self.skipTest("Linux-only rollback helper")
        read_fd, write_fd = os.pipe()
        child = multiprocessing.get_context("fork").Process(
            target=monitor_child, args=(self.policies, read_fd, write_fd, .1))
        child.start()
        os.close(read_fd)
        try:
            self.policies.lower()
            os.write(write_fd, (json.dumps({"owned": DEVICES}) + "\n").encode())
            if eof:
                os.close(write_fd)
                write_fd = None
            deadline = time.monotonic() + 2
            while not self.policies.normal() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.policies.normal(), "independent rollback must restore the fake policies")
        finally:
            if write_fd is not None:
                os.close(write_fd)
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
        self.assertEqual(child.exitcode, 0)

    def test_guard_restores_after_control_pipe_closes(self):
        self.check_guard(eof=True)

    def test_guard_restores_after_lease_expires(self):
        self.check_guard(eof=False)

    def rpc_client(self):
        from serve.idle_power import PolicyClient
        code = r'''import json, os, sys
from pathlib import Path
from serve import idle_power_helper as h
original = h.Policies
h.Policies = lambda devices: original(devices, root=Path(sys.argv[1]))
h.foreign_compute = lambda owner, uid: False
monitor = h.rollback_monitor
h.rollback_monitor = lambda policies, fd: monitor(policies, fd, lease_s=.5)
raise SystemExit(h.serve_helper({'devices': json.loads(sys.argv[2]), 'uid': os.getuid()}))
'''
        client = PolicyClient([sys.executable, '-B', '-c', code, str(self.root), json.dumps(DEVICES)], say=mock.Mock())
        self.addCleanup(client.close)
        return client

    def test_real_stdio_protocol_uses_fake_policies(self):
        client = self.rpc_client()
        self.assertEqual(client.idle(os.getpid())["state"], "low")
        self.assertFalse(self.policies.normal())
        self.assertTrue(client.wake()["normal"])
        self.assertTrue(self.policies.normal())
        client.close()
        self.assertEqual(client.proc.returncode, 0)

    def test_killed_helper_leaves_independent_rollback_alive(self):
        client = self.rpc_client()
        client.idle(os.getpid())
        client.proc.kill()
        client.proc.wait(timeout=2)
        deadline = time.monotonic() + 2
        while not self.policies.normal() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(self.policies.normal())

    def test_stalled_helper_cannot_keep_an_unrenewed_low_lease(self):
        client = self.rpc_client()
        client.idle(os.getpid())
        os.kill(client.proc.pid, signal.SIGSTOP)
        try:
            deadline = time.monotonic() + 2
            while not self.policies.normal() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.policies.normal())
        finally:
            os.kill(client.proc.pid, signal.SIGCONT)
        self.assertTrue(client.wake()["normal"])

    def test_uninstalled_helper_refuses_privileged_use(self):
        with self.assertRaises(ValueError):
            helper.secure_file(Path(__file__))
        with mock.patch.object(helper.os, "geteuid", return_value=1000), mock.patch.object(helper.sys, "argv", ["helper"]):
            with mock.patch.object(helper, "Policies") as policies:
                self.assertEqual(helper.main(), 2)
                policies.assert_not_called()


if __name__ == "__main__":
    unittest.main()
