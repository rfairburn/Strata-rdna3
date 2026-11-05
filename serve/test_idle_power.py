"""Idle power state-machine tests. No GPU, sysfs writes or privileged process is used."""
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import idle_power


class Client:
    def __init__(self):
        self.state = "auto"
        self.calls = []
        self.fail_wake = False
        self.fail_idle = False
        self.foreign = False
        self.closed = False
        self.state_callback = None

    def set_state_callback(self, callback):
        self.state_callback = callback

    def wake(self):
        self.calls.append("auto")
        if self.fail_wake:
            raise OSError("no helper")
        self.state = "auto"
        return {"state": "auto", "normal": True}

    def idle(self, owner):
        self.calls.append(("idle", owner))
        if self.fail_idle:
            self.state = "low"
            raise OSError("partial transition")
        if not self.foreign:
            self.state = "low"
        return {"state": self.state, "normal": self.state == "auto"}

    def verify_normal(self):
        return self.state == "auto"

    def close(self):
        self.closed = True


class Guard(unittest.TestCase):
    def make(self, **kwargs):
        self.now = 0.0
        self.client = Client()
        self.log = mock.Mock()
        guard = idle_power.IdlePower(self.client, owner=lambda: 123, clock=lambda: self.now,
                                     say=self.log, start_thread=False, **kwargs)
        self.addCleanup(guard.close)
        return guard

    def test_idle_deadline_and_wake_before_work(self):
        guard = self.make()
        self.now = 29.9
        self.assertFalse(guard.try_idle())
        self.now = 30
        self.assertTrue(guard.try_idle())
        self.assertEqual(self.client.state, "low")
        with guard.activity():
            self.assertEqual(self.client.state, "auto")
            self.assertEqual(guard.count, 1)
        self.assertEqual(guard.count, 0)
        self.now = 59.9
        self.assertFalse(guard.try_idle())
        self.now = 60
        self.assertTrue(guard.try_idle())

    def test_four_requests_remain_concurrent(self):
        guard = self.make()
        barrier = threading.Barrier(4)
        failures = []
        def work():
            try:
                with guard.activity():
                    barrier.wait(timeout=2)
            except BaseException as error:
                failures.append(error)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(guard.count, 0)

    def test_one_remaining_request_prevents_low(self):
        guard = self.make()
        guard.acquire()
        guard.acquire()
        guard.release()
        self.now = 100
        self.assertFalse(guard.try_idle())
        guard.release()
        self.now = 129
        self.assertFalse(guard.try_idle())
        self.now = 130
        self.assertTrue(guard.try_idle())

    def test_busy_native_drain_defers_idle(self):
        busy = [True]
        guard = self.make(busy=lambda: busy[0])
        self.now = 30
        self.assertFalse(guard.try_idle())
        busy[0] = False
        self.now = 59
        self.assertFalse(guard.try_idle())
        self.now = 60
        self.assertTrue(guard.try_idle())

    def test_foreign_compute_skips_low(self):
        guard = self.make()
        self.client.foreign = True
        self.now = 30
        self.assertFalse(guard.try_idle())
        self.assertEqual(self.client.state, "auto")
        self.assertFalse(guard.disabled)

    def test_external_restore_rearms_idle_without_inference(self):
        guard = self.make()
        self.now = 30
        self.assertTrue(guard.try_idle())
        self.client.state = "auto"
        self.client.foreign = True
        self.now = 40
        self.client.state_callback()
        self.assertTrue(guard.normal)
        self.now = 70
        self.assertFalse(guard.try_idle())
        self.client.foreign = False
        self.now = 100
        self.assertTrue(guard.try_idle())
        self.assertFalse(guard.disabled)

    def test_delayed_restore_notification_cannot_claim_low_is_normal(self):
        guard = self.make()
        self.now = 30
        guard.try_idle()
        self.client.state_callback()
        self.assertEqual(self.client.state, "auto")
        self.assertTrue(guard.normal)
        guard.acquire()
        self.now = 100
        self.client.state_callback()
        self.assertFalse(guard.try_idle())
        guard.release()

    def test_external_restore_wakes_the_real_idle_loop(self):
        client = Client()
        lowered = threading.Event()
        original = client.idle
        def idle(owner):
            answer = original(owner)
            if answer["state"] == "low":
                lowered.set()
            return answer
        client.idle = idle
        guard = idle_power.IdlePower(client, idle_s=.05, owner=lambda: 123, say=mock.Mock())
        self.addCleanup(guard.close)
        self.assertTrue(lowered.wait(timeout=2))
        with guard.cv:
            lowered.clear()
            client.state = "auto"
            client.state_callback()
        self.assertTrue(lowered.wait(timeout=2))
        self.assertEqual(guard.count, 0)
        self.assertFalse(guard.disabled)

    def test_helper_stderr_notification_survives_early_delivery(self):
        client = idle_power.PolicyClient.__new__(idle_power.PolicyClient)
        client.proc = SimpleNamespace(stderr=[b'idle power: another/unknown compute owner; {"normal":true}\n'])
        client.say = mock.Mock()
        client.state_changed = None
        client.state_pending = threading.Event()
        client._errors()
        callback = mock.Mock()
        client.set_state_callback(callback)
        callback.assert_called_once_with()
        self.assertFalse(client.state_pending.is_set())
        client.set_state_callback(None)

    def test_restore_failure_warns_but_never_rejects_work(self):
        guard = self.make()
        self.now = 30
        guard.try_idle()
        self.client.fail_wake = True
        with guard.activity():
            self.assertEqual(guard.count, 1)
            self.assertEqual(self.client.state, "low")
        self.assertTrue(guard.disabled)
        self.assertTrue(any("NORMAL POWER UNVERIFIED" in c.args[0] for c in self.log.call_args_list))
        self.now = 100
        self.assertFalse(guard.try_idle())
        with guard.activity():
            pass

    def test_verify_failure_also_cannot_reject_work(self):
        guard = self.make()
        self.now = 30
        guard.try_idle()
        self.client.fail_wake = True
        self.client.verify_normal = mock.Mock(side_effect=OSError("unreadable"))
        with guard.activity():
            pass
        self.assertTrue(guard.disabled)

    def test_partial_low_failure_attempts_restoration(self):
        guard = self.make()
        self.client.fail_idle = True
        self.now = 30
        self.assertFalse(guard.try_idle())
        self.assertTrue(guard.disabled)
        self.assertEqual(self.client.state, "auto")
        with guard.activity():
            pass

    def test_cancel_exception_releases_request_lease(self):
        guard = self.make()
        with self.assertRaises(ValueError):
            with guard.activity():
                raise ValueError("client cancelled")
        self.assertEqual(guard.count, 0)
        self.now = 30
        self.assertTrue(guard.try_idle())

    def test_inhibit_restores_then_stays_disabled(self):
        guard = self.make()
        self.now = 30
        guard.try_idle()
        guard.inhibit("native cancellation did not drain")
        self.assertEqual(self.client.state, "auto")
        self.assertTrue(guard.disabled)
        self.now = 100
        self.assertFalse(guard.try_idle())

    def test_close_restores_and_is_idempotent(self):
        guard = self.make()
        self.now = 30
        guard.try_idle()
        guard.close()
        self.assertEqual(self.client.state, "auto")
        self.assertTrue(self.client.closed)
        count = len(self.client.calls)
        guard.close()
        self.assertEqual(len(self.client.calls), count)

    def test_admission_cannot_cross_a_low_transition(self):
        guard = self.make()
        entered, finish, worked = threading.Event(), threading.Event(), threading.Event()
        original = self.client.idle
        def lower(owner):
            entered.set()
            if not finish.wait(timeout=2):
                raise TimeoutError("test transition")
            return original(owner)
        self.client.idle = lower
        self.now = 30
        idle_thread = threading.Thread(target=guard.try_idle)
        def request():
            with guard.activity():
                self.assertEqual(self.client.state, "auto")
                worked.set()
        idle_thread.start()
        self.assertTrue(entered.wait(timeout=2))
        request_thread = threading.Thread(target=request)
        request_thread.start()
        self.assertFalse(worked.wait(timeout=.05))
        finish.set()
        idle_thread.join(timeout=2)
        request_thread.join(timeout=2)
        self.assertTrue(worked.is_set())
        self.assertEqual(self.client.state, "auto")

    def test_helper_signal_permission_errors_do_not_escape_cleanup(self):
        client = idle_power.PolicyClient.__new__(idle_power.PolicyClient)
        proc = SimpleNamespace(stdin=mock.Mock(), stdout=mock.Mock(), stderr=mock.Mock(),
                               poll=lambda: None, terminate=mock.Mock(side_effect=PermissionError("root helper")),
                               kill=mock.Mock(side_effect=PermissionError("root helper")),
                               wait=mock.Mock(side_effect=subprocess.TimeoutExpired("helper", 2)))
        client.proc = proc
        client._stop()
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    def test_bad_config_disables_feature_not_inference(self):
        say = mock.Mock()
        for cfg in (True, {}, {"helper_command": "sudo something"},
                    {"helper_command": ["helper"], "idle_s": 0},
                    {"helper_command": ["helper"], "idle_s": float("nan")}):
            with self.subTest(cfg=cfg), mock.patch.object(idle_power, "PolicyClient") as client:
                self.assertIsNone(idle_power.create(cfg, "hip", lambda: None, lambda: False, say))
                client.assert_not_called()
        self.assertTrue(say.called)

    def test_strict_offline_disables_configured_helper_without_any_popen(self):
        say = mock.Mock()
        with mock.patch.object(idle_power.subprocess, "Popen") as popen:
            power = idle_power.create({"helper_command": ["synthetic-helper"]}, "hip", lambda: None,
                                      lambda: False, say, offline=True)
        popen.assert_not_called()
        self.assertIsNone(power)
        diagnostic = " ".join(c.args[0] for c in say.call_args_list)
        self.assertIn("strict offline", diagnostic)
        self.assertIn("low power disabled", diagnostic)
        self.assertIn("inference remains enabled", diagnostic)

    def test_default_and_other_platforms_do_not_start_a_helper(self):
        with mock.patch.object(idle_power, "PolicyClient") as client:
            self.assertIsNone(idle_power.create(None, "hip", lambda: None, lambda: False))
            self.assertIsNone(idle_power.create(False, "hip", lambda: None, lambda: False))
            self.assertIsNone(idle_power.create({"helper_command": ["helper"]}, "cuda", lambda: None,
                                               lambda: False, mock.Mock()))
            with mock.patch.object(idle_power.sys, "platform", "win32"):
                self.assertIsNone(idle_power.create({"helper_command": ["helper"]}, "hip", lambda: None,
                                                   lambda: False, mock.Mock()))
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
