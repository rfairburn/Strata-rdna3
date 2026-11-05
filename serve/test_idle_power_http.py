"""Request-admission and native-drain power tests, using only fake power policies and engines."""
import json
from pathlib import Path
import queue
import threading
import contextlib
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve import server
from serve.idle_power import IdlePower, create
from serve.test_idle_power import Client


class CheckedEngine(server.MockEngine):
    def generate(self, *args, **kwargs):
        self.seen += 1
        if self.power.count <= 0:
            raise AssertionError("engine dispatched without an activity lease")
        if self.client.state != "auto" and not self.allow_low:
            raise AssertionError("engine dispatched before normal power")
        yield from super().generate(*args, **kwargs)


class Http(unittest.TestCase):
    def setUp(self):
        self.telemetry = mock.patch.object(server.Service, "start_telemetry", return_value=None)
        self.telemetry.start()
        self.addCleanup(self.telemetry.stop)
        self.now = 0.0
        self.log = mock.Mock()
        self.client = Client()
        self.power = IdlePower(self.client, clock=lambda: self.now, start_thread=False, say=self.log)
        self.released = threading.Event()
        self.released.set()
        acquire, release = self.power.acquire, self.power.release
        def acquire_checked():
            self.released.clear()
            return acquire()
        def release_checked():
            release()
            if self.power.count == 0:
                self.released.set()
        self.power.acquire, self.power.release = acquire_checked, release_checked
        tok = server.ByteTokenizer()
        self.engine = CheckedEngine(tok, "ok")
        self.engine.power, self.engine.client = self.power, self.client
        self.engine.allow_low, self.engine.seen = False, 0
        self.svc = server.Service(self.engine, tok, server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.svc.set_idle_power(self.power)
        self.http = server.serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.assertEqual(self.power.count, 0)

    def lower(self):
        self.now += 30
        self.assertTrue(self.power.try_idle())
        self.assertEqual(self.client.state, "low")

    def request(self, path, body=None, authorized=True, extra_headers=None):
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if self.svc.api_key and authorized:
            headers["Authorization"] = "Bearer " + self.svc.api_key
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            status, raw = response.status, response.read().decode()
        self.assertTrue(self.released.wait(2), "HTTP handler did not release its activity lease")
        return status, raw

    def test_all_generation_routes_wake_before_dispatch(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    self.lower()
                    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                            "reasoning_effort": "none", "stream": stream}
                    if path == "/v1/responses":
                        body = {"input": "hi", "max_output_tokens": 8, "reasoning": {"effort": "none"},
                                "stream": stream}
                    status, raw = self.request(path, body)
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(self.client.state, "auto")
                    self.assertEqual(self.power.count, 0)
        self.assertEqual(self.engine.seen, 6)

    def test_restoration_failure_warns_and_still_returns_a_generation(self):
        self.lower()
        self.client.fail_wake = True
        self.engine.allow_low = True
        status, raw = self.request("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "reasoning_effort": "none"})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "ok")
        self.assertTrue(self.power.disabled)
        self.assertTrue(any("NORMAL POWER UNVERIFIED" in c.args[0] for c in self.log.call_args_list))

    def test_metadata_and_cpu_settings_do_not_wake(self):
        self.lower()
        self.assertEqual(self.request("/v1/models")[0], 200)
        self.assertEqual(self.request("/settings", {"defaults": None})[0], 200)
        self.assertEqual(self.client.state, "low")
        self.assertEqual(self.engine.seen, 0)

    def test_unauthorized_generation_does_not_wake(self):
        self.svc.api_key = "test-secret"
        self.lower()
        self.assertEqual(self.request("/v1/chat/completions", {}, authorized=False)[0], 401)
        self.assertEqual(self.client.state, "low")
        self.assertEqual(self.engine.seen, 0)

    def test_untrusted_pages_do_not_wake_generation_or_management(self):
        self.lower()
        for path in ("/v1/images/generations", "/load", "/slots/0", "/v1/load", "/v1/vram"):
            with self.subTest(path=path):
                code, raw = self.request(path, {}, extra_headers={"Origin": "https://untrusted.example"})
                self.assertEqual(code, 403, raw)
                self.assertEqual(self.client.state, "low")
                self.assertEqual(self.engine.seen, 0)

    def test_direct_prepare_and_run_are_also_guarded(self):
        self.lower()
        original = self.svc._note_unreadable_tool_images
        def check(*args, **kwargs):
            self.assertEqual(self.client.state, "auto")
            self.assertGreater(self.power.count, 0)
            return original(*args, **kwargs)
        with mock.patch.object(self.svc, "_note_unreadable_tool_images", check):
            ids, thinking, count = self.svc.prepare([{"role": "user", "content": "hi"}], None,
                                                   {"enable_thinking": False}, 8)
        self.lower()
        events = list(self.svc.run(ids, thinking, None, count, {}, threading.Event()))
        self.assertEqual(self.client.state, "auto")
        self.assertIn("done", [k for k, _ in events])

    def test_native_background_cancellation_holds_power_until_bdone(self):
        engine = object.__new__(server.StrataEngine)
        engine.slot_q = [queue.Queue()]
        engine.slot_busy, engine.slot_held = [True], [[1, 2]]
        engine.slot_releasing = set()
        engine.slot_cv = threading.Condition()
        engine._send = mock.Mock()
        engine.idle_power = self.power
        with mock.patch.object(server.threading, "Thread") as thread:
            engine._release_slot_when_done(0, [1, 2])
        self.assertEqual(self.power.count, 1)
        self.now = 60
        self.assertFalse(self.power.try_idle())
        engine.slot_q[0].put("BDONE 0 1 cancel 1\n")
        thread.call_args.kwargs["target"]()
        self.assertEqual(self.power.count, 0)
        self.assertFalse(engine.slot_busy[0])
        self.assertEqual(engine.slot_releasing, set())
        self.now = 89
        self.assertFalse(self.power.try_idle())
        self.now = 90
        self.assertTrue(self.power.try_idle())

    def test_aliases_normalize_before_the_request_lease(self):
        for path in ("/v1/v1/messages", "/v1/v1/messages/count_tokens"):
            self.lower()
            with mock.patch.object(self.power, "acquire", wraps=self.power.acquire) as acquire:
                code, raw = self.request(path, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
            self.assertEqual(code, 200, raw)
            self.assertTrue(acquire.called)
            self.assertEqual(self.client.state, "auto")

    def test_image_generation_holds_the_same_lease_during_gpu_turn(self):
        def check():
            self.assertGreater(self.power.count, 0)
            self.assertEqual(self.client.state, "auto")
            self.now += 60
            self.assertFalse(self.power.try_idle())
        @contextlib.contextmanager
        def turn(*args):
            check()
            yield "keep"
            check()
        def run(*args):
            check()
            return [], {"seconds": 0}
        self.svc.images = SimpleNamespace(cfg={}, run=run, response=lambda *args: {"data": []})
        job = {"width": 64, "height": 64, "n": 1, "steps": 1, "seed": 0}
        self.lower()
        with mock.patch.object(server.images_api, "parse_request", return_value=job), \
                mock.patch.object(server.images_api, "gpu_turn", turn):
            code, raw = self.request("/v1/images/generations", {"prompt": "synthetic"})
        self.assertEqual(code, 200, raw)
        self.assertEqual(self.power.count, 0)

    def test_reserved_preparing_queued_admitting_and_draining_block_low(self):
        for obj, attr, value in ((self.svc, "admission_reserved", 1), (self.svc, "draining", True),
                                 (self.engine, "admitting", 1), (self.engine, "waiting", 1),
                                 (self.engine, "starting", True), (self.engine, "slot_releasing", {3})):
            self.now += 30
            with self.subTest(attr=attr), mock.patch.object(obj, attr, value, create=True):
                self.assertFalse(self.power.try_idle())
            self.assertEqual(self.client.state, "auto")
        self.svc.status["queued"] = 1
        self.now += 30
        self.assertFalse(self.power.try_idle())
        self.svc.status["queued"] = 0
        self.assertEqual(self.power.count, 0)

    def test_offline_helper_failure_does_not_refuse_inference(self):
        self.svc.offline = True
        with mock.patch("serve.idle_power.subprocess.Popen") as popen:
            power = create({"helper_command": ["synthetic-helper"]}, "hip", lambda: None,
                           lambda: False, self.log, offline=self.svc.offline)
        popen.assert_not_called()
        self.assertIsNone(power)
        self.svc.set_idle_power(power)
        self.svc.engine = server.MockEngine(self.svc.tok, "ok")
        code, raw = self.request("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "reasoning_effort": "none"})
        self.assertEqual(code, 200, raw)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "ok")
        self.power.close()

    def test_default_service_is_off(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "ok"), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.assertIsNone(svc.idle_power)
        with svc.power_request("/v1/chat/completions"):
            self.assertIsNone(svc.idle_power)


if __name__ == "__main__":
    unittest.main()
