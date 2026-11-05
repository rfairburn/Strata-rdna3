"""Four-slot protocol and power ownership with in-memory queues; no native or host probes."""
import atexit
import contextlib
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import idle_power, server
from strata_control import private_files
from serve.idle_power import IdlePower
from serve.test_idle_power import Client


class NativeContract(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.client = Client()
        self.power = IdlePower(self.client, clock=lambda: self.now, start_thread=False, say=mock.Mock())
        self.addCleanup(self.power.close)

    def engine(self):
        e = object.__new__(server.StrataEngine)
        e.batch, e.gen = 4, 1
        e.info = {"batch_slots": 4, "batch_groups": 1, "slot_cache": 1}
        e.slot_cv, e.ctl = threading.Condition(), threading.Lock()
        e.slot_busy, e.slot_held = [False] * 4, [[] for _ in range(4)]
        e.slot_live, e.slot_used = [None] * 4, [0.0] * 4
        e.slot_order, e.slot_group, e.slot_groups, e.slot_gs = list(range(4)), [0] * 4, 1, 0
        e.slot_claims, e.slot_releasing = set(), set()
        e.waiting, e.admitting, e.ctl_epoch = 0, 0, 0
        e.wait_lens, e._orphan_seen = [], {}
        e.slot_q, e.lines = [queue.Queue() for _ in range(4)], queue.Queue()
        e.alive, e.exit_code = lambda: True, lambda: None
        e.can_stop, e.silence_s, e.progress, e._yielded = True, 0, None, None
        e.idle_power, e.last = self.power, {}
        return e

    def test_four_slot_last_client_continues_original_input_and_owns_all_legs(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled), mock.patch.dict(server.os.environ, {"STRATA_PARALLEL_SOLO": "1"}):
                e = self.engine()
                e.slot_busy[1:] = [True] * 3
                for _ in range(3):
                    self.power.acquire()
                sent = []
                def send(line):
                    self.assertGreater(self.power.count, 0)
                    self.now += 60
                    self.assertFalse(self.power.try_idle())
                    sent.append(line)
                    if line.startswith("BGEN "):
                        e.lines.put("T 65")
                        e.lines.put("DONE 1 2 5 1 length 0 0 1")
                        e.lines.put("BADM 0 1")
                        e.slot_q[0].put("BT 0 66")
                    elif line == "BSTOP 0":
                        e.slot_q[0].put("BDONE 0 2 cancel 20")
                    elif line.startswith("GEN "):
                        e.lines.put("T 67")
                        e.lines.put("DONE 1 4 1 3 stop 0 0 3")
                e._send = send
                cancel = threading.Event()
                with self.power.activity():
                    gen = e.generate_batched([11, 12], 100, {}, cancel)
                    self.assertEqual(next(gen), 65)
                    self.assertEqual(next(gen), 66)
                    for slot in range(1, 4):
                        e.slot_busy[slot] = False
                        self.power.release()
                    self.assertEqual(self.power.count, 1)
                    if cancelled:
                        cancel.set()
                    self.assertEqual(list(gen), [] if cancelled else [67])
                    self.assertEqual(self.power.count, 1)
                self.assertEqual(self.power.count, 0)
                self.assertIn("BSTOP 0", sent)
                solo = [s for s in sent if s.startswith("GEN ")]
                self.assertEqual(solo, [] if cancelled else ["GEN 98 11,12,65,66"])
                self.assertFalse(any(e.slot_busy))
                self.assertEqual(e.slot_claims, set())
                if not cancelled:
                    self.assertEqual((e.last["generated"], e.last["decode_ms"], e.last["reused"]), (3, 23.0, 1))

    def test_cancelled_consumer_leaves_a_tail_lease_until_bdone_or_eof(self):
        for ending in ("BDONE 0 3 cancel 20", None):
            e = self.engine()
            e.slot_busy[1] = True
            sent = []
            def send(line):
                sent.append(line)
                if line.startswith("BGEN "):
                    e.lines.put("T 65")
                    e.lines.put("DONE 1 2 5 1 length 0 0 0")
                    e.lines.put("BADM 0 1")
                    e.slot_q[0].put("BT 0 66")
            e._send = send
            with self.power.activity():
                gen = e.generate_batched([11, 12], 100, {}, threading.Event())
                self.assertEqual(next(gen), 65)
                self.assertEqual(next(gen), 66)
                with mock.patch.object(server.threading, "Thread") as thread:
                    gen.close()
                self.assertEqual(self.power.count, 2)
            self.assertEqual(self.power.count, 1)
            self.now += 60
            self.assertFalse(self.power.try_idle())
            self.assertIn("BSTOP 0", sent)
            e.slot_q[0].put("BT 0 67")
            e.slot_q[0].put(ending)
            thread.call_args.kwargs["target"]()
            self.assertEqual(self.power.count, 0)
            self.assertFalse(e.slot_busy[0])
            self.assertEqual(e.slot_releasing, set())
            self.assertEqual(e.slot_held[0], [11, 12, 65, 66] if ending else [])

    def test_drain_timeout_exception_and_thread_start_failure_inhibit_low(self):
        for failure in ("timeout", "queue", "thread"):
            with self.subTest(failure=failure):
                e = self.engine()
                e._send = mock.Mock()
                e.slot_busy[0] = True
                self.power.disabled = False
                if failure == "queue":
                    e.slot_q[0] = mock.Mock()
                    e.slot_q[0].get.side_effect = OSError("synthetic queue error")
                with mock.patch.object(server.threading, "Thread") as thread:
                    if failure == "thread":
                        thread.return_value.start.side_effect = RuntimeError("synthetic thread failure")
                        with self.assertRaises(RuntimeError):
                            e._release_slot_when_done(0)
                    else:
                        e._release_slot_when_done(0)
                if failure == "timeout":
                    with mock.patch.object(server.time, "monotonic", side_effect=[0.0, 601.0]):
                        thread.call_args.kwargs["target"]()
                elif failure == "queue":
                    with self.assertRaises(OSError):
                        thread.call_args.kwargs["target"]()
                self.assertEqual(self.power.count, 0)
                self.assertTrue(self.power.disabled)
                self.assertEqual(self.client.state, "auto")

    def test_old_drain_releases_only_its_captured_slot_after_restart(self):
        e = self.engine()
        e._send = mock.Mock()
        e.slot_busy[0] = True
        old_busy, old_q = e.slot_busy, e.slot_q[0]
        with mock.patch.object(server.threading, "Thread") as thread:
            e._release_slot_when_done(0)
        e.slot_busy = [True] * 4
        e.slot_held = [[] for _ in range(4)]
        e.slot_releasing = {0}
        old_q.put("BDONE 0 1 cancel 1")
        thread.call_args.kwargs["target"]()
        self.assertFalse(old_busy[0])
        self.assertEqual(e.slot_busy, [True] * 4)
        self.assertEqual(e.slot_releasing, {0})
        self.assertEqual(self.power.count, 0)

    def test_pump_routes_four_slots_and_keeps_stage_samples_out_of_tokens(self):
        e = self.engine()
        e.proc = SimpleNamespace(stdout=iter(["BSTAT 2 8 6 20", "PSTAT 4 20000 16000 14000 12000 10000",
                                            "BT 3 65", "BDONE 3 1 cancel 3", "DONE 1 2 5 3 stop"]))
        e._pump()
        self.assertEqual(e.slot_q[3].get_nowait(), "BT 3 65")
        self.assertEqual(e.slot_q[3].get_nowait(), "BDONE 3 1 cancel 3")
        self.assertEqual(e.lines.get_nowait(), "DONE 1 2 5 3 stop")
        self.assertIsNone(e.lines.get_nowait())
        self.assertEqual(e.terminal_batch_sample["processed_rows"], 8)
        self.assertEqual(e.terminal_stage_sample["in_flight_us"], [16000, 14000, 12000, 10000])
        stages = server.parse_engine_stage_line("strata serve: layer split: layers 0-11 (CUDA0), 12-23 (CUDA1), "
                                               "24-35 (CUDA2), 36-47 (CUDA3), one hand-off per window")
        self.assertEqual(len(stages), 4)
        e._parse_done("DONE 3 2 5 7 stop 4 2 1 9 10 1 2 3.5 1 0 chunk=32 experts_streamed=7")
        self.assertEqual((e.last["generated"], e.last["chunk"], e.last["experts_streamed"]), (3, 32, 7))

    def test_elastic_vision_restart_and_idle_cleanup_hold_the_same_lease(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "ok"), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        svc.set_idle_power(self.power)
        running = [False]
        def check(*args):
            self.assertGreater(self.power.count, 0)
            self.assertEqual(self.client.state, "auto")
            self.now += 60
            self.assertFalse(self.power.try_idle())
        def restart():
            check()
            running[0] = True
        def unload():
            check()
            running[0] = False
        svc.vision = SimpleNamespace(elastic=True, elastic_mib=16, elastic_idle_s=10,
                                     alive=lambda: running[0], restart=restart, unload=unload)
        svc.engine.vram = mock.Mock(side_effect=lambda reserve: (check(), {})[1])
        self.now += 30
        self.assertTrue(self.power.try_idle())
        with mock.patch.object(svc, "free_vram_mib", return_value=None), mock.patch.object(server.time, "sleep"):
            svc._vision_up()
        self.assertTrue(running[0])
        self.assertEqual(self.power.count, 0)
        self.now += 30
        self.assertTrue(self.power.try_idle())
        self.assertEqual(svc._vision_idle_down(force=True), "down")
        self.assertFalse(running[0])
        self.assertEqual(self.power.count, 0)
        self.now += 30
        self.assertTrue(self.power.try_idle())
        self.assertEqual(svc._vision_idle_down(force=True), "nothing")
        self.assertEqual(self.client.state, "low")

    def test_family_parser_selection_and_declared_only_stay_in_service_run(self):
        from serve.dsml import DsmlOutputParser
        tok = server.ByteTokenizer()
        raw = ('<｜DSML｜ calls><｜DSML｜ invoke name="tools.bash">'
               '<｜DSML｜ parameter name="command" string="true">sample</｜DSML｜ parameter>'
               '</｜DSML｜ invoke></｜DSML｜ calls>')
        svc = server.Service(server.MockEngine(tok, raw), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        svc.set_family("deepseek41")
        self.addCleanup(svc.set_family, None)
        self.assertIs(svc.parser_cls, DsmlOutputParser)
        for enabled in (False, True):
            svc.tool_call_passthrough = enabled
            events = list(svc.run([11], False, [{"name": "bash"}], len(tok.encode(raw)), {},
                                  threading.Event(), declared_only=True))
            calls = [(e.call.name, e.call.arguments) for kind, e in events if kind == "event" and e.kind == "tool_call"]
            self.assertEqual(calls, [("tools.bash", {"command": "sample"})] if enabled else [])

    def test_direct_slot_vram_load_and_unload_wake_before_engine_work(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "ok"), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        svc.set_idle_power(self.power)
        running = [False]
        def check():
            self.assertGreater(self.power.count, 0)
            self.assertEqual(self.client.state, "auto")
            self.now += 60
            self.assertFalse(self.power.try_idle())
        def restart():
            check()
            running[0] = True
        def unload():
            check()
            running[0] = False
        svc.engine.alive = lambda: running[0]
        svc.engine.unloaded = True
        svc.engine.restart, svc.engine.unload = restart, unload
        svc.engine.vram = lambda reserve: (check(), {})[1]
        svc.engine.session_file = lambda *args: (check(), {"tokens": 1, "bytes": 2, "ms": 0})[1]
        # The fixture never creates a session file; only the fake engine receives its temporary path.
        with tempfile.TemporaryDirectory() as tmp:
            svc.slot_save_path = tmp
            for work in (svc.load, lambda: svc.vram(0), lambda: svc.slot_action("0", "save", "synthetic.bin"), svc.unload):
                self.now += 30
                # An unloaded engine is conservatively busy, so lower through an explicit fake idle predicate.
                with mock.patch.object(self.power, "busy", return_value=False):
                    self.assertTrue(self.power.try_idle())
                work()
                self.assertEqual(self.power.count, 0)
                self.assertEqual(self.client.state, "auto")

    def test_startup_ready_failure_and_shutdown_keep_power_and_lifecycle_cleanup(self):
        for fail_ready in (False, True):
            with self.subTest(fail_ready=fail_ready), tempfile.TemporaryDirectory() as tmp:
                client = Client()
                power = IdlePower(client, start_thread=False, say=mock.Mock())
                self.addCleanup(power.close)
                root = Path(tmp)
                cfg = root / "synthetic.json"
                cfg.write_text(json.dumps({"idle_power": {"helper_command": ["synthetic-helper"]}}))
                made, events = [], []
                original = server.MockEngine
                def engine(*args, **kwargs):
                    self.assertEqual(power.count, 1)
                    out = original(*args, **kwargs)
                    out.close = mock.Mock(side_effect=lambda: self.assertGreater(power.count, 0))
                    made.append(out)
                    return out
                def serve(svc, **kwargs):
                    self.assertEqual(power.count, 1)
                    return SimpleNamespace(server_address=("127.0.0.1", 12345),
                                           shutdown=lambda: self.assertGreater(power.count, 0))
                def publish(*args):
                    self.assertEqual(power.count, 1)
                    if fail_ready:
                        raise OSError("synthetic ready-file failure")
                with contextlib.ExitStack() as stack:
                    for patch in (mock.patch.dict(server.os.environ, {}, clear=True),
                                  mock.patch.object(private_files, "adopt_user_defaults"),
                                  mock.patch.object(server.lifeline, "install_stop_handlers"),
                                  mock.patch.object(server, "Server"), mock.patch.object(server, "serve", serve),
                                  mock.patch.object(server, "MockEngine", engine),
                                  mock.patch.object(server, "hub_from_config", return_value=None),
                                  mock.patch.object(idle_power, "create", return_value=power),
                                  mock.patch.object(atexit, "register"),
                                  mock.patch.object(server.Service, "start_idle_unload"),
                                  mock.patch.object(server.Service, "start_vision_idle"),
                                  mock.patch.object(server.ready_file, "preflight", return_value=root / "ready.json"),
                                  mock.patch.object(server.ready_file, "publish", side_effect=publish),
                                  mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt)):
                        stack.enter_context(patch)
                    removed = stack.enter_context(mock.patch.object(server.ready_file, "remove_if_owned"))
                    code = server.main(["--config", str(cfg), "--tokenizer", str(root / "no-tokenizer"),
                                        "--port", "0", "--ready-file", str(root / "ready.json"),
                                        "--shutdown-timeout", "0", "--no-api-key"],
                                       lifecycle_event=lambda kind, payload: events.append(kind))
                self.assertEqual(code, 6 if fail_ready else 0)
                self.assertEqual(power.count, 0)
                self.assertTrue(power.closed)
                self.assertTrue(client.closed)
                made[0].close.assert_called_once()
                if not fail_ready:
                    removed.assert_called_once()
                    self.assertEqual(events, ["ready", "stopping", "draining", "stopped"])

    def test_info_does_not_publish_four_slot_readiness_before_ready(self):
        info_seen, ready, complete = threading.Event(), threading.Event(), threading.Event()
        errors = []
        proc = mock.Mock(stdout=iter(()), stdin=io.StringIO())
        proc.poll.return_value = None
        def ready_pump(process, out):
            out.put("INFO batch_slots=4 batch_groups=1 slot_cache=1")
            info_seen.set()
            if ready.wait(2):
                out.put("READY 4096 stop")
            else:
                out.put(None)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "popen", return_value=proc), \
                mock.patch.object(server, "end_leftover_engines"), mock.patch.object(server, "engine_nice", return_value=0), \
                mock.patch.object(server.StrataEngine, "_ready_pump", staticmethod(ready_pump)):
            e = object.__new__(server.StrataEngine)
            def start():
                try:
                    with self.power.activity():
                        server.StrataEngine.__init__(e, str(Path(tmp) / "synthetic-engine"), ["--batch", "4"])
                except BaseException as error:
                    errors.append(error)
                finally:
                    complete.set()
            thread = threading.Thread(target=start)
            thread.start()
            try:
                self.assertTrue(info_seen.wait(2))
                self.assertFalse(complete.is_set())
                self.assertFalse(e.alive())
                self.assertEqual(self.power.count, 1)
                self.now += 60
                self.assertFalse(self.power.try_idle())
            finally:
                ready.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            e.pump.join(2)
            self.assertEqual((e.batch, e.slot_groups, e.info["slot_cache"]), (4, 1, 1))
            self.assertEqual(len(e.slot_q), 4)
            self.assertEqual(self.power.count, 0)


if __name__ == "__main__":
    unittest.main()
