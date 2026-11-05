"""Opt-in Linux/HIP idle power guard. Inference continues if power control fails."""
from __future__ import annotations

import contextlib
import functools
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time


class PolicyClient:
    def __init__(self, command, say=print, timeout=2.0):
        self.say, self.timeout = say, timeout
        self.proc = None
        self.devices = []
        self.pending = b""
        self.lock = threading.Lock()
        self.failed = False
        self.state_changed = None
        self.state_pending = threading.Event()
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)
            threading.Thread(target=self._errors, daemon=True, name="idle-power-log").start()
            ready = self._read()
            if ready.get("state") != "ready" or ready.get("normal") is not True:
                raise ValueError("helper did not confirm normal power")
            devices = ready.get("devices")
            if not isinstance(devices, list) or not devices or not all(isinstance(x, str) for x in devices):
                raise ValueError("helper did not identify its devices")
            import re
            if any(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) is None for x in devices):
                raise ValueError("invalid helper device address")
            self.devices = devices
        except Exception:
            self.failed = True
            self._stop()
            raise

    def _errors(self):
        proc = self.proc
        try:
            for line in proc.stderr:
                text = line.decode("utf-8", "replace").rstrip()
                self.say("[strata idle power helper] " + text, flush=True)
                if text.startswith("idle power: another/unknown compute owner; "):
                    self.state_pending.set()
                    callback = self.state_changed
                    if callback is not None:
                        self.state_pending.clear()
                        callback()
        except (OSError, ValueError):
            pass

    def set_state_callback(self, callback):
        self.state_changed = callback
        if callback is not None and self.state_pending.is_set():
            self.state_pending.clear()
            callback()

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.pending:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.proc.stdout], [], [], left)[0]:
                raise TimeoutError("power helper response timed out")
            data = os.read(self.proc.stdout.fileno(), 4096)
            if not data:
                raise OSError("power helper exited")
            self.pending += data
            if len(self.pending) > 8192:
                raise ValueError("oversized power helper response")
        line, self.pending = self.pending.split(b"\n", 1)
        answer = json.loads(line)
        if not isinstance(answer, dict):
            raise ValueError("invalid power helper response")
        return answer

    def _rpc(self, command, **fields):
        with self.lock:
            if self.failed or self.proc is None or self.proc.poll() is not None:
                raise OSError("power helper is unavailable")
            try:
                self.proc.stdin.write((json.dumps({"command": command, **fields}) + "\n").encode())
                self.proc.stdin.flush()
                return self._read()
            except Exception:
                self.failed = True
                self._stop()
                raise

    def verify_normal(self):
        try:
            return bool(self.devices) and all(
                (Path("/sys/bus/pci/devices") / d / "power_dpm_force_performance_level").read_text().strip() == "auto"
                for d in self.devices)
        except OSError:
            return False

    def wake(self):
        return self._rpc("active")

    def idle(self, owner):
        return self._rpc("idle", owner=owner)

    def _stop(self):
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self):
        try:
            if not self.failed:
                self._rpc("close")
                self.proc.wait(timeout=self.timeout)
        except Exception:
            pass
        finally:
            self.failed = True
            self._stop()


class IdlePower:
    def __init__(self, client, idle_s=30.0, owner=lambda: None, busy=lambda: False, say=print,
                 clock=time.monotonic, start_thread=True):
        self.client, self.idle_s, self.owner, self.busy = client, idle_s, owner, busy
        self.say, self.clock = say, clock
        self.cv = threading.Condition()
        self.count = 0
        self.normal = False
        self.disabled = False
        self.closed = False
        self.last = clock()
        self.thread = None
        with self.cv:
            self._wake()
        if hasattr(client, "set_state_callback"):
            client.set_state_callback(self._external_restore)
        if start_thread:
            self.thread = threading.Thread(target=self._loop, daemon=True, name="idle-power")
            self.thread.start()

    def _warn(self, text):
        try:
            self.say("[strata] WARNING idle power: " + text, flush=True)
        except Exception:
            pass

    def _verify_normal(self):
        try:
            return self.client.verify_normal()
        except Exception:
            return False

    def _wake(self):
        if self.normal:
            return
        try:
            answer = self.client.wake()
            self.normal = answer.get("normal") is True
            if answer.get("disabled"):
                self.disabled = True
                self._warn("helper disabled the mitigation: " + str(answer.get("reason", "unknown reason")))
            if not self.normal:
                self.normal = self._verify_normal()
            if not self.normal:
                raise OSError("normal GPU power could not be verified")
        except Exception as e:
            self.disabled = True
            self.normal = self._verify_normal()
            self._warn(f"restoration failed ({e}); low-power transitions disabled")
            if not self.normal:
                self._warn("NORMAL POWER UNVERIFIED; continuing inference, which may be throttled")

    def _external_restore(self):
        with self.cv:
            if self.closed:
                return
            self.normal = False
            self._wake()   # verify power; a delayed notification is not authoritative state
            self.last = self.clock()
            self.cv.notify_all()

    def acquire(self):
        with self.cv:
            if self.closed:
                self._warn("guard is closed; continuing without power control")
                return False
            if self.count == 0 or not self.normal:
                self._wake()
            self.count += 1
            self.cv.notify_all()
            return True

    def release(self):
        with self.cv:
            if self.count <= 0:
                self._warn("unbalanced activity release; disabling low power")
                self.disabled = True
                self.normal = False
                self._wake()
                return
            self.count -= 1
            if self.count == 0:
                self.last = self.clock()
            self.cv.notify_all()

    @contextlib.contextmanager
    def activity(self):
        held = self.acquire()
        try:
            yield
        finally:
            if held:
                self.release()

    def inhibit(self, reason):
        with self.cv:
            self.disabled = True
            self._warn("low-power transitions disabled: " + reason)
            self.normal = False
            self._wake()
            self.cv.notify_all()

    def try_idle(self):
        with self.cv:
            if self.closed or self.disabled or self.count or not self.normal:
                return False
            if self.clock() - self.last < self.idle_s:
                return False
            try:
                if self.busy():
                    self.last = self.clock()
                    return False
                answer = self.client.idle(self.owner())
                if answer.get("state") == "low" and not answer.get("disabled"):
                    self.normal = False
                    self.cv.notify_all()
                    self.say(f"[strata] idle power: low after {self.idle_s:g} s; model and KV stay resident", flush=True)
                    return True
                if answer.get("normal") is not True:
                    raise OSError(str(answer.get("reason", "helper did not confirm its power state")))
                self.last = self.clock()
                if answer.get("disabled"):
                    self.disabled = True
                    self._warn("helper disabled the mitigation: " + str(answer.get("reason", "unknown reason")))
            except Exception as e:
                self.disabled = True
                self.normal = False
                self._warn(f"idle transition failed ({e}); restoring normal power")
                self._wake()
            return False

    def _loop(self):
        while True:
            with self.cv:
                if self.closed:
                    return
                if self.count or self.disabled or not self.normal:
                    self.cv.wait()
                    continue
                left = self.idle_s - (self.clock() - self.last)
                if left > 0:
                    self.cv.wait(timeout=left)
                    continue
            self.try_idle()

    def close(self):
        with self.cv:
            if self.closed:
                return
            self.closed = True
            self.normal = False
            self._wake()
            self.cv.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        if hasattr(self.client, "set_state_callback"):
            self.client.set_state_callback(None)
        self.client.close()


def gpu_work(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        power = getattr(self, "idle_power", None)
        if power is None:
            return method(self, *args, **kwargs)
        with power.activity():
            return method(self, *args, **kwargs)
    return wrapped


def engine_busy(engine):
    if engine is None or getattr(engine, "starting", False):
        return True
    if hasattr(engine, "alive") and not engine.alive():
        return True
    cv = getattr(engine, "slot_cv", None)
    with cv if cv is not None else contextlib.nullcontext():
        return bool(any(getattr(engine, "slot_busy", ())) or getattr(engine, "slot_releasing", ()) or
                    getattr(engine, "waiting", 0) or getattr(engine, "admitting", 0) or
                    (getattr(engine, "ctl", None) is not None and engine.ctl.locked()))


def create(config, backend, owner, busy, say=print, *, offline=False):
    if config is None or config is False:
        return None
    client = None
    try:
        if offline:
            raise ValueError("strict offline policy forbids the idle power helper subprocess; low power disabled")
        if not sys.platform.startswith("linux") or backend != "hip":
            raise ValueError("idle power requires Linux and the HIP backend")
        if not isinstance(config, dict) or set(config) - {"helper_command", "idle_s"}:
            raise ValueError("idle_power expects helper_command and optional idle_s")
        command = config.get("helper_command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("helper_command must be a nonempty argument list; shell commands are not accepted")
        idle_s = config.get("idle_s", 30)
        if isinstance(idle_s, bool) or not isinstance(idle_s, (int, float)) or not 0 < idle_s <= 86400:
            raise ValueError("idle_s must be positive and at most 86400")
        client = PolicyClient(command, say=say)
        return IdlePower(client, idle_s=idle_s, owner=owner, busy=busy, say=say)
    except Exception as e:
        if client is not None:
            client.close()
        say(f"[strata] WARNING idle power: setup failed ({e}); normal power is unverified; "
            "inference remains enabled and may be throttled", flush=True)
        return None
