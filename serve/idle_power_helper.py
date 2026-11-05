"""Restricted Linux power-policy helper. Install a root-owned copy before privileged use."""
from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None

import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import time

CONFIG = Path("/etc/strata-idle-power.json")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


class Policies:
    def __init__(self, devices, root=Path("/sys/bus/pci/devices")):
        self.devices = list(devices)
        self.root = Path(root)
        self.owned = set()
        if not self.devices or len(self.devices) > 16 or len(set(self.devices)) != len(self.devices):
            raise ValueError("expected one to sixteen distinct devices")
        for device in self.devices:
            if not isinstance(device, str) or PCI.fullmatch(device) is None:
                raise ValueError("invalid PCI device address")
            if (self.root / device / "vendor").read_text().strip() != "0x1002":
                raise ValueError("configured device is not AMD")
        if not self.normal():
            raise ValueError("all configured GPUs must start in auto; no policies were changed")

    def path(self, device):
        if device not in self.devices:
            raise ValueError("device is not allowlisted")
        return self.root / device / "power_dpm_force_performance_level"

    def read(self, device):
        return self.path(device).read_text().strip()

    def normal(self):
        return all(self.read(d) == "auto" for d in self.devices)

    def lower(self, track=lambda owned: None):
        if not self.normal():
            raise OSError("power policy changed outside this helper")
        for device in self.devices:
            if self.read(device) != "auto":
                raise OSError("power policy changed outside this helper")
            self.owned.add(device)
            track(self.owned)
            self.path(device).write_text("low\n")
            if self.read(device) != "low":
                raise OSError("low policy readback failed for " + device)
        if not all(self.read(d) == "low" for d in self.devices):
            raise OSError("low transition did not remain applied to all devices")

    def restore(self, owned=None):
        targets = self.owned if owned is None else set(owned)
        errors = []
        for device in list(targets):
            try:
                current = self.read(device)
                if current == "low":
                    self.path(device).write_text("auto\n")
                if self.read(device) != "auto":
                    raise OSError("policy is not auto; refusing to overwrite another policy")
                if owned is None:
                    self.owned.discard(device)
            except Exception as e:
                errors.append(f"{device}: {e}")
        if errors:
            raise OSError("; ".join(errors))
        return self.normal()


def foreign_compute(owner, uid, kfd=Path("/sys/class/kfd/kfd/proc"), proc=Path("/proc")):
    if isinstance(owner, bool) or not isinstance(owner, int) or owner <= 0:
        return True
    try:
        if (proc / str(owner)).stat().st_uid != uid or not (kfd / str(owner)).is_dir():
            return True
        for entry in kfd.iterdir():
            if entry.name == ".terraform":
                continue
            if entry.name.isdecimal() and entry.name != str(owner) and entry.is_dir():
                return True
        return False
    except OSError:
        return True


def rollback_monitor(policies, read_fd, lease_s=5.0):
    owned, pending = set(), b""
    deadline = time.monotonic() + lease_s
    def restore():
        if owned:
            try:
                policies.restore(owned)
                print("idle power: rollback restored auto", file=sys.stderr, flush=True)
                owned.clear()
            except Exception as e:
                print(f"WARNING idle power: rollback could not restore auto: {e}", file=sys.stderr, flush=True)
    try:
        while True:
            left = max(0, deadline - time.monotonic())
            ready = select.select([read_fd], [], [], left)[0]
            data = os.read(read_fd, 4096) if ready else b""
            if data:
                pending += data
                if len(pending) > 8192:
                    data = b""
                else:
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        frame = json.loads(line)
                        received = frame.get("owned", [])
                        if not isinstance(received, list) or any(d not in policies.devices for d in received):
                            raise ValueError("invalid rollback device list")
                        owned = set(received)
                        deadline = time.monotonic() + lease_s
                    continue
            restore()
            if ready and not data:
                return
            deadline = time.monotonic() + lease_s
    finally:
        restore()


class Session:
    def __init__(self, policies, uid, foreign=None, track=lambda owned: None):
        self.policies, self.uid, self.track = policies, uid, track
        self.foreign = foreign if foreign is not None else foreign_compute
        self.disabled = False
        self.owner = None

    def restore(self):
        try:
            normal = self.policies.restore()
            self.track(self.policies.owned)
            return {"normal": normal, "state": "auto" if normal else "unknown", "disabled": self.disabled}
        except Exception as e:
            self.disabled = True
            self.track(self.policies.owned)
            return {"normal": False, "state": "unknown", "disabled": True, "reason": str(e)}

    def request(self, frame):
        if not isinstance(frame, dict) or frame.get("command") not in ("active", "idle", "close"):
            self.disabled = True
            return {**self.restore(), "reason": "invalid command; low-power transitions disabled"}
        command = frame["command"]
        if command in ("active", "close"):
            return self.restore()
        self.owner = frame.get("owner")
        if self.disabled or self.foreign(self.owner, self.uid):
            return {**self.restore(), "reason": "low power skipped: disabled, unknown owner or another KFD client"}
        try:
            self.policies.lower(self.track)
            return {"normal": False, "state": "low", "disabled": False}
        except Exception as e:
            self.disabled = True
            return {**self.restore(), "reason": str(e)}

    def heartbeat(self):
        if self.policies.owned and self.foreign(self.owner, self.uid):
            answer = self.restore()
            print("idle power: another/unknown compute owner; " + json.dumps(answer), file=sys.stderr, flush=True)
        self.track(self.policies.owned)


def secure_file(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("privileged files must not use symlinks")
    for item in (path, *path.parents):
        info = item.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("privileged files and parents must be root-owned and not writable by other users")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("expected a regular privileged file")


def load_config(path=CONFIG):
    secure_file(path)
    with open(path, "rb") as f:
        raw = f.read(8193)
    if len(raw) > 8192:
        raise ValueError("oversized helper configuration")
    cfg = json.loads(raw)
    if not isinstance(cfg, dict) or set(cfg) != {"devices", "uid"}:
        raise ValueError("helper configuration expects only devices and uid")
    if isinstance(cfg["uid"], bool) or not isinstance(cfg["uid"], int) or cfg["uid"] <= 0:
        raise ValueError("expected a non-root client uid")
    if not isinstance(cfg["devices"], list):
        raise ValueError("devices must be a list")
    if "SUDO_UID" in os.environ and int(os.environ["SUDO_UID"]) != cfg["uid"]:
        raise PermissionError("sudo caller is not the configured uid")
    return cfg


def take_lock():
    if fcntl is None:
        raise OSError("idle power helper requires Linux")
    fd = os.open("/run/strata-idle-power.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if info.st_uid != 0 or info.st_mode & 0o022 or not stat.S_ISREG(info.st_mode):
            raise PermissionError("unsafe power-helper lock file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except Exception:
        os.close(fd)
        raise


def main():
    if not sys.platform.startswith("linux") or len(sys.argv) != 1 or os.geteuid() != 0:
        print("idle power: this helper requires a root-owned installation and no command-line arguments", file=sys.stderr)
        return 2
    secure_file(Path(__file__))
    cfg = load_config()
    with os.fdopen(take_lock(), "rb"):
        return serve_helper(cfg)


def serve_helper(cfg):
    policies = Policies(cfg["devices"])
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(write_fd)
        os.setsid()
        def guard_stop(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, guard_stop)
        signal.signal(signal.SIGINT, guard_stop)
        try:
            rollback_monitor(policies, read_fd)
        finally:
            os._exit(0)
    os.close(read_fd)

    def track(owned):
        os.write(write_fd, (json.dumps({"owned": sorted(owned)}) + "\n").encode())

    session = Session(policies, cfg["uid"], track=track)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pending = b""
    try:
        print(json.dumps({"state": "ready", "normal": True, "devices": policies.devices}), flush=True)
        while True:
            if not select.select([sys.stdin], [], [], .5)[0]:
                session.heartbeat()
                continue
            data = os.read(sys.stdin.fileno(), 4096)
            if not data:
                break
            pending += data
            if len(pending) > 8192:
                raise ValueError("oversized helper request")
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                frame = json.loads(line)
                answer = session.request(frame)
                print(json.dumps(answer), flush=True)
                if isinstance(frame, dict) and frame.get("command") == "close":
                    return 0
    finally:
        answer = session.restore()
        if answer.get("normal") is not True:
            print("WARNING idle power: exit restoration failed: " + json.dumps(answer), file=sys.stderr, flush=True)
        os.close(write_fd)
        os.waitpid(child, 0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"WARNING idle power helper: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
