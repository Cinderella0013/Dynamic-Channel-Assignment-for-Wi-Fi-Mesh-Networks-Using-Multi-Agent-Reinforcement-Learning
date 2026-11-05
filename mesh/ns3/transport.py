"""JSON-lines pipe to the ns-3 mesh_bridge binary (docs/ns3/protocol.md section 2)."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_BINARY = "~/ns3_validation/ns-3.47/build/scratch/ns3.47-mesh_bridge"


class TransportError(RuntimeError):
    pass


def _lib_dir(binary):
    """<ns-3 root>/build/lib for a binary somewhere under <root>/build/."""
    p = Path(binary)
    for parent in p.parents:
        if parent.name == "build":
            return str(parent / "lib")
    return str(p.parent)


class ProcessTransport:
    """One ns-3 process per episode: open() spawns + init, step() per env step, close() sends end and reaps."""

    def __init__(self, binary=DEFAULT_BINARY, cmd=None, timeout_s=60.0):
        self.binary = os.path.expanduser(binary)
        self.cmd = cmd or [self.binary]
        self.timeout_s = timeout_s
        self.proc = None

    @property
    def closed(self):
        return self.proc is None or self.proc.poll() is not None

    def open(self, init_msg):
        self.close()
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = _lib_dir(self.binary) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        self.err = tempfile.TemporaryFile()  # ns-3 logs on stderr; read back only on failure
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                     text=True, env=env)
        if self._ask(init_msg).get("type") != "ready":
            self._fail("expected ready")

    def step(self, msg):
        reply = self._ask(msg)
        if reply.get("type") != "result" or reply.get("t") != msg["t"]:
            self._fail(f"expected result t={msg['t']}, got {reply.get('type')!r} t={reply.get('t')!r}")
        return reply

    def close(self):
        """Send end and reap; raise if ns-3 exits non-zero. Safe to call twice."""
        p = self.proc
        if p is None or p.returncode is not None:
            return
        try:
            if p.poll() is None:
                p.stdin.write('{"type": "end"}\n')
                p.stdin.flush()
            p.wait(timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            p.kill()
            p.wait()
        if p.returncode != 0:
            raise TransportError(f"ns-3 exited {p.returncode}; stderr: {self._stderr_tail()}")

    def _ask(self, msg):
        if self.closed:
            raise TransportError("transport not open")
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except OSError as e:
            self._fail(f"pipe error: {e}")
        if not line:
            self._fail("ns-3 closed stdout")
        reply = json.loads(line)
        if reply.get("type") == "error":
            self._fail(f"ns-3 error: {reply.get('msg')}")
        return reply

    def _stderr_tail(self, n=2000):
        self.err.seek(0)
        return self.err.read().decode(errors="replace")[-n:]

    def _fail(self, why):
        self.proc.kill()  # no-op if it already exited
        self.proc.wait()
        raise TransportError(f"{why} (exit {self.proc.returncode}); stderr: {self._stderr_tail()}")
