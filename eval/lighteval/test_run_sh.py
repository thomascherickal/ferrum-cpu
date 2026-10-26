"""run.sh with fakes for cargo, ferrum-serve and lighteval (no network, no model)."""
import http.server
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run.sh"


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Fixture:
    """A fake repo: <root>/eval/lighteval/{run.sh,*.py}, fake server, venv and cargo."""

    def __init__(self, server_body):
        self.root = Path(tempfile.mkdtemp())
        lt = self.root / "eval/lighteval"
        lt.mkdir(parents=True)
        for f in ("run.sh", "make_task_data.py", "summarize.py"):
            shutil.copy(HERE / f, lt / f)
        (lt / "ferrum_model.py").write_text("")
        (lt / "ferrum_tasks.py").write_text("")
        exe(self.root / "target/release/ferrum-serve", server_body)
        venv = self.root / "venv"
        exe(venv / "bin/python", f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        self.argv = self.root / "lighteval.argv"
        self.cache_env = self.root / "lighteval.cache_dir"
        exe(
            venv / "bin/lighteval",
            "#!/bin/bash\nprintf '%s\\n' \"$@\" > " + str(self.argv) + "\n"
            "printf '%s' \"$FERRUM_CACHE_DIR\" > " + str(self.cache_env) + "\n"
            "printf '%s\\n%s\\n%s' \"$HF_HOME\" \"$HF_DATASETS_CACHE\" \"$HF_HUB_CACHE\" > " + str(self.root / "lighteval.hf_env") + "\n"
            "echo $$ > " + str(self.root / "lighteval.pid") + "\n"
            '[ "${LE_MODE:-}" = fail ] && exit 3\n'
            '[ "${LE_MODE:-}" = hang ] && exec sleep 300\n'
            'while [ $# -gt 0 ]; do [ "$1" = --output-dir ] && d=$2; shift; done\n'
            'mkdir -p "$d/x"; echo \'{"results": {"all": {"z": 0}, "t|0": {"acc": 0.5}}}\' > "$d/x/results_1.json"\n',
        )
        exe(self.root / "bin/cargo", "#!/bin/bash\nexit 0\n")
        self.test = self.root / "test.txt"
        self.test.write_text("A a. B b. C c. D d. E e.<|endoftext|>F f. G g. H h. I i. J j.")
        self.out = self.root / "res/out.json"

    def _cmd_env(self, extra, env_extra):
        env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
        env.update(env_extra or {})
        env.update(FERRUM_LIGHTEVAL_VENV=str(self.root / "venv"), PATH=f"{self.root/'bin'}:{os.environ['PATH']}", FERRUM_TEST_PIDDIR=str(self.root))
        cmd = ["bash", str(self.root / "eval/lighteval/run.sh"), "--model", "m.bin", "--test", str(self.test), "--out", str(self.out), "--port", str(free_port()), *extra]
        return cmd, env

    def run(self, *extra, timeout=60, env_extra=None):
        cmd, env = self._cmd_env(extra, env_extra)
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)

    def popen(self, *extra, env_extra=None):
        cmd, env = self._cmd_env(extra, env_extra)
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


# fake server: records its pid, binds --port, announces "on http://HOST:PORT" like ferrum-serve, answers /health 200
GOOD_SERVER = (
    "#!/bin/bash\nport=8765\nwhile [ $# -gt 0 ]; do [ \"$1\" = --port ] && port=$2; shift; done\n"
    'echo $$ > "$FERRUM_TEST_PIDDIR/server.pid"\n'
    f"exec {sys.executable} -c \"import http.server as h,sys\n"
    "class H(h.BaseHTTPRequestHandler):\n"
    "  def do_GET(s): s.send_response(200); s.send_header('content-length','2'); s.end_headers(); s.wfile.write(b'ok')\n"
    "  def log_message(s,*a): pass\n"
    "srv=h.HTTPServer(('127.0.0.1',int(sys.argv[1])),H)\n"
    "print('ferrum-serve: m.bin (1 params, context 8, vocab 9) on http://127.0.0.1:%d' % int(sys.argv[1]), file=sys.stderr, flush=True)\n"
    "srv.serve_forever()\" $port\n"
)
# never finishes loading, never binds
HUNG_SERVER = '#!/bin/bash\necho $$ > "$FERRUM_TEST_PIDDIR/server.pid"\nexec sleep 300\n'
# loses the port race: dies after loading without ever announcing a URL
BINDFAIL_SERVER = "#!/bin/bash\necho 'cannot bind 127.0.0.1:1: in use' >&2\nexit 1\n"
DEAD_SERVER = "#!/bin/bash\necho 'fatal: cannot load model' >&2\nexit 3\n"


def alive(pidfile):
    """Is the process whose pid is in `pidfile` still running (zombies count as dead)?"""
    pid = int(Path(pidfile).read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"


class RunShTest(unittest.TestCase):
    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(RUN)]).returncode, 0)

    def test_dead_server_fails_fast_with_server_log(self):
        fx = Fixture(DEAD_SERVER)
        t = time.time()
        r = fx.run()
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(time.time() - t, 20, "must not wait out the health cap")
        self.assertIn("cannot load model", r.stderr)
        self.assertFalse(fx.out.exists())

    def test_happy_path_writes_summary_and_passes_max_samples(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--max-samples", "7")
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = json.loads(fx.out.read_text())
        self.assertEqual(summary["tasks"], {"t|0": {"acc": 0.5}})
        self.assertEqual((summary["model"], summary["max_samples"], summary["lighteval_version"]), ("m.bin", 7, "0.13.0"))
        argv = fx.argv.read_text().split("\n")
        self.assertEqual(argv[argv.index("--max-samples") + 1], "7")
        self.assertTrue((fx.root / "eval/lighteval/out/data/cloze/test.jsonl").exists())

    def test_previous_raw_results_are_kept_as_results_prev(self):
        # regression: run.sh used to `rm -rf out`, deleting the raw results Samples section 8 points at
        fx = Fixture(GOOD_SERVER)
        old = fx.root / "eval/lighteval/out/results/old"
        old.mkdir(parents=True)
        (old / "results_0.json").write_text("{}")
        self.assertEqual(fx.run().returncode, 0)
        out = fx.root / "eval/lighteval/out"
        self.assertTrue((out / "results.prev/old/results_0.json").exists())
        self.assertFalse((out / "results/old").exists())
        self.assertTrue((out / "results/x/results_1.json").exists())
        self.assertEqual(fx.run().returncode, 0)  # a second run replaces results.prev by the first run's
        self.assertFalse((out / "results.prev/old").exists())
        self.assertTrue((out / "results.prev/x/results_1.json").exists())

    def test_huggingface_caches_stay_inside_the_project(self):
        # regression: `datasets` wrote parquet builders and locks into ~/.cache/huggingface
        fx = Fixture(GOOD_SERVER)
        self.assertEqual(fx.run().returncode, 0)
        hf = str(fx.root / "eval/lighteval/.hf-cache")
        self.assertEqual((fx.root / "lighteval.hf_env").read_text().split("\n"), [hf, hf + "/datasets", hf + "/hub"])
        fx2 = Fixture(GOOD_SERVER)
        self.assertEqual(fx2.run(env_extra={"FERRUM_HF_CACHE": str(fx2.root / "elsewhere")}).returncode, 0)
        self.assertEqual((fx2.root / "lighteval.hf_env").read_text().split("\n")[0], str(fx2.root / "elsewhere"))

    def test_lighteval_sample_cache_is_private_to_the_run(self):
        # regression: lighteval's default cache (~/.cache/huggingface/lighteval) is keyed by config, not weights
        fx = Fixture(GOOD_SERVER)
        self.assertEqual(fx.run().returncode, 0)
        self.assertEqual(fx.cache_env.read_text(), str(fx.root / "eval/lighteval/out/cache"))

    def test_no_max_samples_and_wikitext_flag(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--wikitext")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = fx.argv.read_text()
        self.assertNotIn("--max-samples", argv)
        self.assertIn("wikitext:103:document_level|0", argv)
        self.assertIsNone(json.loads(fx.out.read_text())["max_samples"])

    def test_honours_cargo_target_dir(self):
        fx = Fixture(DEAD_SERVER)  # the default location holds a server that dies
        alt = fx.root / "alt-target"
        exe(alt / "release/ferrum-serve", GOOD_SERVER)
        r = fx.run(env_extra={"CARGO_TARGET_DIR": str(alt)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(fx.out.exists())

    def test_bad_arguments(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run("--bogus")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown argument", r.stderr)
        r = subprocess.run(["bash", str(fx.root / "eval/lighteval/run.sh"), "--model", "m"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--test required", r.stderr)

    def test_missing_venv_hint(self):
        fx = Fixture(GOOD_SERVER)
        env = dict(os.environ, FERRUM_LIGHTEVAL_VENV=str(fx.root / "nope"))
        r = subprocess.run(["bash", str(fx.root / "eval/lighteval/run.sh"), "--model", "m", "--test", "t", "--out", "o"], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("python3 -m venv", r.stderr)

    # ── exit paths: the server (and lighteval) must never outlive run.sh ──────────────
    def test_server_is_stopped_after_a_successful_run(self):
        fx = Fixture(GOOD_SERVER)
        self.assertEqual(fx.run().returncode, 0)
        self.assertFalse(alive(fx.root / "server.pid"))

    def test_server_is_stopped_when_lighteval_fails(self):
        fx = Fixture(GOOD_SERVER)
        r = fx.run(env_extra={"LE_MODE": "fail"})
        self.assertEqual(r.returncode, 3)
        self.assertFalse(alive(fx.root / "server.pid"))
        self.assertFalse(fx.out.exists())

    def test_server_is_stopped_when_it_never_becomes_healthy(self):
        # regression: health wait is capped (FERRUM_SERVE_WAIT) and says so, and the hung server is killed
        fx = Fixture(HUNG_SERVER)
        t = time.time()
        r = fx.run(env_extra={"FERRUM_SERVE_WAIT": "1"})
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(time.time() - t, 15)
        self.assertIn("did not become healthy within 1s", r.stderr)
        self.assertFalse(alive(fx.root / "server.pid"))

    def test_sigterm_stops_lighteval_and_server(self):
        fx = Fixture(GOOD_SERVER)
        p = fx.popen(env_extra={"LE_MODE": "hang"})
        try:
            deadline = time.time() + 30
            while not (fx.root / "lighteval.pid").exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue((fx.root / "lighteval.pid").exists(), "lighteval never started")
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=15), 143)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()
            p.stderr.close()
        self.assertFalse(alive(fx.root / "server.pid"))
        self.assertFalse(alive(fx.root / "lighteval.pid"))

    def test_foreign_server_on_the_port_is_never_mistaken_for_ours(self):
        # regression: any process answering /health on --port used to satisfy the wait
        # even when our own server had failed to bind
        class Foreign(http.server.BaseHTTPRequestHandler):
            def do_GET(s):
                s.send_response(200)
                s.send_header("content-length", "2")
                s.end_headers()
                s.wfile.write(b"ok")

            def log_message(s, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Foreign)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            fx = Fixture(BINDFAIL_SERVER)
            r = fx.run("--port", str(srv.server_port))
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot bind", r.stderr)
        self.assertFalse(fx.out.exists())

    # ── argument validation happens before any work ───────────────────────────────────
    def test_bad_values_are_usage_errors_before_the_run(self):
        fx = Fixture(GOOD_SERVER)
        for flag, val, msg in [
            ("--max-samples", "abc", "--max-samples must be a positive integer"),
            ("--max-samples", "0", "--max-samples must be a positive integer"),
            ("--max-samples", "-5", "--max-samples must be a positive integer"),
            ("--port", "http", "--port must be an integer"),
        ]:
            r = fx.run(flag, val)
            self.assertEqual(r.returncode, 2, (flag, val, r.stderr))
            self.assertIn(msg, r.stderr)
        r = fx.run(env_extra={"FERRUM_SERVE_WAIT": "soon"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("FERRUM_SERVE_WAIT", r.stderr)
        self.assertFalse((fx.root / "server.pid").exists(), "server must not start on a usage error")

    def test_flag_without_value_is_a_usage_error_not_an_unbound_variable(self):
        fx = Fixture(GOOD_SERVER)
        for flag in ["--model", "--test", "--out", "--max-samples", "--port"]:
            r = subprocess.run(["bash", str(fx.root / "eval/lighteval/run.sh"), flag], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, flag)
            self.assertIn(f"{flag} needs a value", r.stderr)
            self.assertNotIn("unbound variable", r.stderr)

    def test_no_array_expansion_for_optional_args(self):
        # regression: "${extra[@]}" on an empty array aborts under `set -u` on bash < 4.4 (macOS 3.2).
        # Can't run an old bash here, so forbid the construct.
        code = "\n".join(l for l in RUN.read_text().splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("[@]", code.replace('"$@"', ""))

    def test_optional_max_samples_absent_and_present_under_nounset(self):
        # the optional --max-samples must work both absent and present under `set -u`
        fx = Fixture(GOOD_SERVER)
        self.assertEqual(fx.run().returncode, 0)
        self.assertEqual(fx.run("--max-samples", "3").returncode, 0)


if __name__ == "__main__":
    unittest.main()
