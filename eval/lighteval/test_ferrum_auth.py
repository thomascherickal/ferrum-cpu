"""FerrumModel._call sends `Authorization: Bearer $FERRUM_TOKEN` only when the variable is set.
Runs without LightEval: the lighteval imports of ferrum_model.py are replaced by stand-ins just
for the import, and removed again so the other test modules still skip cleanly."""
import importlib.util
import json
import os
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN = []


def _load_ferrum_model():
    stubs = {}
    for name, attrs in {
        "lighteval": {},
        "lighteval.models": {},
        "lighteval.models.abstract_model": {"LightevalModel": object},
        "lighteval.models.model_output": {"ModelResponse": object},
        "lighteval.tasks": {},
        "lighteval.tasks.requests": {"Doc": object, "SamplingMethod": types.SimpleNamespace(GENERATIVE=0, LOGPROBS=1, PERPLEXITY=2)},
        "lighteval.utils": {},
        "lighteval.utils.cache_management": {"SampleCache": object, "cached": lambda _m: (lambda f: f)},
    }.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        stubs[name] = m
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("ferrum_model_under_auth_test", os.path.join(HERE, "ferrum_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self):
        SEEN.append(self.headers.get("authorization"))
        data = json.dumps({"results": []}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply()

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self._reply()


class BearerHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.mod = _load_ferrum_model()
        cls.model = cls.mod.FerrumModel.__new__(cls.mod.FerrumModel)  # only _call is exercised
        cls.model.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        SEEN.clear()

    def test_header_sent_on_get_and_post_when_token_set(self):
        with mock.patch.dict(os.environ, {"FERRUM_TOKEN": "s3cret"}):
            self.model._call("/v1/info")
            self.model._call("/v1/generate", {"requests": []})
        self.assertEqual(SEEN, ["Bearer s3cret", "Bearer s3cret"])

    def test_no_header_when_token_unset_or_empty(self):
        env = {k: v for k, v in os.environ.items() if k != "FERRUM_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.model._call("/v1/info")
        with mock.patch.dict(os.environ, {"FERRUM_TOKEN": ""}):
            self.model._call("/v1/info")
        self.assertEqual(SEEN, [None, None])

    def test_stand_ins_do_not_leak_into_sys_modules(self):
        before = set(sys.modules)
        _load_ferrum_model()
        self.assertEqual(set(sys.modules), before)


if __name__ == "__main__":
    unittest.main()
