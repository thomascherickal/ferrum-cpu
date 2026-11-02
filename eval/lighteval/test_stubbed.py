"""ferrum_model.py, ferrum_tasks.py and make_task_data.py's CLI guard against fake `lighteval`
modules, so they are exercised without LightEval installed. Plain stdlib."""
import importlib.util
import io
import json
import os
import runpy
import sys
import types
import unittest
import urllib.error
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))


class Rec:
    """Keyword-only record standing in for Doc / ModelResponse / LightevalTaskConfig."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_lighteval():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    metrics = types.SimpleNamespace(word_perplexity="wp", byte_perplexity="bp", bits_per_byte="bpb", loglikelihood_acc="acc")
    sampling = types.SimpleNamespace(GENERATIVE="gen", LOGPROBS="lp")
    cached = lambda _method: (lambda f: f)  # noqa: E731
    cache = type("SampleCache", (), {"__init__": lambda self, config: setattr(self, "config", config)})
    return {
        "lighteval": mod("lighteval"),
        "lighteval.models": mod("lighteval.models"),
        "lighteval.models.abstract_model": mod("lighteval.models.abstract_model", LightevalModel=object),
        "lighteval.models.model_output": mod("lighteval.models.model_output", ModelResponse=Rec),
        "lighteval.tasks": mod("lighteval.tasks"),
        "lighteval.tasks.requests": mod("lighteval.tasks.requests", Doc=Rec, SamplingMethod=sampling),
        "lighteval.tasks.lighteval_task": mod("lighteval.tasks.lighteval_task", LightevalTaskConfig=Rec),
        "lighteval.utils": mod("lighteval.utils"),
        "lighteval.utils.cache_management": mod("lighteval.utils.cache_management", SampleCache=cache, cached=cached),
        "lighteval.metrics": mod("lighteval.metrics"),
        "lighteval.metrics.metrics": mod("lighteval.metrics.metrics", Metrics=metrics),
    }


def load(name):
    with mock.patch.dict(sys.modules, fake_lighteval()):
        spec = importlib.util.spec_from_file_location("stub_" + name, os.path.join(HERE, name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    return m


class FakeResp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.body).encode()


class TasksTest(unittest.TestCase):
    def test_prompts_and_table(self):
        t = load("ferrum_tasks")
        d = t.heldout_prompt({"text": "hi"}, "x")
        self.assertEqual((d.query, d.choices, d.task_name), ("hi", None, "x"))
        c = t.cloze_prompt({"question": "q", "choices": ["a", "b"], "gold": 1})
        self.assertEqual((c.query, c.choices, c.gold_index), ("q", [" a", " b"], 1))
        self.assertEqual([x.name for x in t.TASKS_TABLE], ["ferrum_heldout", "tinystories_cloze"])
        self.assertTrue(t.TASKS_TABLE[0].hf_repo.endswith(os.path.join("heldout")))


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.fm = load("ferrum_model")
        env = {"FERRUM_URL": "http://h:1/", "FERRUM_BATCH": "2", "FERRUM_CACHE_DIR": "/nonexistent-cache"}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("FERRUM_TOKEN", None)
        self.cfg = Rec()
        self.m = self.fm.FerrumModel(self.cfg)

    def test_init_private_cache_and_props(self):
        self.assertEqual(self.cfg.cache_dir, "/nonexistent-cache")
        self.assertEqual(self.m.url, "http://h:1")
        self.assertIsNone(self.m.tokenizer)
        self.assertFalse(self.m.add_special_tokens)

    def test_temp_cache_dir_when_unset(self):
        os.environ.pop("FERRUM_CACHE_DIR")
        with mock.patch.object(self.fm.atexit, "register") as reg:
            d = self.fm._private_cache_dir()
        self.assertTrue(os.path.isdir(d))
        reg.assert_called_once()
        os.rmdir(d)

    def test_batches(self):
        self.assertEqual(list(self.fm._batches([1, 2, 3], 2)), [[1, 2], [3]])

    def test_call_get_post_and_token(self):
        seen = []

        def opener(req, timeout):
            seen.append(req)
            return FakeResp({"ok": 1})

        with mock.patch.object(self.fm.urllib.request, "urlopen", opener):
            self.assertEqual(self.m._call("/v1/info"), {"ok": 1})
            os.environ["FERRUM_TOKEN"] = "t"
            self.m._call("/p", {"a": 1})
        self.assertEqual((seen[0].get_method(), seen[0].data), ("GET", None))
        self.assertNotIn("Authorization", seen[0].headers)
        self.assertEqual(seen[1].get_method(), "POST")
        self.assertEqual(seen[1].headers["Authorization"], "Bearer t")

    def test_call_errors(self):
        http = urllib.error.HTTPError("u", 422, "x", {}, io.BytesIO(b"kaput"))
        with mock.patch.object(self.fm.urllib.request, "urlopen", side_effect=http):
            with self.assertRaisesRegex(RuntimeError, "returned 422: kaput"):
                self.m._call("/g", {})
        with mock.patch.object(self.fm.urllib.request, "urlopen", side_effect=urllib.error.URLError("down")):
            with self.assertRaisesRegex(RuntimeError, "cannot reach ferrum-serve.*down"):
                self.m._call("/g")

    def test_max_length_cached(self):
        with mock.patch.object(self.m, "_call", return_value={"context_len": 128}) as c:
            self.assertEqual((self.m.max_length, self.m.max_length), (128, 128))
        c.assert_called_once_with("/v1/info")

    def test_greedy_until(self):
        sent = []

        def call(path, body):
            sent.append((path, body))
            return {"results": [{"text": "t", "n_tokens": 2} for _ in body["requests"]]}

        docs = [Rec(query="a", generation_size=-1, stop_sequences=None), Rec(query="b", generation_size=5, stop_sequences=["."]),
                Rec(query="c", generation_size=None, stop_sequences=[])]
        with mock.patch.object(self.m, "_call", call):
            out = self.m.greedy_until(docs)
        self.assertEqual(len(sent), 2)  # batch of 2
        reqs = sent[0][1]["requests"] + sent[1][1]["requests"]
        self.assertEqual([r["max_tokens"] for r in reqs], [64, 5, 64])
        self.assertEqual(reqs[1]["stop"], ["."])
        self.assertEqual((out[0].text, out[0].output_tokens), (["t"], [[0, 0]]))

    def test_loglikelihood_and_rolling(self):
        def call(path, body):
            if path == "/v1/loglikelihood":
                return {"results": [{"logprob": -1.0, "greedy": True} for _ in body["requests"]]}
            return {"results": [{"logprob": -2.0} for _ in body["texts"]]}

        docs = [Rec(query="q", choices=[" a", " b"]), Rec(query="r", choices=[" c"])]
        with mock.patch.object(self.m, "_call", call):
            out = self.m.loglikelihood(docs)
            roll = self.m.loglikelihood_rolling(docs)
        self.assertEqual([len(o.logprobs) for o in out], [2, 1])
        self.assertEqual(out[0].argmax_logits_eq_gold, [True, True])
        self.assertEqual(roll[1].logprobs, [-2.0])


class UsageTest(unittest.TestCase):
    def test_make_task_data_usage_exits(self):
        with mock.patch.object(sys, "argv", ["make_task_data.py"]):
            with self.assertRaises(SystemExit) as e:
                runpy.run_path(os.path.join(HERE, "make_task_data.py"), run_name="__main__")
        self.assertIn("usage", str(e.exception))


if __name__ == "__main__":
    unittest.main()
