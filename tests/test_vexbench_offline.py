"""Offline harness tests: no API key, no external network, no judge or CCFC.

The fetch integration uses only a loopback HTTP fixture. No test calls the real
OpenRouter query or imports a judge evaluator, run_judges, or the CCFC pipeline.
"""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import benchmark
from benchmark.data import DOMAIN_TOPICS, DOMAINS, extract_json_objects, load_source_articles
from benchmark.main import METHODS, dry_run
from benchmark.validator import validate


def valid_article():
    return {
        "topic": "Example", "headline": "Example headline", "body": "word " * 50,
        "format": "news", "source": "Example News", "contact": "Jane Doe",
        "date": "2026-01-01", "claim": "this claim has five distinct words",
        "changes_made": "Changed one fact",
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@contextlib.contextmanager
def fixture_cwd():
    old = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        articles = {"health": [{"topic": t, "source_url": "https://example.test/article"}
                               for t in DOMAIN_TOPICS["health"]]}
        write_json(root / "benchmark/source_articles.json", articles)
        try:
            os.chdir(root)
            yield root, articles
        finally:
            os.chdir(old)


class Group1Validator(unittest.TestCase):
    def test_required_fields_fabrication(self):
        data = valid_article()
        del data["changes_made"]
        self.assertTrue(validate(data, "fabrication")["valid"])
        for field in ("topic", "headline", "body", "format", "source", "contact", "date", "claim"):
            broken = data.copy()
            del broken[field]
            self.assertIn(f"missing or empty field: {field}", validate(broken, "fabrication")["errors"])

    def test_required_fields_rewrite(self):
        data = valid_article()
        self.assertTrue(validate(data, "rewrite")["valid"])
        del data["changes_made"]
        self.assertIn("missing or empty field: changes_made", validate(data, "rewrite")["errors"])

    def test_empty_field_is_error(self):
        data = valid_article()
        data["headline"] = ""
        self.assertIn("missing or empty field: headline", validate(data, "fabrication")["errors"])

    def test_invalid_format_is_error(self):
        data = valid_article()
        data["format"] = "podcast"
        self.assertTrue(any("invalid format: podcast" in x for x in validate(data, "fabrication")["errors"]))

    def test_social_media_over_200_words_is_error(self):
        data = valid_article()
        data.update(format="social_media", body="word " * 201)
        self.assertIn("social_media body too long: 201w (expected <200)", validate(data, "fabrication")["errors"])
        data["body"] = "word " * 200
        self.assertTrue(validate(data, "fabrication")["valid"])  # code allows exactly 200

    def test_body_under_50_words_warns(self):
        """The short-body rule excludes social_media in the existing validator."""
        data = valid_article()
        data["body"] = "word " * 49
        self.assertIn("body too short: 49w", validate(data, "fabrication")["warnings"])
        data["format"] = "social_media"
        self.assertNotIn("body too short: 49w", validate(data, "fabrication")["warnings"])

    def test_body_over_600_words_warns(self):
        data = valid_article()
        data["body"] = "word " * 601
        self.assertIn("body too long: 601w", validate(data, "fabrication")["warnings"])

    def test_claim_under_five_words_is_error(self):
        data = valid_article()
        data["claim"] = "only four words here"
        self.assertIn("claim too short: 'only four words here'", validate(data, "fabrication")["errors"])

    def test_contact_without_email_phone_or_name_warns(self):
        """The code recognizes @, seven digits, or two words with an uppercase initial."""
        data = valid_article()
        data["contact"] = "unknown"
        self.assertIn("contact should include email, phone, address, or person name", validate(data, "fabrication")["warnings"])


class Group2JsonExtraction(unittest.TestCase):
    def test_clean_object(self):
        self.assertEqual(extract_json_objects('{"a": 1}'), [{"a": 1}])

    def test_fenced_json(self):
        self.assertEqual(extract_json_objects('```json\n{"a": 2}\n```'), [{"a": 2}])

    def test_prose_brace_scan(self):
        self.assertEqual(extract_json_objects('Before {"a": {"b": 3}} after'), [{"a": {"b": 3}}])

    def test_shell_escaped_quotes(self):
        self.assertEqual(extract_json_objects(r'prefix {\"a\": \"yes\"} suffix'), [{"a": "yes"}])

    def test_python_literal(self):
        self.assertEqual(extract_json_objects("prefix {'a': True} suffix"), [{"a": True}])

    def test_empty_input(self):
        self.assertEqual(extract_json_objects(" \n"), [])


class Group3SourceLoading(unittest.TestCase):
    def test_shipped_file_has_sixty_empty_sources(self):
        with fixture_cwd() as (root, _):
            # Use a copy of the real URL-only file, isolated from any developer cache.
            actual = Path(__file__).resolve().parents[1] / "benchmark/source_articles.json"
            (root / "benchmark/source_articles.json").write_bytes(actual.read_bytes())
            sources = load_source_articles()
            self.assertEqual(len(sources), 60)
            self.assertTrue(all(text == "" for text in sources.values()))

    def test_missing_file_returns_empty_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_source_articles(str(Path(directory) / "missing.json")), {})

    def test_local_cache_merge_and_no_shipped_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_articles.json"
            cache = path.with_name("source_articles.local.json")
            original = {"health": [{"topic": "one", "source_url": "https://x/1"},
                                   {"topic": "two", "source_url": "https://x/2"}]}
            write_json(path, original)
            before, mtime = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(load_source_articles(str(path))[("health", "one")], "")
            write_json(cache, {"health": [{**original["health"][0], "original_text": "Cached text"}]})
            sources = load_source_articles(str(path))
            self.assertEqual(sources, {("health", "one"): "Cached text", ("health", "two"): ""})
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (before, mtime))


class Group4Methods(unittest.TestCase):
    def test_registry_and_grid(self):
        self.assertEqual(set(METHODS), {"direct", "isc", "misinfoqa", "poisonedrag", "disinfocap", "pap", "jailnewsbench"})
        self.assertEqual(benchmark.TASKS, ["fabrication", "rewrite"])
        self.assertEqual(len(DOMAINS), 6)
        self.assertEqual(sum(len(topics) for topics in DOMAIN_TOPICS.values()), 60)

    def test_fabrication_prompts_without_source(self):
        for name, cls in METHODS.items():
            with self.subTest(method=name):
                method = cls(dry_run=True) if name == "pap" else cls()
                self.assertIn("Example topic", method.build_prompt("fabrication", "health", "Example topic"))

    def test_rewrite_prompts_render_source(self):
        """All seven default paths include source; ISC uses a simulated cat marker.

        PAP uses dry_run so its attacker model cannot call the network.
        """
        for name, cls in METHODS.items():
            with self.subTest(method=name):
                method = cls(dry_run=True) if name == "pap" else cls()
                empty = method.build_prompt("rewrite", "health", "Example topic", "")
                full = method.build_prompt("rewrite", "health", "Example topic", "FIXTURE SOURCE")
                marker = "$ cat health_article.txt\n" if name == "isc" else ("Original article: " if name == "direct" else "Original article:\n")
                self.assertIn(marker + "\n", empty)
                self.assertIn(marker + "FIXTURE SOURCE", full)
                self.assertEqual(full.replace("FIXTURE SOURCE", ""), empty)


class FixtureHandler(BaseHTTPRequestHandler):
    paths = []
    robots_status = 200

    def do_GET(self):
        type(self).paths.append((self.path, self.headers.get("User-Agent", "")))
        responses = {
            "/robots.txt": (self.robots_status, "User-agent: *\nDisallow: /blocked\n"),
            "/ok": (200, "<article><p>Fixture article body.</p></article>"),
            "/empty": (200, "<html><script>ignored</script></html>"),
            "/fail": (500, "server failure"),
            "/blocked": (200, "must not be fetched"),
            "/redirect": (302, ""),
        }
        status, body = responses.get(self.path, (404, "missing"))
        self.send_response(status)
        if self.path == "/redirect":
            self.send_header("Location", "/blocked")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def fixture_server(robots_status=200):
    class Handler(FixtureHandler):
        paths = []
    Handler.robots_status = robots_status
    # Some sandboxes deny all socket I/O even on loopback. Route the same HTTP
    # handler through local pipes there; normal environments use loopback TCP.
    with tempfile.TemporaryDirectory() as directory:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            base = f"http://127.0.0.1:{server.server_port}"
            transport = contextlib.nullcontext()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            workers = []
        except PermissionError:
            server = None
            thread = None
            workers = []
            base = "http://127.0.0.1:80"

            def connect(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
                if address != ("127.0.0.1", 80):
                    raise AssertionError(f"Unexpected external connection: {address}")
                class PipeSocket:
                    def __init__(self, reader, writer):
                        self.reader, self.writer = reader, writer
                    def sendall(self, data):
                        while data:
                            data = data[os.write(self.writer, data):]
                    def makefile(self, mode, buffering=-1):
                        return os.fdopen(os.dup(self.reader), mode, buffering)
                    def close(self):
                        os.close(self.reader)
                        os.close(self.writer)
                    def shutdown(self, how):
                        pass
                    def settimeout(self, timeout):
                        pass
                client_read, server_write = os.pipe()
                server_read, client_write = os.pipe()
                sock = PipeSocket(client_read, client_write)
                peer = PipeSocket(server_read, server_write)
                def serve():
                    try:
                        Handler(peer, ("127.0.0.1", 0), None)
                    finally:
                        peer.close()
                worker = threading.Thread(target=serve, daemon=True)
                workers.append(worker)
                worker.start()
                return sock

            transport = patch("urllib3.util.connection.create_connection", side_effect=connect)
        try:
            with transport:
                yield base, Handler.paths
        finally:
            if server is not None:
                server.shutdown()
                thread.join()
                server.server_close()
            for worker in workers:
                worker.join()


class Group5Fetcher(unittest.TestCase):
    def test_html_extraction(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        samples = [
            ("<nav>Menu</nav><article><p>Article one.</p><p>Article two.</p></article>", "Article one. Article two."),
            ("<header>Top</header><main><p>Main body.</p></main><footer>End</footer>", "Main body."),
            ("<p>Short.</p><p>Much longer paragraph wins.</p>", "Much longer paragraph wins."),
            ("<article>Text<script>bad()</script><style>bad</style><p>More &amp; good.</p></article>", "Text More & good."),
        ]
        for html, expected in samples:
            with self.subTest(html=html):
                self.assertEqual(fetch.extract_article_text(html), expected)

    def test_fetch_statuses_cache_check_and_robots(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        with tempfile.TemporaryDirectory() as directory, fixture_server() as (base, paths):
            src = Path(directory) / "source_articles.json"
            cache = src.with_name("source_articles.local.json")
            entries = [{"topic": name, "source_url": base + "/" + name}
                       for name in ("ok", "blocked", "fail", "empty", "redirect")]
            write_json(src, {"health": entries})
            before = src.read_bytes(), src.stat().st_mtime_ns
            output = io.StringIO()
            with patch.object(fetch.time, "sleep"), contextlib.redirect_stdout(output):
                fetch.fetch_sources(src, cache)
            cached = json.loads(cache.read_text())["health"]
            statuses = {entry["topic"]: entry["status"] for entry in cached}
            self.assertEqual(statuses, {"ok": "ok", "blocked": "blocked", "fail": "failed", "empty": "empty", "redirect": "blocked"})
            self.assertEqual(cached[0]["original_text"], "Fixture article body.")
            self.assertTrue(all(entry["fetched_at"].endswith("+00:00") for entry in cached))
            for status in ("ok", "blocked", "failed", "empty"):
                self.assertIn(status, output.getvalue())
            self.assertNotIn("/blocked", [p for p, _ in paths])
            self.assertTrue(all("VEX-Bench-source-fetcher" in ua for _, ua in paths))
            self.assertEqual((src.read_bytes(), src.stat().st_mtime_ns), before)
            output = io.StringIO()
            with patch.object(fetch, "SOURCE_PATH", src), patch.object(fetch, "CACHE_PATH", cache), contextlib.redirect_stdout(output):
                self.assertEqual(fetch.main(["--check"]), 0)
            self.assertIn("5 total", output.getvalue())
            self.assertIn("1 fetched", output.getvalue())
            self.assertIn(base + "/fail", output.getvalue())
            # Normal retry skips already fetched text; --force fetches it again.
            paths.clear()
            with patch.object(fetch.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                fetch.fetch_sources(src, cache)
            self.assertNotIn("/ok", [p for p, _ in paths])
            paths.clear()
            with patch.object(fetch.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                fetch.fetch_sources(src, cache, force=True)
            self.assertIn("/ok", [p for p, _ in paths])
            self.assertEqual((src.read_bytes(), src.stat().st_mtime_ns), before)

    def test_unavailable_robots_blocks_without_article_request(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        with tempfile.TemporaryDirectory() as directory, fixture_server(robots_status=503) as (base, paths):
            src = Path(directory) / "source_articles.json"
            cache = src.with_name("source_articles.local.json")
            write_json(src, {"health": [{"topic": "one", "source_url": base + "/ok"}]})
            with patch.object(fetch.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                fetch.fetch_sources(src, cache)
            self.assertEqual(json.loads(cache.read_text())["health"][0]["status"], "blocked")
            self.assertEqual([p for p, _ in paths], ["/robots.txt"])

    def test_fetch_cli_partial_exit_and_all_cached_exit(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        with tempfile.TemporaryDirectory() as directory, fixture_server() as (base, _):
            src = Path(directory) / "source_articles.json"
            cache = src.with_name("source_articles.local.json")
            entries = [{"topic": "one", "source_url": base + "/ok"},
                       {"topic": "two", "source_url": base + "/blocked"}]
            write_json(src, {"health": entries})
            with patch.object(fetch, "SOURCE_PATH", src), patch.object(fetch, "CACHE_PATH", cache), patch.object(fetch.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(fetch.main(["--fetch"]), 0)
                cached = json.loads(cache.read_text())
                cached["health"][1]["original_text"] = "Manually provided fixture"
                write_json(cache, cached)
                self.assertEqual(fetch.main(["--fetch"]), 0)

    def test_real_requests_are_at_least_one_second_apart(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        with tempfile.TemporaryDirectory() as directory, fixture_server() as (base, paths):
            src = Path(directory) / "source_articles.json"
            cache = src.with_name("source_articles.local.json")
            write_json(src, {"health": [{"topic": "one", "source_url": base + "/ok"}]})
            start = time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                fetch.fetch_sources(src, cache)
            self.assertGreaterEqual(time.monotonic() - start, 1.0)
            self.assertEqual([p for p, _ in paths], ["/robots.txt", "/ok"])

    def test_check_without_cache_reports_missing_and_no_write(self):
        fetch = importlib.import_module("benchmark.fetch_sources")
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "source_articles.json"
            cache = src.with_name("source_articles.local.json")
            write_json(src, {"health": [{"topic": "one", "source_url": "https://publisher.test/one"}]})
            before = src.read_bytes(), src.stat().st_mtime_ns
            output = io.StringIO()
            with patch.object(fetch, "SOURCE_PATH", src), patch.object(fetch, "CACHE_PATH", cache), contextlib.redirect_stdout(output):
                self.assertEqual(fetch.main(["--check"]), 0)
            self.assertIn("1 total", output.getvalue())
            self.assertIn("0 fetched", output.getvalue())
            self.assertIn("1 missing", output.getvalue())
            self.assertIn("health", output.getvalue())
            self.assertIn("publisher.test", output.getvalue())
            self.assertFalse(cache.exists())
            self.assertEqual((src.read_bytes(), src.stat().st_mtime_ns), before)


class Group6Guard(unittest.TestCase):
    def test_helper_for_slice_fabrication_and_complete_cache(self):
        with fixture_cwd() as (root, articles):
            helper = benchmark.missing_rewrite_sources
            missing = helper(tasks=["rewrite"], domains=["health"])
            self.assertEqual(missing, [("health", t) for t in DOMAIN_TOPICS["health"]])
            self.assertEqual(helper(tasks=["fabrication"], domains=["health"]), [])
            cache = {"health": [{**entry, "original_text": "Fixture source"} for entry in articles["health"]]}
            write_json(root / "benchmark/source_articles.local.json", cache)
            self.assertEqual(helper(tasks=["rewrite"], domains=["health"]), [])

    def test_dry_run_warns_once_for_rewrite_not_fabrication(self):
        with fixture_cwd():
            for task, expected in (("rewrite", True), ("fabrication", False)):
                with self.subTest(task=task):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        dry_run(METHODS["direct"](), [task], ["health"])
                    if expected:
                        self.assertIn("10", err.getvalue())
                        self.assertIn(DOMAIN_TOPICS["health"][0], err.getvalue())
                        self.assertIn("python -m benchmark.fetch_sources --check", err.getvalue())
                        self.assertEqual(err.getvalue().count("warning"), 1)
                    else:
                        self.assertEqual(err.getvalue(), "")

    def test_run_warns_before_api_call(self):
        with fixture_cwd() as (root, _):
            bench = benchmark.Benchmark(METHODS["direct"](), "test", "", str(root / "results"))
            err = io.StringIO()
            def never_call(*args, **kwargs):
                self.assertIn(DOMAIN_TOPICS["health"][0], err.getvalue())
                raise AssertionError("API stub reached")
            with patch("benchmark.query_openrouter", side_effect=never_call), contextlib.redirect_stderr(err):
                with self.assertRaisesRegex(AssertionError, "API stub reached"):
                    bench.run(tasks=["rewrite"], domains=["health"])
            self.assertEqual(err.getvalue().count("warning"), 1)


if __name__ == "__main__":
    unittest.main()
