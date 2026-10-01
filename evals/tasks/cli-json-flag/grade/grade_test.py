import json
import subprocess
import sys
import tempfile
import unittest

TEXT = "the cat and the hat and the bat sat"


def run(*args):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(TEXT)
    return subprocess.run([sys.executable, "wordfreq.py", f.name, *args], capture_output=True, text=True)


class Grade(unittest.TestCase):
    def test_text_unchanged(self):
        self.assertEqual(run("--top", "2").stdout, "the: 3\nand: 2\n")

    def test_json(self):
        result = run("--json", "--top", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data, {"the": 3, "and": 2})
        self.assertEqual(list(data), ["the", "and"])

    def test_json_default_top(self):
        self.assertEqual(len(json.loads(run("--json").stdout)), 5)
