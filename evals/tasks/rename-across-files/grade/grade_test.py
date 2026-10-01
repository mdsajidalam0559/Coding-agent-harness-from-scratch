import os
import subprocess
import sys
import unittest


class Grade(unittest.TestCase):
    def test_renamed_everywhere(self):
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name.endswith(".py") and name != "grade_test.py":
                    self.assertNotIn("calc_total", open(os.path.join(root, name)).read(), name)
        from shop.pricing import compute_subtotal
        self.assertEqual(compute_subtotal([(2, 3)]), 6)

    def test_output(self):
        out = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "30.0")
