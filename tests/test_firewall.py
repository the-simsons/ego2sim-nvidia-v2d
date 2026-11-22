"""The mocap firewall (plans/RECONSTRUCTION_PLAN.md P0.5): reconstruction code must not import e2s.mocap.

Allowed importers: e2s/mocap itself and bench/ (the scoring side). Everything else (tools/, scripts/*.py, the rest
of e2s/) is reconstruction or training and must not see motion capture. Run: python -m unittest discover tests"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = [ROOT / "tools", ROOT / "scripts", ROOT / "e2s"]
ALLOWED = [ROOT / "e2s" / "mocap", ROOT / "bench", ROOT / "tests"]


def mocap_imports(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.startswith("e2s.mocap")]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("e2s.mocap") or (mod == "e2s" and any(a.name == "mocap" for a in node.names)):
                hits.append(mod)
    return [(node_name, path) for node_name in hits]


class FirewallTest(unittest.TestCase):
    def test_no_reconstruction_code_imports_mocap(self):
        bad = []
        for d in CHECKED:
            for f in d.rglob("*.py"):
                if any(f.is_relative_to(a) for a in ALLOWED):
                    continue
                bad += [f"{p.relative_to(ROOT)} imports {m}" for m, p in mocap_imports(f)]
        self.assertEqual(bad, [], "reconstruction code must not import e2s.mocap:\n" + "\n".join(bad))

    def test_detector_catches_an_import(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.py"
            p.write_text("from e2s.mocap import motive\nimport e2s.mocap.sync\nfrom e2s import mocap\n")
            self.assertEqual(len(mocap_imports(p)), 3)


if __name__ == "__main__":
    unittest.main()
