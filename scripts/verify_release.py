"""End-to-end release verification.

Run from the release root in a fresh Python 3.10+ virtual environment:

    python -m venv .venv && . .venv/bin/activate
    python scripts/verify_release.py

The script installs both packages into the current environment, runs their
test suites, and then checks the dataset manifest, the dashboard zip, tree
hygiene, and git cleanliness. It prints one PASS line per check and exits
non-zero on the first failure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# SHA-256 of the released dashboard_standalone.html revision (the exact
# served file that was zipped into dashboard/dashboard_standalone.html.zip).
DASHBOARD_SHA256 = \
    "d9aebf31adf760dbd4dd0de621b540ae030cb58e5c22b6fb28af70318d57b732"

MAX_FILE_BYTES = 25 * 1024 * 1024
ZIP_RELPATH = "dashboard/dashboard_standalone.html.zip"

PASSED = []


def ok(name: str, detail: str = ""):
    PASSED.append(name)
    print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))


def fail(name: str, detail: str):
    print(f"FAIL  {name}: {detail}")
    sys.exit(1)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, **kw)


# ---------------------------------------------------------------------- 1
def check_install_and_tests():
    r = run([sys.executable, "-m", "pip", "install", "--quiet",
             "./forking_paths", "./otrecon", "pytest>=8"])
    if r.returncode != 0:
        fail("install", "pip install failed")
    r = run([sys.executable, "-c",
             "import forking_paths, otrecon; "
             "print(forking_paths.__version__, otrecon.__name__)"],
            capture_output=True, text=True)
    if r.returncode != 0:
        fail("import", r.stderr.strip()[-500:])
    ok("install+import", r.stdout.strip())

    for suite in ("otrecon/tests", "forking_paths/tests"):
        r = run([sys.executable, "-m", "pytest", suite, "-q"],
                capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()
        if r.returncode != 0:
            print(r.stdout[-4000:])
            fail(f"pytest {suite}", "test failures")
        ok(f"pytest {suite}", tail[-1] if tail else "")


# ---------------------------------------------------------------------- 2
def check_data():
    _spec = importlib.util.spec_from_file_location(
        "release_loader", os.path.join(DATA, "loader.py"))
    loader = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(loader)
    import numpy as np

    with open(os.path.join(DATA, "MANIFEST.json")) as f:
        manifest = json.load(f)

    expected = {f"s200/{t}/row{i:03d}.json.gz"
                for t in ("llama", "deepseek") for i in range(100)}
    expected |= {"s1000/llama/row012_stride4.json.gz",
                 "s1000/llama/row039_stride4.json.gz",
                 "s1000/llama/row039_everytoken.json.gz"}
    on_disk = {rel for rel, _ in loader.iter_stores(DATA)}
    if on_disk != expected:
        fail("data coverage",
             f"disk vs expected: +{sorted(on_disk - expected)[:5]} "
             f"-{sorted(expected - on_disk)[:5]}")
    if set(manifest["files"]) != expected:
        fail("data coverage", "manifest file set != expected set")
    ok("data coverage",
       "s200 rows 0-99 x {llama,deepseek} + 3 s1000 stores (203 files)")

    for rel in sorted(expected):
        path = os.path.join(DATA, rel)
        raw = open(path, "rb").read()
        entry = manifest["files"][rel]
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            fail("data sha256", rel)
        if len(raw) != entry["bytes"]:
            fail("data size", rel)
    ok("data sha256", f"{len(expected)}/{len(expected)} files match the manifest")

    # identity spot fields for every file
    for rel, entry in manifest["files"].items():
        want_s = 200 if rel.startswith("s200/") else 1000
        want_track = rel.split("/")[1]
        want_row = int(re.search(r"row(\d{3})", rel).group(1))
        if (entry["s"], entry["track"], entry["row_id"]) != \
                (want_s, want_track, want_row):
            fail("data identity", f"{rel}: {entry}")
    ok("data identity", "s / track / row_id consistent with filenames")

    # loader round-trip: recompute the full-pool o_t of the reference store
    ref = manifest["reference_o_t"]
    rec = loader.load_store(os.path.join(DATA, ref["file"]))
    _, o = loader.o_t(rec)
    curve = np.round(o, 6).tolist()
    if curve != rec["o_t_full"]:
        fail("loader round-trip", f"{ref['file']}: recomputed != recorded")
    sha = hashlib.sha256(
        json.dumps(curve, separators=(",", ":")).encode()).hexdigest()
    if sha != ref["sha256"]:
        fail("loader round-trip", "curve sha != manifest reference sha")
    ok("loader round-trip",
       f"{ref['file']}: recomputed o_t == recorded o_t_full exactly")


# ---------------------------------------------------------------------- 3
def check_dashboard_zip():
    path = os.path.join(ROOT, ZIP_RELPATH)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if names != ["dashboard_standalone.html"]:
            fail("dashboard zip", f"unexpected contents: {names}")
        html = z.read("dashboard_standalone.html")
    sha = hashlib.sha256(html).hexdigest()
    if sha != DASHBOARD_SHA256:
        fail("dashboard zip", f"extracted sha {sha} != released revision")
    ok("dashboard zip",
       f"byte-identical to the released revision ({len(html)/1e6:.1f} MB)")

    # zero external network references: nothing in the document loads a
    # remote resource when opened (plotly.js is inlined; the http(s) strings
    # inside it are namespace/doc-link constants, not load-bearing tags).
    forbidden = [b"<script src=", b"<link ", b"<img ", b"<iframe",
                 b"<embed ", b"<object ", b"srcset=", b"@import",
                 b"url(http", b"url(\"http", b"url('http"]
    hits = [p.decode() for p in forbidden if p in html]
    if hits:
        fail("dashboard offline", f"load-bearing reference(s): {hits}")
    ok("dashboard offline", "no external network references")


# ---------------------------------------------------------------------- 4
def check_tree_hygiene():
    r = run(["git", "ls-files"], capture_output=True, text=True)
    if r.returncode != 0:
        fail("git ls-files", r.stderr.strip())
    tracked = [p for p in r.stdout.splitlines() if p]

    for p in tracked:
        if p.endswith(".zst") or ".jsonl.zst" in p or "_text/" in p \
                or p.endswith("_text.jsonl"):
            fail("sidecar exclusion", f"raw-completion sidecar present: {p}")
    ok("sidecar exclusion", "no raw-completion sidecars in the tree")

    for p in tracked:
        size = os.path.getsize(os.path.join(ROOT, p))
        if size > MAX_FILE_BYTES and p != ZIP_RELPATH:
            fail("file size", f"{p} is {size/1e6:.1f} MB (> 25 MB)")
    ok("file size", "no tracked file over 25 MB except the dashboard zip")

    # internal-path scan over prose and code (README, python, toml, config);
    # the store JSONs keep their collection-time provenance fields and the
    # zipped dashboard is the pinned served artifact, so both are exempt.
    pats = ["/mnt" + "/", "/srv" + "/", "_flat" + "/", "exp_01" + "k",
            "sil" + "ico"]
    scan_ext = (".py", ".md", ".toml", ".cfg", ".txt", ".gitignore")
    bad = []
    for p in tracked:
        if not (p.endswith(scan_ext) or os.path.basename(p) == ".gitignore"):
            continue
        text = open(os.path.join(ROOT, p), encoding="utf-8",
                    errors="replace").read().lower()
        for pat in pats:
            if pat in text:
                bad.append((p, pat))
    if bad:
        fail("internal paths", str(bad[:10]))
    ok("internal paths", "no internal cluster paths in README/code/docs")


# ---------------------------------------------------------------------- 5
def check_git_clean():
    r = run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if r.returncode != 0:
        fail("git status", r.stderr.strip())
    if r.stdout.strip():
        fail("git clean", f"uncommitted changes:\n{r.stdout[:2000]}")
    ok("git clean", "working tree matches HEAD")


def main():
    check_install_and_tests()
    check_data()
    check_dashboard_zip()
    check_tree_hygiene()
    check_git_clean()
    print(f"\nALL CHECKS PASSED ({len(PASSED)} checks)")


if __name__ == "__main__":
    main()
