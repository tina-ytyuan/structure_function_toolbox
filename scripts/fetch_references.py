"""Download the published normative references into outputs/.

The reference files are ~15 MB each and are distributed as a GitHub release
rather than committed to the repository. This fetches them, verifies each
against the published SHA-256 checksums, and places them where the app looks
(``outputs/measure_ref_<measure>.npz``).

Usage
-----
    python scripts/fetch_references.py                 # all 11 measures
    python scripts/fetch_references.py rsfa alff reho  # only these
    python scripts/fetch_references.py --force         # re-download everything

Only the standard library is used, so this works before installing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "tina-ytyuan/structure_function_toolbox"
TAG = "refs-v1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

MEASURES = [
    "rsfa", "alff", "falff", "reho",
    "alff_slow4", "alff_slow5", "falff_slow4", "falff_slow5",
    "int", "coherence_reho", "mse",
]


def _ssl_context():
    """A context with a usable certificate bundle.

    Python installed from python.org on macOS ships its own certificate store
    and does not read the system one, so HTTPS fails with
    CERTIFICATE_VERIFY_FAILED until the bundled Install Certificates command is
    run. certifi provides the same bundle and arrives with the toolbox's
    dependencies, so prefer it and fall back to the system default.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(url: str, dest: Path, label: str) -> bool:
    """Stream a file to disk, showing progress. Returns True on success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, context=_ssl_context()) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r  {label}: {pct:5.1f}%  "
                              f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)",
                              end="", flush=True)
        print(f"\r  {label}: done ({done / 1e6:.0f} MB)          ")
        tmp.replace(dest)
        return True
    except urllib.error.HTTPError as e:
        print(f"\r  {label}: HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        reason = str(e.reason)
        print(f"\r  {label}: {reason}")
        if "CERTIFICATE_VERIFY_FAILED" in reason:
            print("     Your Python cannot verify HTTPS certificates. Fix with:")
            print("       pip install certifi")
            print("     or, on macOS with python.org Python, run once:")
            print('       open "/Applications/Python 3.x/Install Certificates.command"')
    finally:
        tmp.unlink(missing_ok=True)
    return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(out_dir: Path) -> dict[str, str]:
    """Fetch SHA256SUMS.txt so downloads can be verified."""
    dest = out_dir / "SHA256SUMS.txt"
    if not dest.exists() and not download(f"{BASE}/SHA256SUMS.txt", dest,
                                          "SHA256SUMS.txt"):
        return {}
    sums = {}
    for line in dest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    return sums


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("measures", nargs="*", default=None,
                    help=f"Measures to fetch (default: all). One of: "
                         f"{', '.join(MEASURES)}")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file is already present")
    args = ap.parse_args()

    wanted = args.measures or MEASURES
    unknown = [m for m in wanted if m not in MEASURES]
    if unknown:
        raise SystemExit(f"Unknown measure(s): {', '.join(unknown)}\n"
                         f"Available: {', '.join(MEASURES)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(wanted)} reference(s) from {REPO} ({TAG})")
    print(f"into {out_dir.resolve()}\n")

    sums = load_checksums(out_dir)
    if not sums:
        print("  (checksums unavailable; downloads will not be verified)\n")

    ok, skipped, failed = 0, 0, []
    for m in wanted:
        name = f"measure_ref_{m}.npz"
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"  {name}: already present, skipping")
            skipped += 1
            continue
        if not download(f"{BASE}/{name}", dest, name):
            failed.append(m)
            continue
        expected = sums.get(name)
        if expected:
            actual = sha256(dest)
            if actual != expected:
                print(f"  {name}: CHECKSUM MISMATCH, removing")
                dest.unlink(missing_ok=True)
                failed.append(m)
                continue
            print(f"  {name}: checksum ok")
        ok += 1

    print(f"\n{ok} downloaded, {skipped} already present, {len(failed)} failed")
    if failed:
        print(f"failed: {', '.join(failed)}")
        print("If every download failed, check that the release exists and its "
              "assets are public.")
        sys.exit(1)
    print("\nStart the app with:  python -m sftoolbox.webapp")


if __name__ == "__main__":
    main()
