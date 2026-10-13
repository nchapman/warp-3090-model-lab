"""Upload the frozen data sets to the HF Hub mirror
(nchapman/warp-3090-quant-lab-frozen-data).

Dual-homing rationale (README data section): revision-hash citability,
pull-by-pods without a repo clone, license/provenance surface via dataset card.
Run in either venv (huggingface_hub is a dependency of both):

  .venv-quant/bin/python scripts/upload_frozen_data.py [--repo nchapman/warp-3090-quant-lab-frozen-data]

Verifies MANIFEST.sha256 before uploading anything; aborts on drift.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CARD = ROOT / "hf" / "frozen-data" / "README.md"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="nchapman/warp-3090-quant-lab-frozen-data")
    ap.add_argument("--public", action="store_true",
                    help="publish publicly (default: private repo)")
    ap.add_argument("--allow-nc-public", action="store_true",
                    help="required with --public: probe-guard contains "
                         "CC-BY-NC-4.0 no_robots rows; this flag records an "
                         "explicit decision to redistribute them publicly")
    args = ap.parse_args()

    NC_LICENSED = {"probe-guard.jsonl"}  # no_robots-derived rows (CC-BY-NC-4.0)
    # up-front gate: refuse the public+NC combination BEFORE touching the Hub
    # (creating a public repo and then refusing would still be a mutation)
    if args.public and not args.allow_nc_public:
        sys.exit(f"refusing --public: {sorted(NC_LICENSED)} carry CC-BY-NC-4.0 "
                 f"rows; pass --allow-nc-public to record the redistribution "
                 f"decision, or upload private (the default)")

    # integrity gate: never upload drifted data, and upload EXACTLY the
    # manifest's file list — stray files under data/ abort instead of
    # silently shipping unpinned
    manifest = []
    for line in (DATA / "MANIFEST.sha256").read_text().splitlines():
        fields = line.split(None, 1)
        if not fields:
            continue
        if len(fields) != 2:
            sys.exit(f"malformed MANIFEST.sha256 line: {line!r}")
        manifest.append(fields[1].strip())
    # tolerate both ./prefixed and bare manifest path forms
    manifest = [m[2:] if m.startswith("./") else m for m in manifest]
    on_disk = sorted(q.relative_to(DATA).as_posix() for q in DATA.rglob("*")
                     if q.is_file() and q.name != "MANIFEST.sha256")
    if on_disk != sorted(manifest):
        extra = set(on_disk) - set(manifest)
        missing = set(manifest) - set(on_disk)
        sys.exit(f"data/ does not match MANIFEST.sha256 file list "
                 f"(extra: {sorted(extra)}, missing: {sorted(missing)}) — "
                 f"pin or remove before uploading")
    try:
        subprocess.run(["sha256sum", "-c", "MANIFEST.sha256", "--quiet"],
                       cwd=DATA, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        sys.exit(f"hash drift or sha256sum unavailable ({type(e).__name__}): "
                 f"data/ must match MANIFEST.sha256 before upload")
    print("MANIFEST verified (hashes + exact file set)")

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    try:
        api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                        exist_ok=True)
        if args.public:
            # create_repo(exist_ok=True) never flips an existing repo's
            # visibility — apply it explicitly, else --public against the
            # existing private mirror silently uploads to a still-private repo
            api.update_repo_settings(args.repo, repo_type="dataset",
                                     private=False)
        # belt-and-suspenders: verify ACTUAL visibility before NC-licensed
        # files go anywhere (a pre-existing PUBLIC repo would bypass the
        # up-front flag gate entirely)
        try:
            is_private = api.repo_info(args.repo, repo_type="dataset").private
        except RepositoryNotFoundError:
            is_private = not args.public  # just created; visibility as requested
        if not is_private and not args.allow_nc_public:
            sys.exit(f"refusing upload: target {args.repo} is PUBLIC and "
                     f"{sorted(NC_LICENSED)} carry CC-BY-NC-4.0 rows; pass "
                     f"--allow-nc-public, or make the repo private first")
        if args.public and is_private:
            sys.exit(f"--public requested but {args.repo} is still private after "
                     f"update_repo_settings — make it public manually before "
                     f"re-running")
    except Exception as e:  # clean abort: HfHubHTTPError, conn errors, etc.
        sys.exit(f"Hub error during create/publish of {args.repo}: {type(e).__name__}: {e}")

    try:
        api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset",
                        commit_message="dataset card: provenance + licenses")
    except Exception as e:
        sys.exit(f"Hub error uploading dataset card to {args.repo}: "
                 f"{type(e).__name__}: {e}")
    files = sorted(manifest)
    for rel in files:
        try:
            api.upload_file(path_or_fileobj=str(DATA / rel), path_in_repo=rel,
                            repo_id=args.repo, repo_type="dataset",
                            commit_message=f"frozen data: {rel}")
        except Exception as e:
            sys.exit(f"Hub error uploading {rel} to {args.repo}: "
                     f"{type(e).__name__}: {e}")
        print(f"uploaded {rel}")
    try:
        api.upload_file(path_or_fileobj=str(DATA / "MANIFEST.sha256"),
                        path_in_repo="MANIFEST.sha256",
                        repo_id=args.repo, repo_type="dataset",
                        commit_message="hash pin (uploaded last, after contents)")
    except Exception as e:
        sys.exit(f"Hub error uploading MANIFEST.sha256 to {args.repo}: "
                 f"{type(e).__name__}: {e}")
    print(f"\nDONE -> https://huggingface.co/datasets/{args.repo}")
    print("record the final commit sha in the repo README (HF mirror line)")


if __name__ == "__main__":
    sys.exit(main())
