"""Install verified weights from GitHub Releases or a local asset directory."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checksum(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "configs/weights.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *manifest["models"]], default="all")
    parser.add_argument("--repo", default=manifest["repository"], help="GitHub owner/repository hosting the model release")
    parser.add_argument("--tag", default=manifest["release_tag"])
    parser.add_argument("--local-dir", type=Path, help="Copy named release assets without network access")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    if args.repo and not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must be owner/repository")
    if not re.fullmatch(r"[\w.-]+", args.tag):
        parser.error("Invalid release tag")
    names = manifest["models"] if args.dataset == "all" else [args.dataset]
    for name in names:
        item = manifest["models"][name]
        destination = args.output / item["path"]
        if destination.is_file() and checksum(destination) == item["sha256"]:
            print(f"Verified existing {name}"); continue
        if not args.local_dir and not args.repo:
            parser.error("Specify a model source with --local-dir ASSETS or --repo OWNER/REPO.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if args.local_dir:
                shutil.copyfile(args.local_dir / item["asset"], temporary)
            else:
                url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{item['asset']}"
                request = urllib.request.Request(url, headers={"User-Agent": "DLO-FWI-weight-downloader"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if temporary.stat().st_size != item["bytes"] or checksum(temporary) != item["sha256"]:
                raise ValueError(f"Download checksum/size mismatch: {name}")
            temporary.replace(destination)
            print(f"Installed {name}: {destination}")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
