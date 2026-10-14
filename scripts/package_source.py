"""Build a source archive with an explicit allowlist (no model weights or outputs)."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/DLO-FWI-source.zip")
    args = parser.parse_args()
    files = [ROOT / n for n in (".gitignore", "LICENSE", "MODEL_LICENSE.md", "README.md",
             "THIRD_PARTY_NOTICES.md", "requirements.txt")]
    for directory, extensions in {"dlo": {".py"}, "scripts": {".py"}, "tests": {".py"},
                                 "configs": {".json"}, "examples": {".npz", ".json", ".md"},
                                 "assets": {".png", ".csv", ".json"}}.items():
        files += sorted(p for p in (ROOT / directory).iterdir() if p.suffix in extensions)
    lock = ROOT / "requirements-lock.txt"
    if lock.exists():
        files.append(lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 10 * 1024**2:
                raise ValueError(f"Unexpected source asset: {path}")
            archive.write(path, Path("DLO-FWI") / path.relative_to(ROOT))
    print(f"{args.output}: {len(files)} files, {args.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
