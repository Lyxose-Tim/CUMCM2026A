"""Package the reviewed manuscript and runnable project without touching old archives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archive(path, files, base):
    manifest = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            name = p.relative_to(base).as_posix()
            z.write(p, name)
            manifest[name] = digest(p)
        z.writestr("CONTENTS_SHA256.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"Invalid archive: {path}")
        for name, sha in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise RuntimeError(f"Archive hash mismatch: {name}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
            "files": len(manifest), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path,
                        help="Reviewed compiled PDF; relative paths are rooted in this project")
    parser.add_argument("--output-dir", type=Path, default=Path("deliverables/post_contest"),
                        help="Destination within this project; use a new folder to preserve a previous edition")
    args = parser.parse_args()
    source_pdf = args.pdf if args.pdf.is_absolute() else ROOT / args.pdf
    if not source_pdf.is_file() or not source_pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("A compiled PDF is required")
    target = (ROOT / args.output_dir).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("The artifact destination must stay within this project")
    target.mkdir(parents=True, exist_ok=True)
    pdf = target / "paper_review.pdf"
    shutil.copy2(source_pdf, pdf)
    extensions = {".tex", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".json", ".py", ".md"}
    paper_files = [p for p in (ROOT / "paper").rglob("*")
                   if p.is_file() and p.suffix.lower() in extensions
                   and "__pycache__" not in p.parts]
    compiler = archive(target / "paper_overleaf_review.zip", paper_files, ROOT / "paper")
    dirs = ("src", "tests", "config", "附件", "outputs", "exports", "reports", "scripts", "paper", "skills")
    support = []
    for folder in dirs:
        support.extend(p for p in (ROOT/folder).rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix.lower() not in {".pyc", ".log", ".aux"})
    support.extend(ROOT/p for p in ("README.md", "pyproject.toml", "requirements.txt", "A题.pdf", "AI工具使用声明.md"))
    reproduction = archive(target / "reproduction_support.zip", support, ROOT)
    summary = {"pdf": {"path": pdf.relative_to(ROOT).as_posix(), "sha256": digest(pdf),
                       "bytes": pdf.stat().st_size},
               "overleaf": compiler, "reproduction_support": reproduction,
               "note": "Archives contain explicit file hashes; competition paper_overleaf.zip is untouched."}
    (target/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
