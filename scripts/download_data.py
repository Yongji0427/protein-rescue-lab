"""Fetch only GB1 assay members from official ProteinGym v1.3 archives."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
ASSAY = "SPG1_STRSG_Olson_2014.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(filename, output_name):
    archive = ROOT / "data" / "source_archives" / filename
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + filename
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        partial.rename(archive)
    with zipfile.ZipFile(archive) as package:
        matches = [name for name in package.namelist() if Path(name).name == ASSAY]
        if len(matches) != 1:
            raise ValueError(f"Expected one GB1 member in {filename}, found {matches}")
        member = matches[0]
        output = ROOT / "data" / "evaluator_only" / output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        with package.open(member) as source, output.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
    info = {
        "url": url, "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": sha256(archive), "member": member,
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
    }
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


def main():
    metadata_dir = ROOT / "data" / "provenance"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    commit = requests.get(
        "https://api.github.com/repos/OATML-Markslab/ProteinGym/commits/main", timeout=30
    )
    commit.raise_for_status()
    revision = commit.json()["sha"]
    url = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{revision}/reference_files/DMS_substitutions.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    metadata = metadata_dir / "DMS_substitutions.csv"
    metadata.write_bytes(response.content)
    jobs = [
        ("DMS_ProteinGym_substitutions.zip", "gb1_processed.csv"),
        ("substitutions_raw_DMS.zip", "gb1_raw.csv"),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda job: fetch_archive(*job), jobs))
    manifest = {
        "assay": ASSAY, "release": "ProteinGym_v1.3",
        "metadata_commit": revision, "metadata_url": url,
        "metadata_sha256": sha256(metadata), "downloads": records,
        "paper": "https://doi.org/10.1016/j.cub.2014.09.072",
    }
    (metadata_dir / "sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
