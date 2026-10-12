#!/usr/bin/env python3
"""Fetch and validate the selected Function Vector tasks for the broad study.

Selected rows are fetched from the pinned upstream revision and verified before
writing the local cache. See third_party/function_vectors/README.md for dataset
attribution and licensing status.

--output is checked before any network request: symlinks, non-regular files and
every repository path other than the default git-ignored cache
(experiments/broad_study/function_vectors.json) are refused. The default cache or an
existing regular file outside the repository may be replaced by a re-fetch; the
file is written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import ssl
import tempfile
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("function_vectors.json")
DEFAULT_OUTPUT_RELATIVE = DEFAULT_OUTPUT.relative_to(REPO_ROOT).as_posix()

UPSTREAM_REPOSITORY = "https://github.com/jorispos/ConceptorSteering"
UPSTREAM_COMMIT = "5e134153f4eeefa631bda39788f1f996bbf052a1"
RAW_HOST = "raw.githubusercontent.com"
RAW_PATH_ROOT = f"/jorispos/ConceptorSteering/{UPSTREAM_COMMIT}/data/functionvectors"
RAW_ROOT = f"https://{RAW_HOST}{RAW_PATH_ROOT}"
MAX_SOURCE_BYTES = 5 * 1024 * 1024

# These indices reproduce the selections embedded in the executed broad-study
# notebooks. Most tasks use rows 0--31. English--French skips source row 18;
# singular--plural uses the listed subset.
SELECTIONS = {
    "antonym": ("antonyms.json", list(range(32))),
    "country_capital": ("country-capital.json", list(range(32))),
    "present_past": ("present-past.json", list(range(32))),
    "english_french": (
        "english-french.json",
        list(range(18)) + list(range(19, 33)),
    ),
    "singular_plural": (
        "singular-plural.json",
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
            17, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 36,
            37, 38,
        ],
    ),
    "capitalize": ("capitalize.json", list(range(32))),
}

# SHA-256 of pair_canonical_json(pairs): the exact [input, output] pair list
# written for each task, in written order. The digests were computed from the
# pair lists embedded in the executed Qwen2.5-1.5B and Qwen2.5-7B broad-study notebooks,
# which agree for all six tasks (32 distinct pairs each).
EXPECTED_PAIR_SHA256 = {
    "antonym": "73656d00e60f97864cb5b9211ffbea84404a2d841678dfae28de89885455a6cf",
    "country_capital": "e9a4023769aefee6f65e3620425c1c6e8af4085625d937e7ed02f94cc10cde8e",
    "present_past": "cb80ad5ac036d8ef0ea19386739c875775e61f84cb450340a78f063ec4db1780",
    "english_french": "2d9db403feb4e18f661b9cbdce120af71b385b85d98f973a0751cb5cf2c9b4f2",
    "singular_plural": "843e03d90e5affc805b949b23a6bfc3e52a98eea5ff5daabb8497aae797a1d50",
    "capitalize": "9bcee89790888ae834e8e7b7ce4b90745e54bc92c1ca8297eec21bef5ebb4d37",
}


def pair_canonical_json(pairs: list[list[str]]) -> str:
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))


def pair_sha256(pairs: list[list[str]]) -> str:
    return hashlib.sha256(pair_canonical_json(pairs).encode("utf-8")).hexdigest()


def require_expected_pairs(functions: Mapping[str, object]) -> None:
    """Raise unless every task's pairs match the executed broad-study selection exactly."""
    if set(functions) != set(EXPECTED_PAIR_SHA256):
        raise ValueError(
            "Function Vector tasks must be exactly "
            f"{sorted(EXPECTED_PAIR_SHA256)}, got {sorted(functions)}"
        )
    for task, expected in EXPECTED_PAIR_SHA256.items():
        specification = functions[task]
        pairs = specification.get("pairs") if isinstance(specification, Mapping) else None
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
            for pair in pairs
        ):
            raise ValueError(f"Malformed Function Vector pairs for task {task}")
        actual = pair_sha256(pairs)
        if actual != expected:
            raise ValueError(
                f"Function Vector pairs for task {task} do not match the executed broad-study "
                f"selection (expected sha256 {expected}, got {actual}); refusing to "
                "use them."
            )


def fetch_json(filename: str) -> tuple[list[dict[str, str]], str]:
    allowed = {selection[0] for selection in SELECTIONS.values()}
    if filename not in allowed or Path(filename).name != filename:
        raise ValueError(f"Refusing unexpected source filename: {filename}")
    path = f"{RAW_PATH_ROOT}/{filename}"
    # Python 3.11+ and an explicit default context guarantee certificate and
    # hostname verification. The static-analysis suppression is scoped to this
    # fully pinned host/path client.
    connection = http.client.HTTPSConnection(  # nosemgrep: python.lang.security.audit.httpsconnection-detected.httpsconnection-detected
        RAW_HOST,
        port=443,
        timeout=60,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"User-Agent": "conceptor-complement-gain/1.0"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"Pinned source returned HTTP {response.status} for {filename}"
            )
        payload = response.read(MAX_SOURCE_BYTES + 1)
    finally:
        connection.close()
    if len(payload) > MAX_SOURCE_BYTES:
        raise RuntimeError(f"Pinned source exceeds {MAX_SOURCE_BYTES} bytes: {filename}")
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list from {RAW_ROOT}/{filename}")
    return rows, hashlib.sha256(payload).hexdigest()


def repository_relative(resolved: Path) -> str | None:
    """Repository-relative POSIX path of a resolved path, or None if outside (alias safe)."""
    repository = REPO_ROOT.resolve()
    if resolved == repository or repository in resolved.parents:
        return resolved.relative_to(repository).as_posix()
    repository_stat = repository.stat()
    parts: list[str] = []
    for candidate in (resolved, *resolved.parents):
        try:
            if os.path.samestat(candidate.stat(), repository_stat):
                return "/".join(reversed(parts)) or "."
        except OSError:
            pass
        parts.append(candidate.name)
    return None


def validate_output(output: Path) -> Path:
    """Return the resolved --output, refusing unsafe destinations (checked before fetching)."""
    output = output.expanduser()
    if output.is_symlink():
        raise ValueError(f"refusing symlink --output {output}")
    resolved = output.resolve()
    if os.path.lexists(resolved) and not resolved.is_file():
        raise ValueError(f"--output {resolved} exists and is not a regular file")
    relative = repository_relative(resolved)
    if relative is None or (relative == DEFAULT_OUTPUT_RELATIVE and resolved == DEFAULT_OUTPUT):
        return resolved
    raise ValueError(
        f"refusing to write --output to repository path {relative}: inside the repository "
        f"only the default git-ignored cache {DEFAULT_OUTPUT_RELATIVE} may be used "
        "(pair text must never be committed)"
    )


def write_atomically(resolved: Path, text: str) -> None:
    if validate_output(resolved) != resolved:
        raise ValueError(f"--output {resolved} changed after validation")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=resolved.parent, prefix=f".{resolved.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        if os.path.lexists(temporary):
            os.unlink(temporary)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output file (default: the git-ignored cache next to this script). Any other "
            "path must be outside the repository."
        ),
    )
    args = parser.parse_args(argv)
    try:
        output = validate_output(args.output)
    except ValueError as error:
        parser.error(str(error))

    functions: dict[str, dict[str, object]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for task, (filename, indices) in SELECTIONS.items():
        url = f"{RAW_ROOT}/{filename}"
        rows, digest = fetch_json(filename)
        if max(indices) >= len(rows):
            raise ValueError(f"Selection for {task} exceeds {filename}")
        pairs = []
        for index in indices:
            row = rows[index]
            if (
                not isinstance(row, dict)
                or not {"input", "output"} <= set(row)
                or not isinstance(row["input"], str)
                or not isinstance(row["output"], str)
            ):
                raise ValueError(f"Malformed row {index} in {filename}")
            pairs.append([row["input"], row["output"]])
        if len(pairs) != 32 or len({tuple(pair) for pair in pairs}) != 32:
            raise ValueError(f"Expected 32 distinct pairs for {task}")
        functions[task] = {"pairs": pairs}
        source_files[filename] = {"url": url, "sha256": digest}

    # Fail closed before any output is written.
    require_expected_pairs(functions)

    document = {
        "provenance": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "note": "Fetched locally; not redistributed by this repository.",
            "source_files": source_files,
        },
        "functions": functions,
    }
    write_atomically(output, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output} with {len(functions)} tasks and 32 pairs per task")


if __name__ == "__main__":
    main()
