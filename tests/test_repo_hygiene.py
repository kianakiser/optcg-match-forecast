"""Enforce the MLOps HS26 course repository rules in CI.

The Repository Guide fixes a handful of things: required paths, no secrets, no data or
model artifacts in git. Those are grading criteria, so they are worth failing a build over
rather than discovering at a milestone deadline.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


# --------------------------------------------------------------------------- required layout


@pytest.mark.parametrize(
    "path",
    ["README.md", "docs", "tests", ".github/workflows", "Dockerfile", ".env.example", ".gitignore"],
)
def test_required_path_exists(path: str) -> None:
    """The Repository Guide fixes these locations."""
    assert (REPO / path).exists(), f"{path} is required by the Repository Guide"


def test_dependencies_are_pinned() -> None:
    """'Pinned dependencies: uv.lock or requirements.txt with exact versions.'"""
    assert (REPO / "uv.lock").exists() or (REPO / "requirements.txt").exists(), (
        "need uv.lock or requirements.txt with exact versions"
    )


# --------------------------------------------------------------------------- secrets


def test_no_env_file_tracked() -> None:
    """.env must never be committed; .env.example must be."""
    tracked = tracked_files()
    leaked = [f for f in tracked if Path(f).name == ".env" or f.endswith("/.env")]
    assert not leaked, f"secret file(s) tracked in git: {leaked}"
    assert ".env.example" in tracked, ".env.example should be committed"


SECRETISH_KEY = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.I)
PLACEHOLDERS = {"changeme", "placeholder", "xxx", "todo", "none", "your-key-here", ""}


def test_env_example_has_no_real_values() -> None:
    """.env.example carries variable NAMES with placeholder values only.

    Only secret-looking keys are checked: non-secret config such as GCP_REGION=europe-west6
    is legitimately a real value and is not what the rule is about.
    """
    example = REPO / ".env.example"
    suspicious: list[str] = []
    for line in example.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if not SECRETISH_KEY.search(key):
            continue
        if value.lower() in PLACEHOLDERS:
            continue
        # A path to a credential file is a pointer, not the credential itself.
        if value.startswith(("./", "/", "$", "${")):
            continue
        suspicious.append(key)
    assert not suspicious, f".env.example may contain real secret values for: {suspicious}"


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub token
    re.compile(r"sk-[A-Za-z0-9]{32,}"),  # OpenAI-style key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # private key block
    re.compile(r"\"private_key_id\"\s*:"),  # GCP service-account JSON
]

TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".env",
    ".example",
    ".tf",
    ".js",
    ".ts",
    "",
}


def test_no_secrets_in_tracked_files() -> None:
    findings: list[str] = []
    for rel in tracked_files():
        path = REPO / rel
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if rel.startswith("tests/"):  # this file necessarily contains the patterns
            continue
        try:
            content = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(f"{rel}: matched {pattern.pattern}")
    assert not findings, "possible secrets committed:\n" + "\n".join(findings)


# --------------------------------------------------------------------------- no data / models

FORBIDDEN_SUFFIXES = {
    ".parquet",
    ".csv",
    ".tsv",
    ".feather",
    ".pkl",
    ".joblib",
    ".onnx",
    ".h5",
    ".pt",
    ".pth",
    ".ckpt",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}
ALLOWED_DATA = {"tests/fixtures", "docs"}


def test_no_data_or_model_artifacts_tracked() -> None:
    """'no data, model files or mlruns/ in git' — and 'no video files in the repo'."""
    offenders = [
        f
        for f in tracked_files()
        if Path(f).suffix.lower() in FORBIDDEN_SUFFIXES
        and not any(f.startswith(prefix) for prefix in ALLOWED_DATA)
    ]
    assert not offenders, f"data/model/video artifacts tracked in git: {offenders}"


def test_no_mlruns_tracked() -> None:
    offenders = [f for f in tracked_files() if f.startswith(("mlruns/", "mlartifacts/"))]
    assert not offenders, f"mlruns/ must not be in git: {offenders}"


def test_gitignore_covers_the_required_patterns() -> None:
    """The guide names exactly what .gitignore must cover."""
    text = (REPO / ".gitignore").read_text()
    for needed in [".env", "mlruns/"]:
        assert needed in text, f".gitignore must cover {needed}"


# --------------------------------------------------------------------------- notebooks


def test_pipelines_do_not_import_from_notebooks() -> None:
    """'pipelines never import from notebooks' — an explicit rule in the Repository Guide."""
    src = REPO / "src"
    if not src.exists():
        pytest.skip("src/ not created yet")
    offenders = [
        f"{path.relative_to(REPO)}"
        for path in src.rglob("*.py")
        if re.search(r"^\s*(from|import)\s+.*notebooks", path.read_text(errors="ignore"), re.M)
    ]
    assert not offenders, f"pipeline code imports from notebooks: {offenders}"


# --------------------------------------------------------------------------- working area

NEVER_PUBLISH = ("notes/", "course_exercises/")


@pytest.mark.parametrize("prefix", NEVER_PUBLISH)
def test_local_working_area_is_never_tracked(prefix: str) -> None:
    """notes/ and course_exercises/ live in the repo folder but must never be published.

    notes/ holds peer-review drafts, and the Review Guide is explicit: "never commit reviews
    or drafts to your repo". course_exercises/ is the lecturer's material, which he said he
    will make private again after the lecture.

    .gitignore covers both, but `git add -f` bypasses it, so this asserts the outcome rather
    than trusting the mechanism.
    """
    tracked = [f for f in tracked_files() if f.startswith(prefix)]
    assert not tracked, (
        f"{prefix} must never be committed; found {len(tracked)} file(s): {tracked[:5]}"
    )


@pytest.mark.parametrize("prefix", NEVER_PUBLISH)
def test_local_working_area_is_absent_from_history(prefix: str) -> None:
    """Also check it was never committed and later removed — the rule is about the history too."""
    out = subprocess.run(
        ["git", "log", "--all", "--diff-filter=A", "--name-only", "--format="],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    ever = {line for line in out.stdout.splitlines() if line.startswith(prefix)}
    assert not ever, f"{prefix} appears in git history: {sorted(ever)[:5]}"
