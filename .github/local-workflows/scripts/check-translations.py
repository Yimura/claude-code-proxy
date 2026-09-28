#!/usr/bin/env python3
"""Fail when .github/local-workflows/ has drifted from .github/workflows/.

Copied into a repository by the authoring-local-workflows skill and run by
.github/local-workflows/parity.yml, so it is part of the local suite rather than
a billed GitHub job.

Drift is detected by comparing each translated source's recorded commit against
the commit that last touched it. That needs no understanding of workflow
semantics, which is the whole reason it was chosen: a check that has to reason
about coverage is a check that can be wrong about it.

A comment-only edit to a source flags stale. That is deliberate -- resolving it
means reading the real diff, which is the thing that should happen.

CONTRACT -- the authoring skill's instruction files point here, so keep this accurate:

  * Invoked argument-free, from the repository root.
  * Reads `local_workflows` from the manifest; resolves each `local:` name
    against it. Does not hardcode the directory.
  * Globs `.github/workflows/*.yml` and `*.yaml`.
  * Aggregates every problem, sorted, and reports them all. Never aborts on the
    first -- an authoring loop needs the whole list.
  * Problems to stderr; the clean summary to stdout.
  * Whether git itself is usable is checked once, up front (`git rev-parse
    --git-dir`) -- not inferred from a per-file result. If git is unusable
    there, that is exit 2: an inability to answer at all.
  * A workflow file that exists on disk but git has no commit for yet (added,
    not yet committed) is answerable: it is a parity problem, reported and
    aggregated like any other, flowing through to exit 1 -- not folded into
    "git is unusable" and not exit 2. This is the normal incremental-authoring
    case: add a workflow, run parity before committing it.
  * Exit 0 clean, 1 on problems, 2 when it cannot answer at all (missing
    manifest, unparseable manifest, git itself unusable in this tree).
  * Recorded commits are full 40-character hashes compared by exact string
    equality, never by prefix.
  * The problem classes below are a floor, not a closed set. A repository may
    extend them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

MANIFEST = Path(".claude/c7/github.yml")
WORKFLOWS = Path(".github/workflows")
DEFAULT_LOCAL = ".github/local-workflows"


def problems(
    facts: dict,
    on_disk: set[str],
    heads: dict[str, str | None],
    locals_present: set[str],
) -> list[str]:
    """Every way the manifest and the two directories can disagree, sorted.

    `heads` is assumed collected after git itself was confirmed usable, so an
    explicit `None` value for a path means only one thing: git has no commit
    for that file yet. That is reported once, for the file itself, rather than
    compared against any recorded commit -- there is nothing yet to compare.
    A path simply absent from `heads` (rather than present with value `None`)
    is not this case -- production always populates every `on_disk` path.
    """
    found: list[str] = []
    declared: set[str] = set()

    for entry in facts.get("uncovered") or []:
        declared.add(entry["path"])

    for path in on_disk:
        if path in heads and heads[path] is None:
            found.append(
                f"{path} exists on disk but has no commits yet; commit it before it can be stamped"
            )

    for entry in facts.get("translations") or []:
        local = entry["local"]
        if local not in locals_present:
            found.append(f"{local} is named in the manifest but does not exist")
        for source in entry.get("sources") or []:
            path, recorded = source["path"], source["commit"]
            declared.add(path)
            if path not in on_disk:
                found.append(f"{local} translates {path}, which no longer exists")
                continue
            if path in heads and heads[path] is None:
                continue  # already flagged above; nothing yet to compare against
            if heads.get(path) != recorded:
                found.append(
                    f"{local} is stale: {path} changed since {recorded} (now {heads.get(path)})"
                )

    for path in on_disk - declared:
        found.append(f"{path} has no translations entry and is not listed as uncovered")

    return sorted(found)


def git_usable() -> bool:
    """Whether git answers at all in this tree, checked once, up front.

    Deliberately separate from any per-file lookup: a file with no commits is
    a normal, answerable state (see `problems`), and must not be confused with
    git itself being broken or absent.
    """
    result = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, text=True)
    return result.returncode == 0


def git_head(path: str) -> str | None:
    """The commit that last touched this file, or None when it has no commits.

    Assumes `git_usable()` was already confirmed true; under that assumption
    an empty result means only "no commit touches this path yet", not "git is
    broken".
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", path], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> int:
    if not MANIFEST.is_file():
        print(f"{MANIFEST} is missing.", file=sys.stderr)
        return 2
    try:
        loaded = yaml.safe_load(MANIFEST.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        print(f"{MANIFEST} does not parse ({exc}).", file=sys.stderr)
        return 2
    facts = (loaded or {}).get("ci") or {}

    if not git_usable():
        # Refusing rather than passing. This is issue #14's shape: a check that
        # cannot read history must say so, not report success having verified
        # nothing.
        print("git is not usable in this tree (`git rev-parse --git-dir` failed).", file=sys.stderr)
        print("Run this from the repository root, in a tree where git works.", file=sys.stderr)
        return 2

    on_disk = {str(p) for p in sorted(WORKFLOWS.glob("*.yml"))}
    on_disk |= {str(p) for p in sorted(WORKFLOWS.glob("*.yaml"))}

    heads = {path: git_head(path) for path in on_disk}

    local_dir = Path(facts.get("local_workflows", DEFAULT_LOCAL))
    locals_present = {p.name for p in local_dir.glob("*.yml")} | {
        p.name for p in local_dir.glob("*.yaml")
    }

    found = problems(facts, on_disk, heads, locals_present)
    for problem in found:
        print(f"parity: {problem}", file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} parity problem(s). Invoke the authoring-local-workflows "
            "skill to translate or declare each one.",
            file=sys.stderr,
        )
        return 1
    print(f"parity: {len(on_disk)} workflow(s) declared, no drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
