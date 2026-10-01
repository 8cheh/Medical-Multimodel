"""Normalise report/REPORT.md heading structure and make its anchors resolve.

Two defects this fixes:
  1. Five level-1 headings (title + four parts) -> MD025. Parts become H2 and
     their subsections H3, so the document has exactly one H1.
  2. Figure headings used U+3000 (ideographic space) between "图N" and the title.
     GitHub's slugger turns an ASCII space into '-' but drops U+3000, so all 11
     index links resolved to nothing. U+3000 becomes a plain space, which makes
     the generated anchors match the links exactly.

Fenced code blocks are skipped: the reproduction section contains bash comments
that start with '#' and must not be treated as headings.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report", "REPORT.md")
KEEP_H2 = {"目标达成状态"}


def slug(text):
    """Approximate GitHub's heading anchor (lowercase, drop punctuation, spaces->-)."""
    out = []
    for ch in text.strip().lower():
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    return "".join(out)


def transform(text):
    lines = text.splitlines()
    out, in_fence = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        m = re.match(r"^# (第[一二三四五六七八九十]+部分)\u3000(.*)$", line)
        if m:
            out.append(f"## {m.group(1)} {m.group(2)}")
            continue

        m = re.match(r"^## (图\d+)\u3000(.*)$", line)
        if m:
            out.append(f"### {m.group(1)} {m.group(2)}")
            continue

        m = re.match(r"^## (.+)$", line)
        if m and m.group(1).strip() not in KEEP_H2:
            out.append(f"### {m.group(1)}")
            continue

        m = re.match(r"^# (.+)$", line)
        if m and not m.group(1).startswith("VitalDB"):
            out.append(f"## {m.group(1)}")
            continue

        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def heading_slugs(text):
    in_fence = False
    found = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6}) (.+)$", line)
        if m:
            found.append((len(m.group(1)), m.group(2), slug(m.group(2))))
    return found


def main():
    try:
        with open(REPORT, encoding="utf-8") as fh:
            original = fh.read()
    except OSError as exc:
        print(f"cannot read {REPORT}: {exc}")
        return 1

    fixed = transform(original)
    try:
        with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(fixed)
    except OSError as exc:
        print(f"cannot write {REPORT}: {exc}")
        return 1

    heads = heading_slugs(fixed)
    h1 = [h for h in heads if h[0] == 1]
    print(f"headings: {len(heads)}  H1: {len(h1)}")
    print("U+3000 remaining in headings:", sum(1 for h in heads if "\u3000" in h[1]))

    available = {h[2] for h in heads}
    links = re.findall(r"\]\(#([^)]+)\)", fixed)
    missing = [ln for ln in links if ln not in available]
    print(f"internal links: {len(links)}  unresolved: {len(missing)}")
    for ln in missing:
        print(f"  ! #{ln}")
    return 0 if not missing and len(h1) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
