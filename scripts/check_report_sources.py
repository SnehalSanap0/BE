"""Static sanity checks of the LaTeX report sources (for machines without a LaTeX compiler).

Checks control characters, brace balance, matching begin/end environments, undefined cross-references,
missing figure/table files and citation keys. It does not replace compiling the report (Overleaf or
pdflatex), but catches the most common mistakes early.

    python -m scripts.check_report_sources
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "Final_Year_Project_Report"
ENVIRONMENTS = ["table", "figure", "longtable", "itemize", "enumerate", "algorithm", "tabular", "minipage",
                "center", "equation", "align", "thebibliography", "document", "flushright", "flushleft"]


def main() -> int:
    files = [ROOT / n for n in ("main.tex", "Abstract.tex", "Ack.tex", "Certificate.tex", "References.tex")]
    files += sorted(ROOT.glob("chapters/*.tex")) + sorted(ROOT.glob("tables/*.tex"))
    problems: list[str] = []
    texts = {f: f.read_text(encoding="utf-8") for f in files}

    for f, text in texts.items():
        controls = {hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\r\t"}
        if controls:
            problems.append(f"{f.name}: control characters {sorted(controls)}")
        body = text.replace("\\\\", "")  # a double backslash is a line break, not an escape
        body = re.sub(r"(?<!\\)%.*", "", body).replace(r"\{", "").replace(r"\}", "")
        if body.count("{") != body.count("}"):
            problems.append(f"{f.name}: unbalanced braces ({body.count('{')} open, {body.count('}')} close)")

    alltext = "\n".join(texts.values())
    for env in ENVIRONMENTS:
        begins = len(re.findall(r"\\begin\{%s\}" % env, alltext))
        ends = len(re.findall(r"\\end\{%s\}" % env, alltext))
        if begins != ends:
            problems.append(f"environment {env}: {begins} begin vs {ends} end")

    labels = set(re.findall(r"\\label\{([^}]+)\}", alltext))
    refs = set(re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", alltext))
    if refs - labels:
        problems.append(f"undefined references: {sorted(refs - labels)}")

    graphics = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltext))
    missing = sorted(g for g in graphics if not (ROOT / g).exists())
    if missing:
        problems.append(f"missing graphics: {missing}")
    tables = set(re.findall(r"\\input\{(tables/[^}]+)\}", alltext))
    missing_tables = sorted(t for t in tables if not (ROOT / t).exists())
    if missing_tables:
        problems.append(f"missing table files: {missing_tables}")

    keys = set(re.findall(r"\\bibitem\{([^}]+)\}", alltext))
    cited = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", alltext) for k in m.split(",")}
    if cited - keys:
        problems.append(f"cited but undefined: {sorted(cited - keys)}")

    print(f"Checked {len(files)} files: {len(graphics)} graphics, {len(tables)} tables, {len(cited)} cited works, "
          f"{len(labels)} labels.")
    for problem in problems:
        print("PROBLEM:", problem)
    print("OK - no problems found." if not problems else f"{len(problems)} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
