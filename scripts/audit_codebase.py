"""Read-only first-party syntax and module inventory (never imports project code)."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"vendor", "node_modules", "models", ".git", ".venv", "venv", "build", "dist",
            "__pycache__", ".pytest_cache", ".pytest-tmp", ".sweep-runtime"}


def audit(root: Path = ROOT) -> dict:
    files, lines, suffixes = Counter(), Counter(), Counter()
    errors = []
    modules = []
    for directory, dirs, names in root.walk():
        dirs[:] = [d for d in dirs if d not in EXCLUDED and not d.startswith(".")]
        for name in sorted(names):
            path = directory / name
            if path.suffix not in {".py", ".ts", ".tsx", ".cpp", ".h", ".hpp"}:
                continue
            suffixes[path.suffix] += 1
            if path.suffix != ".py":
                continue
            relative = path.relative_to(root)
            source = path.read_text(encoding="utf-8-sig")
            component = relative.parts[0] if len(relative.parts) > 1 else "root"
            files[component] += 1
            lines[component] += len(source.splitlines())
            try:
                tree = ast.parse(source, filename=str(relative))
                modules.append({"path": relative.as_posix(),
                                "purpose": (ast.get_docstring(tree) or "").split("\n")[0],
                                "interfaces": [n.name for n in tree.body if isinstance(
                                    n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]})
            except SyntaxError as exc:
                errors.append({"path": str(relative), "line": exc.lineno, "error": exc.msg})
    return {"python_files": sum(files.values()), "files_by_component": dict(files),
            "lines_by_component": dict(lines), "source_file_types": dict(suffixes),
            "syntax_errors": errors, "modules": modules}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["syntax_errors"]))
