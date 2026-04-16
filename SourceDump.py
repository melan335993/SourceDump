import argparse
import fnmatch
import io
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


DEFAULT_IGNORE = {
    ".git",
    "__pycache__",
    "node_modules",
    ".DS_Store",
    "uv.lock",
    ".venv",
}

DIFF_INSTRUCTION = """
SYSTEM INSTRUCTION:
You are an expert software engineer modifying an existing repository.
The user's request and a snapshot of the repository structure and file contents are provided above.
Your job is to produce the smallest correct patch that can be applied directly with `git apply` (or `patch`).
OUTPUT MODES:
Choose EXACTLY ONE:
1. PATCH MODE
   Output EXACTLY one fenced code block with language tag `diff` containing ONLY a valid unified diff.
2. NEED_HELP MODE
   If the change cannot be made safely from the snapshot, output EXACTLY:
   NEED_HELP: <concise description of the missing or ambiguous information>
Use NEED_HELP instead of guessing whenever:
* required files are missing, truncated, or incomplete
* the request depends on unseen code, symbols, APIs, types, functions, macros, or config
* the correct edit location is ambiguous
* the change conflicts with the visible code
* correctness would require inspecting files not present in the snapshot
3. NO_CHANGES MODE
   If the snapshot already satisfies the request, output EXACTLY:
   NO_CHANGES
PATCH RULES:
If you choose PATCH MODE, follow ALL rules below.
1. Output MUST contain exactly one fenced code block:
```diff <unified diff here>
```
Do NOT include any text before or after the block.
2. Inside the block, output ONLY valid unified diff syntax.
3. Output ONLY the minimal unified diff. Do NOT include:
* `diff --git`
* `index <hash>..<hash>`
* mode changes
* `new file mode` / `deleted file mode`
* rename or similarity headers
  unless explicitly requested.
4. Use repo-relative POSIX paths with forward slashes.
   Paths MUST be relative to the repository root shown after `Root:`.
   Do NOT repeat that root directory name inside diff headers.
Example:
If the snapshot shows `Root: NanoComrade/`, use:
--- a/agent/nanocomrade_agent/models/messages.py
+++ b/agent/nanocomrade_agent/models/messages.py
NOT:
--- a/NanoComrade/agent/nanocomrade_agent/models/messages.py
+++ b/NanoComrade/agent/nanocomrade_agent/models/messages.py
5. File headers MUST be:
* modified file:
  --- a/path/to/file
  +++ b/path/to/file
* new file:
  --- /dev/null
  +++ b/path/to/file
* deleted file:
  --- a/path/to/file
  +++ /dev/null
6. Every hunk MUST use the full canonical form:
   @@ -old_start,old_count +new_start,new_count @@
Never omit counts, even when count = 1.
7. Hunk counts MUST be mathematically correct:
* old_count = number of context lines + number of removed lines
* new_count = number of context lines + number of added lines
* do NOT count the hunk header itself
8. Context lines and removed lines MUST match the snapshot EXACTLY, including:
* spaces vs tabs
* indentation
* trailing spaces
* punctuation
* capitalization
* line order
Use canonical line prefixes only:
* context = single leading space
* removed = `-`
* added = `+`
For an empty context line, emit a single space followed by newline.
9. Provide enough surrounding context for unambiguous matching.
   Prefer 3–6 lines when available.
   Use more if needed for uniqueness.
   Do NOT invent context.
10. Keep the patch minimal and local:
* change only what is required
* do not reformat unrelated code
* do not touch unrelated whitespace
* do not reorder imports unless required
* do not rename or move code unless required
* do not rewrite whole files for small edits
* for multiple distant edits, emit multiple hunks
11. Do NOT use placeholders or omissions:
* no `...`
* no `<existing code>`
* no `// rest unchanged`
* no prose
* no pseudo-code
* no comments describing omitted code
12. Do NOT invent anything not grounded in the snapshot or explicitly required:
* no new files unless necessary
* no new imports unless necessary
* no new helpers unless necessary
* no unseen project internals
* no new symbols, APIs, config keys, macros, or abstractions unless clearly required and consistent with visible code
13. Only modify files present in the snapshot, unless the request explicitly requires creating a new file.
14. If added code depends on existing names or patterns, use only ones visible in the snapshot.
    If a required dependency is not visible, output NEED_HELP.
15. Preserve visible local conventions:
* naming
* formatting
* brace style
* indentation
* error handling
* logging
* comment style
16. Preserve newline behavior:
* if the patched file should end without trailing newline, use `\ No newline at end of file` where appropriate
* otherwise do not add that marker
17. Never include snapshot line numbers unless they are actual file content.
18. Large-change safety rule:
    Prefer a correct multi-hunk patch over a broad speculative rewrite.
    Do not modify code outside the smallest region required by the request.
    If a broader refactor seems useful but was not explicitly requested, do not include it.
FINAL SAFETY CHECK:
Before emitting the patch, verify internally that:
* every touched path is valid relative to the repo root
* every removed line exists exactly in the snapshot
* every context line exists exactly in the snapshot
* every added line is consistent with visible code style and symbols
* every hunk header matches its body exactly
* no unrelated code is silently changed
If any of these checks fail, output NEED_HELP instead of guessing.
PRIORITY ORDER:
1. Patch correctness and applicability
2. Faithfulness to the snapshot
3. Minimality of change
4. Completeness of the requested fix
If uncertain, prefer NEED_HELP over a risky patch.
""".strip()

def ensure_utf8_stdout() -> None:
    """Пытается переключить stdout на UTF-8, если это возможно."""
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding and encoding.lower() == "utf-8":
        return

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            return
        except Exception:
            pass

    if hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SourceDump: упаковка структуры проекта и содержимого файлов для LLM"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Пути для сканирования",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help="Дополнительные шаблоны исключений",
    )
    parser.add_argument(
        "-ef",
        "--exclude-file",
        help="Файл с шаблонами исключений (например, .gitignore)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Выходной файл",
    )
    parser.add_argument(
        "-dp",
        "--diff-prompt",
        action="store_true",
        help="Добавить инструкцию для генерации Unified Diff",
    )
    parser.add_argument(
        "-s",
        "--strip",
        action="store_true",
        help="Удалить пустые строки из содержимого файлов (работает только с --full)",
    )
    parser.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="Включить содержимое файлов. Без флага выводится только дерево проекта",
    )
    return parser.parse_args()


def load_exclude_patterns(extra_patterns: Sequence[str], exclude_file: str | None) -> list[str]:
    patterns = set(DEFAULT_IGNORE)
    patterns.update(p.strip('"\'') for p in extra_patterns if p.strip())

    if exclude_file:
        exclude_path = Path(exclude_file)
        if exclude_path.exists():
            for line in exclude_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.add(line.strip('"\''))
    return sorted(patterns)


def is_excluded(path: Path, exclude_patterns: Sequence[str]) -> bool:
    posix_path = path.as_posix()
    name = path.name

    for pattern in exclude_patterns:
        normalized = pattern.rstrip("/\\")
        stripped = pattern.strip("/\\")

        if fnmatch.fnmatch(name, normalized):
            return True
        if fnmatch.fnmatch(posix_path, pattern):
            return True
        if fnmatch.fnmatch(posix_path, f"*/{stripped}"):
            return True

    return False


def sorted_dir_items(path: Path, exclude_patterns: Sequence[str]) -> list[Path]:
    items = [item for item in path.iterdir() if not is_excluded(item, exclude_patterns)]
    return sorted(items, key=lambda p: (not p.is_dir(), p.name.lower()))


def generate_tree(root_dir: Path, exclude_patterns: Sequence[str], prefix: str = "") -> list[str]:
    try:
        items = sorted_dir_items(root_dir, exclude_patterns)
    except PermissionError:
        return [f"{prefix}└── [Permission Denied]"]

    tree_lines: list[str] = []

    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            child_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines.extend(generate_tree(item, exclude_patterns, child_prefix))

    return tree_lines


def is_probably_binary(file_path: Path, sample_size: int = 2048) -> bool:
    try:
        with file_path.open("rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return True


def format_display_path(root_path: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root_path)
    return (Path(root_path.name) / relative).as_posix()


def normalize_content(content: str, strip_empty: bool) -> str:
    if not strip_empty:
        return content
    return "\n".join(line for line in content.splitlines() if line.strip())


def iter_file_blocks(
    root_path: Path,
    exclude_patterns: Sequence[str],
    strip_empty: bool = False,
) -> Iterator[str]:
    def walk(current_path: Path) -> Iterator[str]:
        try:
            items = sorted_dir_items(current_path, exclude_patterns)
        except PermissionError:
            return

        for item in items:
            if item.is_dir():
                yield from walk(item)
                continue

            if not item.is_file():
                continue

            if is_probably_binary(item):
                continue

            try:
                content = item.read_text(encoding="utf-8", errors="ignore")
            except (PermissionError, OSError, UnicodeDecodeError):
                continue

            content = normalize_content(content, strip_empty)
            display_path = format_display_path(root_path, item)
            yield f'\n<file path="{display_path}">\n{content}\n</file>'

    yield from walk(root_path)


def resolve_existing_paths(raw_paths: Sequence[str]) -> list[Path]:
    return [Path(p).resolve() for p in raw_paths if Path(p).exists()]


def get_token_count(text: str) -> tuple[int, bool]:
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text, disallowed_special=())), True
        except Exception:
            pass

    return max(1, len(text) // 4), False


def build_output(
    paths: Sequence[Path],
    exclude_patterns: Sequence[str],
    include_diff_instruction: bool,
    include_file_contents: bool,
    strip_empty: bool,
) -> str:
    output: list[str] = []

    if include_diff_instruction:
        output.append(DIFF_INSTRUCTION)
        output.append("")

    output.append("=== PROJECT STRUCTURE ===")
    for root in paths:
        root_label = root.name or str(root)
        output.append(f"\nRoot: {root_label}/")
        output.extend(generate_tree(root, exclude_patterns))

    if include_file_contents:
        output.append("\n" + "=" * 25)
        output.append("=== FILE CONTENTS ===")
        for root in paths:
            output.extend(iter_file_blocks(root, exclude_patterns, strip_empty=strip_empty))

    return "\n".join(output).rstrip() + "\n"


def write_or_print_result(result: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(result, encoding="utf-8")
        print(f"✅ Дамп сохранён в: {output_path}")
    else:
        print(result, end="")


def main() -> None:
    ensure_utf8_stdout()
    args = parse_args()

    paths = resolve_existing_paths(args.paths)
    if not paths:
        print("❌ Не найдено ни одного существующего пути для сканирования.", file=sys.stderr)
        sys.exit(1)

    exclude_patterns = load_exclude_patterns(args.exclude, args.exclude_file)

    result = build_output(
        paths=paths,
        exclude_patterns=exclude_patterns,
        include_diff_instruction=args.diff_prompt,
        include_file_contents=args.full,
        strip_empty=args.strip and args.full,
    )

    write_or_print_result(result, args.output)

    tokens, is_exact = get_token_count(result)
    print(f"\n{'-' * 30}")
    print(f"📊 {'Точно' if is_exact else 'Примерно'}: {tokens} токенов.")
    print(f"{'-' * 30}")


if __name__ == "__main__":
    main()