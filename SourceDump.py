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
The user's change request is provided earlier in the same message or conversation.
Below is a snapshot of the project structure and file contents.

Use the snapshot below as the source of truth for the codebase.
When fulfilling the user's request, respond ONLY with a valid unified diff.

STRICT OUTPUT RULES:
1. Output ONLY unified diff text and nothing else.
2. Do NOT include explanations, comments, analysis, markdown fences, or headings.
3. Use repo-relative POSIX paths.
4. Use this format for modified files:
   --- a/path/to/file
   +++ b/path/to/file
5. For a new file use:
   --- /dev/null
   +++ b/path/to/file
6. For file deletion use:
   --- a/path/to/file
   +++ /dev/null
7. Every hunk must have a correct header:
   @@ -start,count +start,count @@
8. Context lines must match the provided file contents exactly, including indentation and empty lines.
9. Do NOT omit required context lines.
10. Do NOT use placeholders like:
   ...
   <existing code>
   rest unchanged
11. Keep the patch minimal and focused.
12. Do NOT modify unrelated code, formatting, imports, or file structure unless required.
13. Do NOT invent files, symbols, or code that are not supported by the provided snapshot.
14. If the requested change cannot be made safely from the provided snapshot, output exactly:
   NEED_MORE_CONTEXT
15. If no code changes are required, output exactly:
   NO_CHANGES

PATCH QUALITY RULES:
- Prefer the smallest correct patch.
- Preserve unchanged code exactly.
- The patch should be directly applicable by git apply / patch.
==================================================================
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