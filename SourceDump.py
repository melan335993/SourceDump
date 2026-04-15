import os
import sys
import io
import fnmatch
import argparse
from pathlib import Path

# Попытка импорта tiktoken для точного подсчета
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Исправление кодировки для Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Базовые исключения (всегда активны)
DEFAULT_IGNORE = {".git", "__pycache__", "node_modules", ".DS_Store", "uv.lock", ".venv"}

DIFF_INSTRUCTION = """
SYSTEM INSTRUCTION:
You are an expert AI developer. Analyze the provided codebase.
When asked for changes, respond ONLY with valid Unified Diff format (---/+++/@@).

CRITICAL RULES FOR DIFFS:
1. CONTEXT MATCHING: Include exactly matching context lines around your changes. DO NOT skip or omit empty lines within the context blocks. Empty lines in the original file MUST be represented as empty context lines (starting with a single space).
2. LINE COUNTS: Accurately calculate the line counts in the chunk headers (@@ -start,count +start,count @@). The count must exactly match the actual number of lines in the chunk.
3. KEEP IT SHORT: Do not rewrite entire files. Keep chunks small and focused on the modified lines.
==================================================================
"""

def is_excluded(path: Path, exclude_patterns):
    posix_path = path.as_posix()
    return any(fnmatch.fnmatch(path.name, pattern.rstrip('/\\')) or fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch(posix_path, f"*/{pattern.strip('/\\')}") for pattern in exclude_patterns)

def generate_tree(root_dir, exclude_patterns, prefix=""):
    tree = []
    try:
        items = sorted([item for item in root_dir.iterdir() if not is_excluded(item, exclude_patterns)])
    except PermissionError:
        return [f"{prefix}└── [Permission Denied]"]

    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        tree.append(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree.extend(generate_tree(item, exclude_patterns, new_prefix))
    return tree

def get_files_content(root_path, exclude_patterns, strip_empty=False):
    def walk(current_path):
        try:
            for item in sorted(current_path.iterdir()):
                if is_excluded(item, exclude_patterns):
                    continue
                
                if item.is_dir():
                    yield from walk(item)
                elif item.is_file():
                    try:
                        content = item.read_text(encoding='utf-8', errors='ignore')
                        if strip_empty:
                            content = "\n".join(line for line in content.splitlines() if line.strip())
                        
                        rel_path = item.relative_to(root_path.parent)
                        yield f'\n<file path="{rel_path}">\n{content}\n</file>'
                    except (PermissionError, OSError):
                        continue
        except PermissionError:
            pass
    
    yield from walk(root_path)

def get_token_count(text):
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            # disallowed_special=() позволяет кодировать спец. токены (напр. <|endoftext|>) как обычный текст
            return len(encoding.encode(text, disallowed_special=())), True
        except: return len(text) // 4, False
    return len(text) // 4, False

def main():
    parser = argparse.ArgumentParser(description="SourceDump: Упаковка проекта для LLM")
    parser.add_argument("paths", nargs="*", default=["."], help="Пути для сканирования")
    parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Доп. исключения")
    parser.add_argument("-ef", "--exclude-file", help="Файл исключений (напр. .gitignore)")
    parser.add_argument("-o", "--output", help="Выходной файл")
    parser.add_argument("--diff", action="store_true", help="Добавить инструкцию для генерации диффов")
    parser.add_argument("--strip", action="store_true", help="Удалить пустые строки из файлов для экономии токенов")

    args = parser.parse_args()

    # Сбор всех паттернов исключений
    final_excludes = set(DEFAULT_IGNORE)
    if args.exclude:
        final_excludes.update(args.exclude)

    if args.exclude_file and (ef := Path(args.exclude_file)).exists():
        with ef.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    final_excludes.update(line.split() if " " in line and not any(c in line for c in "*?[]") else [line])

    exclude_list = [p.strip('"\'') for p in final_excludes]
    full_output = []

    if args.diff:
        full_output.append(DIFF_INSTRUCTION)

    paths = [Path(p).resolve() for p in args.paths if Path(p).exists()]
    
    full_output.append("=== PROJECT STRUCTURE ===")
    for root in paths:
        full_output.append(f"\nRoot: {root.name}/")
        full_output.extend(generate_tree(root, exclude_list))

    full_output.append("\n" + "="*25 + "\n=== FILE CONTENTS ===")
    for root in paths:
        full_output.extend(get_files_content(root, exclude_list, strip_empty=args.strip))

    result = "\n".join(full_output)
    tokens, is_exact = get_token_count(result)

    if args.output:
        Path(args.output).write_text(result, encoding='utf-8')
        print(f"✅ Дамп сохранен в: {args.output}")
    else:
        print(result)

    print(f"\n{'-'*30}\n📊 {'Точно' if is_exact else 'Примерно'}: {tokens} токенов.\n{'-'*30}")

if __name__ == "__main__":
    main()