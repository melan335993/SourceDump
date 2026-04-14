import os
import sys
import io
import fnmatch
import argparse
from pathlib import Path

# Попытка импорта tiktoken
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Базовые исключения (всегда активны)
DEFAULT_IGNORE = {".git", "__pycache__", "node_modules", ".DS_Store"}

def is_excluded(name, exclude_patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns)

def generate_tree(root_dir, exclude_patterns, prefix=""):
    tree = []
    try:
        items = sorted([item for item in root_dir.iterdir() if not is_excluded(item.name, exclude_patterns)])
    except PermissionError:
        return [f"{prefix}└── [Permission Denied]"]

    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        tree.append(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree.extend(generate_tree(item, exclude_patterns, new_prefix))
    return tree

def get_files_content(root_path, exclude_patterns):
    lines = []
    def walk(current_path):
        try:
            for item in sorted(current_path.iterdir()):
                if is_excluded(item.name, exclude_patterns):
                    continue
                if item.is_file():
                    try:
                        content = item.read_text(encoding='utf-8')
                        rel_path = item.relative_to(root_path.parent)
                        lines.append(f'\n<file path="{rel_path}">\n{content}\n</file>')
                    except: continue
                elif item.is_dir():
                    walk(item)
        except PermissionError: pass
    walk(root_path)
    return lines

def get_token_count(text):
    if HAS_TIKTOKEN:
        try:
            # Используем кодировку для GPT-4 / GPT-3.5
            encoding = tiktoken.get_encoding("cl100k_base")
            # disallowed_special=() разрешает кодировать спец. токены как обычный текст
            return len(encoding.encode(text, disallowed_special=())), True
        except Exception as e:
            return len(text) // 4, False
    return len(text) // 4, False
    
def main():
    parser = argparse.ArgumentParser(description="Утилита для создания дампа проекта.")
    parser.add_argument("paths", nargs="*", default=["."], help="Пути к проекту")
    parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Доп. паттерны")
    parser.add_argument("-ef", "--exclude-file", help="Файл исключений")
    parser.add_argument("-o", "--output", help="Выходной файл")

    args = parser.parse_args()

    final_excludes = set(DEFAULT_IGNORE)
    
    # 1. Из командной строки (argparse сам корректно кушает пробелы в кавычках)
    if args.exclude:
        final_excludes.update(args.exclude)

    # 2. Из файла (теперь читаем ПОСТРОЧНО)
    if args.exclude_file:
        ef = Path(args.exclude_file)
        if ef.exists():
            with open(ef, 'r', encoding='utf-8') as f:
                for line in f:
                    # Убираем только символы переноса строки, но сохраняем внутренние пробелы
                    clean_line = line.strip()
                    # Игнорируем комментарии и пустые строки
                    if clean_line and not clean_line.startswith('#'):
                        # Если строка выглядит как список через пробел (старый формат), 
                        # она все равно сработает, но лучше писать по одной на строку.
                        # Чтобы поддержать и то и то, проверяем наличие пробелов:
                        if " " in clean_line and not any(c in clean_line for c in "*?[]"):
                            # Если пробел есть, но нет спецсимволов — скорее всего это список
                            for part in clean_line.split():
                                final_excludes.add(part.strip('"').strip("'"))
                        else:
                            # Это либо паттерн с пробелом, либо обычный паттерн
                            final_excludes.add(clean_line.strip('"').strip("'"))
        else:
            print(f"⚠️ Файл {args.exclude_file} не найден", file=sys.stderr)

    exclude_list = list(final_excludes)
    full_output = ["=== PROJECT STRUCTURE ==="]

    for p in args.paths:
        root = Path(p).resolve()
        if root.exists():
            full_output.append(f"\nRoot: {root.name}/")
            full_output.extend(generate_tree(root, exclude_list))

    full_output.append("\n" + "="*25 + "\n=== FILE CONTENTS ===")

    for p in args.paths:
        root = Path(p).resolve()
        if root.exists():
            full_output.extend(get_files_content(root, exclude_list))

    result = "\n".join(full_output)
    tokens, is_exact = get_token_count(result)

    if args.output:
        Path(args.output).write_text(result, encoding='utf-8')
        print(f"✅ Готово! Дамп: {args.output}")
    else:
        print(result)

    print(f"\n{'-'*30}\n📊 {'Точно' if is_exact else 'Примерно'}: {tokens} токенов.\n{'-'*30}")

if __name__ == "__main__":
    main()