import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin.
# Если поддержка Kotlin не включена в пакет по умолчанию, можно использовать tree-sitter-kotlin.
language = get_language('kotlin')
parser = get_parser('kotlin')


def count_nodes(node):
    """Рекурсивно считает количество узлов в AST."""
    return 1 + sum(count_nodes(child) for child in node.children)


# Получаем путь к директории от пользователя
directory = input("Введите путь к директории: ").strip()

# Рекурсивный обход директории
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".kt"):
            kt_filepath = os.path.join(root, file)
            try:
                with open(kt_filepath, "r", encoding="utf8") as f:
                    code = f.read()
                # Парсинг кода
                tree = parser.parse(code.encode("utf8"))
                program_size = count_nodes(tree.root_node)
                print(f"{kt_filepath}: Program Size = {program_size}")

                # Формирование имени CSV файла (например, myfile.kt -> myfile.csv)
                base_name, _ = os.path.splitext(file)
                csv_filename = base_name + ".csv"
                csv_filepath = os.path.join(root, csv_filename)

                # Запись результата в CSV файл
                with open(csv_filepath, "w", newline="", encoding="utf8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Program Size"])
                    writer.writerow([program_size])

            except Exception as e:
                print(f"Ошибка обработки файла {kt_filepath}: {e}")
