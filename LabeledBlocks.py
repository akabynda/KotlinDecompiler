import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin.
language = get_language('kotlin')
parser = get_parser('kotlin')


def count_labeled_blocks(node):
    """
    Рекурсивно подсчитывает количество узлов, представляющих Labeled Blocks.
    Предполагается, что в грамматике tree-sitter для Kotlin такие конструкции имеют тип "labeled_expression".
    """
    count = 1 if node.type == "label" else 0
    for child in node.children:
        count += count_labeled_blocks(child)
    return count


def append_column_to_csv(csv_filepath, column_name, column_value):
    """
    Если CSV-файл существует, считывает его содержимое, добавляет новый столбец (как к заголовкам, так и к данным)
    и записывает обновлённое содержимое обратно. Если файла нет, создаёт новый CSV с единственным столбцом.
    Предполагается, что CSV содержит одну строку заголовков и одну строку с данными.
    """
    if os.path.exists(csv_filepath):
        with open(csv_filepath, "r", newline="", encoding="utf8") as csvfile:
            reader = list(csv.reader(csvfile))
        if reader:
            header = reader[0]
            data = reader[1] if len(reader) > 1 else []
        else:
            header, data = [], []
        header.append(column_name)
        data.append(str(column_value))
        with open(csv_filepath, "w", newline="", encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)
    else:
        with open(csv_filepath, "w", newline="", encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([column_name])
            writer.writerow([column_value])


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
                # Парсинг исходного кода
                tree = parser.parse(code.encode("utf8"))
                labeled_count = count_labeled_blocks(tree.root_node)
                if labeled_count != 0:
                    print(f"{kt_filepath}: Labeled Blocks = {labeled_count}")

                # Формируем имя CSV-файла: например, MyFile.kt -> MyFile.csv
                base_name, _ = os.path.splitext(file)
                csv_filename = base_name + ".csv"
                csv_filepath = os.path.join(root, csv_filename)

                # Добавляем столбец "Labeled Blocks" с рассчитанным значением
                append_column_to_csv(csv_filepath, "Labeled Blocks", labeled_count)
            except Exception as e:
                print(f"Ошибка обработки файла {kt_filepath}: {e}")
