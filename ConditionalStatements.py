import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin.
language = get_language('kotlin')
parser = get_parser('kotlin')


def count_if_statements(node):
    """
    Рекурсивно считает узлы AST, соответствующие if-выражениям.
    Предполагается, что для Kotlin узлы if-выражения имеют тип "if_expression".
    """
    count = 1 if node.type == "if_expression" else 0
    for child in node.children:
        count += count_if_statements(child)
    return count


def append_column_to_csv(csv_filepath, column_name, column_value):
    """
    Если CSV-файл существует, считывает его содержимое, добавляет новый столбец и записывает обратно.
    Если файла нет, создаёт его с одним столбцом.
    Предполагается, что CSV содержит две строки: первая – заголовки, вторая – данные.
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

# Рекурсивно обходим директорию
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".kt"):
            kt_filepath = os.path.join(root, file)
            try:
                with open(kt_filepath, "r", encoding="utf8") as f:
                    code = f.read()
                tree = parser.parse(code.encode("utf8"))
                conditional_count = count_if_statements(tree.root_node)
                print(f"{kt_filepath}: Conditional Statements = {conditional_count}")

                # Формируем имя CSV-файла для данного .kt файла (например, MyFile.kt -> MyFile.csv)
                base_name, _ = os.path.splitext(file)
                csv_filename = base_name + ".csv"
                csv_filepath = os.path.join(root, csv_filename)

                # Добавляем столбец "Conditional Statements" с рассчитанным значением
                append_column_to_csv(csv_filepath, "Conditional Statements", conditional_count)

            except Exception as e:
                print(f"Ошибка обработки файла {kt_filepath}: {e}")
