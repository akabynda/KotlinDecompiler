import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin.
language = get_language('kotlin')
parser = get_parser('kotlin')


def count_local_variables(node, in_function=False):
    """
    Рекурсивно подсчитывает количество локальных переменных.
    Считаем, что объявления переменных в функциях представлены узлами типа "property".
    Параметр in_function = True, если текущий узел (или его предки) являются частью тела функции.
    """
    count = 0
    # Если узел представляет объявление функции, переходим в режим in_function.
    if node.type in ["function_declaration", "function_definition"]:
        in_function = True
    # Если мы находимся внутри функции и узел – это свойство, считаем его как локальную переменную.
    if in_function and node.type == "property_declaration":
        count += 1
    for child in node.children:
        count += count_local_variables(child, in_function)
    return count


def append_column_to_csv(csv_filepath, column_name, column_value):
    """
    Если CSV-файл существует, считывает его содержимое, добавляет новый столбец (как в заголовке, так и в строке данных)
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


# Запрашиваем путь к директории у пользователя
directory = input("Введите путь к директории: ").strip()

# Рекурсивно обходим директорию
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".kt"):
            kt_filepath = os.path.join(root, file)
            try:
                with open(kt_filepath, "r", encoding="utf8") as f:
                    code = f.read()
                # Парсим исходный код и строим AST
                tree = parser.parse(code.encode("utf8"))
                local_vars_count = count_local_variables(tree.root_node, in_function=False)
                if local_vars_count != 0:
                    print(f"{kt_filepath}: Local Variables = {local_vars_count}")

                # Формируем имя CSV-файла (например, MyFile.kt -> MyFile.csv)
                base_name, _ = os.path.splitext(file)
                csv_filename = base_name + ".csv"
                csv_filepath = os.path.join(root, csv_filename)

                # Добавляем столбец "Local Variables" с рассчитанным значением
                append_column_to_csv(csv_filepath, "Local Variables", local_vars_count)
            except Exception as e:
                print(f"Ошибка обработки файла {kt_filepath}: {e}")
