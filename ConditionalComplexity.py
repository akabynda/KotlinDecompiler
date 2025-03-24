import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin.
language = get_language('kotlin')
parser = get_parser('kotlin')


def compute_conditional_complexity(node):
    """
    Рекурсивно вычисляет сложность boolean-выражения.

    - Если узел – лист (например, идентификатор или литерал): сложность = 1.
    - Для унарного оператора (например, "!"): 0.5 + сложность операнда.
    - Для бинарного оператора:
        * Если оператор – один из: "<", ">", "<=", ">=", "==", "!=": 0.5 + сложность левого + сложность правого.
        * Если оператор – один из: "&&", "||": 1 + сложность левого + сложность правого.
    - В остальных случаях сложность равна сумме сложностей всех дочерних узлов.
    """
    # Если узел не имеет дочерних узлов – считаем его сложность равной 1.
    if len(node.children) == 0:
        return 1
    if node.type == "unary_expression":
        # Предполагаем, что структура: [operator, operand]
        return 0.5 + compute_conditional_complexity(node.children[-1])
    if node.type == "binary_expression" and len(node.children) >= 3:
        op_node = node.children[1]
        # Пытаемся получить текст оператора
        op = op_node.text.decode('utf8') if hasattr(op_node, 'text') else op_node.type
        if op in ["<", ">", "<=", ">=", "==", "!="]:
            return 0.5 + compute_conditional_complexity(node.children[0]) + compute_conditional_complexity(
                node.children[2])
        elif op in ["&&", "||"]:
            return 1 + compute_conditional_complexity(node.children[0]) + compute_conditional_complexity(
                node.children[2])
        else:
            return sum(compute_conditional_complexity(child) for child in node.children)
    # Для остальных типов узлов суммируем сложности всех детей.
    return sum(compute_conditional_complexity(child) for child in node.children)


def get_condition_expressions(node):
    """
    Рекурсивно ищет все условные выражения, которые являются условиями в if-выражениях.
    Предполагается, что if-выражение содержит условие в узле типа "parenthesized_expression".
    """
    conditions = []
    if node.type == "if_expression":
        for child in node.children:
            if child.type == "parenthesized_expression":
                # Если внутри скобок есть реальное выражение, берем его.
                if child.child_count > 0:
                    conditions.append(child.children[0])
                else:
                    conditions.append(child)
                break
    for child in node.children:
        conditions.extend(get_condition_expressions(child))
    return conditions


def append_column_to_csv(csv_filepath, column_name, column_value):
    """
    Если CSV-файл существует, считывает его содержимое, добавляет новый столбец и перезаписывает файл.
    Если файла нет, создаёт его с одним столбцом.
    Предполагается, что CSV имеет одну строку заголовков и одну строку данных.
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


# Запрос пути к директории
directory = input("Введите путь к директории: ").strip()

# Рекурсивный обход директории
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".kt"):
            kt_filepath = os.path.join(root, file)
            try:
                with open(kt_filepath, "r", encoding="utf8") as f:
                    code = f.read()
                tree = parser.parse(code.encode("utf8"))
                # Находим все условные выражения (условия в if-выражениях)
                conditions = get_condition_expressions(tree.root_node)
                if conditions:
                    total_complexity = sum(compute_conditional_complexity(cond) for cond in conditions)
                    average_complexity = total_complexity / len(conditions)
                else:
                    average_complexity = 0
                print(f"{kt_filepath}: Average Conditional Complexity = {average_complexity:.2f}")

                # Формируем имя CSV-файла (например, MyFile.kt -> MyFile.csv)
                base_name, _ = os.path.splitext(file)
                csv_filename = base_name + ".csv"
                csv_filepath = os.path.join(root, csv_filename)

                # Добавляем столбец "Conditional Complexity" с рассчитанным значением
                append_column_to_csv(csv_filepath, "Conditional Complexity", f"{average_complexity:.2f}")
            except Exception as e:
                print(f"Ошибка обработки файла {kt_filepath}: {e}")
