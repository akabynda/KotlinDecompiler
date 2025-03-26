import os
import csv
from tree_sitter_languages import get_language, get_parser

# Загружаем грамматику и парсер для Kotlin
language = get_language('kotlin')
parser = get_parser('kotlin')


# ============================
# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК
# ============================

def count_nodes(node):
    """Рекурсивно считает общее количество узлов в AST."""
    return 1 + sum(count_nodes(child) for child in node.children)


def count_abrupt_control_flow(node):
    """
    Рекурсивно подсчитывает узлы, соответствующие операторам прерывания:
    типы "break_expression" и "continue_expression".
    """
    count = 1 if node.type in ["jump_expression", "break_expression", "continue_expression"] else 0
    for child in node.children:
        count += count_abrupt_control_flow(child)
    return count


def count_if_statements(node):
    """
    Рекурсивно считает узлы if‑выражений.
    Предполагается, что для Kotlin узлы if‑выражения имеют тип "if_expression".
    """
    count = 1 if node.type == "if_expression" else 0
    for child in node.children:
        count += count_if_statements(child)
    return count


def get_condition_expressions(node):
    """
    Рекурсивно находит условные выражения (условия if‑выражений).
    Предполагается, что if‑выражение содержит условие в узле типа "parenthesized_expression".
    """
    conditions = []
    if node.type == "if_expression":
        for child in node.children:
            if child.type == "parenthesized_expression":
                # Если внутри скобок есть реальное выражение, берем первого ребенка
                if child.child_count > 0:
                    conditions.append(child.children[0])
                else:
                    conditions.append(child)
                break
    for child in node.children:
        conditions.extend(get_condition_expressions(child))
    return conditions


def compute_conditional_complexity(node):
    """
    Рекурсивно вычисляет сложность boolean-выражения.
    Схема:
      - Если узел – лист (например, идентификатор или литерал): сложность = 1.
      - Если узел – унарное выражение (например, "!"): 0.5 + сложность операнда.
      - Если узел – бинарное выражение:
            * для операторов "<", ">", "<=", ">=", "==", "!=": 0.5 + сложность левого + сложность правого,
            * для операторов "&&", "||": 1 + сложность левого + сложность правого.
      - Иначе – сумма сложностей всех детей.
    """
    if len(node.children) == 0:
        return 1
    if node.type == "unary_expression":
        return 0.5 + compute_conditional_complexity(node.children[-1])
    if node.type == "binary_expression" and len(node.children) >= 3:
        op_node = node.children[1]
        op = op_node.text.decode('utf8') if hasattr(op_node, 'text') else op_node.type
        if op in ["<", ">", "<=", ">=", "==", "!="]:
            return 0.5 + compute_conditional_complexity(node.children[0]) + compute_conditional_complexity(
                node.children[2])
        elif op in ["&&", "||"]:
            return 1 + compute_conditional_complexity(node.children[0]) + compute_conditional_complexity(
                node.children[2])
        else:
            return sum(compute_conditional_complexity(child) for child in node.children)
    return sum(compute_conditional_complexity(child) for child in node.children)


def count_labeled_blocks(node):
    """
    Рекурсивно подсчитывает узлы, соответствующие Labeled Blocks.
    Здесь в качестве примера проверяем тип узла "label" (при необходимости можно изменить).
    """
    count = 1 if node.type == "label" else 0
    for child in node.children:
        count += count_labeled_blocks(child)
    return count


def count_local_variables(node, in_function=False):
    """
    Рекурсивно подсчитывает количество локальных переменных.
    Предполагается, что объявления переменных внутри функций представлены узлами типа "property_declaration".
    При входе в функцию (узел типа "function_declaration" или "function_definition") in_function становится True.
    """
    count = 0
    if node.type in ["function_declaration", "function_definition"]:
        in_function = True
    if in_function and node.type == "property_declaration":
        count += 1
    for child in node.children:
        count += count_local_variables(child, in_function)
    return count


# Функция для вычисления метрик для одного .kt файла
def compute_metrics_for_file(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        code = f.read()
    tree = parser.parse(code.encode("utf8"))
    metrics = {}
    metrics["program_size"] = count_nodes(tree.root_node)
    metrics["abrupt_control_flow"] = count_abrupt_control_flow(tree.root_node)
    metrics["conditional_statements"] = count_if_statements(tree.root_node)
    conditions = get_condition_expressions(tree.root_node)
    if conditions:
        total_complexity = sum(compute_conditional_complexity(cond) for cond in conditions)
        metrics["conditional_complexity_sum"] = total_complexity
        metrics["conditional_complexity_count"] = len(conditions)
        metrics["conditional_complexity"] = total_complexity / len(conditions)
    else:
        metrics["conditional_complexity_sum"] = 0
        metrics["conditional_complexity_count"] = 0
        metrics["conditional_complexity"] = 0
    metrics["labeled_blocks"] = count_labeled_blocks(tree.root_node)
    metrics["local_variables"] = count_local_variables(tree.root_node, in_function=False)
    return metrics


# Если результат состоит из нескольких файлов, комбинируем метрики (суммируем счетчики, а для conditional_complexity – вычисляем среднее по всем условиям)
def combine_metrics(metrics_list):
    combined = {}
    combined["program_size"] = sum(m["program_size"] for m in metrics_list)
    combined["abrupt_control_flow"] = sum(m["abrupt_control_flow"] for m in metrics_list)
    combined["conditional_statements"] = sum(m["conditional_statements"] for m in metrics_list)
    combined["labeled_blocks"] = sum(m["labeled_blocks"] for m in metrics_list)
    combined["local_variables"] = sum(m["local_variables"] for m in metrics_list)
    total_sum = sum(m["conditional_complexity_sum"] for m in metrics_list)
    total_count = sum(m["conditional_complexity_count"] for m in metrics_list)
    if total_count > 0:
        combined["conditional_complexity"] = total_sum / total_count
    else:
        combined["conditional_complexity"] = 0
    return combined


def get_kt_files(path):
    """Возвращает список всех .kt файлов в path (если path – файл, возвращает его в списке)."""
    if os.path.isfile(path) and path.endswith(".kt"):
        return [path]
    kt_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".kt"):
                kt_files.append(os.path.join(root, f))
    return kt_files


def compute_metrics_for_result(result_path):
    """Если result_path – файл или папка, находит все .kt файлы и объединяет их метрики."""
    kt_files = get_kt_files(result_path)
    if not kt_files:
        return None
    metrics_list = []
    for file in kt_files:
        try:
            m = compute_metrics_for_file(file)
            metrics_list.append(m)
        except Exception as e:
            print(f"Ошибка обработки файла {file}: {e}")
    if metrics_list:
        return combine_metrics(metrics_list)
    return None


# Функция для записи словаря метрик в CSV (одна строка с заголовками и одна строка с данными)
def write_metrics_to_csv(csv_filepath, metrics):
    headers = ["program_size", "abrupt_control_flow", "conditional_statements", "conditional_complexity",
               "labeled_blocks", "local_variables"]
    with open(csv_filepath, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        row = [metrics[h] for h in headers]
        writer.writerow(row)


def normalize_metrics(result_metrics, origin_metrics):
    """
    Нормализует метрики результата относительно метрик оригинала.
    Для каждой метрики: normalized = (result value) / (origin value), если origin value != 0, иначе 0.
    """
    normalized = {}
    for key in ["program_size", "abrupt_control_flow", "conditional_statements", "conditional_complexity",
                "labeled_blocks", "local_variables"]:
        orig = origin_metrics.get(key, 0)
        normalized[key] = result_metrics.get(key, 0) / orig if orig != 0 else 0
    return normalized


# ============================
# ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ
# ============================

# Запрашиваем путь к директории, содержащей тестовые примеры (например, папка с тестовыми примерами, такими как builder, ...)
main_dir = input("Введите путь к директории тестовых примеров: ").strip()

# Обрабатываем каждый тестовый пример (предполагается, что они находятся на первом уровне в main_dir)
for test_folder in os.listdir(main_dir):
    test_folder_path = os.path.join(main_dir, test_folder)
    if not os.path.isdir(test_folder_path):
        continue
    print(f"\nОбработка тестового примера: {test_folder}")

    # Ищем оригинальный тестовый файл (находится непосредственно в test_folder, а не в поддиректориях)
    origin_files = [f for f in os.listdir(test_folder_path) if
                    f.endswith(".kt") and os.path.isfile(os.path.join(test_folder_path, f))]
    if not origin_files:
        print(f"Оригинальный тестовый файл не найден в {test_folder}")
        continue
    origin_file_path = os.path.join(test_folder_path, origin_files[0])
    try:
        origin_metrics = compute_metrics_for_file(origin_file_path)
    except Exception as e:
        print(f"Ошибка обработки оригинального файла {origin_file_path}: {e}")
        continue
    # Записываем метрики оригинала в Origin.csv в папке тестового примера
    origin_base = os.path.splitext(origin_files[0])[0]
    origin_csv = os.path.join(test_folder_path, f"Origin{origin_base}.csv")
    write_metrics_to_csv(origin_csv, origin_metrics)
    print(f"Записаны метрики оригинала в {origin_csv}")

    # Задаем имена инструментов (папок) с результатами декомпиляции
    tool_names = ["Bytecode", "CFR", "Fernflower", "JDGUI"]
    for tool in tool_names:
        tool_folder = os.path.join(test_folder_path, tool)
        if not os.path.isdir(tool_folder):
            continue
        # Определяем допустимые идентификаторы результата
        if tool == "Bytecode":
            result_ids = ["ChatGPT"]
        else:
            result_ids = ["J2K", "ChatGPT", "CodeConvert"]
        # Проходим по элементам в папке инструмента
        for item in os.listdir(tool_folder):
            # Если имя элемента содержит один из идентификаторов (без учета регистра)
            if any(result_id.lower() in item.lower() for result_id in result_ids):
                result_path = os.path.join(tool_folder, item)
                result_metrics = compute_metrics_for_result(result_path)
                if result_metrics is None:
                    print(f"В {result_path} не найдено .kt файлов.")
                    continue
                # Формируем имя CSV файла: ToolName + <название элемента> + ".csv"
                base_item = os.path.splitext(item)[0]
                new_base_item = base_item
                for result_id in result_ids:
                    if base_item.lower().endswith(result_id.lower()):
                        part_without = base_item[:-len(result_id)]
                        new_base_item = result_id + part_without
                        break
                csv_name = tool + new_base_item + ".csv"
                csv_path = os.path.join(tool_folder, csv_name)
                write_metrics_to_csv(csv_path, result_metrics)
                print(f"Записаны метрики для {result_path} в {csv_path}")
