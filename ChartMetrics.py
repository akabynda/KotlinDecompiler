import os
import csv
import matplotlib.pyplot as plt

# Запрашиваем путь к главной директории (где лежат все CSV-файлы)
main_dir = input("Введите путь к главной директории с CSV-файлами: ").strip()

# Задаём ожидаемые категории.
# Для Origin – имя файла оригинала не учитывается, просто категоризируем все как "Origin".
# Для остальных ожидаем, что имя файла начинается с одного из следующих префиксов.
expected_categories = [
    "Origin",
    "BytecodeChatGPT",
    "CFRJ2K", "CFRChatGPT",
    "FernflowerJ2K", "FernflowerChatGPT",
    "JDGUIJ2K", "JDGUIChatGPT"
]

# Инициализируем словарь для накопления метрик по категориям
categories = {cat: {"program_size": [],
                    "abrupt_control_flow": [],
                    "conditional_statements": [],
                    "conditional_complexity": [],
                    "labeled_blocks": [],
                    "local_variables": []} for cat in expected_categories}

def extract_category(csv_filepath):
    base = os.path.basename(csv_filepath)
    # Если имя начинается с "Origin", вернём "Origin"
    if base.startswith("Origin"):
        return "Origin"
    for cat in expected_categories:
        if cat != "Origin" and base.startswith(cat):
            return cat
    return None

# Рекурсивно собираем все CSV-файлы
csv_files = []
for root, dirs, files in os.walk(main_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

# Читаем CSV-файлы и распределяем метрики по категориям
for csv_file in csv_files:
    cat = extract_category(csv_file)
    if not cat:
        continue
    try:
        with open(csv_file, "r", newline="", encoding="utf8") as f:
            reader = list(csv.reader(f))
        if len(reader) < 2:
            continue
        header = reader[0]
        values = reader[1]
        data = {h: float(v) for h, v in zip(header, values)}
    except Exception as e:
        print(f"Ошибка чтения файла {csv_file}: {e}")
        continue
    for metric in categories[cat].keys():
        categories[cat][metric].append(data.get(metric, 0))

# Усредняем метрики по каждой категории
averages = {}
for cat, metrics in categories.items():
    averages[cat] = {}
    for metric, vals in metrics.items():
        if vals:
            averages[cat][metric] = sum(vals) / len(vals)
        else:
            averages[cat][metric] = 0

# Определим порядок категорий: пусть Origin всегда будет первым, затем остальные по алфавиту
sorted_categories = ["Origin"] + sorted([cat for cat in expected_categories if cat != "Origin"])

# Для каждой метрики строим график
metric_names = ["program_size", "abrupt_control_flow", "conditional_statements", "conditional_complexity", "labeled_blocks", "local_variables"]

for metric in metric_names:
    cats = []
    values = []
    for cat in sorted_categories:
        cats.append(cat)
        values.append(averages[cat][metric])
    plt.figure(figsize=(10,6))
    plt.bar(cats, values, color='skyblue')
    plt.xlabel("Комбинация")
    plt.ylabel(metric)
    plt.title(f"Среднее значение метрики {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_file = os.path.join(main_dir, f"chart_{metric}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Сохранён график для {metric} в {output_file}")

    output_csv = os.path.join(main_dir, f"data_{metric}.csv")
    with open(output_csv, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", metric])
        for cat in sorted_categories:
            writer.writerow([cat, averages[cat][metric]])
    print(f"Сохранены агрегированные данные для метрики {metric} в {output_csv}")
