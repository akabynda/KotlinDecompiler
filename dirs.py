import os


def print_directories_and_kt_content(path):
    # Перебираем все элементы в указанной директории
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        # Если элемент является папкой
        if os.path.isdir(full_path):
            # Ищем в данной папке файлы с расширением .kt
            kt_files = [f for f in os.listdir(full_path) if f.endswith(".kt")]
            if kt_files:
                for kt_file in kt_files:
                    kt_full_path = os.path.join(full_path, kt_file)
                    print(f"\nСодержимое файла {kt_file} в директории {item}:")
                    try:
                        with open(kt_full_path, "r", encoding="utf8") as f:
                            content = f.read()
                        print(content)
                    except Exception as e:
                        print(f"Ошибка при чтении файла {kt_full_path}: {e}")
                    print("\n" + "-" * 40 + "\n")
            else:
                print("В этой директории нет файлов с расширением .kt.\n")


if __name__ == "__main__":
    directory = input("Введите путь к папке: ").strip()
    print("Найденные папки и содержимое их .kt файлов:\n")
    print_directories_and_kt_content(directory)
