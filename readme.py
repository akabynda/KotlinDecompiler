import os

def print_readme_contents(directory):
    """
    Рекурсивно ищет файлы README.md в указанной директории и её поддиректориях,
    затем выводит путь к файлу и его содержимое.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == "readme.md":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf8") as f:
                        content = f.read()
                        print(content)
                        print("\n")
                except Exception as e:
                    print(f"Ошибка чтения файла {file_path}: {e}")

# Пример использования:
if __name__ == "__main__":
    directory_path = input("Введите путь к директории: ").strip()
    print_readme_contents(directory_path)
