import os

# Укажите путь к нужной папке
folder_path = input()

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".csv"):
            file_path = os.path.join(root, file)
            print(f"Удаление файла: {file_path}")
            os.remove(file_path)
