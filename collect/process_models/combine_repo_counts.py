import pandas as pd

file1 = input("First file: ")
file2 = input("Second file: ")
output_file = input("Output: ")

try:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    combined = pd.concat([df1, df2])
    result = combined.groupby("field", as_index=False)["repo_count"].sum()

    result = result.sort_values(by="repo_count", ascending=False)

    result.to_csv(output_file, index=False)

except FileNotFoundError:
    print("Ошибка: один из файлов не найден. Проверьте пути.")
except Exception as e:
    print(f"Произошла ошибка: {e}")
