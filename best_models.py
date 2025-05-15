import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine, chebyshev

metrics_path = input("Путь к CSV с метриками: ")
comp_path = input("Путь к CSV с компиляциями: ")
output_path = input("Путь для сохранения результата: ")

try:
    df_metrics = pd.read_csv(metrics_path)
    df_comp = pd.read_csv(comp_path)
except FileNotFoundError:
    print("Ошибка: один из файлов не найден.")
    exit(1)

features = [
    'CE', 'CondE', 'Conditional Complexity',
    'Halstead Distinct Operators', 'Halstead Vocabulary',
    'JSD', 'KL', 'LM_CondE', 'LM_JSD'
]

for f in features + ['model']:
    if f not in df_metrics.columns:
        raise ValueError(f"Не хватает столбца: {f}")

df_avg = df_metrics.groupby('model')[features].median().reset_index()

if 'kt_source' not in df_avg['model'].values:
    raise ValueError("kt_source не найден в model")
ref_vec = df_avg[df_avg['model'] == 'kt_source'][features].iloc[0].values


def compute_distances(row):
    v = row[features].values
    return pd.Series({
        'euclidean': euclidean(v, ref_vec),
        'manhattan': cityblock(v, ref_vec),
        'cosine': cosine(v, ref_vec),
        'chebyshev': chebyshev(v, ref_vec)
    })


distances = df_avg.apply(compute_distances, axis=1)
df_with_dist = pd.concat([df_avg, distances], axis=1)

df_with_dist = df_with_dist[df_with_dist['model'] != 'kt_source']

df_result = df_with_dist.merge(
    df_comp.rename(columns={'field': 'model'}),
    on='model',
    how='left'
).fillna({'repo_count': 0}).astype({'repo_count': int})

df_result = df_result.sort_values(
    by=['chebyshev', 'repo_count'],
    ascending=[True, False]
)

df_result.to_csv(output_path, index=False)
print(f"Готово! Результат сохранён в: {output_path}")

print(df_result[['model', 'repo_count', 'euclidean', 'manhattan', 'cosine', 'chebyshev']].to_string(index=False))
