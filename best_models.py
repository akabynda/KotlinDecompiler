import sys
from typing import Tuple

import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine, chebyshev

FEATURES = [
    'CE', 'CondE', 'Conditional Complexity',
    'Halstead Distinct Operators', 'Halstead Vocabulary',
    'JSD', 'KL', 'LM_CondE', 'LM_JSD'
]


def load_data(metrics_path: str, comp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df_metrics = pd.read_csv(metrics_path)
        df_comp = pd.read_csv(comp_path)
        return df_metrics, df_comp
    except FileNotFoundError:
        print("Ошибка: один из файлов не найден.")
        sys.exit(1)


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = FEATURES + ['model']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Не хватает столбцов: {', '.join(missing)}")


def compute_reference_vector(df: pd.DataFrame) -> pd.Series:
    if 'kt_source' not in df['model'].values:
        raise ValueError("kt_source не найден в model")
    return df[df['model'] == 'kt_source'][FEATURES].iloc[0]


def compute_distances(row: pd.Series, ref_vec: pd.Series) -> pd.Series:
    v = row[FEATURES].values
    return pd.Series({
        'euclidean': euclidean(v, ref_vec),
        'manhattan': cityblock(v, ref_vec),
        'cosine': cosine(v, ref_vec),
        'chebyshev': chebyshev(v, ref_vec)
    })


def process(metrics_path: str, comp_path: str, output_path: str) -> None:
    df_metrics, df_comp = load_data(metrics_path, comp_path)
    validate_columns(df_metrics)

    df_avg = df_metrics.groupby('model')[FEATURES].median().reset_index()
    ref_vec = compute_reference_vector(df_avg)

    distances = df_avg.apply(lambda row: compute_distances(row, ref_vec), axis=1)
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


def main() -> None:
    metrics_path = input("Путь к CSV с метриками: ")
    comp_path = input("Путь к CSV с компиляциями: ")
    output_path = input("Путь для сохранения результата: ")
    process(metrics_path, comp_path, output_path)


if __name__ == '__main__':
    main()
