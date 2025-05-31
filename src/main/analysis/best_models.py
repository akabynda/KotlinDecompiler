import sys
from typing import Tuple

import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine, chebyshev

from src.global_config import FEATURES, COV_EPS


class MetricProcessor:
    """
    Processes metrics data by calculating distances to a reference vector
    and merging results with compilation data.
    """

    def __init__(self, metrics_path: str, comp_path: str, output_path: str) -> None:
        self.metrics_path = metrics_path
        self.comp_path = comp_path
        self.output_path = output_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load metrics and compilation data from CSV files.
        """
        try:
            df_metrics = pd.read_csv(self.metrics_path)
            df_comp = pd.read_csv(self.comp_path)
            return df_metrics, df_comp
        except FileNotFoundError:
            print("Error: One of the files was not found.")
            sys.exit(1)

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns are present in the dataset.
        """
        required_columns = FEATURES + ["model"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

    def compute_reference_vector(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the reference vector for 'kt_source' model.
        """
        if "kt_source" not in df["model"].values:
            raise ValueError("'kt_source' not found in 'model' column.")

        ref_vec = df[df["model"] == "kt_source"][FEATURES].iloc[0].copy()
        for feat in ["CondE", "JSD", "KL"]:
            if feat in ref_vec:
                ref_vec[feat] = 0.0
        return ref_vec

    def compute_distances(
        self, row: pd.Series, ref_vec: pd.Series, coverage: float
    ) -> pd.Series:
        """
        Compute various distances between a row and the reference vector.
        """
        v = row[FEATURES].values
        cov = max(coverage, COV_EPS)
        return pd.Series(
            {
                "euclidean_cov": euclidean(v, ref_vec) / cov,
                "manhattan_cov": cityblock(v, ref_vec) / cov,
                "cosine_cov": cosine(v, ref_vec) / cov,
                "chebyshev_cov": chebyshev(v, ref_vec) / cov,
                "euclidean": euclidean(v, ref_vec),
                "manhattan": cityblock(v, ref_vec),
                "cosine": cosine(v, ref_vec),
                "chebyshev": chebyshev(v, ref_vec),
            }
        )

    def process(self) -> None:
        """
        Main processing pipeline: load data, validate, compute distances,
        merge results, and save to a CSV file.
        """
        df_metrics, df_comp = self.load_data()
        self.validate_columns(df_metrics)

        df_avg = df_metrics.groupby("model")[FEATURES].median().reset_index()
        ref_vec = self.compute_reference_vector(df_avg)

        total_cases = df_metrics[df_metrics["model"] == "kt_source"].shape[0]
        case_counts = df_metrics.groupby("model").size().rename("case_count")
        df_avg = df_avg.merge(case_counts, on="model")

        distances = df_avg.apply(
            lambda row: self.compute_distances(
                row, ref_vec, row["case_count"] / total_cases
            ),
            axis=1,
        )

        df_with_dist = pd.concat([df_avg, distances], axis=1)
        df_with_dist = df_with_dist[df_with_dist["model"] != "kt_source"]

        df_result = (
            df_with_dist.merge(
                df_comp.rename(columns={"field": "model"}), on="model", how="left"
            )
            .fillna({"repo_count": 0})
            .astype({"repo_count": int})
        )

        df_result = df_result.sort_values(
            by=["chebyshev_cov", "repo_count"], ascending=[True, False]
        )

        df_result.to_csv(self.output_path, index=False)
        print(f"Done! Results saved to: {self.output_path}")


def main() -> None:
    """
    Entry point for running the script.
    """
    metrics_path = input("Path to metrics CSV: ").strip()
    comp_path = input("Path to compilations CSV: ").strip()
    output_path = input("Path to save results: ").strip()

    processor = MetricProcessor(metrics_path, comp_path, output_path)
    processor.process()


if __name__ == "__main__":
    main()
