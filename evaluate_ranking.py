import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def get_entity_prefix(entity_id: str) -> str:
    """Return entity type prefix including separator, e.g. 'Compound::'."""
    text = str(entity_id)
    if "::" in text:
        return text.split("::", 1)[0] + "::"
    return ""


def compact_entity_label(entity_id: str) -> str:
    """Shorten long entity IDs for plotting readability."""
    text = str(entity_id)
    if "::" in text:
        return text.split("::", 1)[1]
    return text


def standardize_triple_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with canonical columns: head, interaction, tail."""
    columns_lower = {c.lower(): c for c in df.columns}

    head_col = columns_lower.get("head")
    tail_col = columns_lower.get("tail")
    interaction_col = columns_lower.get("interaction") or columns_lower.get("relation")

    if head_col and tail_col and interaction_col:
        out = df[[head_col, interaction_col, tail_col]].copy()
        out.columns = ["head", "interaction", "tail"]
        return out

    if df.shape[1] < 3:
        raise ValueError("Dataset must contain at least 3 columns (head, interaction/relation, tail).")

    out = df.iloc[:, :3].copy()
    out.columns = ["head", "interaction", "tail"]
    return out


def infer_task_relation(ranking: list[dict]) -> str | None:
    if not ranking:
        return None
    series = pd.Series([x.get("relation") for x in ranking]).dropna()
    if series.empty:
        return None
    return series.mode().iloc[0]


def infer_dominant_prefix(series: pd.Series) -> str | None:
    prefixes = series.astype(str).map(get_entity_prefix)
    prefixes = prefixes[prefixes != ""]
    if prefixes.empty:
        return None
    return prefixes.mode().iloc[0]


def get_top1_prediction_per_head(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        raise ValueError("Ranking dataframe is empty after filtering.")

    ranking_df["confidence"] = pd.to_numeric(ranking_df["confidence"], errors="coerce")
    if ranking_df["confidence"].isna().any():
        raise ValueError("Found non-numeric confidence values in ranking JSON.")

    # In complete ranking tasks, each head should have same number of candidate tails.
    rows_per_head = ranking_df.groupby("head").size()
    if rows_per_head.min() != rows_per_head.max():
        print(
            "[w] Inconsistent candidate count per head: "
            f"min={rows_per_head.min()}, max={rows_per_head.max()} "
            "(continuing anyway)."
        )

    # Pick tail with max confidence per head.
    idx = ranking_df.groupby("head", sort=False)["confidence"].idxmax()
    top1 = ranking_df.loc[idx, ["head", "tail", "confidence"]].copy()
    top1 = top1.rename(columns={"head": "head_id", "tail": "pred_tail", "confidence": "pred_confidence"})
    return top1


def build_prediction_debug_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Return top-1 and top-2 confidence per head for quick sanity checks."""
    ranking_df = ranking_df.copy()
    ranking_df["confidence"] = pd.to_numeric(ranking_df["confidence"], errors="coerce")
    ranking_df = ranking_df.sort_values(["head", "confidence"], ascending=[True, False]).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.groupby("head").cumcount() + 1
    top2 = ranking_df[ranking_df["rank"] <= 2].copy()
    top2["slot"] = top2["rank"].map({1: "top1", 2: "top2"})
    out = (
        top2.pivot(index="head", columns="slot", values=["tail", "confidence"])
        .reset_index()
    )
    out.columns = [
        "head_id",
        "top1_tail",
        "top2_tail",
        "top1_confidence",
        "top2_confidence",
    ]
    out["margin_top1_top2"] = out["top1_confidence"] - out["top2_confidence"]
    return out


def get_ground_truth_per_head(
    gt_df: pd.DataFrame,
    ranking_relation: str | None,
    head_prefix: str,
    tail_prefix: str,
) -> pd.DataFrame:
    triples = standardize_triple_columns(gt_df)

    # Focus on the same entity types present in ranking, e.g. Compound:: -> ExtGene::.
    head_mask = triples["head"].astype(str).str.startswith(head_prefix)
    tail_mask = triples["tail"].astype(str).str.startswith(tail_prefix)
    subset = triples[head_mask & tail_mask].copy()

    if subset.empty:
        raise ValueError(
            f"No ground-truth rows matching {head_prefix} -> {tail_prefix} found in dataset."
        )

    if ranking_relation is not None:
        rel_subset = subset[subset["interaction"].astype(str) == ranking_relation]
        if not rel_subset.empty:
            subset = rel_subset

    # If a head has multiple tails, keep the most frequent one as GT label.
    gt = (
        subset.groupby("head")["tail"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={"head": "head_id", "tail": "gt_tail"})
    )
    return gt


def plot_confusion_heatmap(matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    if plt is None:
        print(f"[w] Matplotlib unavailable, skipping heatmap: {output_path}")
        return
    labels = matrix.columns.tolist()
    data = matrix.values

    n = len(labels)
    fig_w = min(24, max(9, int(0.45 * n)))
    fig_h = min(22, max(7, int(0.40 * n)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([compact_entity_label(x) for x in labels], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([compact_entity_label(x) for x in labels], fontsize=8)
    ax.set_xlabel("Predicted tail")
    ax.set_ylabel("Ground truth tail")
    ax.set_title(title)

    # Annotate cells only for small matrices; large ones are unreadable with text overlay.
    annotate = n <= 20
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def reduce_confusion_for_display(matrix: pd.DataFrame, max_labels: int) -> pd.DataFrame:
    """
    Keep only the most relevant labels (by GT support + predicted support) for readable plots.
    """
    if len(matrix) <= max_labels:
        return matrix

    gt_support = matrix.sum(axis=1)
    pred_support = matrix.sum(axis=0)
    score = gt_support.add(pred_support, fill_value=0)
    keep_labels = score.sort_values(ascending=False).head(max_labels).index.tolist()
    return matrix.reindex(index=keep_labels, columns=keep_labels, fill_value=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GT-vs-top1-prediction confusion matrix and heatmap from ranking."
    )
    parser.add_argument(
        "--ranking_json",
        type=str,
        default="models/target_PathogenKG_n74_20260223_142722/compgcn_run0_ranking.json",
        help="Path to ranking JSON produced by train_and_eval.py",
    )
    parser.add_argument(
        "--ground_truth_tsv",
        type=str,
        default="dataset/PathogenKG_n19.tsv",
        help="Path to dataset TSV with ground truth triples",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="models",
        help="Output directory for CSV and heatmaps",
    )
    parser.add_argument(
        "--max_plot_labels",
        type=int,
        default=30,
        help="Max number of labels shown in heatmap plots (default: 30).",
    )
    args = parser.parse_args()

    ranking_path = Path(args.ranking_json)
    gt_path = Path(args.ground_truth_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ranking_path.open("r", encoding="utf-8") as f:
        ranking = json.load(f)
    ranking_df = pd.DataFrame(ranking)

    if ranking_df.empty:
        raise ValueError("Ranking JSON is empty.")

    required_cols = {"head", "tail", "confidence"}
    missing = required_cols - set(ranking_df.columns)
    if missing:
        raise ValueError(f"Ranking JSON missing required keys: {sorted(missing)}")

    ranking_relation = infer_task_relation(ranking)
    if ranking_relation is not None and "relation" in ranking_df.columns:
        ranking_df = ranking_df[ranking_df["relation"].astype(str) == ranking_relation].copy()
        if ranking_df.empty:
            raise ValueError(f"No ranking rows left after filtering by relation '{ranking_relation}'.")

    head_prefix = infer_dominant_prefix(ranking_df["head"])
    tail_prefix = infer_dominant_prefix(ranking_df["tail"])
    if head_prefix is None or tail_prefix is None:
        raise ValueError("Could not infer entity prefixes from ranking head/tail columns.")

    ranking_df = ranking_df[
        ranking_df["head"].astype(str).str.startswith(head_prefix)
        & ranking_df["tail"].astype(str).str.startswith(tail_prefix)
    ].copy()
    if ranking_df.empty:
        raise ValueError(
            f"No ranking rows left after prefix filtering {head_prefix} -> {tail_prefix}."
        )

    gt_raw = pd.read_csv(gt_path, sep="\t", dtype=str)

    pred_top1 = get_top1_prediction_per_head(ranking_df)
    pred_debug = build_prediction_debug_table(ranking_df)
    gt_per_head = get_ground_truth_per_head(gt_raw, ranking_relation, head_prefix, tail_prefix)

    merged = gt_per_head.merge(pred_top1, on="head_id", how="inner")
    if merged.empty:
        raise ValueError("No overlap between ground truth heads and ranking heads.")

    merged = merged.sort_values(["head_id"]).reset_index(drop=True)

    labels = sorted(set(merged["gt_tail"]).union(set(merged["pred_tail"])))

    confusion_counts = pd.crosstab(
        merged["gt_tail"],
        merged["pred_tail"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    confusion_norm = confusion_counts.div(confusion_counts.sum(axis=1).replace(0, 1), axis=0)

    match_acc = (merged["gt_tail"] == merged["pred_tail"]).mean()

    merged_csv = out_dir / "head_top1_vs_ground_truth.csv"
    debug_csv = out_dir / "head_top2_debug.csv"
    counts_csv = out_dir / "confusion_matrix_counts.csv"
    norm_csv = out_dir / "confusion_matrix_row_normalized.csv"
    counts_png = out_dir / "confusion_matrix_heatmap_counts.png"
    norm_png = out_dir / "confusion_matrix_heatmap_row_normalized.png"

    merged.to_csv(merged_csv, index=False)
    pred_debug.to_csv(debug_csv, index=False)
    confusion_counts.to_csv(counts_csv)
    confusion_norm.to_csv(norm_csv)

    confusion_counts_plot = reduce_confusion_for_display(confusion_counts, args.max_plot_labels)
    confusion_norm_plot = reduce_confusion_for_display(confusion_norm, args.max_plot_labels)

    plot_confusion_heatmap(
        confusion_counts_plot,
        counts_png,
        title=(
            f"Confusion Matrix (Counts, top {len(confusion_counts_plot)}) | "
            f"heads={len(merged)} | top1-acc={match_acc:.4f}"
        ),
    )
    plot_confusion_heatmap(
        confusion_norm_plot,
        norm_png,
        title=(
            f"Confusion Matrix (Row-Normalized, top {len(confusion_norm_plot)}) | "
            f"heads={len(merged)} | top1-acc={match_acc:.4f}"
        ),
    )

    print(f"[i] Evaluated heads: {len(merged)}")
    print(f"[i] Entity types: {head_prefix} -> {tail_prefix}")
    print(f"[i] Relation filter: {ranking_relation}")
    print(f"[i] Top-1 tail accuracy: {match_acc:.4f}")
    print(f"[i] Pred top-1 distribution: {merged['pred_tail'].value_counts().to_dict()}")
    print(f"[i] Avg margin top1-top2: {pred_debug['margin_top1_top2'].mean():.6f}")
    print(f"[i] Saved: {merged_csv}")
    print(f"[i] Saved: {debug_csv}")
    print(f"[i] Saved: {counts_csv}")
    print(f"[i] Saved: {norm_csv}")
    print(f"[i] Saved: {counts_png}")
    print(f"[i] Saved: {norm_png}")


if __name__ == "__main__":
    main()
