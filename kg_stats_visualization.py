
"""
Docstring for kg_stats_visualization

This script loads the PathogenKG Knowledge Graph from a TSV file and displays statistics.

file head:
head	interaction	tail	source	type
ExtGene::Uniprot:Rv0001	GENE_BIND	ExtGene::Uniprot:Rv0058	STRING	ExtGene-ExtGene
ExtGene::Uniprot:Rv0001	GENE_BIND	ExtGene::Uniprot:Rv0002	STRING	ExtGene-ExtGene
ExtGene::Uniprot:Rv0002	GENE_BIND	ExtGene::Uniprot:Rv1629	STRING	ExtGene-ExtGene

I want to create the plots that best represent the main statistics of the knowledge graph.

I also want to create a well-defined image of the KnowledgeGraph schema using networkx and matplotlib and save the image as a PNG file in "figures/kg_schema.png".

The nodes representing the classes must be proportional to the number of instances of each class in the KnowledgeGraph. The relationships between classes must be represented as arcs between nodes. and create a graphical visualisation of the graph schema.

This is a biomedical KG with different types of nodes and relationships, which come from 19 different human pathogens. Gene, Compound, GO:BP, GO:MF, GO:CC, Disease, etc. See below for a description of the TSV file columns:

"""
from __future__ import annotations

#%%
import os
import pandas as pd
# Load the merged PathogenKG TSV file
pathogenkg = pd.read_csv("dataset/pathogenkg/PathogenKG_merged.tsv", sep="\t", dtype=str)
pathogenkg.head()
#%%
# describe the columns
pathogenkg.describe(include='object').T.to_clipboard()
"""
    count	unique	top	freq
head	2944846	76248	ExtGene::Uniprot:A0A5K4FAJ8	566
interaction	2944846	12	BiologicalProcess	1541757
tail	2944846	41779	GO::GO:0110165	61823
source	2944846	2	STRING	2943997
type	2944846	4	ExtGene-GeneOnthology	2723643

"""
#%%
pathogenkg.interaction.value_counts().to_clipboard()
"""
interaction	count

# GO:
    type == ExtGene-GeneOnthology:
        BiologicalProcess	1541757
        MolecularFunction	657558
        CellularComponent	524328

# STRING:
    type == ExtGene-ExtGene:
        gene_OTHER_gene	129382
        GENE_BIND	16614
        ACTIVATION	2118
        REACTION	156
        CATALYSIS	40
        PTMOD	9
        INHIBITION	4
    type == ExtGene-OrthologyGroup:
        ORTHOLOGY	72031
# DrugBank:
    type == Compound-ExtGene:
        TARGET	849
"""
#%%
pathogenkg.type.value_counts().to_clipboard()
"""
type	count
ExtGene-GeneOnthology	2723643
ExtGene-ExtGene	148323
ExtGene-OrthologyGroup	72031
Compound-ExtGene	849
"""
#%%
"""kg_stats_visualization

Loads a PathogenKG edge list (TSV) and produces a compact set of plots that
summarize the main statistics of the Knowledge Graph, plus a schema diagram.

The input TSV is expected to contain *one edge/triple per row* with columns:

- ``head``: source node identifier (string)
- ``interaction``: predicate / relation label (string)
- ``tail``: target node identifier (string)
- ``source``: provenance (e.g., STRING, DrugBank)
- ``type``: high-level edge family (e.g., ExtGene-GeneOnthology)

Example rows::

    head\tinteraction\ttail\tsource\ttype
    ExtGene::Uniprot:Rv0001\tGENE_BIND\tExtGene::Uniprot:Rv0058\tSTRING\tExtGene-ExtGene
    ExtGene::Uniprot:Rv0001\tGENE_BIND\tExtGene::Uniprot:Rv0002\tSTRING\tExtGene-ExtGene
    ExtGene::Uniprot:Rv0002\tGENE_BIND\tExtGene::Uniprot:Rv1629\tSTRING\tExtGene-ExtGene

Outputs (saved under ``--outdir``; default: ``figures``):

- ``kg_schema.png``: a schema / metagraph visualization (NetworkX + Matplotlib)
    where nodes represent *classes* (e.g., ExtGene, Compound, GO:BP/GO:MF/GO:CC)
    and node size is proportional to the number of *instances* of that class.
    Directed arcs represent relationships between classes, with thickness
    proportional to the number of edges.
- Additional summary plots (counts by type/interaction/source and node classes).

Notes on node classes
---------------------
Node IDs are assumed to follow the convention ``<Class>::<LocalId>``.
For GO terms (e.g., ``GO::GO:0110165``) the node ID alone does not specify
which GO sub-ontology it belongs to. To represent GO:BP / GO:MF / GO:CC as
separate classes, this script infers the GO sub-ontology from the
``interaction`` value on ``ExtGene-GeneOnthology`` edges.

Usage
-----
    python kg_stats_visualization.py \
        --tsv dataset/pathogenkg/PathogenKG_merged.tsv \
        --outdir figures
"""



import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_TSV = Path("dataset/pathogenkg/PathogenKG_merged.tsv")


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _load_kg(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    expected = {"head", "interaction", "tail", "source", "type"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _base_class_from_node_id(node_ids: pd.Series) -> pd.Series:
    """Extract base class from IDs like 'ExtGene::Uniprot:...' -> 'ExtGene'."""

    node_ids = node_ids.fillna("")
    return node_ids.str.split("::", n=1, expand=False).str[0].replace({"": "Unknown"})


def _infer_go_subontology(df: pd.DataFrame) -> tuple[dict[str, str], pd.Series]:
    """Infer GO sub-ontology label for each GO node ID.

    Returns:
        - mapping from GO node id (tail) -> one of {'GO:BP','GO:MF','GO:CC'}
        - counts of unique GO nodes by sub-ontology (Series)
    """

    go_edges = df[df["type"] == "ExtGene-GeneOnthology"].copy()
    if go_edges.empty:
        return {}, pd.Series(dtype=int)

    # Only consider valid GO ids in tail.
    go_edges = go_edges[go_edges["tail"].fillna("").str.startswith("GO::")]
    if go_edges.empty:
        return {}, pd.Series(dtype=int)

    interaction_to_go = {
        "BiologicalProcess": "GO:BP",
        "MolecularFunction": "GO:MF",
        "CellularComponent": "GO:CC",
    }
    go_edges["go_class"] = go_edges["interaction"].map(interaction_to_go)
    go_edges = go_edges.dropna(subset=["go_class"])
    if go_edges.empty:
        return {}, pd.Series(dtype=int)

    # If a GO term appears with multiple interactions, pick the most frequent.
    counts = (
        go_edges.groupby(["tail", "go_class"], dropna=False)
        .size()
        .rename("edge_count")
        .reset_index()
    )
    idx = counts.groupby("tail")["edge_count"].idxmax()
    top = counts.loc[idx, ["tail", "go_class"]]
    go_map = dict(zip(top["tail"].astype(str), top["go_class"].astype(str)))

    unique_terms_per_onto = go_edges.groupby("go_class")["tail"].nunique().sort_values(ascending=False)
    return go_map, unique_terms_per_onto


def _node_class_series(
    node_ids: pd.Series,
    go_map: dict[str, str],
    default_unknown: str = "Unknown",
) -> pd.Series:
    node_ids = node_ids.fillna("")
    base = _base_class_from_node_id(node_ids)
    is_go = node_ids.str.startswith("GO::")
    if go_map:
        mapped = node_ids.map(go_map)
        base = base.where(~is_go, mapped.fillna("GO"))
    else:
        base = base.where(~is_go, "GO")
    return base.replace({"": default_unknown}).fillna(default_unknown)


def _compute_node_class_counts(df: pd.DataFrame, go_map: dict[str, str]) -> pd.Series:
    nodes = pd.unique(pd.concat([df["head"], df["tail"]], ignore_index=True).astype(str))
    node_series = pd.Series(nodes, name="node")
    classes = _node_class_series(node_series, go_map)
    return classes.value_counts().sort_values(ascending=False)


def _compute_schema_edge_counts(df: pd.DataFrame, go_map: dict[str, str]) -> pd.DataFrame:
    head_class = _node_class_series(df["head"], go_map)
    tail_class = _node_class_series(df["tail"], go_map)

    # For GO edges, the interaction directly encodes BP/MF/CC; prefer that when available.
    interaction_to_go = {
        "BiologicalProcess": "GO:BP",
        "MolecularFunction": "GO:MF",
        "CellularComponent": "GO:CC",
    }
    go_tail_override = df["interaction"].map(interaction_to_go)
    is_go_onto_edge = (df["type"] == "ExtGene-GeneOnthology") & df["tail"].fillna("").str.startswith("GO::")
    tail_class = tail_class.where(~is_go_onto_edge, go_tail_override.fillna(tail_class))

    edge_counts = (
        pd.DataFrame({"head_class": head_class, "tail_class": tail_class})
        .groupby(["head_class", "tail_class"], dropna=False)
        .size()
        .rename("edge_count")
        .reset_index()
        .sort_values("edge_count", ascending=False)
    )
    return edge_counts


def _plot_bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    top_n: int | None = None,
    logy: bool = False,
) -> None:
    if top_n is not None:
        series = series.head(top_n)

    plt.figure(figsize=(10, max(4, 0.35 * len(series))))
    sns.barplot(x=series.values, y=series.index, orient="h", color=sns.color_palette()[0])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.xscale("log")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def _draw_schema(
    node_class_counts: pd.Series,
    edge_counts: pd.DataFrame,
    outpath: Path,
    title: str = "PathogenKG schema (class-level metagraph)",
) -> None:
    g = nx.DiGraph()

    for cls, count in node_class_counts.items():
        g.add_node(str(cls), count=int(count))

    for _, row in edge_counts.iterrows():
        u = str(row["head_class"])
        v = str(row["tail_class"])
        w = int(row["edge_count"])
        if u == "Unknown" or v == "Unknown":
            # keep the schema readable; unknowns are usually artifacts
            continue
        if g.has_edge(u, v):
            g[u][v]["weight"] += w
        else:
            g.add_edge(u, v, weight=w)

    # Node sizes proportional to instance counts, scaled for visibility.
    counts = np.array([g.nodes[n].get("count", 1) for n in g.nodes], dtype=float)
    if len(counts) == 0:
        raise ValueError("Schema graph has no nodes.")
    min_size, max_size = 800.0, 8000.0
    if counts.max() == counts.min():
        sizes = np.full_like(counts, (min_size + max_size) / 2.0)
    else:
        sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min()) * (max_size - min_size)

    # Edge widths scaled by log(weight).
    weights = np.array([g[u][v]["weight"] for u, v in g.edges], dtype=float)
    if len(weights) == 0:
        widths = []
    else:
        widths = 0.5 + (np.log10(weights + 1.0) / np.log10(weights.max() + 1.0)) * 6.0

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(g, seed=42, k=0.9)

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=sizes,
        node_color=sns.color_palette("deep", n_colors=1)[0],
        alpha=0.90,
        linewidths=1.0,
        edgecolors="white",
    )
    nx.draw_networkx_labels(g, pos, font_size=10, font_color="white")
    nx.draw_networkx_edges(
        g,
        pos,
        width=widths,
        alpha=0.55,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.15",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PathogenKG statistics and schema")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="Path to PathogenKG TSV edge list")
    parser.add_argument("--outdir", type=Path, default=Path("figures"), help="Output directory for figures")
    parser.add_argument(
        "--topn",
        type=int,
        default=25,
        help="Top-N bars for categorical plots (interaction/type/node-class)",
    )
    args = parser.parse_args()

    _ensure_outdir(args.outdir)
    sns.set_style("whitegrid")

    df = _load_kg(args.tsv)
    go_map, unique_go_terms = _infer_go_subontology(df)

    # Core counts
    node_class_counts = _compute_node_class_counts(df, go_map)
    type_counts = df["type"].value_counts()
    interaction_counts = df["interaction"].value_counts()
    source_counts = df["source"].value_counts()

    # Save summary plots
    _plot_bar(
        node_class_counts,
        title="Node instances by class",
        xlabel="# unique nodes",
        ylabel="Class",
        outpath=args.outdir / "node_class_counts.png",
        top_n=None,
    )
    _plot_bar(
        type_counts,
        title="Edges by high-level type",
        xlabel="# edges",
        ylabel="Type",
        outpath=args.outdir / "edge_type_counts.png",
        top_n=args.topn,
        logy=True,
    )
    _plot_bar(
        interaction_counts,
        title="Edges by interaction (predicate)",
        xlabel="# edges",
        ylabel="Interaction",
        outpath=args.outdir / "interaction_counts.png",
        top_n=args.topn,
        logy=True,
    )
    _plot_bar(
        source_counts,
        title="Edges by source",
        xlabel="# edges",
        ylabel="Source",
        outpath=args.outdir / "edge_source_counts.png",
        top_n=None,
        logy=True,
    )
    if not unique_go_terms.empty:
        _plot_bar(
            unique_go_terms,
            title="Unique GO terms used (inferred sub-ontology)",
            xlabel="# unique GO nodes",
            ylabel="GO class",
            outpath=args.outdir / "go_terms_by_ontology.png",
            top_n=None,
        )

    # Schema plot
    edge_counts = _compute_schema_edge_counts(df, go_map)
    _draw_schema(
        node_class_counts=node_class_counts,
        edge_counts=edge_counts,
        outpath=args.outdir / "kg_schema.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
