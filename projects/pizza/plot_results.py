"""Generate pizza trend-prediction visualizations from a ranked_test.csv run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import patheffects

plt.rcParams["font.family"] = "DejaVu Sans"


def latest_run(project_dir: Path) -> Path:
    runs = sorted((project_dir / "runs").glob("run_*"))
    if not runs:
        raise SystemExit("No runs found. Run `alphapy` first.")
    return runs[-1]


CUISINE_COLORS = {
    "indian":            "#e8522f",
    "korean":            "#d4371a",
    "korean_fusion":     "#c22d3a",
    "japanese":          "#7a4f8a",
    "japanese_american": "#5f3f79",
    "chinese":           "#b03f3f",
    "sichuan":           "#a12c2c",
    "thai":              "#4b8b4f",
    "vietnamese":        "#5a9a5e",
    "filipino":          "#d17c1f",
    "southeast_asian":   "#3f8a6f",
    "fusion":            "#c06b8a",
    "italian":           "#2e7d32",
    "italian_american":  "#3e8e41",
    "italian_med":       "#5a9440",
    "italian_calabrian": "#8a3d2a",
    "french":            "#3a5f9a",
    "french_italian":    "#4a6ea6",
    "mediterranean":     "#6a8d3a",
    "levantine":         "#9a8a3a",
    "greek":             "#1b5e91",
    "spanish":           "#c94a27",
    "turkish":           "#9a3a5a",
    "persian":           "#6a4d8a",
    "moroccan":          "#b76a3a",
    "north_african":     "#a85a2a",
    "balkan":            "#8a6a4a",
    "american":          "#1f5fb3",
    "american_south":    "#1c4e8a",
    "dutch_american":    "#2d6fa6",
    "nordic":            "#6aa8d4",
    "peruvian":          "#b8862a",
    "ethiopian":         "#7a3a2a",
    "mexican":           "#c44a1f",
    "mexican_american":  "#b8441f",
    "indo_chinese":      "#d44a4a",
    "argentinian":       "#4a9ac4",
    "modern":            "#6e7a8a",
    "modern_plant":      "#4a7a5a",
    "hawaiian":          "#d4a028",
    "new_haven":         "#264d73",
    "german":            "#8a8a4a",
    "sardinian":         "#3a6a8a",
    "european":          "#556677",
    "asian":             "#9a4a7a",
    "american_deli":     "#3a6fb3",
}


def color_for(cuisine: str) -> str:
    return CUISINE_COLORS.get(cuisine, "#888888")


def plot_top_overall(df: pl.DataFrame, out: Path, n: int = 20) -> None:
    top = df.sort("pred_test_xgrk", descending=True).head(n)
    names = top["name"].to_list()[::-1]
    scores = top["pred_test_xgrk"].to_list()[::-1]
    cuisines = top["cuisine"].to_list()[::-1]
    colors = [color_for(c) for c in cuisines]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    bars = ax.barh(names, scores, color=colors, edgecolor="#2a2a2a")
    for bar, score, cuisine in zip(bars, scores, cuisines):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}  ·  {cuisine}",
            va="center", ha="left", fontsize=9, color="#e0e0e0",
        )

    ax.set_title(
        f"Top {n} Emerging & Undiscovered Pizza Toppings\n"
        "Predicted trend score — XGBoost pairwise ranker",
        color="#ffcc33", fontsize=14, pad=18, weight="bold",
    )
    ax.set_xlabel("Predicted trend score (higher = more breakout potential)",
                  color="#cccccc", fontsize=10)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(axis="x", linestyle=":", color="#3a3a3a", alpha=0.6)
    ax.set_xlim(0, max(scores) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_category_leaders(df: pl.DataFrame, out: Path, per_cat: int = 3) -> None:
    cats = sorted(df["category"].unique().to_list())
    n = len(cats)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows))
    fig.patch.set_facecolor("#1a1a1a")
    axes = axes.flatten()

    for ax, cat in zip(axes, cats):
        top = df.filter(pl.col("category") == cat).sort("pred_test_xgrk", descending=True).head(per_cat)
        names = top["name"].to_list()[::-1]
        scores = top["pred_test_xgrk"].to_list()[::-1]
        cuisines = top["cuisine"].to_list()[::-1]
        colors = [color_for(c) for c in cuisines]

        ax.set_facecolor("#1a1a1a")
        bars = ax.barh(names, scores, color=colors, edgecolor="#2a2a2a")
        for bar, score, cuisine in zip(bars, scores, cuisines):
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f} · {cuisine}",
                va="center", ha="left", fontsize=8, color="#e0e0e0",
            )
        ax.set_title(cat.upper(), color="#ffcc33", fontsize=11, weight="bold", loc="left")
        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(axis="x", linestyle=":", color="#3a3a3a", alpha=0.5)
        if scores:
            ax.set_xlim(0, max(scores) * 1.4)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Category leaders — emerging pizza toppings",
                 color="#ffcc33", fontsize=15, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_flavor_radar(df: pl.DataFrame, out: Path, n: int = 6) -> None:
    axes_names = ["sweet", "salt", "umami", "heat", "acid", "bitter", "fat"]
    top = df.sort("pred_test_xgrk", descending=True).head(n)

    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    palette = plt.cm.plasma(np.linspace(0.15, 0.85, n))
    for i, row in enumerate(top.iter_rows(named=True)):
        values = [row[a] for a in axes_names]
        values += values[:1]
        color = palette[i]
        ax.plot(angles, values, color=color, linewidth=2,
                label=f"{row['name']} ({row['pred_test_xgrk']:.2f})")
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a.upper() for a in axes_names], color="#e0e0e0", fontsize=11)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#888888", fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(color="#444444", alpha=0.6)
    ax.spines["polar"].set_color("#444444")

    title = ax.set_title(f"Flavor profiles of top {n} predicted breakout toppings",
                         color="#ffcc33", fontsize=14, weight="bold", pad=28)
    title.set_path_effects([patheffects.withStroke(linewidth=2, foreground="#1a1a1a")])

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05),
                       fontsize=9, facecolor="#222222", edgecolor="#444444")
    for text in legend.get_texts():
        text.set_color("#e0e0e0")

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


FAMILY_COLORS = {
    "spicy":            "#d4371a",
    "umami_bomb":       "#7a4f8a",
    "sweet":            "#e8a838",
    "bright_acidic":    "#4a9ac4",
    "herbal":           "#4b8b4f",
    "savory":           "#8a6a4a",
    "smoky":            "#6a3f2a",
    "pungent":          "#9a3a5a",
    "funky_fermented":  "#c06b8a",
}


def plot_score_by_family(df: pl.DataFrame, out: Path) -> None:
    families = sorted(df["flavor_family"].unique().to_list(),
                      key=lambda f: df.filter(pl.col("flavor_family") == f)["pred_test_xgrk"].mean(),
                      reverse=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    rng = np.random.default_rng(42)

    for i, fam in enumerate(families):
        scores = df.filter(pl.col("flavor_family") == fam)["pred_test_xgrk"].to_list()
        color = FAMILY_COLORS.get(fam, "#888888")
        jitter = rng.uniform(-0.18, 0.18, size=len(scores))
        ax.scatter([i + j for j in jitter], scores,
                   color=color, s=70, alpha=0.85, edgecolor="#1a1a1a", linewidth=0.5, zorder=3)
        if len(scores) >= 2:
            bp = ax.boxplot(scores, positions=[i], widths=0.5, patch_artist=True,
                            showfliers=False, zorder=2)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.25)
                patch.set_edgecolor(color)
            for line in bp["whiskers"] + bp["caps"] + bp["medians"]:
                line.set_color(color)
        mean_score = float(np.mean(scores))
        ax.text(i, max(scores) + 0.12, f"μ={mean_score:.2f}",
                ha="center", fontsize=8, color="#cccccc")

    # Annotate top 2 in each family with name
    for i, fam in enumerate(families):
        sub = df.filter(pl.col("flavor_family") == fam).sort("pred_test_xgrk", descending=True).head(2)
        for j, r in enumerate(sub.iter_rows(named=True)):
            ax.annotate(r["name"],
                        xy=(i, r["pred_test_xgrk"]),
                        xytext=(8, -2 - j * 10), textcoords="offset points",
                        fontsize=7, color="#e0e0e0", alpha=0.9)

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, color="#cccccc", fontsize=10, rotation=20, ha="right")
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(axis="y", linestyle=":", color="#3a3a3a", alpha=0.6)
    ax.set_ylabel("Predicted trend score", color="#cccccc", fontsize=11)
    ax.set_title("Predicted trend score by flavor family",
                 color="#ffcc33", fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cuisine_momentum(df: pl.DataFrame, out: Path) -> None:
    agg = (df.group_by("cuisine")
             .agg([pl.col("pred_test_xgrk").mean().alias("mean_score"),
                   pl.col("pred_test_xgrk").count().alias("n")])
             .filter(pl.col("n") >= 1)
             .sort("mean_score", descending=True))

    cuisines = agg["cuisine"].to_list()[::-1]
    means = agg["mean_score"].to_list()[::-1]
    counts = agg["n"].to_list()[::-1]
    colors = [color_for(c) for c in cuisines]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(cuisines))))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    bars = ax.barh(cuisines, means, color=colors, edgecolor="#2a2a2a")
    for bar, mean, n in zip(bars, means, counts):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{mean:.2f}  (n={n})",
                va="center", ha="left", fontsize=8, color="#e0e0e0")

    ax.set_title("Cuisine momentum — mean predicted trend score",
                 color="#ffcc33", fontsize=13, weight="bold", pad=14)
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(axis="x", linestyle=":", color="#3a3a3a", alpha=0.5)
    ax.set_xlim(0, max(means) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", default=None, help="Run directory (defaults to latest)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    run_dir = (project_dir / "runs" / args.rundir) if args.rundir else latest_run(project_dir)
    ranked = run_dir / "output" / "ranked_test.csv"
    if not ranked.exists():
        raise SystemExit(f"Missing {ranked}")

    df = pl.read_csv(ranked)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_top_overall(df,         plots_dir / "top20_emerging.png")
    plot_category_leaders(df,    plots_dir / "category_leaders.png")
    plot_flavor_radar(df,        plots_dir / "flavor_radar_top6.png")
    plot_cuisine_momentum(df,    plots_dir / "cuisine_momentum.png")
    plot_score_by_family(df,     plots_dir / "score_by_flavor_family.png")

    print(f"Wrote 5 plots to {plots_dir}")


if __name__ == "__main__":
    main()
