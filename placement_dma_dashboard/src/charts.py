"""Matplotlib figures for the Placement DMA dashboard.

Every function returns a ``matplotlib.figure.Figure`` (or ``None`` when the data
needed for the chart is not available) so the app can simply skip it.
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

PLACED_C = "#0F766E"      # teal
NOT_PLACED_C = "#D97706"  # amber
ACCENT = "#1D4ED8"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#7A5195", "#8C8C8C"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#F3F4F6", "#99D5CF", "#0F766E"])
DIV = LinearSegmentedColormap.from_list("div", ["#B45309", "#F9FAFB", "#0F766E"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
})


def _fig(w=6.2, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    return fig, ax


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _wrap(text, width=34, lines=2):
    parts = textwrap.wrap(str(text), width=width)
    if len(parts) > lines:
        parts = parts[:lines]
        parts[-1] = parts[-1][: max(0, width - 1)] + "…"
    return "\n".join(parts)


def _has(df, *cols):
    return all(c in df.columns for c in cols)


def _outcome_df(df):
    return df.dropna(subset=["placed"]) if "placed" in df.columns else df.iloc[0:0]


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def outcome_bar(df):
    d = _outcome_df(df)
    if d.empty:
        return None
    counts = d["placed"].value_counts().reindex([1, 0]).fillna(0).astype(int)
    fig, ax = _fig(5.2, 3.4)
    bars = ax.bar(["Placed", "Not placed"], counts.values, color=[PLACED_C, NOT_PLACED_C], width=0.55)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({value / total:.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Students")
    ax.set_title("Placement outcome")
    _grid(ax)
    fig.tight_layout()
    return fig


def cgpa_distribution(df):
    d = _outcome_df(df).dropna(subset=["cgpa"]) if _has(df, "cgpa") else None
    if d is None or d.empty:
        return None
    fig, ax = _fig(5.2, 3.4)
    bins = np.linspace(d["cgpa"].min(), d["cgpa"].max() + 1e-9, 15)
    for flag, label, color in ((0, "Not placed", NOT_PLACED_C), (1, "Placed", PLACED_C)):
        values = d.loc[d["placed"] == flag, "cgpa"]
        if len(values):
            ax.hist(values, bins=bins, alpha=0.65, color=color, label=f"{label} (n={len(values)})",
                    density=True, edgecolor="white")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Density")
    ax.set_title("CGPA distribution by outcome")
    ax.legend(frameon=False, fontsize=8.5)
    _grid(ax)
    fig.tight_layout()
    return fig


def cgpa_aptitude_scatter(df):
    if not _has(df, "cgpa", "aptitude_score", "placed"):
        return None
    d = _outcome_df(df).dropna(subset=["cgpa", "aptitude_score"])
    if d.empty:
        return None
    fig, ax = _fig(5.2, 3.4)
    for flag, label, color, z in ((0, "Not placed", NOT_PLACED_C, 1), (1, "Placed", PLACED_C, 2)):
        sub = d[d["placed"] == flag]
        ax.scatter(sub["cgpa"], sub["aptitude_score"], s=16, alpha=0.65, color=color, label=label,
                   edgecolor="white", linewidth=0.3, zorder=z)
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Aptitude score")
    ax.set_title("CGPA vs aptitude score")
    ax.legend(frameon=False, fontsize=8.5)
    _grid(ax, "both")
    fig.tight_layout()
    return fig


def placement_rate_panel(df):
    """Placement rate by internship, projects, backlogs and certifications."""
    d = _outcome_df(df)
    panels = [(c, t) for c, t in (("internship", "Internship"), ("projects", "Projects"),
                                  ("backlogs", "Backlogs"), ("certifications", "Certifications"))
              if c in d.columns and d[c].notna().any()]
    if d.empty or not panels:
        return None
    overall = d["placed"].mean()
    cols = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols / 1.0, 2.9 * rows), dpi=110, squeeze=False)
    for ax, (col, title) in zip(axes.ravel(), panels):
        g = d.dropna(subset=[col]).groupby(col)["placed"].agg(["mean", "size"])
        labels = [("Yes" if i == 1 else "No") if col == "internship" else f"{int(i)}" for i in g.index]
        bars = ax.bar(labels, g["mean"] * 100, color=PLACED_C, width=0.6)
        for bar, (_, row) in zip(bars, g.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['mean']:.0%}\nn={int(row['size'])}",
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)
        ax.axhline(overall * 100, color=NOT_PLACED_C, linestyle="--", linewidth=1)
        ax.set_ylim(0, max(g["mean"].max() * 100 * 1.4, overall * 100 * 1.4, 10))
        ax.set_title(f"Placement rate by {title.lower()}")
        ax.set_ylabel("% placed")
        _grid(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.text(0.99, 0.005, f"dashed line = overall rate ({overall:.1%})", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def skills_chart(df, top=12):
    if not _has(df, "skills", "placed"):
        return None
    d = _outcome_df(df)
    rows = []
    for skills, placed in zip(d["skills"], d["placed"]):
        for skill in {s.strip() for s in str(skills).replace(",", ";").split(";") if s.strip() and s.strip().lower() != "nan"}:
            rows.append((skill, placed))
    if not rows:
        return None
    s = pd.DataFrame(rows, columns=["skill", "placed"]).groupby("skill")["placed"].agg(["mean", "size"])
    s = s.sort_values("size", ascending=False).head(top).sort_values("mean")
    fig, ax = _fig(6.2, 0.33 * len(s) + 1.3)
    bars = ax.barh(s.index, s["mean"] * 100, color=PLACED_C, height=0.62)
    overall = d["placed"].mean() * 100
    for bar, (_, row) in zip(bars, s.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{row['mean']:.1%}  (n={int(row['size'])})",
                va="center", fontsize=8, color=MUTED)
    ax.axvline(overall, color=NOT_PLACED_C, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(s["mean"].max() * 100 * 1.35, overall * 1.35))
    ax.set_xlabel(f"% placed among students with the skill (dashed = overall {overall:.1f}%)")
    ax.set_title("Placement rate by skill (most common skills)")
    _grid(ax, "x")
    fig.tight_layout()
    return fig


def correlation_heatmap(df):
    cols = [c for c in ["cgpa", "aptitude_score", "communication_score", "projects", "certifications",
                        "backlogs", "internship", "placed", "salary_lpa"] if c in df.columns]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    n = len(corr)
    fig, ax = _fig(max(5.0, 0.62 * n + 2), max(4.0, 0.55 * n + 1.6))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(n), corr.columns)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Correlation matrix")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    return fig


def salary_histogram(df):
    if not _has(df, "salary_lpa"):
        return None
    s = pd.to_numeric(df["salary_lpa"], errors="coerce").dropna()
    if "placed" in df.columns:
        s = s[df.loc[s.index, "placed"] == 1]
    if s.empty:
        return None
    fig, ax = _fig(5.2, 3.4)
    ax.hist(s, bins=15, color=PLACED_C, edgecolor="white", alpha=0.9)
    ax.axvline(s.mean(), color=NOT_PLACED_C, linestyle="--", linewidth=1.3, label=f"Mean {s.mean():.2f} LPA")
    ax.axvline(s.median(), color=ACCENT, linestyle=":", linewidth=1.3, label=f"Median {s.median():.2f} LPA")
    ax.set_xlabel("Salary (LPA)")
    ax.set_ylabel("Placed students")
    ax.set_title("Salary distribution (placed students)")
    ax.legend(frameon=False, fontsize=8.5)
    _grid(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Association rules
# --------------------------------------------------------------------------- #
def top_itemsets(itemsets: pd.DataFrame, top=15):
    if itemsets is None or itemsets.empty:
        return None
    s = itemsets.head(top).iloc[::-1]
    fig, ax = _fig(6.4, 0.34 * len(s) + 1.3)
    colors = [PLACED_C if "Placement_Placed" in i else ACCENT for i in s["itemset"]]
    ax.barh([_wrap(i, 46, 1) for i in s["itemset"]], s["support"], color=colors, height=0.65)
    for y, v in enumerate(s["support"]):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, s["support"].max() * 1.12)
    ax.set_xlabel("Support")
    ax.set_title(f"Top {len(s)} frequent itemsets")
    _grid(ax, "x")
    fig.tight_layout()
    return fig


def rule_scatter(rules: pd.DataFrame, max_points=4000):
    if rules is None or rules.empty:
        return None
    r = rules.sample(max_points, random_state=0) if len(rules) > max_points else rules
    fig, ax = _fig(6.0, 4.0)
    sc = ax.scatter(r["support"], r["confidence"], c=r["lift"], cmap="viridis", s=18, alpha=0.75,
                    edgecolor="none")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Support vs confidence ({len(r):,} rules)")
    _grid(ax, "both")
    fig.tight_layout()
    return fig


def top_rules(rules: pd.DataFrame, top=10):
    if rules is None or rules.empty:
        return None
    r = rules.sort_values(["lift", "confidence"], ascending=False).head(top).iloc[::-1]
    labels = [f"{_wrap(a, 34, 2)}\n→ {_wrap(c, 30, 1)}" for a, c in zip(r["antecedents"], r["consequents"])]
    fig, ax = _fig(7.0, 0.62 * len(r) + 1.4)
    bars = ax.barh(labels, r["lift"], color=ACCENT, height=0.62)
    ax.axvline(1, color=NOT_PLACED_C, linestyle="--", linewidth=1)
    for bar, (_, row) in zip(bars, r.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"lift {row['lift']:.2f} | conf {row['confidence']:.2f} | sup {row['support']:.2f}",
                va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, r["lift"].max() * 1.55)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("Lift (dashed line = independence)")
    ax.set_title(f"Top {len(r)} rules by lift")
    _grid(ax, "x")
    fig.tight_layout()
    return fig


def rule_heatmap(rules: pd.DataFrame, size=12):
    """Lift heat-map of single-item → single-item rules."""
    if rules is None or rules.empty:
        return None
    single = rules[~rules["antecedents"].str.contains(",") & ~rules["consequents"].str.contains(",")]
    if len(single) < 4:
        return None
    ants = single["antecedents"].value_counts().head(size).index
    cons = single["consequents"].value_counts().head(size).index
    pivot = single.pivot_table(index="antecedents", columns="consequents", values="lift", aggfunc="max")
    pivot = pivot.reindex(index=ants, columns=cons)
    if pivot.notna().sum().sum() < 4:
        return None
    fig, ax = _fig(max(5.8, 0.55 * pivot.shape[1] + 2.4), max(4.0, 0.42 * pivot.shape[0] + 1.8))
    im = ax.imshow(pivot.values, cmap=SEQ, aspect="auto", vmin=min(1.0, np.nanmin(pivot.values)))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    mid = np.nanmax(pivot.values) * 0.6 + np.nanmin(pivot.values) * 0.4
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > mid else INK)
    ax.set_xlabel("Consequent")
    ax.set_ylabel("Antecedent")
    ax.set_title("Lift of single-item rules")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label("Lift")
    fig.tight_layout()
    return fig


def metric_distribution(rules: pd.DataFrame):
    """Histograms of confidence and lift for the discovered rules."""
    if rules is None or rules.empty:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=110)
    a1.hist(rules["confidence"], bins=20, color=ACCENT, edgecolor="white")
    a1.set_xlabel("Confidence"); a1.set_ylabel("Rules"); a1.set_title("Confidence")
    a2.hist(rules["lift"], bins=20, color=PLACED_C, edgecolor="white")
    a2.axvline(1, color=NOT_PLACED_C, linestyle="--", linewidth=1)
    a2.set_xlabel("Lift"); a2.set_title("Lift")
    _grid(a1); _grid(a2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# K-Means
# --------------------------------------------------------------------------- #
def cluster_scatter(projection: pd.DataFrame, ratio):
    if projection is None or projection.empty:
        return None
    fig, ax = _fig(6.0, 4.0)
    for i, cluster in enumerate(sorted(projection["cluster"].unique())):
        sub = projection[projection["cluster"] == cluster]
        ax.scatter(sub["pc1"], sub["pc2"], s=18, alpha=0.7, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   label=f"Cluster {cluster} (n={len(sub)})", edgecolor="white", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    ax.set_title("Clusters in PCA space")
    ax.legend(frameon=False, fontsize=8.5)
    _grid(ax, "both")
    fig.tight_layout()
    return fig


def elbow_chart(diag: pd.DataFrame, chosen_k=None):
    if diag is None or diag.empty:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=110)
    a1.plot(diag["k"], diag["inertia"], marker="o", color=ACCENT)
    a1.set_xlabel("K"); a1.set_ylabel("Inertia"); a1.set_title("Elbow method")
    a2.plot(diag["k"], diag["silhouette"], marker="o", color=PLACED_C)
    a2.set_xlabel("K"); a2.set_ylabel("Silhouette"); a2.set_title("Silhouette score")
    for ax in (a1, a2):
        _grid(ax, "both")
        ax.set_xticks(diag["k"])
        if chosen_k in set(diag["k"]):
            ax.axvline(chosen_k, color=NOT_PLACED_C, linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def cluster_profile_heatmap(summary: pd.DataFrame):
    if summary is None or summary.empty:
        return None
    feats = [c for c in summary.columns if c not in ("cluster", "students")]
    if not feats:
        return None
    values = summary[feats].to_numpy(dtype=float)
    std = values.std(axis=0)
    z = (values - values.mean(axis=0)) / np.where(std == 0, 1, std)
    fig, ax = _fig(max(5.4, 0.8 * len(feats) + 1.5), 0.62 * len(summary) + 1.8)
    im = ax.imshow(z, cmap=DIV, vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right")
    ax.set_yticks(range(len(summary)), [f"Cluster {c}\n(n={n})" for c, n in zip(summary["cluster"], summary["students"])])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(z[i, j]) > 1.1 else INK)
    ax.set_title("Cluster profiles (colour = relative to other clusters)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig


def cluster_size_rate(labels: pd.Series, placed: pd.Series | None):
    sizes = labels.value_counts().sort_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=110)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(sizes))]
    a1.bar([f"C{c}" for c in sizes.index], sizes.values, color=colors)
    for x, v in enumerate(sizes.values):
        a1.text(x, v, str(v), ha="center", va="bottom", fontsize=8.5)
    a1.set_title("Cluster sizes"); a1.set_ylabel("Students"); _grid(a1)
    if placed is not None and placed.notna().any():
        rate = placed.groupby(labels).mean().reindex(sizes.index) * 100
        a2.bar([f"C{c}" for c in rate.index], rate.values, color=colors)
        for x, v in enumerate(rate.values):
            a2.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5)
        a2.set_ylim(0, max(np.nanmax(rate.values) * 1.3, 10))
        a2.set_title("Placement rate by cluster"); a2.set_ylabel("% placed"); _grid(a2)
    else:
        a2.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Classification and regression
# --------------------------------------------------------------------------- #
def confusion_chart(cm):
    fig, ax = _fig(4.2, 3.6)
    cm = np.asarray(cm)
    ax.imshow(cm, cmap=SEQ)
    labels = ["Not placed", "Placed"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_title("Confusion matrix (test set)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig


def importance_chart(importance: pd.DataFrame, title="Feature importance"):
    if importance is None or importance.empty:
        return None
    s = importance.iloc[::-1]
    fig, ax = _fig(5.4, 0.38 * len(s) + 1.2)
    ax.barh(s["feature"], s["importance"], color=ACCENT, height=0.6)
    for y, v in enumerate(s["importance"]):
        ax.text(v + 0.003, y, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(s["importance"].max() * 1.15, 0.05))
    ax.set_xlabel("Importance")
    ax.set_title(title)
    _grid(ax, "x")
    fig.tight_layout()
    return fig


def roc_chart(roc):
    fpr, tpr, area = roc
    fig, ax = _fig(4.6, 3.6)
    ax.plot(fpr, tpr, color=PLACED_C, linewidth=2, label=f"Decision tree (AUC = {area:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _grid(ax, "both")
    fig.tight_layout()
    return fig


def tree_chart(model, feature_names):
    fig, ax = plt.subplots(figsize=(13, 6), dpi=100)
    plot_tree(model, feature_names=list(feature_names), class_names=["Not placed", "Placed"], filled=True,
              rounded=True, impurity=False, proportion=True, fontsize=7, ax=ax)
    ax.set_title("Fitted decision tree")
    fig.tight_layout()
    return fig


def actual_vs_predicted(actual, predicted):
    fig, ax = _fig(4.8, 3.8)
    ax.scatter(actual, predicted, s=28, color=PLACED_C, alpha=0.8, edgecolor="white", linewidth=0.4)
    lo, hi = min(np.min(actual), np.min(predicted)), max(np.max(actual), np.max(predicted))
    ax.plot([lo, hi], [lo, hi], color=NOT_PLACED_C, linestyle="--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual salary (LPA)"); ax.set_ylabel("Predicted salary (LPA)")
    ax.set_title("Actual vs predicted (test set)")
    ax.legend(frameon=False, fontsize=8.5)
    _grid(ax, "both")
    fig.tight_layout()
    return fig


def salary_vs_cgpa(df):
    if not _has(df, "cgpa", "salary_lpa", "placed"):
        return None
    d = df[(df["placed"] == 1)].dropna(subset=["cgpa", "salary_lpa"])
    if d.empty:
        return None
    fig, ax = _fig(5.0, 3.8)
    if "internship" in d.columns:
        for flag, label, color in ((0, "No internship", NOT_PLACED_C), (1, "Internship", PLACED_C)):
            sub = d[d["internship"] == flag]
            ax.scatter(sub["cgpa"], sub["salary_lpa"], s=26, alpha=0.8, color=color, label=label,
                       edgecolor="white", linewidth=0.4)
        ax.legend(frameon=False, fontsize=8.5)
    else:
        ax.scatter(d["cgpa"], d["salary_lpa"], s=26, alpha=0.8, color=PLACED_C)
    if len(d) > 2:
        slope, intercept = np.polyfit(d["cgpa"], d["salary_lpa"], 1)
        xs = np.array([d["cgpa"].min(), d["cgpa"].max()])
        ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.2, linestyle="--")
    ax.set_xlabel("CGPA"); ax.set_ylabel("Salary (LPA)")
    ax.set_title("Salary vs CGPA (placed students)")
    _grid(ax, "both")
    fig.tight_layout()
    return fig
